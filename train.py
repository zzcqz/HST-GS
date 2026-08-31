#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import time
import uuid
import torch
from random import randint
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except ImportError:
    SPARSE_ADAM_AVAILABLE = False

try:
    from lpipsPyTorch.modules.lpips import LPIPS
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS = None
    LPIPS_AVAILABLE = False


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m}m{s}s"
    return f"{m}m{s}s"


def _print_and_save_training_timings(scene, lines):
    text = "\n".join(lines)
    print("\n" + text)
    path = os.path.join(scene.model_path, "training_timings.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print("[Timing log written to {}]".format(path))
    except OSError as e:
        print("[Could not write training_timings.txt: {}]".format(e))


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    fast_train=False,
    disable_tensorboard=False,
    sr=False,
    sr_refine_stride=2,
    sr_stride4_after=5000,
    sr_stride6_after=9000,
    sr_full_fb_window=0,
    hts_stats_interval=0,
    sr_log_interval=0,
    metrics_curve_csv="",
    video_render_every=0,
    video_out_dir="",
):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            "sparse_adam requires SparseGaussianAdam in diff_gaussian_rasterization; "
            "install the matching rasterizer build."
        )

    t_setup0 = time.perf_counter()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, disable_tensorboard=disable_tensorboard)

    hts_stats_file = None
    hts_stats_reset = hts_stats_st_cand = hts_stats_st_kept = None
    if hts_stats_interval and hts_stats_interval > 0:
        try:
            from diff_gaussian_rasterization import (
                hts_stats_reset,
                hts_stats_st_cand,
                hts_stats_st_kept,
            )
        except Exception as e:
            sys.exit("hts_stats_interval requires rebuilt rasterizer with HTS stats: {}".format(e))
        stats_path = os.path.join(dataset.model_path, "hts_supertile_stats.csv")
        hts_stats_file = open(stats_path, "w")
        hts_stats_file.write("iteration,st_cand,st_kept,cull_ratio\n")
        hts_stats_file.flush()
        print("[HTS] logging Super-Tile stats every {} iters -> {}".format(hts_stats_interval, stats_path))

    sr_log_file = None
    if sr_log_interval and sr_log_interval > 0:
        sr_log_path = os.path.join(dataset.model_path, "sr_schedule.csv")
        sr_log_file = open(sr_log_path, "w")
        sr_log_file.write("iter,refine_stride,skip_fb,in_full_fb,phase\n")
        sr_log_file.flush()
        print("[SR] logging schedule every {} iters -> {}".format(sr_log_interval, sr_log_path))

    metrics_curve_path = (metrics_curve_csv or "").strip()
    lpips_model = None
    if metrics_curve_path:
        if not LPIPS_AVAILABLE:
            print("WARNING: lpipsPyTorch unavailable; metrics curve will omit LPIPS")
        else:
            lpips_model = LPIPS("vgg", "0.1").cuda()
        os.makedirs(os.path.dirname(metrics_curve_path) or ".", exist_ok=True)
        with open(metrics_curve_path, "w") as f:
            f.write("iter,PSNR,SSIM,LPIPS\n")

    # Video frames: render a fixed test camera in-loop at scheduled iterations.
    video_out_dir = (video_out_dir or "").strip()
    video_frame_iters = set()
    video_camera = None
    video_times_file = None
    if video_out_dir:
        os.makedirs(video_out_dir, exist_ok=True)
        video_frame_iters = {it for it in testing_iterations if it > 0}
        if video_render_every and video_render_every > 0:
            video_frame_iters |= set(range(video_render_every, opt.iterations + 1, video_render_every))
        video_frame_iters.add(opt.iterations)
        video_times_file = open(os.path.join(video_out_dir, "frame_times.csv"), "w")
        video_times_file.write("iter,wall_sec\n")
        video_times_file.flush()
        print("[VIDEO] rendering fixed test view at {} iterations -> {}".format(
            len(video_frame_iters), video_out_dir))

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(
        dataset,
        gaussians,
        load_test_cameras=not (fast_train and sr and not metrics_curve_path),
    )
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    setup_wall_s = time.perf_counter() - t_setup0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0

    # FastGS-style: fixed background tensor reused each iter unless random_background.
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    log_train_scalars = not fast_train
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    sum_fb_gpu_ms = 0.0
    sum_optim_gpu_ms = 0.0
    sum_forward_loss_bwd_wall_s = 0.0
    sum_no_grad_wall_s = 0.0
    sum_log_eval_wall_s = 0.0
    sum_save_ply_wall_s = 0.0
    sum_densify_wall_s = 0.0
    sum_misc_print_wall_s = 0.0
    sum_optimizer_wall_s = 0.0
    sum_checkpoint_wall_s = 0.0

    if sr:
        print(
            "[SR] growth until {}; refine_stride={}; full_fb_window={}".format(
                opt.densify_until_iter, sr_refine_stride, sr_full_fb_window
            )
        )

    t_loop0 = time.perf_counter()
    sum_refine_skip_iters = 0
    for iteration in range(first_iter, opt.iterations + 1):
        if opt.random_background:
            bg = torch.rand((3), device="cuda")

        gaussians.update_learning_rate(iteration)

        refine_stride = sr_refine_stride
        if sr and iteration > opt.densify_until_iter + sr_stride4_after:
            refine_stride = max(refine_stride, 4)
        if sr and iteration > opt.densify_until_iter + sr_stride6_after:
            refine_stride = max(refine_stride, 6)

        in_full_fb = (
            sr
            and sr_full_fb_window > 0
            and opt.densify_until_iter < iteration <= opt.densify_until_iter + sr_full_fb_window
        )
        skip_fb = (
            sr
            and iteration > opt.densify_until_iter
            and not in_full_fb
            and refine_stride > 1
            and iteration % refine_stride != 0
        )
        if sr_log_file and (iteration % sr_log_interval == 0 or iteration == 1):
            phase = "growth" if iteration <= opt.densify_until_iter else (
                "full_fb" if in_full_fb else "refine"
            )
            sr_log_file.write(
                "{},{},{},{},{}\n".format(
                    iteration, refine_stride, int(skip_fb), int(in_full_fb), phase
                )
            )
            sr_log_file.flush()
        if skip_fb:
            sum_refine_skip_iters += 1
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            # SR skip must not drop scheduled/final PLY or checkpoints.
            with torch.no_grad():
                if iteration in saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                if iteration in checkpoint_iterations:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save(
                        (gaussians.capture(), iteration),
                        scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                    )
            if iteration == opt.iterations:
                progress_bar.close()
            continue

        t_fwd0 = time.perf_counter()
        iter_start.record()

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        collect_hts = (
            hts_stats_file is not None
            and hts_stats_interval > 0
            and (iteration % hts_stats_interval == 0 or iteration == 1 or iteration == opt.iterations)
        )
        if collect_hts:
            os.environ["HSTGS_HTS_STATS"] = "1"
            hts_stats_reset()
        else:
            os.environ.pop("HSTGS_HTS_STATS", None)

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            separate_sh=use_sparse_adam,
        )
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if collect_hts:
            cand = hts_stats_st_cand()
            kept = hts_stats_st_kept()
            cull = 0.0 if cand == 0 else 1.0 - (kept / float(cand))
            hts_stats_file.write("{},{},{},{:.6f}\n".format(iteration, cand, kept, cull))
            hts_stats_file.flush()
            os.environ.pop("HSTGS_HTS_STATS", None)

        if viewpoint_cam.alpha_mask is not None:
            image = image * viewpoint_cam.alpha_mask.cuda()

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        loss.backward()
        torch.cuda.synchronize()
        iter_end.record()
        torch.cuda.synchronize()
        iter_ms = float(iter_start.elapsed_time(iter_end))
        sum_fb_gpu_ms += iter_ms
        sum_forward_loss_bwd_wall_s += time.perf_counter() - t_fwd0

        t_ng0 = time.perf_counter()
        with torch.no_grad():
            t0 = time.perf_counter()
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if video_times_file is not None and iteration in video_frame_iters:
                if video_camera is None:
                    test_cams = scene.getTestCameras()
                    video_camera = test_cams[0] if test_cams else None
                if video_camera is not None:
                    frame = torch.clamp(
                        render(
                            video_camera,
                            gaussians,
                            pipe,
                            background,
                            1.0,
                            use_sparse_adam,
                            None,
                        )["render"],
                        0.0,
                        1.0,
                    )
                    from torchvision.utils import save_image as _save_image
                    _save_image(
                        frame,
                        os.path.join(video_out_dir, "frame_{:05d}.png".format(iteration)),
                    )
                wall_sec = time.perf_counter() - t_loop0
                video_times_file.write("{},{:.6f}\n".format(iteration, wall_sec))
                video_times_file.flush()

            eval_iters = (
                testing_iterations
                if (not fast_train or metrics_curve_path)
                else []
            )
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_ms,
                eval_iters,
                scene,
                render,
                (
                    pipe,
                    background,
                    1.0,
                    use_sparse_adam,
                    None,
                ),
                log_train_scalars=log_train_scalars,
                metrics_curve_csv=metrics_curve_path,
                lpips_model=lpips_model,
            )
            sum_log_eval_wall_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            sum_save_ply_wall_s += time.perf_counter() - t0

            optim_start.record()
            t0 = time.perf_counter()
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 25 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                        iteration=iteration,
                    )

                if not fast_train and iteration % 500 == 0:
                    gaussians.reduce_opacity()

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()
            sum_densify_wall_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            if not fast_train and iteration % 500 == 0:
                print("Gaussian number: {}".format(gaussians.get_xyz.shape[0]))
            sum_misc_print_wall_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
            sum_optimizer_wall_s += time.perf_counter() - t0

            optim_end.record()
            torch.cuda.synchronize()
            sum_optim_gpu_ms += float(optim_start.elapsed_time(optim_end))

            t0 = time.perf_counter()
            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
            sum_checkpoint_wall_s += time.perf_counter() - t0

        sum_no_grad_wall_s += time.perf_counter() - t_ng0

    if hts_stats_file is not None:
        hts_stats_file.close()
    if sr_log_file is not None:
        sr_log_file.close()
    if video_times_file is not None:
        video_times_file.close()

    torch.cuda.synchronize()
    loop_wall_s = time.perf_counter() - t_loop0
    sum_fb_gpu_s = sum_fb_gpu_ms / 1000.0
    sum_optim_gpu_s = sum_optim_gpu_ms / 1000.0
    fastgs_training_s = sum_fb_gpu_s + sum_optim_gpu_s
    accounted_loop_s = sum_forward_loss_bwd_wall_s + sum_no_grad_wall_s
    overhead_loop_s = max(0.0, loop_wall_s - accounted_loop_s)
    sub_no_grad_s = (
        sum_log_eval_wall_s
        + sum_save_ply_wall_s
        + sum_densify_wall_s
        + sum_misc_print_wall_s
        + sum_optimizer_wall_s
        + sum_checkpoint_wall_s
    )
    no_grad_gap_s = max(0.0, sum_no_grad_wall_s - sub_no_grad_s)

    lines = [
        "========== Training timing summary ==========",
        "  (wall = CPU timer; CUDA FB = GPU event sum over iterations)",
        "  Setup (Scene + Gaussians + checkpoint restore):  {:>10}  ({:.2f} s)".format(
            _format_duration(setup_wall_s), setup_wall_s
        ),
        "  Total training loop (wall):                       {:>10}  ({:.2f} s)".format(
            _format_duration(loop_wall_s), loop_wall_s
        ),
        "  --- Per-iteration sums ---",
        "  Forward + loss + backward (wall):                 {:>10}  ({:.2f} s)".format(
            _format_duration(sum_forward_loss_bwd_wall_s), sum_forward_loss_bwd_wall_s
        ),
        "  Forward + backward (CUDA events, sum):          {:>10}  ({:.2f} s)".format(
            _format_duration(sum_fb_gpu_s), sum_fb_gpu_s
        ),
        "  Post-backward block (wall, whole no_grad):        {:>10}  ({:.2f} s)".format(
            _format_duration(sum_no_grad_wall_s), sum_no_grad_wall_s
        ),
        "  --- Inside post-backward ---",
        "    Progress + TensorBoard + test eval:             {:>10}  ({:.2f} s)".format(
            _format_duration(sum_log_eval_wall_s), sum_log_eval_wall_s
        ),
        "    Save PLY (save_iterations):                     {:>10}  ({:.2f} s)".format(
            _format_duration(sum_save_ply_wall_s), sum_save_ply_wall_s
        ),
        "    Densify / prune / opacity hooks:                {:>10}  ({:.2f} s)".format(
            _format_duration(sum_densify_wall_s), sum_densify_wall_s
        ),
        "    Gaussian count prints:                          {:>10}  ({:.2f} s)".format(
            _format_duration(sum_misc_print_wall_s), sum_misc_print_wall_s
        ),
        "    Optimizer step:                                 {:>10}  ({:.2f} s)".format(
            _format_duration(sum_optimizer_wall_s), sum_optimizer_wall_s
        ),
        "    Checkpoints:                                    {:>10}  ({:.2f} s)".format(
            _format_duration(sum_checkpoint_wall_s), sum_checkpoint_wall_s
        ),
        "  Post-backward subtotal (sum of above):            {:>10}  ({:.2f} s)".format(
            _format_duration(sub_no_grad_s), sub_no_grad_s
        ),
    ]
    if no_grad_gap_s > 0.01:
        lines.append(
            "  (tiny gap vs no_grad block:                       {:>10}  ({:.2f} s))".format(
                _format_duration(no_grad_gap_s), no_grad_gap_s
            )
        )
    if overhead_loop_s > 0.1:
        lines.append(
            "  Loop overhead (tqdm/scheduler, unpartitioned):    {:>10}  ({:.2f} s)".format(
                _format_duration(overhead_loop_s), overhead_loop_s
            )
        )
    lines.append(
        "  Final Gaussian count:                             {}".format(gaussians.get_xyz.shape[0])
    )
    if sum_refine_skip_iters > 0:
        lines.append(
            "  Refine stride skip iterations:                    {}".format(sum_refine_skip_iters)
        )
    total_wall_s = setup_wall_s + loop_wall_s
    lines.append(
        "  Total wall (setup + loop):                          {:>10}  ({:.2f} s)".format(
            _format_duration(total_wall_s), total_wall_s
        )
    )
    lines.append("  --- FastGS-style timing (CUDA events, active iters only) ---")
    lines.append(
        "  Forward + backward (CUDA events, sum):            {:>10}  ({:.2f} s)".format(
            _format_duration(sum_fb_gpu_s), sum_fb_gpu_s
        )
    )
    lines.append(
        "  Optimizer + densify hooks (CUDA events, sum):     {:>10}  ({:.2f} s)".format(
            _format_duration(sum_optim_gpu_s), sum_optim_gpu_s
        )
    )
    lines.append(
        "  Training time (FastGS: FB+optim CUDA sum):        {:>10}  ({:.2f} s)".format(
            _format_duration(fastgs_training_s), fastgs_training_s
        )
    )
    lines.append("============================================")
    print(f"Training time (FastGS): {fastgs_training_s:.2f}s")

    _print_and_save_training_timings(scene, lines)


def prepare_output_and_logger(args, disable_tensorboard=False):
    if disable_tensorboard:
        if not args.model_path:
            if os.getenv("OAR_JOB_ID"):
                unique_str = os.getenv("OAR_JOB_ID")
            else:
                unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
        print("Output folder: {}".format(args.model_path))
        os.makedirs(args.model_path, exist_ok=True)
        with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))
        print("TensorBoard logging disabled (--disable_tensorboard).")
        return None

    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss_fn,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    log_train_scalars=True,
    metrics_curve_csv="",
    lpips_model=None,
):
    if tb_writer and log_train_scalars:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        if metrics_curve_csv:
            validation_configs = ({"name": "test", "cameras": scene.getTestCameras()},)
        else:
            validation_configs = (
                {"name": "test", "cameras": scene.getTestCameras()},
                {
                    "name": "train",
                    "cameras": [
                        scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                        for idx in range(5, 30, 5)
                    ],
                },
            )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    if lpips_model is not None:
                        lpips_test += lpips_model(
                            image.unsqueeze(0), gt_image.unsqueeze(0)
                        ).mean().double()
                n = len(config["cameras"])
                psnr_test /= n
                l1_test /= n
                ssim_test /= n
                if lpips_model is not None:
                    lpips_test /= n
                lpips_msg = "" if lpips_model is None else " LPIPS {}".format(lpips_test)
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}{}".format(
                        iteration, config["name"], l1_test, psnr_test, ssim_test, lpips_msg
                    )
                )
                if metrics_curve_csv and config["name"] == "test":
                    lpips_val = float(lpips_test) if lpips_model is not None else float("nan")
                    with open(metrics_curve_csv, "a") as f:
                        f.write(
                            "{},{:.6f},{:.6f},{:.6f}\n".format(
                                iteration, float(psnr_test), float(ssim_test), lpips_val
                            )
                        )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument(
        "--fast_train",
        action="store_true",
        default=False,
        help="FastGS-style: no per-iter TensorBoard scalars, no periodic reduce_opacity / Gaussian count prints.",
    )
    parser.add_argument(
        "--disable_tensorboard",
        action="store_true",
        default=False,
        help="Do not create a SummaryWriter (still writes cfg_args).",
    )
    parser.add_argument(
        "--disable_hts",
        action="store_true",
        default=False,
        help="Disable HTS (same as --hts_mode off). For ablation vs baseline.",
    )
    parser.add_argument(
        "--hts_mode",
        type=str,
        default="conservative",
        choices=["off", "conservative"],
        help="HTS mode. Default: conservative (compact AABB + hierarchical min_q + 24-bit depth + FastGS buckets). "
        "off / --disable_hts = native 3DGS binning (circular getRect, 32-bit depth, classic backward).",
    )
    parser.add_argument(
        "--hts_stats_interval",
        type=int,
        default=0,
        help="If >0, log Super-Tile candidate/kept counts every N iters to hts_supertile_stats.csv.",
    )
    parser.add_argument(
        "--sr",
        action="store_true",
        default=False,
        help="SR refine schedule: skip FB after growth freeze (anchored window + stride annealing).",
    )
    parser.add_argument(
        "--sr_refine_stride",
        type=int,
        default=2,
        help="After growth freeze, skip FB every N-1 of N iterations (SR only).",
    )
    parser.add_argument("--sr_stride4_after", type=int, default=5000)
    parser.add_argument("--sr_stride6_after", type=int, default=9000)
    parser.add_argument(
        "--sr_full_fb_window",
        type=int,
        default=0,
        help="Post-freeze iterations with no refine skip (full FB each step).",
    )
    parser.add_argument(
        "--sr_log_interval",
        type=int,
        default=0,
        help="If >0, log SR schedule state every N iters to sr_schedule.csv.",
    )
    parser.add_argument(
        "--metrics_curve_csv",
        type=str,
        default="",
        help="If set, evaluate test PSNR/SSIM/LPIPS at --test_iterations and append rows to this CSV "
        "(works with --fast_train).",
    )
    parser.add_argument(
        "--video_render_every",
        type=int,
        default=0,
        help="If >0 (and --video_out_dir set), also render the fixed video test view every N iters.",
    )
    parser.add_argument(
        "--video_out_dir",
        type=str,
        default="",
        help="If set, render the fixed test camera in-loop at --test_iterations (plus every "
        "--video_render_every iters) into frame_XXXXX.png here, and write frame_times.csv "
        "(iter,wall_sec since training-loop start).",
    )
    parser.add_argument(
        "--depth_sort_bits",
        type=int,
        default=24,
        choices=[16, 20, 24, 32],
        help="HTS depth key bit width (forced 32 when --disable_hts).",
    )
    parser.add_argument(
        "--hts_hier",
        type=str,
        default="st4_st2",
        choices=["st4_st2", "st4_only"],
        help="HTS hierarchy ablation mode (ignored when --disable_hts).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for python/numpy/torch (default: 0).",
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    hts_mode = args.hts_mode
    if getattr(args, "disable_hts", False):
        hts_mode = "off"

    if hts_mode == "off":
        os.environ["HSTGS_DISABLE_HTS"] = "1"
        os.environ.pop("HSTGS_HTS_MODE", None)
        os.environ.pop("HSTGS_HTS_HIER", None)
        os.environ.pop("HSTGS_DEPTH_SORT_BITS", None)
        print("HTS disabled (mode=off)")
    else:
        hts_mode = "conservative"
        os.environ.pop("HSTGS_DISABLE_HTS", None)
        os.environ["HSTGS_HTS_MODE"] = "conservative"
        os.environ["HSTGS_HTS_HIER"] = args.hts_hier
        os.environ["HSTGS_DEPTH_SORT_BITS"] = str(args.depth_sort_bits)
        print("HTS mode=conservative hier={} depth_bits={}".format(
            args.hts_hier, args.depth_sort_bits
        ))

    # Persist mode for matching render.py eval
    try:
        os.makedirs(args.model_path, exist_ok=True)
        with open(os.path.join(args.model_path, "hts_mode.txt"), "w") as f:
            f.write(hts_mode + "\n")
    except Exception:
        pass

    # TFR is eval-only (HST-GS/submodules/tfr). Never train with it.
    if getattr(args, "use_tfr", False) or os.environ.get("HSTGS_USE_TFR", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        print(
            "ERROR: TFR (--use_tfr / HSTGS_USE_TFR) is eval-only.\n"
            "  Train with native diff-gaussian-rasterization.\n"
            "  For validation FPS/PSNR use: python render.py ... --use_tfr\n"
            "  See HST-GS/submodules/TFR.md"
        )
        sys.exit(2)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet, seed=args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        fast_train=args.fast_train,
        disable_tensorboard=args.disable_tensorboard,
        sr=args.sr,
        sr_refine_stride=args.sr_refine_stride,
        sr_stride4_after=args.sr_stride4_after,
        sr_stride6_after=args.sr_stride6_after,
        sr_full_fb_window=args.sr_full_fb_window,
        hts_stats_interval=args.hts_stats_interval,
        sr_log_interval=args.sr_log_interval,
        metrics_curve_csv=args.metrics_curve_csv,
        video_render_every=args.video_render_every,
        video_out_dir=args.video_out_dir,
    )

    print("\nTraining complete.")
