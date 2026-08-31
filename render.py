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

import json
import time
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    frame_times = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rendering = render(view, gaussians, pipeline, background, separate_sh=separate_sh)["render"]
        torch.cuda.synchronize()
        frame_times.append(time.perf_counter() - t0)
        gt = view.original_image[0:3, :, :]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    num_frames = len(views)
    total_time = sum(frame_times)
    avg_time = total_time / num_frames if num_frames > 0 else 0.0
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    print(
        f"[{name}] Rendered {num_frames} frames in {total_time:.2f}s. Average FPS: {fps:.2f}"
    )
    return {"frames": num_frames, "total_s": total_time, "avg_s": avg_time, "fps": fps}

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, backend_tag: str):
    fps_report = {}
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            fps_report["train"] = render_set(
                dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                gaussians, pipeline, background, separate_sh,
            )

        if not skip_test:
            fps_report["test"] = render_set(
                dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                gaussians, pipeline, background, separate_sh,
            )

    if fps_report:
        fps_path = os.path.join(dataset.model_path, "render_fps_{}.json".format(backend_tag))
        with open(fps_path, "w") as f:
            json.dump(fps_report, f, indent=2)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # get_combined_args drops None-valued attrs when merging cfg_args,
    # so use a non-None sentinel for "read from hts_mode.txt".
    parser.add_argument(
        "--hts_mode",
        type=str,
        default="auto",
        choices=["auto", "off", "conservative"],
        help="HTS mode for rendering. auto: read model hts_mode.txt, else conservative. "
        "off = native 3DGS binning (no compact AABB / 24-bit / buckets).",
    )
    parser.add_argument("--disable_hts", action="store_true", default=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    hts_mode = args.hts_mode
    if getattr(args, "disable_hts", False):
        hts_mode = "off"
    if hts_mode == "auto":
        mode_file = os.path.join(args.model_path, "hts_mode.txt")
        if os.path.isfile(mode_file):
            hts_mode = open(mode_file).read().strip() or "conservative"
        else:
            hts_mode = "conservative"
    # Legacy checkpoints may still say "aggressive"; map to conservative.
    if hts_mode not in ("off", "conservative"):
        print("WARNING: unknown hts_mode={!r}, using conservative".format(hts_mode))
        hts_mode = "conservative"
    if hts_mode == "off":
        os.environ["HSTGS_DISABLE_HTS"] = "1"
        os.environ.pop("HSTGS_HTS_MODE", None)
    else:
        os.environ.pop("HSTGS_DISABLE_HTS", None)
        os.environ["HSTGS_HTS_MODE"] = "conservative"
    print("HTS mode={}".format(hts_mode))

    pipe = pipeline.extract(args)
    backend_tag = "tfr" if getattr(pipe, "use_tfr", False) else "native"

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipe, args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, backend_tag)