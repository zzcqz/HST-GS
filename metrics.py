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

from pathlib import Path
import os
import shutil

# Must run before `import torch`: libcudnn_cnn_infer needs libnvrtc from the same CUDA stack.
# Cluster jobs often omit conda lib from the linker path.
_conda_prefix = os.environ.get("CONDA_PREFIX")
if _conda_prefix:
    _conda_lib = os.path.join(_conda_prefix, "lib")
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _conda_lib not in _ld.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = (
            _conda_lib + (os.pathsep + _ld if _ld else "")
        )

import json
from argparse import ArgumentParser

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from utils.loss_utils import ssim
from utils.image_utils import psnr

try:
    from lpipsPyTorch import lpips as lpips_fn
except ImportError:
    lpips_fn = None


def readImages(renders_dir, gt_dir, device: torch.device):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].to(device))
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].to(device))
        image_names.append(fname)
    return renders, gts, image_names


def cleanup_render_dirs(scene_dir: str):
    """Delete the on-disk PNG render dirs (test/train) once metrics are computed.

    Only the rendered PNGs are transient; results.json / per_view.json are kept.
    """
    for name in ("test", "train"):
        d = Path(scene_dir) / name
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            print("Removed render dir:", d)


def evaluate(model_paths, device: torch.device, skip_lpips: bool, keep_images: bool = False):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    failed = False

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir, device)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    if skip_lpips or lpips_fn is None:
                        lpipss.append(torch.tensor(0.0, device=device))
                    else:
                        lpipss.append(lpips_fn(renders[idx], gts[idx], net_type="vgg"))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                if skip_lpips or lpips_fn is None:
                    print("  LPIPS: (skipped)")
                else:
                    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                row = {
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                }
                if not skip_lpips and lpips_fn is not None:
                    row["LPIPS"] = torch.tensor(lpipss).mean().item()

                full_dict[scene_dir][method].update(row)
                per_view_dict[scene_dir][method].update(
                    {
                        "SSIM": {
                            name: v for v, name in zip(torch.tensor(ssims).tolist(), image_names)
                        },
                        "PSNR": {
                            name: v for v, name in zip(torch.tensor(psnrs).tolist(), image_names)
                        },
                    }
                )
                if not skip_lpips and lpips_fn is not None:
                    per_view_dict[scene_dir][method]["LPIPS"] = {
                        name: v for v, name in zip(torch.tensor(lpipss).tolist(), image_names)
                    }

            with open(scene_dir + "/results.json", "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)

            if not keep_images:
                cleanup_render_dirs(scene_dir)
        except Exception as e:
            failed = True
            print("Unable to compute metrics for model", scene_dir, ":", e)

    if failed:
        raise RuntimeError(
            "One or more scenes failed metric evaluation; refusing success exit "
            "so callers do not reuse a stale results.json."
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--model_paths", "-m", required=True, nargs="+", type=str, default=[]
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="cuda avoids host LPIPS slowness; cpu avoids broken GPU NVRTC/cuDNN.",
    )
    parser.add_argument(
        "--no_lpips",
        action="store_true",
        help="Only SSIM/PSNR (no VGG LPIPS / cuDNN).",
    )
    parser.add_argument(
        "--keep_images",
        action="store_true",
        help="Keep the rendered test/train PNG dirs. By default they are "
        "deleted after metrics are computed to save disk space.",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    evaluate(args.model_paths, device, skip_lpips=args.no_lpips, keep_images=args.keep_images)
