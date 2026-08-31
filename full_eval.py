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
from argparse import ArgumentParser
import time

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--fast", action="store_true")




args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()


def run_cmd(cmd: str) -> None:
    """Run a shell command and abort the eval pipeline on failure."""
    print(">>", cmd, flush=True)
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command failed with exit status {ret}: {cmd}")


m360_timing = None
tandt_timing = None
db_timing = None

if not args.skip_training:
    # Only persist the final checkpoint (no mid-run 7000 save/render).
    common_args = (
        " --disable_viewer --quiet --eval --test_iterations -1 "
        " --save_iterations 30000 "
    )

    if args.fast:
        common_args += " --optimizer_type sparse_adam "

    start_time = time.time()
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        run_cmd("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        run_cmd("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args)
    m360_timing = (time.time() - start_time)/60.0

    start_time = time.time()
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        run_cmd("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    tandt_timing = (time.time() - start_time)/60.0

    start_time = time.time()
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        run_cmd("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    db_timing = (time.time() - start_time)/60.0

if not args.skip_training:
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "timing.txt"), "w") as file:
        file.write(
            f"m360: {m360_timing} minutes \n tandt: {tandt_timing} minutes \n db: {db_timing} minutes\n"
        )

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)
    
    common_args = " --quiet --eval --skip_train"

    for scene, source in zip(all_scenes, all_sources):
        run_cmd(
            "python render.py --iteration 30000 -s "
            + source
            + " -m "
            + args.output_path
            + "/"
            + scene
            + common_args
        )

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    run_cmd("python metrics.py -m " + scenes_string)
