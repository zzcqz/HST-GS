"""Load Scene + GaussianModel (HST-GS code, native/vanilla checkpoints preferred)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple

# tfr/python/tfr → repo root = HST-GS/
HSTGS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
TFR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VANILLA_ROOT = os.path.join(TFR_ROOT, "outputs", "vanilla_3dgs")


def ensure_hstgs_path():
    if HSTGS_ROOT not in sys.path:
        sys.path.insert(0, HSTGS_ROOT)


@dataclass
class LoadedScene:
    gaussians: Any
    scene: Any
    background: Any
    source_path: str
    model_path: str
    baseline: str  # "vanilla_3dgs" | "best_psnr26"


# Source always under HST-GS/data; model prefers TFR vanilla then falls back.
SCENE_REGISTRY = {
    "bicycle": {"source": "data/Mip-NeRF360/bicycle", "legacy_model": "data/Mip-NeRF360/bicycle/best_psnr26"},
    "flowers": {"source": "data/Mip-NeRF360/flowers", "legacy_model": "data/Mip-NeRF360/flowers/best_psnr26"},
    "garden": {"source": "data/Mip-NeRF360/garden", "legacy_model": "data/Mip-NeRF360/garden/best_psnr26"},
    "stump": {"source": "data/Mip-NeRF360/stump", "legacy_model": "data/Mip-NeRF360/stump/best_psnr26"},
    "treehill": {"source": "data/Mip-NeRF360/treehill", "legacy_model": "data/Mip-NeRF360/treehill/best_psnr26"},
    "room": {"source": "data/Mip-NeRF360/room", "legacy_model": "data/Mip-NeRF360/room/best_psnr26"},
    "counter": {"source": "data/Mip-NeRF360/counter", "legacy_model": "data/Mip-NeRF360/counter/best_psnr26"},
    "kitchen": {"source": "data/Mip-NeRF360/kitchen", "legacy_model": "data/Mip-NeRF360/kitchen/best_psnr26"},
    "bonsai": {"source": "data/Mip-NeRF360/bonsai", "legacy_model": "data/Mip-NeRF360/bonsai/best_psnr26"},
    "playroom": {"source": "data/Deep_Blending/playroom", "legacy_model": "data/Deep_Blending/playroom/best_psnr26"},
    "drjohnson": {"source": "data/Deep_Blending/drjohnson", "legacy_model": "data/Deep_Blending/drjohnson/best_psnr26"},
    "Truck": {"source": "data/Tanks_Temples/Truck", "legacy_model": "data/Tanks_Temples/Truck/best_psnr26"},
    "Train": {"source": "data/Tanks_Temples/Train", "legacy_model": "data/Tanks_Temples/Train/best_psnr26"},
}


def _has_ply(model_path: str, iteration: int = 30000) -> bool:
    ply = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    return os.path.isfile(ply)


def resolve_scene(
    name: str,
    hstgs_root: Optional[str] = None,
    prefer_vanilla: bool = True,
    force_legacy: bool = False,
) -> Tuple[str, str, str]:
    """Returns (source_path, model_path, baseline_tag)."""
    root = hstgs_root or HSTGS_ROOT
    if name not in SCENE_REGISTRY:
        raise KeyError(f"Unknown scene {name}; choose from {list(SCENE_REGISTRY)}")
    meta = SCENE_REGISTRY[name]
    source = os.path.join(root, meta["source"])
    vanilla = os.path.join(VANILLA_ROOT, name)
    legacy = os.path.join(root, meta["legacy_model"])
    if force_legacy:
        return source, legacy, "best_psnr26"
    if prefer_vanilla and _has_ply(vanilla):
        return source, vanilla, "vanilla_3dgs"
    return source, legacy, "best_psnr26"


def load_scene(
    source_path: str,
    model_path: str,
    sh_degree: int = 3,
    iteration: int = 30000,
    eval_mode: bool = True,
    resolution: int = -1,
    baseline: str = "unknown",
) -> LoadedScene:
    ensure_hstgs_path()
    import torch
    from arguments import GroupParams
    from scene import Scene
    from gaussian_renderer import GaussianModel

    dataset = GroupParams()
    dataset.source_path = os.path.abspath(source_path)
    dataset.model_path = os.path.abspath(model_path)
    dataset.sh_degree = sh_degree
    dataset.eval = eval_mode
    dataset.resolution = resolution
    dataset.data_device = "cuda"
    dataset.images = "images"
    dataset.white_background = False

    gaussians = GaussianModel(sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")
    return LoadedScene(gaussians, scene, background, dataset.source_path, dataset.model_path, baseline)
