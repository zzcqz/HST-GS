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

import math
import os
import sys

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def _tfr_enabled(pipe) -> bool:
    if getattr(pipe, "use_tfr", False):
        return True
    return os.environ.get("HSTGS_USE_TFR", "").strip().lower() in ("1", "true", "yes")


def _tfr_roots():
    """Prefer HST-GS/submodules/tfr; fall back to sibling TFR or TFR_ROOT."""
    here = os.path.dirname(os.path.abspath(__file__))
    hstgs_root = os.path.abspath(os.path.join(here, ".."))
    candidates = []
    env = os.environ.get("TFR_ROOT", "").strip()
    if env:
        candidates.append(env)
    candidates.append(os.path.join(hstgs_root, "submodules", "tfr"))
    candidates.append(os.path.abspath(os.path.join(hstgs_root, "..", "TFR")))
    return candidates


def _import_tfr_render():
    last_err = None
    for root in _tfr_roots():
        if not root or not os.path.isdir(root):
            continue
        for sub in ("python", "cuda"):
            p = os.path.join(root, sub)
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        try:
            from tfr.render import render as tfr_render
            return tfr_render, root
        except Exception as e:
            last_err = e
            continue
    raise ImportError(
        "TFR eval submodule not found/built. Expected HST-GS/submodules/tfr "
        "with cuda extension built. "
        f"Last import error: {last_err}"
    )


def _render_with_tfr(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, separate_sh, override_color):
    """Eval-only TFR forward. Training must use native diff-gaussian-rasterization."""
    if torch.is_grad_enabled():
        raise RuntimeError(
            "TFR is an eval-only submodule and must not run under autograd. "
            "Use native rasterization for training (do not pass --use_tfr to train.py)."
        )
    tfr_render, _root = _import_tfr_render()
    return tfr_render(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier=scaling_modifier,
        separate_sh=separate_sh,
        override_color=override_color,
        with_aux=False,
    )


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    Optional TFR backend (**eval / render.py only**): ``pipe.use_tfr=True`` or
    env ``HSTGS_USE_TFR=1``. Training always uses diff-gaussian-rasterization.
    """
    if _tfr_enabled(pipe):
        out = _render_with_tfr(
            viewpoint_camera, pc, pipe, bg_color, scaling_modifier,
            separate_sh, override_color,
        )
    else:
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
            antialiasing=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None

        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                if separate_sh:
                    dc, shs = pc.get_features_dc, pc.get_features_rest
                else:
                    shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        if separate_sh:
            rendered_image, radii, _invdepth = rasterizer(
                means3D = means3D,
                means2D = means2D,
                dc = dc,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
        else:
            rendered_image, radii, _invdepth = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

        out = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            }

    out["render"] = out["render"].clamp(0, 1)
    return out
