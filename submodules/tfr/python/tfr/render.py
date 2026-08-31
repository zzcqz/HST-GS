"""TFR renderer: HST-GS-compatible ``render()`` with optional Autograd (Phase 1)."""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, Optional, Sequence, Union

import torch


def _tfr_cuda():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cuda"))
    if root not in sys.path:
        sys.path.insert(0, root)
    import tfr_cuda

    return tfr_cuda


def _as_bg_tensor(bg: Union[torch.Tensor, Sequence[float]], device: torch.device) -> torch.Tensor:
    if isinstance(bg, torch.Tensor):
        t = bg.detach().to(device=device, dtype=torch.float32).reshape(-1)
    else:
        t = torch.tensor(list(bg), dtype=torch.float32, device=device)
    if t.numel() == 1:
        t = t.expand(3).contiguous()
    assert t.numel() == 3, "bg_color must have 3 channels"
    return t.contiguous()


class _TfrRasterize(torch.autograd.Function):
    """Phase 2: PerGaussian sample-bucket raster + preprocess backward."""

    @staticmethod
    def forward(
        ctx,
        means3D,
        scales,
        rotations,
        opacities,
        sh_dc,
        sh_rest,
        viewmatrix,
        projmatrix,
        campos,
        bg_color,
        screenspace_points,
        W,
        H,
        tan_fovx,
        tan_fovy,
        sh_degree,
        scale_modifier,
    ):
        tfr_cuda = _tfr_cuda()
        outs = tfr_cuda.forward_packed_train(
            means3D,
            scales,
            rotations,
            opacities,
            sh_dc,
            viewmatrix,
            projmatrix,
            campos,
            bg_color,
            int(W),
            int(H),
            float(tan_fovx),
            float(tan_fovy),
            int(sh_degree),
            float(scale_modifier),
            False,  # already-activated get_* inputs
            sh_rest,
        )
        (
            image,
            radii,
            means2d,
            conic_opacity,
            rgb,
            cov3Ds,
            clamped,
            ranges,
            point_list,
            final_T,
            n_contrib,
            max_contrib,
            pixel_colors,
            bucket_off,
            bucket_to_tile,
            sampled_T,
            sampled_ar,
            num_buckets_t,
            bg,
        ) = outs

        ctx.save_for_backward(
            means3D,
            scales,
            rotations,
            opacities,
            sh_dc,
            sh_rest,
            viewmatrix,
            projmatrix,
            campos,
            means2d,
            conic_opacity,
            rgb,
            cov3Ds,
            clamped,
            ranges,
            point_list,
            final_T,
            n_contrib,
            max_contrib,
            pixel_colors,
            bucket_off,
            bucket_to_tile,
            sampled_T,
            sampled_ar,
            num_buckets_t,
            bg,
            radii,
        )
        ctx.meta = (int(W), int(H), float(tan_fovx), float(tan_fovy), int(sh_degree), float(scale_modifier))
        return image, radii

    @staticmethod
    def backward(ctx, grad_image, grad_radii):
        (
            means3D,
            scales,
            rotations,
            opacities,
            sh_dc,
            sh_rest,
            viewmatrix,
            projmatrix,
            campos,
            means2d,
            conic_opacity,
            rgb,
            cov3Ds,
            clamped,
            ranges,
            point_list,
            final_T,
            n_contrib,
            max_contrib,
            pixel_colors,
            bucket_off,
            bucket_to_tile,
            sampled_T,
            sampled_ar,
            num_buckets_t,
            bg,
            radii,
        ) = ctx.saved_tensors
        W, H, tan_fovx, tan_fovy, sh_degree, scale_modifier = ctx.meta
        tfr_cuda = _tfr_cuda()

        tiles_x = (W + 15) // 16
        tiles_y = (H + 15) // 16

        if grad_image is None:
            grad_image = torch.zeros((3, H, W), device=means2d.device, dtype=means2d.dtype)

        dL_dmean2D, dL_dconic, dL_dopacity, dL_dcolors = tfr_cuda.render_sorted_list_backward(
            ranges,
            point_list,
            means2d,
            conic_opacity,
            rgb,
            bg,
            final_T,
            n_contrib,
            max_contrib,
            pixel_colors,
            bucket_off,
            bucket_to_tile,
            sampled_T,
            sampled_ar,
            grad_image.contiguous(),
            W,
            H,
            tiles_x,
            tiles_y,
            int(num_buckets_t.item()),
        )

        dL_dopacity = dL_dopacity.view(-1)

        # preprocess_backward expects combined (P, M, 3) SH layout
        shs = torch.cat([sh_dc, sh_rest], dim=1) if sh_rest.numel() > 0 else sh_dc
        dL_dmeans, dL_dscale, dL_drot, dL_dopacity_out, dL_dsh = tfr_cuda.preprocess_backward(
            means3D,
            radii,
            shs,
            clamped,
            scales,
            rotations,
            cov3Ds,
            viewmatrix,
            projmatrix,
            campos,
            opacities.view(-1),
            dL_dmean2D,
            dL_dconic,
            dL_dopacity,
            dL_dcolors,
            W,
            H,
            tan_fovx,
            tan_fovy,
            sh_degree,
            scale_modifier,
        )

        grad_screenspace = dL_dmean2D
        dL_dsh_dc = dL_dsh[:, : sh_dc.shape[1]]
        dL_dsh_rest = dL_dsh[:, sh_dc.shape[1] :] if sh_rest.numel() > 0 else None

        return (
            dL_dmeans,
            dL_dscale,
            dL_drot,
            dL_dopacity_out.view_as(opacities),
            dL_dsh_dc,
            dL_dsh_rest,
            None,  # viewmatrix
            None,  # projmatrix
            None,  # campos
            None,  # bg
            grad_screenspace,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def render_from_gaussians(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    features: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    campos: torch.Tensor,
    width: int,
    height: int,
    tanfovx: float,
    tanfovy: float,
    sh_degree: int = 3,
    bg: Union[torch.Tensor, Sequence[float]] = (0.0, 0.0, 0.0),
    scale_modifier: float = 1.0,
    profile: bool = False,
    activate_params: bool = False,
    sh_rest: torch.Tensor = None,
) -> Dict[str, Any]:
    """Low-level tensor forward via ``tfr_cuda.forward_packed`` (inference)."""
    tfr_cuda = _tfr_cuda()
    assert tfr_cuda.available(), "tfr_cuda required"

    bg_t = _as_bg_tensor(bg, means3d.device)
    if profile:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    image, radii, n_tiles_t, L_t = tfr_cuda.forward_packed(
        means3d,
        scales,
        rotations,
        opacities,
        features,
        viewmatrix,
        projmatrix,
        campos,
        bg_t,
        width,
        height,
        tanfovx,
        tanfovy,
        sh_degree,
        scale_modifier,
        activate_params,
        sh_rest,
    )

    out: Dict[str, Any] = {
        "image": image,
        "radii": radii,
        "active_tiles": n_tiles_t,
        "fixed_tiles": ((width + 15) // 16) * ((height + 15) // 16),
        "assignment_L": L_t,
    }
    if profile:
        torch.cuda.synchronize()
        t_e2e = time.perf_counter() - t0
        out["timings_s"] = {
            "e2e": t_e2e,
            "e2e_prep_pack_raster": t_e2e,
            "raster": t_e2e,
        }
    return out


def render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    separate_sh: bool = False,
    override_color: Optional[torch.Tensor] = None,
    with_aux: bool = False,
) -> Dict[str, Any]:
    """HST-GS-compatible render. Autograd enabled when ``torch.is_grad_enabled()``.

    Training path uses activated ``get_*`` buffers (like HST-GS) + Phase 1 backward.
    Eval path uses raw ``_scaling/_rotation/_opacity`` + CUDA activation for FPS.
    """
    if separate_sh:
        raise NotImplementedError("TFR render does not support separate_sh=True")
    if override_color is not None:
        raise NotImplementedError("TFR render does not support override_color")
    if getattr(pipe, "convert_SHs_python", False):
        raise NotImplementedError("TFR render does not support convert_SHs_python")
    if getattr(pipe, "compute_cov3D_python", False):
        raise NotImplementedError("TFR render does not support compute_cov3D_python")

    w = int(viewpoint_camera.image_width)
    h = int(viewpoint_camera.image_height)
    means3d = pc.get_xyz
    bg_t = _as_bg_tensor(bg_color, means3d.device)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    sh_degree = int(pc.active_sh_degree)

    if torch.is_grad_enabled():
        screenspace_points = torch.zeros(
            (means3d.shape[0], 4), dtype=means3d.dtype, device=means3d.device, requires_grad=True
        )
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        image, radii = _TfrRasterize.apply(
            means3d,
            pc.get_scaling,
            pc.get_rotation,
            pc.get_opacity,
            pc.get_features_dc,
            pc.get_features_rest,
            viewpoint_camera.world_view_transform,
            viewpoint_camera.full_proj_transform,
            viewpoint_camera.camera_center,
            bg_t,
            screenspace_points,
            w,
            h,
            tanfovx,
            tanfovy,
            sh_degree,
            float(scaling_modifier),
        )
        rendered_image = image.clamp(0, 1)

        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "active_tiles": 0,
            "assignment_L": 0,
        }

    # Inference / eval (no grad)
    packed = render_from_gaussians(
        means3d=means3d,
        scales=pc._scaling,
        rotations=pc._rotation,
        opacities=pc._opacity,
        features=pc._features_dc,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        width=w,
        height=h,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        sh_degree=sh_degree,
        bg=bg_t,
        scale_modifier=float(scaling_modifier),
        profile=False,
        activate_params=True,
        sh_rest=pc._features_rest,
    )

    rendered_image = packed["image"].clamp_(0, 1)
    radii = packed["radii"]

    out: Dict[str, Any] = {
        "render": rendered_image,
        "radii": radii,
        "active_tiles": packed["active_tiles"],
        "assignment_L": packed["assignment_L"],
    }
    if with_aux:
        out["viewspace_points"] = torch.zeros(
            (means3d.shape[0], 4), dtype=means3d.dtype, device=means3d.device
        )
        out["visibility_filter"] = (radii > 0).nonzero()
    else:
        out["viewspace_points"] = rendered_image.new_empty((0, 4))
        out["visibility_filter"] = rendered_image.new_empty((0, 1), dtype=torch.long)
    return out
