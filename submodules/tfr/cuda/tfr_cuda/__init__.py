"""TFR CUDA ops: preprocess + fused forward + train backward."""

import torch

try:
    from . import _C
except ImportError:  # pragma: no cover
    _C = None


def available() -> bool:
    return _C is not None and torch.cuda.is_available()


def _empty_sh_rest(like: torch.Tensor) -> torch.Tensor:
    return torch.empty(0, device=like.device, dtype=like.dtype)


def preprocess_gaussians(
    means3D,
    scales,
    rotations,
    opacities,
    sh,
    viewmatrix,
    projmatrix,
    campos,
    W,
    H,
    tan_fovx,
    tan_fovy,
    sh_degree=3,
    scale_modifier=1.0,
    activate_params=False,
    sh_rest=None,
):
    assert available(), "tfr_cuda not built or CUDA unavailable"
    if sh_rest is None:
        sh_rest = _empty_sh_rest(sh)
    return _C.preprocess_gaussians(
        means3D.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        opacities.contiguous(),
        sh.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        int(W),
        int(H),
        float(tan_fovx),
        float(tan_fovy),
        int(sh_degree),
        float(scale_modifier),
        int(bool(activate_params)),
        sh_rest.contiguous(),
    )


def forward_packed(
    means3D,
    scales,
    rotations,
    opacities,
    sh,
    viewmatrix,
    projmatrix,
    campos,
    bg_color,
    W,
    H,
    tan_fovx,
    tan_fovy,
    sh_degree=3,
    scale_modifier=1.0,
    activate_params=False,
    sh_rest=None,
):
    assert available(), "tfr_cuda not built or CUDA unavailable"
    if sh_rest is None:
        sh_rest = _empty_sh_rest(sh)
    return _C.forward_packed(
        means3D.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        opacities.contiguous(),
        sh.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        bg_color.contiguous().to(torch.float32),
        int(W),
        int(H),
        float(tan_fovx),
        float(tan_fovy),
        int(sh_degree),
        float(scale_modifier),
        int(bool(activate_params)),
        sh_rest.contiguous(),
    )


def forward_packed_train(
    means3D,
    scales,
    rotations,
    opacities,
    sh,
    viewmatrix,
    projmatrix,
    campos,
    bg_color,
    W,
    H,
    tan_fovx,
    tan_fovy,
    sh_degree=3,
    scale_modifier=1.0,
    activate_params=False,
    sh_rest=None,
):
    assert available(), "tfr_cuda not built or CUDA unavailable"
    if sh_rest is None:
        sh_rest = _empty_sh_rest(sh)
    return _C.forward_packed_train(
        means3D.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        opacities.contiguous(),
        sh.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        bg_color.contiguous().to(torch.float32),
        int(W),
        int(H),
        float(tan_fovx),
        float(tan_fovy),
        int(sh_degree),
        float(scale_modifier),
        int(bool(activate_params)),
        sh_rest.contiguous(),
    )


def render_sorted_list_backward(
    ranges,
    point_list,
    means2d,
    conic_opacity,
    rgb,
    bg_color,
    final_T,
    n_contrib,
    max_contrib,
    pixel_colors,
    per_tile_bucket_offset,
    bucket_to_tile,
    sampled_T,
    sampled_ar,
    dL_dpixels,
    W,
    H,
    tiles_x,
    tiles_y,
    num_buckets,
):
    assert available(), "tfr_cuda not built or CUDA unavailable"
    return _C.render_sorted_list_backward(
        ranges.contiguous(),
        point_list.contiguous(),
        means2d.contiguous(),
        conic_opacity.contiguous(),
        rgb.contiguous(),
        bg_color.contiguous(),
        final_T.contiguous(),
        n_contrib.contiguous(),
        max_contrib.contiguous(),
        pixel_colors.contiguous(),
        per_tile_bucket_offset.contiguous(),
        bucket_to_tile.contiguous(),
        sampled_T.contiguous(),
        sampled_ar.contiguous(),
        dL_dpixels.contiguous(),
        int(W),
        int(H),
        int(tiles_x),
        int(tiles_y),
        int(num_buckets),
    )


def preprocess_backward(
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
    opacities,
    dL_dmean2D,
    dL_dconic,
    dL_dopacity,
    dL_dcolor,
    W,
    H,
    tan_fovx,
    tan_fovy,
    sh_degree,
    scale_modifier,
):
    assert available(), "tfr_cuda not built or CUDA unavailable"
    return _C.preprocess_backward(
        means3D.contiguous(),
        radii.contiguous(),
        shs.contiguous(),
        clamped.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        cov3Ds.contiguous(),
        viewmatrix.contiguous(),
        projmatrix.contiguous(),
        campos.contiguous(),
        opacities.contiguous(),
        dL_dmean2D.contiguous(),
        dL_dconic.contiguous(),
        dL_dopacity.contiguous(),
        dL_dcolor.contiguous(),
        int(W),
        int(H),
        float(tan_fovx),
        float(tan_fovy),
        int(sh_degree),
        float(scale_modifier),
    )
