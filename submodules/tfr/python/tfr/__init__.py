"""TFR: eval-only fused fixed-tile forward for native 3DGS / HST-GS validation.

Training should use HST-GS ``diff-gaussian-rasterization``. TFR is exposed to
HST-GS via ``submodules/tfr`` + ``render.py --use_tfr``.
"""

from .metrics import psnr
from .render import render, render_from_gaussians

__all__ = [
    "psnr",
    "render",
    "render_from_gaussians",
]
