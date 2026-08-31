# Golden (fixed-tile) path: forked copy of HST-GS diff-gaussian-rasterization.
# Prefer the already-installed HST-GS package for A/B:
#   from diff_gaussian_rasterization import GaussianRasterizer

This folder is a snapshot of the fixed-tile rasterizer (CUB sort / tile ranges /
renderCUDA) used as the numerical and performance reference.

TFR fused forward lives in:
- `../csrc/tfr_preprocess.cu`
- `../csrc/tfr_binning.cu` (tiles_touched → CUB scan → duplicate → radix sort → ranges)
- `../csrc/tfr_render.cu` (`render_sorted_list`)
- Python entry: `../../python/tfr/render.py` → `tfr_cuda.forward_packed`
