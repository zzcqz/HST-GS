# TFR Experiments

## Role

TFR is an **eval-only** raster submodule for HST-GS validation / `render.py`.
Training always uses native `diff-gaussian-rasterization`.

Layout:

- Research tree: `ZhouZheng/TFR/`
- HST-GS submodule: `HST-GS/submodules/tfr` → `../../TFR` (symlink)
- Notes: [`HST-GS/submodules/TFR.md`](../HST-GS/submodules/TFR.md)

```python
from tfr import render  # forward / no_grad
out = render(camera, gaussians, pipe, bg)
```

- **Eval**: fused CUDA forward (~2.26× mean on vanilla 13-scene @ 30k PLY)
- Autograd / train path exists in-tree for research only; **not** wired into HST-GS `train.py`

Unsupported: `separate_sh`, `override_color`, `convert_SHs_python`, `compute_cov3D_python`.

### HST-GS drop-in (validation only)

```bash
cd HST-GS/submodules/tfr/cuda && python setup.py build_ext --inplace   # GPU node
cd HST-GS
python render.py -m <model> --use_tfr
# or
HSTGS_USE_TFR=1 python render.py -m <model>
```

`train.py --use_tfr` / `HSTGS_USE_TFR=1` during training **exits with an error**.

## Eval: vanilla 13-scene (iteration_30000 PLY)

`scripts/bench_tfr_render.py --all-13 --warmup 2 --repeats 5`
→ `outputs/bench_render/bench_render_13.json`

Full-scene bench uses **all test cameras** per scene (omit `--max-views`; use it only to cap views for quick debugging). Primary reported metrics are **Native FPS** and **TFR end-to-end FPS** (`1000 / ms`).

All checkpoints: `outputs/vanilla_3dgs/<scene>/point_cloud/iteration_30000/point_cloud.ply`.

### Metric glossary

| Metric | Meaning |
|--------|---------|
| **Native FPS** | HST-GS `gaussian_renderer.render()` end-to-end frames/sec (`1000 / ms`) |
| **TFR FPS** | TFR `render()` end-to-end frames/sec (preprocess + sort/bin + raster) |
| **Speedup** | `TFR_FPS / Native_FPS` (same as `native_ms / tfr_ms`) |
| **PSNR vs native** | Image agreement vs HST-GS output (not vs GT); ~99 dB ≈ near-identical |
| warmup / repeats | Discard first N frames, then average over M timed frames |
| max-views | Optional debug cap on test cameras (default: all test views) |

| Scene | Native FPS | TFR FPS | Speedup | PSNR vs native |
|-------|-----------:|---------:|--------:|---------------:|
| bicycle | 99 | 212 | 2.14× | 99.0 |
| flowers | 158 | 374 | 2.37× | 99.0 |
| garden | 118 | 254 | 2.16× | 99.0 |
| stump | 124 | 288 | 2.32× | 99.0 |
| treehill | 130 | 295 | 2.27× | 99.0 |
| room | 208 | 464 | 2.23× | 99.0 |
| counter | 203 | 431 | 2.13× | 99.0 |
| kitchen | 175 | 375 | 2.15× | 99.0 |
| bonsai | 237 | 558 | 2.35× | 99.0 |
| playroom | 204 | 481 | 2.36× | 95.1 |
| drjohnson | 162 | 372 | 2.30× | 99.0 |
| Truck | 186 | 436 | 2.35× | 99.0 |
| Train | 211 | 473 | 2.24× | 94.9 |
| **Mean** | **~159** | **~370** | **~2.26×** | **~98.4** |

### Paper figures & written comparison

详细文字版（与原版 3DGS 的异同、加速杠杆、定量表）：[`docs/vs_vanilla_3dgs.md`](vs_vanilla_3dgs.md)。

```bash
python scripts/make_vs_3dgs_figure.py     # Vanilla vs TFR (reviewer)
python scripts/make_mechanism_figure.py  # how acceleration works
python scripts/make_paper_figures.py     # FPS / speedup / PSNR charts
# → outputs/figures/fig_vs_vanilla_3dgs*.{pdf,png}
# → outputs/figures/fig_mechanism_speedup*.{pdf,png}
# → outputs/figures/fig{1..6}_*.{pdf,png}
```

| Figure | Use in paper |
|------|----------------|
| `fig_vs_vanilla_3dgs` | Shared skeleton + eval-path diffs + why faster |
| `fig_mechanism_speedup` | Method mechanism: host bottlenecks → fused CUDA → 3 levers |
| `fig1_pipeline` | Short overview strip |
| `fig2_fps_bars` | Main result: Native vs TFR FPS |
| `fig3_speedup` | Per-scene acceleration |
| `fig4_quality_psnr` | Fidelity vs native (not GT) |
| `fig5_mem_saving` | Peak memory |
| `fig6_fps_scatter` | Compact throughput correlation |

Prefer PDF for LaTeX. See `outputs/figures/README.md`.

## Eval: best_psnr26 FPS

| | HST-GS | TFR |
|--|------:|-----:|
| Mean FPS | **~304** | **~770** |
| Speedup | — | **~2.52×** |

## Historical: Phase 2 train research (not product path)

Earlier experiments wired TFR into `train.py` for gradient alignment.
Those runs showed PerGaussian backward can match native PSNR at 700/3k iters,
but the **supported product posture is eval-only** (this document).

Artifacts remain under `outputs/smoke_train_compare_*` / `outputs/train_compare_*`
for reference only.

## Baseline notes

`best_psnr26` is HST-GS, not native 3DGS. See `docs/baseline_mismatch.md`.

HST-GS method stack: **HTS** + **SR** + ACW densify; optional **TFR** at eval.
