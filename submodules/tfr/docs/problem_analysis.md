# Phase 1 — Problem Analysis

## 1. Observed raster bottlenecks (3D-GS / HST-GS)

From `diff-gaussian-rasterization` path `preprocess → duplicateWithKeys → radix_sort → identifyTileRanges → render`:

1. **Fixed 16×16 tiles** create severe load imbalance on real indoor/outdoor views (dense near-field vs empty sky/walls).
2. **Scattered SoA reads**: after sort, each tile walks a list of Gaussian IDs into global means/cov/SH — irregular global memory.
3. **Empty tile launches**: many tile slots touch zero or near-zero Gaussians but still exist in the grid.

HTS in HST-GS only culls within fixed granularity; it does not *generate* variable tiles or *repack* memory.

## 2. Gaps in current TFR v0 prototype

| Item | Status | Gap |
|------|--------|-----|
| Complexity estimator | OK (CPU/CUDA) | Tuned on synthetic; uncalibrated on real footprints |
| Adaptive merge {16,32,64} | OK | Defaults `C_merge=200` may be wrong for real density |
| Visibility priority | OK | Not yet driving CUDA grid launch order in golden kernel |
| Assignment + sort keys | OK | Wave-split on `C_split` incomplete |
| GatherPack | OK | Fed only soft/CPU preview, not production tile raster |
| Real-scene PSNR check | Missing | Need HST-GS golden vs GT + TFR consistency |
| Differentiable backward | Deferred | After forward quality/speed proven |

## 3. Measurement plan

For each real view:

- \(N\): Gaussian count
- \(T_{\mathrm{fixed}}\): ceil(W/16)·ceil(H/16)
- \(T_{\mathrm{adapt}}\): active adaptive tiles
- \(L\): Gaussian–tile associations (duplicate list length)
- \(C\) map stats: mean/max/\% empty base tiles
- Golden render: PSNR / SSIM vs GT (checkpoint quality floor)
- Pack: contiguous touch time vs scatter touch time (proxy bandwidth)

## 4. Hypotheses to test next

**H1:** Real views have ≥30% empty or near-empty base tiles → adaptive merge yields \(T_{\mathrm{adapt}}/T_{\mathrm{fixed}} \le 0.5\) without touching dense cores.

**H2:** Optimal `C_merge` is scene-dependent but a single shared config can stay within 10% of per-scene best on tile reduction.

**H3:** GatherPack preserves list order ⇒ identical soft compositing (PSNR ∞ / 99); any future packed raster must match golden within 0.05 dB when using the same depth order on fixed 16×16 (ablation control).

## 5. Decisions for Phase 2

1. Project Gaussians in PyTorch (view + Jacobian 2D cov + SH→RGB) without modifying HST-GS submodule.
2. **Rethink (playroom first pass):** raw `C(b)` mean ≈ 2300 with `w_alpha+w_area` counting — absolute `C_merge≤800` never merges. Switch merge criterion to **density** \(C(T)/\#bases \le C_{\mathrm{merge}}\).
3. Sweep density `C_merge ∈ {800…3200}`, `C_split ∈ {2500…8000}` on playroom.
4. Promote winning config to `configs/real.yaml`, then ablate and full-bench 4 scenes.
5. Keep Mip-NeRF360 out of full table until ply exists; document explicitly.
