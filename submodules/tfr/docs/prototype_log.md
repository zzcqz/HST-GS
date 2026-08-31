# TFR Prototype / Research Log

Date: 2026-07-15  
Loop: `scripts/run_research_loop.py` (phases 1→5)

## Phase 1 — Analysis

Bottlenecks: fixed 16×16 imbalance, scattered SoA reads, empty-ish work.  
See `docs/problem_analysis.md`.

## Phase 2 — Optimize (with rethink)

**First tune failure:** absolute `C_merge≤800` never merged on playroom (mean \(C(b)\approx 2295\)).  

**Fix:** merge uses **density** \(C(T)/\#bases\). Retuned grid → promoted:

```
C_merge: 4000
C_split: 5000
```

(`configs/real.yaml`)

## Phase 3 — Verify (playroom test view0)

| Metric | Value |
|--------|-------|
| Golden PSNR vs GT | **29.88** |
| Fixed base tiles | 4108 |
| Adaptive tiles | **325** (ratio 0.079) |
| Levels 16/32/64 | 52 / 26 / 247 |
| H1 tile_ratio ≤ 0.5 | **True** |
| Backend | CUDA complexity + assignment |

## Phase 4 — Ablation (playroom)

| Variant | Tiles | Reduction vs fixed |
|---------|-------|--------------------|
| A0 fixed L0 | 4108 | 0 |
| A1 merge L1 | 1069 | 0.740 |
| A2 full ATS L2 | **325** | **0.921** |
| A3 ATS no pack | 325 | 0.921 |
| A5 conservative | 478 | 0.884 |

Takeaway: Contribution 1 (ATS / merge levels) drives tile count; pack is orthogonal I/O prep (same tile counts).

## Phase 5 — Full (4 scenes w/ `best_psnr26`)

| Scene | Reduction | Adaptive/Fixed | PSNR mean (all test) | PSNR view0 (misleading) |
|-------|-----------|----------------|----------------------|-------------------------|
| playroom | 0.921 | 325/4108 | **30.75** | 29.88 |
| drjohnson | 0.914 | 399/4620 | **29.57** | 35.25 (=max!) |
| Truck | 0.838 | 352/2170 | **24.75** | 26.18 |
| Train | 0.806 | 421/2170 | **20.82** | 23.25 |
| **Mean** | **0.870** | — | **26.47** | — |

**Bug found:** earlier table used single `view0` PSNR; for drjohnson view0 is the easiest frame (35.25). Fixed to full-test mean (matches `results.json`).


## Remaining work (next iteration)

1. **Packed forward raster kernel** into golden fork — measure true ms vs `diff_gaussian_rasterization` (current golden FPS is checkpoint render only).
2. Cap/wave-split long lists (`C_split` work items) — L still millions.
3. Train Mip-NeRF360 checkpoints or link existing plies for 13-scene table.
4. Differentiable scatter backward (v1).

## Reproduce

```bash
srun -N 1 -n 1 --gres=gpu:1 -t 02:00:00 -p gpu \
  bash ZhouZheng/TFR/scripts/run_all_phases.sh
```
