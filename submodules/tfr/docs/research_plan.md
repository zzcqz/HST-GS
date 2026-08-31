# TFR Research Plan (iterative)

## Goal

Bring TFR from synthetic smoke to real 3D-GS checkpoints with:
problem analysis → optimize → single-scene verify → ablation → multi-scene full bench.
Rethink thresholds and packing after each phase.

## Available assets

| Scene | Checkpoint | Images/Sparse |
|-------|------------|---------------|
| Deep_Blending/playroom | `best_psnr26` | yes |
| Deep_Blending/drjohnson | `best_psnr26` | yes |
| Tanks_Temples/Truck | `best_psnr26` | yes |
| Tanks_Temples/Train | `best_psnr26` | yes |
| Mip-NeRF360/* | **no ply yet** | yes (source only) |

Full-volume this round = **4 scenes with `best_psnr26`**. Mip-NeRF360 deferred until checkpoints exist.

## Phase checklist

1. **Analyze** — bottleneck + gap vs golden (`docs/problem_analysis.md`)
2. **Optimize** — real projection, threshold sweep, pack path, CUDA ops hardening
3. **Verify** — playroom single-view: tile stats + golden PSNR(GT) + TFR internal consistency
4. **Ablate** — ATS off/on, merge levels, pack on/off, C_merge/C_split grid
5. **Full** — 4 scenes × best config → `docs/experiments.md`

## Success criteria (this iteration)

- Real-scene complexity heatmaps + adaptive tile reduction vs fixed 16×16
- Threshold config promoted from synthetic defaults into `configs/real.yaml`
- Ablation table showing each contribution’s incremental effect
- 4-scene summary; note Mip-NeRF360 as blocked by missing ply
