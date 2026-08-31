# TFR: Fixed-Tile Splatting (3DGS-style binning)

Independent prototype track (not tied to HST-GS training stack).

**Focus:** golden-aligned CUDA preprocess + CUB sort/binning + 16×16 tile raster on **native 3DGS** checkpoints.

Pipeline:

```
Gaussian → CUDA preprocess → tiles_touched → CUB scan → duplicateWithKeys
         → CUB radix sort → identifyTileRanges → sorted-list raster → Image
```

## Layout

```
TFR/
  python/tfr/            # render_from_gaussians, metrics, scene bridge
  cuda/                   # preprocess / binning / render
  scripts/                # bench + train vanilla 3DGS
  configs/
  docs/
```

## Quick start

```bash
conda activate 3dgs_cu118
cd ZhouZheng/TFR
export PYTHONPATH="$PWD/python:$PWD/cuda:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Build CUDA extension (on GPU node)
cd cuda && python setup.py build_ext --inplace && cd ..

# Bench vs native (GPU required)
srun -N 1 -n 1 --gres=gpu:1 -t 01:00:00 -p gpu \
  bash scripts/bench_tfr_render.sh --all-13 --max-views 5 --warmup 2 --repeats 5
```

## Relation to HST-GS / FastGS

TFR attaches to **native 3DGS** densify+optimize (see `configs/vanilla_3dgs.yaml`, `scripts/train_vanilla_3dgs.py`).

Do **not** use HST-GS `best_psnr26` or FastGS-Big weights as the quality base — see `docs/baseline_mismatch.md`.  
HST-GS is only reused as a **library** (scene IO / train entrypoint) with ACW and sr disabled.
