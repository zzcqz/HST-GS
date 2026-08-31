# TFR (eval-only submodule)

TFR lives under `HST-GS/submodules/tfr`.

TFR is a **validation / render-time** accelerator for fixed-tile 3DGS forward.
It is **not** part of the training loop.

## Build

```bash
cd submodules/tfr/cuda
# on a GPU node
python setup.py build_ext --inplace
```

## Use (eval / `render.py` only)

```bash
cd HST-GS
python render.py -m <model> --use_tfr
# or
HSTGS_USE_TFR=1 python render.py -m <model>
```

Optional: `TFR_ROOT=/path/to/TFR` if needed.

## Training

Do **not** pass `--use_tfr` to `train.py`. Training always uses
`diff-gaussian-rasterization`. `train.py` will exit if TFR is requested.

## Method stack

HST-GS training: **HTS** (Super-Tile cull) + **SR** (refine FB schedule) + ACW densify.  
Eval acceleration: optional **TFR**.

Full mechanism (Chinese): [`tfr/docs/TFR_MECHANISM_HSTGS.md`](tfr/docs/TFR_MECHANISM_HSTGS.md).
