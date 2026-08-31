# Baseline mismatch: why Truck/Train PSNR looked “too low”

## Symptom

TFR Phase5 reported (full test mean) on `best_psnr26`:

| Scene | Ours (reported) | FastGS-Big ref | FastGS-Big N |
|-------|-----------------|----------------|--------------|
| Truck | **24.75** / 0.73M | **26.09** / 0.63M | |
| Train | **20.82** / 0.41M | **22.68** / 0.46M | |

## Root cause (not a PSNR-code bug)

We evaluated **HST-GS `best_psnr26`**, an *accelerated* training recipe:

- earlier densify freeze (`densify_until≈8000`)
- sparser densify interval (`500` vs vanilla/`FastGS` often `100`)
- SR FB skipping
- ACW densify weighting

AAAI draft already notes Truck/Train PSNR lags FastGS-Big by **~1.3–1.9 dB** under this recipe. TFR merely **read that checkpoint**; it did not cause the quality drop.

So comparing those numbers to FastGS-Big is an **apples-to-oranges baseline**.

## Resolution

1. **Stop using `best_psnr26` as TFR quality base.**
2. **Re-host TFR on native 3DGS** (Kerbl densify schedule, no SR, ACW disabled).
3. Re-measure PSNR on vanilla checkpoints; then attach Adaptive Tile / Pack on top of that stack.

Vanilla recipe used in `scripts/train_vanilla_3dgs.py`:

- `densify_until_iter=15000`
- `densification_interval=100`
- `percent_dense=0.01`
- `grad_weight_base=1`, `grad_weight_scale=0` (disable ACW)
- no `--sr`, no `--fast_train`

## Resolution status

Native Truck/Train trained and TFR-benched (`outputs/native_bench/report.json`):

| Scene | vanilla PSNR | FastGS-Big | Δ |
|-------|--------------|------------|---|
| Truck | 25.36 | 26.09 | −0.73 |
| Train | 21.73 | 22.68 | −0.95 |

Loader prefers `TFR/outputs/vanilla_3dgs/{scene}` over `best_psnr26`.

