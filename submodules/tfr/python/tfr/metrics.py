"""Small evaluation helpers."""

from __future__ import annotations

import numpy as np


def psnr(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """PSNR for images in [0,1], shape (H,W,3) or (3,H,W)."""
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(gt, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = np.transpose(a, (1, 2, 0))
    if b.ndim == 3 and b.shape[0] == 3 and b.shape[-1] != 3:
        b = np.transpose(b, (1, 2, 0))
    mse = np.mean((a - b) ** 2)
    if mse < eps:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))
