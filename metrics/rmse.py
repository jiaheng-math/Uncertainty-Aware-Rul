from __future__ import annotations

import numpy as np


def compute_rmse(pred, true) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2)))
