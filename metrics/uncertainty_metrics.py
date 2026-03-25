from __future__ import annotations

import numpy as np


def compute_picp(lower, upper, true) -> float:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.mean((true >= lower) & (true <= upper)))


def compute_mpiw(lower, upper) -> float:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return float(np.mean(upper - lower))
