from __future__ import annotations

import numpy as np


def compute_phm_score(pred, true) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    d = pred - true
    score = np.where(d < 0.0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))
