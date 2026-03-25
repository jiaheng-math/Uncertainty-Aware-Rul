from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StandardScalerState:
    mean_: np.ndarray
    scale_: np.ndarray
    feature_names: list[str]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "feature_names": self.feature_names,
        }
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "StandardScalerState":
        with Path(path).open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return cls(
            mean_=np.asarray(payload["mean"], dtype=np.float64),
            scale_=np.asarray(payload["scale"], dtype=np.float64),
            feature_names=list(payload["feature_names"]),
        )


class FeatureStandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.feature_names: list[str] = []

    def fit(self, x: np.ndarray, feature_names: list[str]) -> "FeatureStandardScaler":
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x.shape}")
        self.mean_ = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale
        self.feature_names = list(feature_names)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before calling transform.")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray, feature_names: list[str]) -> np.ndarray:
        self.fit(x, feature_names)
        return self.transform(x)

    def state_dict(self) -> StandardScalerState:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return StandardScalerState(
            mean_=self.mean_.copy(),
            scale_=self.scale_.copy(),
            feature_names=list(self.feature_names),
        )

    def load_state_dict(self, state: StandardScalerState) -> None:
        self.mean_ = state.mean_.copy()
        self.scale_ = state.scale_.copy()
        self.feature_names = list(state.feature_names)

    def save(self, path: str | Path) -> None:
        self.state_dict().save(path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureStandardScaler":
        scaler = cls()
        scaler.load_state_dict(StandardScalerState.load(path))
        return scaler
