from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.scaler import FeatureStandardScaler


BASE_COLUMNS = ["unit_id", "cycle", "op1", "op2", "op3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
ALL_COLUMNS = BASE_COLUMNS + SENSOR_COLUMNS


@dataclass
class FeatureProcessor:
    include_op_settings: bool
    var_threshold: float
    kept_sensor_columns: list[str]
    removed_sensor_columns: list[str]
    feature_columns: list[str]
    scaler: FeatureStandardScaler

    def transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature_columns].to_numpy(dtype=np.float32)
        return self.scaler.transform(x).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "include_op_settings": self.include_op_settings,
            "var_threshold": self.var_threshold,
            "kept_sensor_columns": self.kept_sensor_columns,
            "removed_sensor_columns": self.removed_sensor_columns,
            "feature_columns": self.feature_columns,
            "scaler": {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
                "feature_names": self.scaler.feature_names,
            },
        }
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)


class CMAPSSWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        unit_ids: np.ndarray,
        cycles: np.ndarray,
        mode: str,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.unit_ids = np.asarray(unit_ids, dtype=np.int64)
        self.cycles = np.asarray(cycles, dtype=np.int64)
        self.mode = mode

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


@dataclass
class CMAPSSDataBundle:
    train_dataset: CMAPSSWindowDataset
    val_dataset: CMAPSSWindowDataset
    test_dataset: CMAPSSWindowDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    feature_processor: FeatureProcessor
    train_units: list[int]
    val_units: list[int]
    input_dim: int
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_features: np.ndarray
    val_features: np.ndarray
    test_features: np.ndarray


def ensure_subset_files(data_dir: str | Path, subset: str, zip_path: str | Path = "CMAPSSData.zip") -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    expected_files = [
        f"train_{subset}.txt",
        f"test_{subset}.txt",
        f"RUL_{subset}.txt",
    ]
    missing = [name for name in expected_files if not (data_dir / name).exists()]
    if not missing:
        return

    archive_path = Path(zip_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing data archive: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as archive:
        for name in expected_files:
            archive.extract(name, path=data_dir)


def load_cmapss_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if frame.shape[1] < len(ALL_COLUMNS):
        raise ValueError(f"Unexpected file shape {frame.shape} for {path}")
    frame = frame.iloc[:, : len(ALL_COLUMNS)].copy()
    frame.columns = ALL_COLUMNS
    frame = frame.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    return frame


def load_rul_targets(path: str | Path, unit_ids: list[int], rul_clip: int) -> dict[int, float]:
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    values = frame.iloc[:, 0].to_numpy(dtype=np.float32)
    if len(values) != len(unit_ids):
        raise ValueError("Mismatch between test units and RUL target count.")
    return {unit_id: float(min(values[idx], rul_clip)) for idx, unit_id in enumerate(sorted(unit_ids))}


def add_train_rul(frame: pd.DataFrame, rul_clip: int) -> pd.DataFrame:
    frame = frame.copy()
    max_cycle = frame.groupby("unit_id")["cycle"].transform("max")
    rul_raw = max_cycle - frame["cycle"]
    frame["RUL_raw"] = rul_raw
    frame["RUL"] = np.minimum(rul_raw, rul_clip).astype(np.float32)
    return frame


def split_train_val_units(frame: pd.DataFrame, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    unit_ids = sorted(frame["unit_id"].unique().tolist())
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unit_ids)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_units = sorted(shuffled[:val_count].tolist())
    train_units = sorted(shuffled[val_count:].tolist())
    return train_units, val_units


def fit_feature_processor(train_frame: pd.DataFrame, include_op_settings: bool, var_threshold: float) -> FeatureProcessor:
    sensor_variances = train_frame[SENSOR_COLUMNS].var(axis=0, ddof=0)
    kept_sensor_columns = sensor_variances[sensor_variances >= var_threshold].index.tolist()
    removed_sensor_columns = sensor_variances[sensor_variances < var_threshold].index.tolist()

    feature_columns = []
    if include_op_settings:
        feature_columns.extend(["op1", "op2", "op3"])
    feature_columns.extend(kept_sensor_columns)

    scaler = FeatureStandardScaler()
    scaler.fit(train_frame[feature_columns].to_numpy(dtype=np.float32), feature_columns)

    return FeatureProcessor(
        include_op_settings=include_op_settings,
        var_threshold=var_threshold,
        kept_sensor_columns=kept_sensor_columns,
        removed_sensor_columns=removed_sensor_columns,
        feature_columns=feature_columns,
        scaler=scaler,
    )


def pad_sequence_left(sequence: np.ndarray, target_len: int, mode: str) -> np.ndarray:
    if len(sequence) >= target_len:
        return sequence[-target_len:]
    pad_len = target_len - len(sequence)
    if mode == "repeat":
        pad_values = np.repeat(sequence[:1], pad_len, axis=0)
    elif mode == "zero":
        pad_values = np.zeros((pad_len, sequence.shape[1]), dtype=sequence.dtype)
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")
    return np.concatenate([pad_values, sequence], axis=0)


def make_sliding_window_dataset(
    frame: pd.DataFrame,
    features: np.ndarray,
    window_size: int,
    stride: int,
    mode: str,
    padding_mode: str,
    test_rul_map: dict[int, float] | None = None,
) -> CMAPSSWindowDataset:
    windows = []
    labels = []
    unit_ids = []
    cycles = []

    for unit_id, unit_frame in frame.groupby("unit_id", sort=True):
        idx = unit_frame.index.to_numpy()
        unit_features = features[idx]
        unit_cycles = unit_frame["cycle"].to_numpy()

        if mode == "test":
            window = pad_sequence_left(unit_features, window_size, padding_mode)
            windows.append(window)
            labels.append(test_rul_map[int(unit_id)])
            unit_ids.append(unit_id)
            cycles.append(unit_cycles[-1])
            continue

        unit_rul = unit_frame["RUL"].to_numpy(dtype=np.float32)
        if len(unit_features) < window_size:
            continue
        end_indices = range(window_size - 1, len(unit_features), stride)
        for end_idx in end_indices:
            start_idx = end_idx - window_size + 1
            windows.append(unit_features[start_idx : end_idx + 1])
            labels.append(unit_rul[end_idx])
            unit_ids.append(unit_id)
            cycles.append(unit_cycles[end_idx])

    x = np.stack(windows).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    unit_ids_arr = np.asarray(unit_ids, dtype=np.int64)
    cycles_arr = np.asarray(cycles, dtype=np.int64)
    return CMAPSSWindowDataset(x=x, y=y, unit_ids=unit_ids_arr, cycles=cycles_arr, mode=mode)


def build_unit_trajectory_windows(
    frame: pd.DataFrame,
    features: np.ndarray,
    unit_id: int,
    window_size: int,
    padding_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit_frame = frame[frame["unit_id"] == unit_id].copy()
    idx = unit_frame.index.to_numpy()
    unit_features = features[idx]
    unit_cycles = unit_frame["cycle"].to_numpy(dtype=np.int64)
    unit_rul = unit_frame["RUL"].to_numpy(dtype=np.float32)

    windows = []
    for end_idx in range(len(unit_features)):
        start_idx = max(0, end_idx - window_size + 1)
        window = unit_features[start_idx : end_idx + 1]
        window = pad_sequence_left(window, window_size, padding_mode)
        windows.append(window)
    return np.stack(windows).astype(np.float32), unit_rul, unit_cycles


def build_dataloaders(config: dict) -> CMAPSSDataBundle:
    data_cfg = config["data"]
    training_cfg = config["training"]

    subset = data_cfg["subset"]
    data_dir = Path(data_cfg["data_dir"])
    ensure_subset_files(data_dir=data_dir, subset=subset)

    train_frame = add_train_rul(load_cmapss_frame(data_dir / f"train_{subset}.txt"), data_cfg["rul_clip"])
    test_frame = load_cmapss_frame(data_dir / f"test_{subset}.txt")

    train_units, val_units = split_train_val_units(train_frame, data_cfg["val_ratio"], training_cfg["seed"])
    train_split = train_frame[train_frame["unit_id"].isin(train_units)].reset_index(drop=True)
    val_split = train_frame[train_frame["unit_id"].isin(val_units)].reset_index(drop=True)

    feature_processor = fit_feature_processor(
        train_frame=train_split,
        include_op_settings=data_cfg["include_op_settings"],
        var_threshold=data_cfg["var_threshold"],
    )

    train_features = feature_processor.transform_frame(train_split)
    val_features = feature_processor.transform_frame(val_split)
    test_features = feature_processor.transform_frame(test_frame)

    test_rul_map = load_rul_targets(
        data_dir / f"RUL_{subset}.txt",
        unit_ids=sorted(test_frame["unit_id"].unique().tolist()),
        rul_clip=data_cfg["rul_clip"],
    )

    train_dataset = make_sliding_window_dataset(
        frame=train_split,
        features=train_features,
        window_size=data_cfg["window_size"],
        stride=1,
        mode="train",
        padding_mode=data_cfg["padding_mode"],
    )
    val_dataset = make_sliding_window_dataset(
        frame=val_split,
        features=val_features,
        window_size=data_cfg["window_size"],
        stride=1,
        mode="val",
        padding_mode=data_cfg["padding_mode"],
    )
    test_dataset = make_sliding_window_dataset(
        frame=test_frame,
        features=test_features,
        window_size=data_cfg["window_size"],
        stride=1,
        mode="test",
        padding_mode=data_cfg["padding_mode"],
        test_rul_map=test_rul_map,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return CMAPSSDataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        feature_processor=feature_processor,
        train_units=train_units,
        val_units=val_units,
        input_dim=len(feature_processor.feature_columns),
        train_df=train_split,
        val_df=val_split,
        test_df=test_frame,
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
    )
