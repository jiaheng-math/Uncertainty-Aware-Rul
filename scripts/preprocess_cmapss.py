from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.cmapss_dataset import build_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CMAPSS data for RUL prediction.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    bundle = build_dataloaders(config)
    data_cfg = config["data"]
    output_cfg = config["output"]

    scaler_path = Path(output_cfg["checkpoint_dir"]) / f"scaler_{data_cfg['subset']}.json"
    bundle.feature_processor.save(scaler_path)

    print(f"Subset: {data_cfg['subset']}")
    print(f"Train units: {len(bundle.train_units)} | Val units: {len(bundle.val_units)}")
    print(f"Removed sensors: {bundle.feature_processor.removed_sensor_columns}")
    print(f"Kept sensors: {bundle.feature_processor.kept_sensor_columns}")
    print(f"Final feature columns: {bundle.feature_processor.feature_columns}")
    print(f"Train windows: {len(bundle.train_dataset)}")
    print(f"Val windows: {len(bundle.val_dataset)}")
    print(f"Test engines: {len(bundle.test_dataset)}")
    print(f"Scaler saved to: {scaler_path}")


if __name__ == "__main__":
    main()
