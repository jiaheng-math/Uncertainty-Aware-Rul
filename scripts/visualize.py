from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.cmapss_dataset import build_dataloaders, build_unit_trajectory_windows
from models.tcn_rul_model import TCNPointModel, TCNUncertaintyModel
from utils.plotting import plot_engine_degradation, plot_loss_curve, plot_test_predictions, plot_warning_demo
from utils.warning import get_warning_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CMAPSS training and prediction outputs.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    return parser.parse_args()


def build_model(config: dict, input_dim: int) -> torch.nn.Module:
    kwargs = {
        "n_features": input_dim,
        "num_channels": config["model"]["num_channels"],
        "kernel_size": config["model"]["kernel_size"],
        "dropout": config["model"]["dropout"],
    }
    if config["model"]["type"] == "point":
        return TCNPointModel(**kwargs)
    return TCNUncertaintyModel(**kwargs)


def predict_dataset(model, loader, dataset, device, model_type: str) -> dict:
    model.eval()
    mu_list = []
    logvar_list = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if model_type == "point":
                mu = model(x)
                logvar = None
            else:
                mu, logvar = model(x)
            mu_list.append(mu.cpu().numpy())
            if logvar is not None:
                logvar_list.append(logvar.cpu().numpy())

    mu = np.concatenate(mu_list)
    payload = {
        "unit_ids": dataset.unit_ids.copy(),
        "true_rul": dataset.y.cpu().numpy(),
        "pred_mu": mu,
    }
    if logvar_list:
        logvar = np.concatenate(logvar_list)
        sigma = np.exp(0.5 * logvar)
        payload["lower"] = mu - 1.96 * sigma
        payload["upper"] = mu + 1.96 * sigma
        payload["logvar"] = logvar
    return payload


def predict_unit_trajectory(model, windows: np.ndarray, device, model_type: str) -> dict:
    model.eval()
    x = torch.as_tensor(windows, dtype=torch.float32).to(device)
    with torch.no_grad():
        if model_type == "point":
            mu = model(x).cpu().numpy()
            return {"pred_mu": mu, "lower": None, "upper": None, "logvar": None}
        mu, logvar = model(x)
        mu = mu.cpu().numpy()
        logvar = logvar.cpu().numpy()
        sigma = np.exp(0.5 * logvar)
        return {
            "pred_mu": mu,
            "lower": mu - 1.96 * sigma,
            "upper": mu + 1.96 * sigma,
            "logvar": logvar,
        }


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    subset = config["data"]["subset"]
    model_type = config["model"]["type"]
    bundle = build_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, bundle.input_dim).to(device)

    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / f"best_model_{subset}_{model_type}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    figures_dir = Path(config["output"]["figures_dir"])
    history_path = Path(config["output"]["logs_dir"]) / f"history_{subset}_{model_type}.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")
    history = pd.read_csv(history_path)
    plot_loss_curve(history, best_epoch=int(checkpoint["epoch"]), output_path=figures_dir / f"loss_curve_{subset}_{model_type}.png")

    test_payload = predict_dataset(model, bundle.test_loader, bundle.test_dataset, device, model_type)
    plot_test_predictions(
        unit_ids=test_payload["unit_ids"],
        true_rul=test_payload["true_rul"],
        pred_mu=test_payload["pred_mu"],
        lower=test_payload.get("lower"),
        upper=test_payload.get("upper"),
        output_path=figures_dir / f"rul_prediction_{subset}_{model_type}.png",
    )

    selected_units = bundle.val_units[: min(4, len(bundle.val_units))]
    degradation_payloads = []
    warning_payloads = []
    for unit_id in selected_units:
        windows, true_rul, cycles = build_unit_trajectory_windows(
            frame=bundle.val_df,
            features=bundle.val_features,
            unit_id=unit_id,
            window_size=config["data"]["window_size"],
            padding_mode=config["data"]["padding_mode"],
        )
        pred_payload = predict_unit_trajectory(model, windows, device, model_type)
        degradation_payloads.append(
            {
                "unit_id": unit_id,
                "cycles": cycles,
                "true_rul": true_rul,
                "pred_mu": pred_payload["pred_mu"],
                "lower": pred_payload["lower"],
                "upper": pred_payload["upper"],
            }
        )

        if model_type == "uncertainty":
            warning_levels = [
                get_warning_level(mu=float(mu), logvar=float(logvar), config=config)["level"]
                for mu, logvar in zip(pred_payload["pred_mu"], pred_payload["logvar"])
            ]
            warning_payloads.append(
                {
                    "unit_id": unit_id,
                    "cycles": cycles,
                    "pred_mu": pred_payload["pred_mu"],
                    "lower": pred_payload["lower"],
                    "warning_levels": warning_levels,
                }
            )
        else:
            pseudo_logvar = np.full_like(pred_payload["pred_mu"], fill_value=np.log(1.0), dtype=np.float32)
            warning_levels = [
                get_warning_level(mu=float(mu), logvar=float(logvar), config=config)["level"]
                for mu, logvar in zip(pred_payload["pred_mu"], pseudo_logvar)
            ]
            warning_payloads.append(
                {
                    "unit_id": unit_id,
                    "cycles": cycles,
                    "pred_mu": pred_payload["pred_mu"],
                    "lower": pred_payload["pred_mu"],
                    "warning_levels": warning_levels,
                }
            )

    plot_engine_degradation(
        degradation_payloads,
        figures_dir / f"engine_degradation_{subset}.png",
    )
    plot_warning_demo(
        warning_payloads,
        figures_dir / f"warning_demo_{subset}.png",
    )
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
