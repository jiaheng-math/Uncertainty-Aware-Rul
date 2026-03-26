from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.cmapss_dataset import build_dataloaders
from losses.gaussian_nll import gaussian_nll_loss, weighted_point_loss
from metrics.phm_score import compute_phm_score
from metrics.rmse import compute_rmse
from metrics.uncertainty_metrics import compute_mpiw, compute_picp
from models import build_model
from utils.experiment import get_experiment_name
from utils.logger import get_timestamp, save_json, setup_logger
from utils.rul import clip_rul_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TCN model on CMAPSS test set.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    return parser.parse_args()


def evaluate(model, loader, dataset, device, model_type: str, config: dict) -> dict:
    model.eval()
    losses = []
    mu_list = []
    true_list = []
    logvar_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if model_type == "point":
                mu = model(x)
                loss = weighted_point_loss(
                    mu,
                    y,
                    loss_name=config["training"].get("point_loss", "mse"),
                    low_rul_threshold=config["training"].get("low_rul_threshold"),
                    low_rul_weight=float(config["training"].get("low_rul_weight", 1.0)),
                    smooth_l1_beta=float(config["training"].get("smooth_l1_beta", 1.0)),
                )
                logvar = None
            else:
                mu, logvar = model(x)
                loss = gaussian_nll_loss(mu, logvar, y)
            losses.append(loss.item())
            mu_list.append(mu.cpu().numpy())
            true_list.append(y.cpu().numpy())
            if logvar is not None:
                logvar_list.append(logvar.cpu().numpy())

    mu = np.concatenate(mu_list)
    if config["training"].get("clip_predictions", False):
        mu = clip_rul_array(mu, min_value=0.0, max_value=float(config["data"]["rul_clip"]))
    true = np.concatenate(true_list)
    payload = {
        "loss": float(np.mean(losses)),
        "test_rmse": compute_rmse(mu, true),
        "test_phm_score": compute_phm_score(mu, true),
        "unit_ids": dataset.unit_ids.tolist(),
        "true_rul": true.tolist(),
        "pred_mu": mu.tolist(),
    }
    if logvar_list:
        logvar = np.concatenate(logvar_list)
        sigma = np.exp(0.5 * logvar)
        lower = mu - 1.96 * sigma
        upper = mu + 1.96 * sigma
        payload["test_picp"] = compute_picp(lower, upper, true)
        payload["test_mpiw"] = compute_mpiw(lower, upper)
        payload["sigma_mean"] = float(np.mean(sigma))
        payload["sigma_std"] = float(np.std(sigma))
        payload["lower"] = lower.tolist()
        payload["upper"] = upper.tolist()
    return payload


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    subset = config["data"]["subset"]
    model_type = config["model"]["type"]
    experiment_name = get_experiment_name(config, args.config)
    timestamp = get_timestamp()
    logger = setup_logger(
        name=f"eval_{experiment_name}",
        log_dir=config["output"]["logs_dir"],
        filename=f"evaluate_{experiment_name}_{timestamp}.log",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_dataloaders(config)
    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / f"best_model_{experiment_name}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(config, bundle.input_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    payload = evaluate(model, bundle.test_loader, bundle.test_dataset, device, model_type, config)
    logger.info("Loss: %.4f", payload["loss"])
    logger.info("Test RMSE: %.4f", payload["test_rmse"])
    logger.info("Test PHM Score: %.4f", payload["test_phm_score"])
    if model_type == "uncertainty":
        logger.info("Test PICP: %.4f", payload["test_picp"])
        logger.info("Test MPIW: %.4f", payload["test_mpiw"])

    eval_path = Path(config["output"]["logs_dir"]) / f"evaluation_{experiment_name}_{timestamp}.json"
    save_json(payload, eval_path)
    print(f"Evaluation saved to: {eval_path}")


if __name__ == "__main__":
    main()
