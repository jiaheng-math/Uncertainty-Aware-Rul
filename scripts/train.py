from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.cmapss_dataset import build_dataloaders
from losses.gaussian_nll import gaussian_nll_loss, mse_loss
from metrics.phm_score import compute_phm_score
from metrics.rmse import compute_rmse
from metrics.uncertainty_metrics import compute_mpiw, compute_picp
from models.tcn_rul_model import TCNPointModel, TCNUncertaintyModel
from utils.logger import append_results_summary, get_timestamp, save_history, save_json, setup_logger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TCN model on CMAPSS.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def ensure_output_dirs(config: dict) -> None:
    for key in ["results_dir", "figures_dir", "checkpoint_dir", "logs_dir"]:
        Path(config["output"][key]).mkdir(parents=True, exist_ok=True)


def build_model(config: dict, input_dim: int) -> torch.nn.Module:
    model_cfg = config["model"]
    common_kwargs = {
        "n_features": input_dim,
        "num_channels": model_cfg["num_channels"],
        "kernel_size": model_cfg["kernel_size"],
        "dropout": model_cfg["dropout"],
    }
    if model_cfg["type"] == "point":
        return TCNPointModel(**common_kwargs)
    if model_cfg["type"] == "uncertainty":
        return TCNUncertaintyModel(**common_kwargs)
    raise ValueError(f"Unsupported model type: {model_cfg['type']}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def run_epoch(model, loader, optimizer, device, model_type: str, train: bool, epoch: int, total_epochs: int) -> dict:
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    preds = []
    targets = []
    mus = []
    logvars = []

    stage = "Train" if train else "Val"
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [{stage}]",
        leave=False,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )

    for batch_idx, (x, y) in enumerate(progress, start=1):
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if model_type == "point":
                pred = model(x)
                loss = mse_loss(pred, y)
                batch_mu = pred
                batch_logvar = None
            else:
                batch_mu, batch_logvar = model(x)
                loss = gaussian_nll_loss(batch_mu, batch_logvar, y)

            if train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        preds.append(batch_mu.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
        mus.append(batch_mu.detach().cpu().numpy())
        if batch_logvar is not None:
            logvars.append(batch_logvar.detach().cpu().numpy())

        progress.set_postfix(loss=f"{np.mean(losses):.4f}", batch=batch_idx)

    pred_arr = np.concatenate(preds)
    true_arr = np.concatenate(targets)
    mu_arr = np.concatenate(mus)
    metrics = {
        "loss": float(np.mean(losses)),
        "rmse": compute_rmse(pred_arr, true_arr),
        "phm_score": compute_phm_score(pred_arr, true_arr),
        "pred": pred_arr,
        "true": true_arr,
        "mu": mu_arr,
    }

    if logvars:
        logvar_arr = np.concatenate(logvars)
        sigma_arr = np.exp(0.5 * logvar_arr)
        lower = mu_arr - 1.96 * sigma_arr
        upper = mu_arr + 1.96 * sigma_arr
        metrics.update(
            {
                "logvar": logvar_arr,
                "sigma_mean": float(np.mean(sigma_arr)),
                "sigma_std": float(np.std(sigma_arr)),
                "picp": compute_picp(lower, upper, true_arr),
                "mpiw": compute_mpiw(lower, upper),
                "lower": lower,
                "upper": upper,
            }
        )
    return metrics


def evaluate_on_test(model, loader, dataset, device, model_type: str) -> dict:
    metrics = run_epoch(model=model, loader=loader, optimizer=None, device=device, model_type=model_type, train=False)
    result = {
        "test_rmse": metrics["rmse"],
        "test_phm_score": metrics["phm_score"],
        "unit_ids": dataset.unit_ids.tolist(),
        "true_rul": dataset.y.cpu().numpy().tolist(),
        "pred_mu": metrics["mu"].tolist(),
    }
    if model_type == "uncertainty":
        result["test_picp"] = metrics["picp"]
        result["test_mpiw"] = metrics["mpiw"]
        result["lower"] = metrics["lower"].tolist()
        result["upper"] = metrics["upper"].tolist()
        result["sigma_mean"] = metrics["sigma_mean"]
        result["sigma_std"] = metrics["sigma_std"]
    return result


def save_checkpoint(path: Path, model: torch.nn.Module, epoch: int, best_val_loss: float, input_dim: int, config: dict) -> None:
    payload = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "input_dim": input_dim,
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    set_seed(config["training"]["seed"])

    subset = config["data"]["subset"]
    model_type = config["model"]["type"]
    timestamp = get_timestamp()
    logger = setup_logger(
        name=f"train_{subset}_{model_type}",
        log_dir=config["output"]["logs_dir"],
        filename=f"train_{subset}_{model_type}_{timestamp}.log",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading data for subset %s", subset)
    bundle = build_dataloaders(config)
    bundle.feature_processor.save(Path(config["output"]["checkpoint_dir"]) / f"scaler_{subset}_{model_type}.json")
    logger.info("Train units: %d | Val units: %d", len(bundle.train_units), len(bundle.val_units))
    logger.info("Removed sensors: %s", bundle.feature_processor.removed_sensor_columns)
    logger.info("Kept sensors: %s", bundle.feature_processor.kept_sensor_columns)
    logger.info("Input features: %s", bundle.feature_processor.feature_columns)
    logger.info(
        "Dataset sizes | train windows: %d | val windows: %d | test engines: %d",
        len(bundle.train_dataset),
        len(bundle.val_dataset),
        len(bundle.test_dataset),
    )

    model = build_model(config, bundle.input_dim).to(device)
    logger.info("Model parameters: %d", count_parameters(model))

    optimizer_name = config["training"]["optimizer"]
    weight_decay = config["training"].get("weight_decay", 0.0)
    if optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=config["training"]["lr"], weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=weight_decay)
    else:
        raise ValueError("Only Adam and AdamW optimizers are implemented.")

    if config["training"]["scheduler"] != "ReduceLROnPlateau":
        raise ValueError("Only ReduceLROnPlateau scheduler is implemented.")
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config["training"]["scheduler_patience"],
        factor=config["training"]["scheduler_factor"],
    )

    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / f"best_model_{subset}_{model_type}.pth"
    history_path = Path(config["output"]["logs_dir"]) / f"history_{subset}_{model_type}.csv"
    train_summary_path = Path(config["output"]["logs_dir"]) / f"train_summary_{subset}_{model_type}.json"
    results_summary_path = Path(config["output"]["results_dir"]) / "results_summary.csv"

    monitor_name = config["training"].get("early_stopping_monitor", "val_loss")
    if monitor_name != "val_loss":
        raise ValueError("Only val_loss is supported for early stopping monitor.")

    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    total_epochs = config["training"]["epochs"]
    for epoch in range(1, total_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=bundle.train_loader,
            optimizer=optimizer,
            device=device,
            model_type=model_type,
            train=True,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        val_metrics = run_epoch(
            model=model,
            loader=bundle.val_loader,
            optimizer=optimizer,
            device=device,
            model_type=model_type,
            train=False,
            epoch=epoch,
            total_epochs=total_epochs,
        )

        scheduler.step(val_metrics["loss"])

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"],
            "val_phm_score": val_metrics["phm_score"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if model_type == "uncertainty":
            record["val_sigma_mean"] = val_metrics["sigma_mean"]
            record["val_sigma_std"] = val_metrics["sigma_std"]
        history.append(record)

        log_line = (
            f"Epoch {epoch}/{total_epochs} | Train Loss {train_metrics['loss']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | Val RMSE {val_metrics['rmse']:.2f} | "
            f"Val Score {val_metrics['phm_score']:.2f}"
        )
        if model_type == "uncertainty":
            log_line += (
                f" | Val Sigma Mean {val_metrics['sigma_mean']:.2f}"
                f" | Val Sigma Std {val_metrics['sigma_std']:.2f}"
            )
        logger.info(log_line)

        current_monitor = val_metrics["loss"]
        if current_monitor < best_val_loss:
            best_val_loss = current_monitor
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_path, model, epoch, best_val_loss, bundle.input_dim, config)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config["training"]["early_stopping_patience"]:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    save_history(history, history_path)
    save_json(
        {
            "subset": subset,
            "model_type": model_type,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "history_path": str(history_path),
            "checkpoint_path": str(checkpoint_path),
        },
        train_summary_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate_on_test(model, bundle.test_loader, bundle.test_dataset, device, model_type)

    result_record = {
        "timestamp": timestamp,
        "subset": subset,
        "model_type": model_type,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_rmse": test_result["test_rmse"],
        "test_phm_score": test_result["test_phm_score"],
    }
    if model_type == "uncertainty":
        result_record["test_picp"] = test_result["test_picp"]
        result_record["test_mpiw"] = test_result["test_mpiw"]
    append_results_summary(result_record, results_summary_path)

    test_log_path = Path(config["output"]["logs_dir"]) / f"test_metrics_{subset}_{model_type}_{timestamp}.json"
    save_json(test_result, test_log_path)

    logger.info("Best checkpoint saved to %s", checkpoint_path)
    logger.info("Test RMSE: %.4f | Test PHM Score: %.4f", test_result["test_rmse"], test_result["test_phm_score"])
    if model_type == "uncertainty":
        logger.info("Test PICP: %.4f | Test MPIW: %.4f", test_result["test_picp"], test_result["test_mpiw"])


if __name__ == "__main__":
    main()
