from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.cmapss_dataset import build_dataloaders
from models import build_model, count_parameters
from utils.experiment import get_experiment_name
from utils.logger import append_results_summary, get_timestamp, save_history, save_json, setup_logger
from utils.seed import set_seed
from utils.training import (
    compute_engine_level_metrics,
    evaluate_on_test,
    get_monitor_value,
    run_epoch,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TCN model on CMAPSS.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def ensure_output_dirs(config: dict) -> None:
    for key in ["results_dir", "figures_dir", "checkpoint_dir", "logs_dir"]:
        Path(config["output"][key]).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    set_seed(config["training"]["seed"])

    subset = config["data"]["subset"]
    model_type = config["model"]["type"]
    experiment_name = get_experiment_name(config, args.config)
    timestamp = get_timestamp()
    logger = setup_logger(
        name=f"train_{experiment_name}",
        log_dir=config["output"]["logs_dir"],
        filename=f"train_{experiment_name}_{timestamp}.log",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading data for subset %s", subset)
    bundle = build_dataloaders(config)
    bundle.feature_processor.save(Path(config["output"]["checkpoint_dir"]) / f"scaler_{experiment_name}.json")
    logger.info("Train units: %d | Val units: %d", len(bundle.train_units), len(bundle.val_units))
    logger.info("Removed sensors: %s", bundle.feature_processor.removed_sensor_columns)
    logger.info("Kept sensors: %s", bundle.feature_processor.kept_sensor_columns)
    logger.info("Input features: %s", bundle.feature_processor.feature_columns)
    logger.info(
        "Dataset sizes | train windows: %d | val windows: %d | val eval samples: %d | test engines: %d",
        len(bundle.train_dataset),
        len(bundle.val_dataset),
        len(bundle.val_eval_dataset),
        len(bundle.test_dataset),
    )
    logger.info("Validation mode: %s", bundle.validation_mode)

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

    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / f"best_model_{experiment_name}.pth"
    # 断点续训用的 latest checkpoint（每个 epoch 都保存，包含完整训练状态）
    latest_checkpoint_path = Path(config["output"]["checkpoint_dir"]) / f"latest_model_{experiment_name}.pth"
    history_path = Path(config["output"]["logs_dir"]) / f"history_{experiment_name}.csv"
    train_summary_path = Path(config["output"]["logs_dir"]) / f"train_summary_{experiment_name}.json"
    results_summary_path = Path(config["output"]["results_dir"]) / "results_summary.csv"

    monitor_name = config["training"].get("early_stopping_monitor", "val_loss")
    scheduler_monitor = config["training"].get("scheduler_monitor", monitor_name)

    best_monitor_value = math.inf
    best_val_loss = math.inf
    best_val_rmse = math.inf
    best_epoch = 0
    epochs_without_improvement = 0  # 早停计数器
    history = []
    start_epoch = 1

    # ========== 断点续训：从 latest checkpoint 恢复完整训练状态 ==========
    if args.resume and latest_checkpoint_path.exists():
        ckpt = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_epoch = ckpt.get("best_epoch", ckpt["epoch"])
        best_monitor_value = ckpt.get("best_monitor_value", math.inf)
        best_val_loss = ckpt["best_val_loss"]
        best_val_rmse = ckpt.get("best_val_rmse", math.inf)
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        history = ckpt.get("history", [])
        logger.info(
            "Resumed from epoch %d (best_val_loss=%.4f, patience=%d/%d)",
            ckpt["epoch"],
            best_val_loss,
            epochs_without_improvement,
            config["training"]["early_stopping_patience"],
        )
    elif args.resume:
        logger.info("No checkpoint found at %s, training from scratch.", latest_checkpoint_path)

    # ========== 训练循环 ==========
    total_epochs = config["training"]["epochs"]
    for epoch in range(start_epoch, total_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=bundle.train_loader,
            optimizer=optimizer,
            device=device,
            model_type=model_type,
            train=True,
            config=config,
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
            config=config,
            epoch=epoch,
            total_epochs=total_epochs,
        )

        if bundle.validation_mode == "pseudo_test":
            val_eval_metrics = run_epoch(
                model=model,
                loader=bundle.val_eval_loader,
                optimizer=optimizer,
                device=device,
                model_type=model_type,
                train=False,
                config=config,
                epoch=epoch,
                total_epochs=total_epochs,
                stage_name="ValPseudoTest",
            )
            val_engine = {
                "rmse": val_eval_metrics["rmse"],
                "phm_score": val_eval_metrics["phm_score"],
            }
        else:
            # window 模式下，聚合每台发动机最后一个窗口的预测
            val_engine = compute_engine_level_metrics(
                val_metrics["pred"],
                val_metrics["true"],
                bundle.val_dataset.unit_ids,
                bundle.val_dataset.cycles,
            )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_rmse": val_engine["rmse"],
            "val_phm_score": val_engine["phm_score"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if model_type == "uncertainty":
            record["val_sigma_mean"] = val_metrics["sigma_mean"]
            record["val_sigma_std"] = val_metrics["sigma_std"]
        history.append(record)

        scheduler.step(get_monitor_value(val_metrics["loss"], val_engine["rmse"], scheduler_monitor))

        log_line = (
            f"Epoch {epoch}/{total_epochs} | Train Loss {train_metrics['loss']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | Val RMSE {val_engine['rmse']:.2f} | "
            f"Val Score {val_engine['phm_score']:.2f}"
        )
        if model_type == "uncertainty":
            log_line += (
                f" | Val Sigma Mean {val_metrics['sigma_mean']:.2f}"
                f" | Val Sigma Std {val_metrics['sigma_std']:.2f}"
            )
        logger.info(log_line)

        current_monitor = get_monitor_value(val_metrics["loss"], val_engine["rmse"], monitor_name)
        if current_monitor < best_monitor_value:
            best_monitor_value = current_monitor
            best_val_loss = val_metrics["loss"]
            best_val_rmse = val_engine["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            # 保存最佳模型（用于最终评估）
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler,
                epoch, best_epoch, best_monitor_value, best_val_loss, best_val_rmse,
                epochs_without_improvement, bundle.input_dim, config, history,
            )
        else:
            epochs_without_improvement += 1

        # 每个 epoch 都保存 latest checkpoint（用于断点续训）
        save_checkpoint(
            latest_checkpoint_path, model, optimizer, scheduler,
            epoch, best_epoch, best_monitor_value, best_val_loss, best_val_rmse,
            epochs_without_improvement, bundle.input_dim, config, history,
        )

        if epochs_without_improvement >= config["training"]["early_stopping_patience"]:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    # ========== 训练结束，保存结果并在测试集上评估 ==========
    save_history(history, history_path)
    save_json(
        {
            "subset": subset,
            "model_type": model_type,
            "experiment_name": experiment_name,
            "best_epoch": best_epoch,
            "best_monitor": monitor_name,
            "best_monitor_value": best_monitor_value,
            "best_val_loss": best_val_loss,
            "best_val_rmse": best_val_rmse,
            "history_path": str(history_path),
            "checkpoint_path": str(checkpoint_path),
        },
        train_summary_path,
    )

    # 加载最佳 checkpoint（而非最后一个 epoch 的模型）进行测试
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate_on_test(model, bundle.test_loader, bundle.test_dataset, device, model_type, config)

    result_record = {
        "timestamp": timestamp,
        "subset": subset,
        "model_type": model_type,
        "experiment_name": experiment_name,
        "best_epoch": best_epoch,
        "best_monitor": monitor_name,
        "best_monitor_value": best_monitor_value,
        "best_val_loss": best_val_loss,
        "best_val_rmse": best_val_rmse,
        "test_rmse": test_result["test_rmse"],
        "test_phm_score": test_result["test_phm_score"],
    }
    if model_type == "uncertainty":
        result_record["test_picp"] = test_result["test_picp"]
        result_record["test_mpiw"] = test_result["test_mpiw"]
    append_results_summary(result_record, results_summary_path)

    test_log_path = Path(config["output"]["logs_dir"]) / f"test_metrics_{experiment_name}_{timestamp}.json"
    save_json(test_result, test_log_path)

    # ========== 训练总结 ==========
    logger.info("=" * 60)
    logger.info("Training complete")
    logger.info("  Best epoch       : %d / %d", best_epoch, total_epochs)
    logger.info("  Best val loss    : %.4f", best_val_loss)
    logger.info("  Best val RMSE    : %.4f", best_val_rmse)
    logger.info("  Checkpoint       : %s", checkpoint_path)
    logger.info("-" * 60)
    logger.info("  Test RMSE        : %.4f", test_result["test_rmse"])
    logger.info("  Test PHM Score   : %.4f", test_result["test_phm_score"])
    if model_type == "uncertainty":
        logger.info("  Test PICP        : %.4f", test_result["test_picp"])
        logger.info("  Test MPIW        : %.4f", test_result["test_mpiw"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
