from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from losses.gaussian_nll import composite_uncertainty_loss, gaussian_nll_loss, weighted_point_loss
from metrics.phm_score import compute_phm_score
from metrics.rmse import compute_rmse
from metrics.uncertainty_metrics import compute_mpiw, compute_picp
from utils.rul import clip_rul_array


def get_monitor_value(val_loss: float, val_rmse: float, monitor_name: str) -> float:
    if monitor_name == "val_loss":
        return val_loss
    if monitor_name == "val_rmse":
        return val_rmse
    raise ValueError(f"Unsupported monitor: {monitor_name}")


def maybe_clip_predictions(pred: np.ndarray, config: dict) -> np.ndarray:
    if not config["training"].get("clip_predictions", False):
        return pred
    return clip_rul_array(pred, min_value=0.0, max_value=float(config["data"]["rul_clip"]))


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    model_type: str,
    train: bool,
    config: dict,
    epoch: int | None = None,
    total_epochs: int | None = None,
    stage_name: str | None = None,
) -> dict:
    """执行一个 epoch 的训练或评估。

    统一处理 point 和 uncertainty 两种模型类型：
    - point: 前向传播 → MSE 损失
    - uncertainty: 前向传播 → (μ, logvar) → 高斯 NLL 损失

    返回包含 loss、RMSE、PHM score 的字典；
    uncertainty 模型额外返回 PICP、MPIW 等不确定性指标。
    """
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    preds = []
    targets = []
    mus = []
    logvars = []

    stage = stage_name or ("Train" if train else "Eval")
    epoch_desc = f"Epoch {epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else "Inference"
    progress = tqdm(
        loader,
        desc=f"{epoch_desc} [{stage}]",
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
                loss = weighted_point_loss(
                    pred,
                    y,
                    loss_name=config["training"].get("point_loss", "mse"),
                    low_rul_threshold=config["training"].get("low_rul_threshold"),
                    low_rul_weight=float(config["training"].get("low_rul_weight", 1.0)),
                    smooth_l1_beta=float(config["training"].get("smooth_l1_beta", 1.0)),
                )
                batch_mu = pred
                batch_logvar = None
            else:
                batch_mu, batch_logvar = model(x)
                plw = config["training"].get("point_loss_weight")
                if plw is not None and float(plw) > 0:
                    loss = composite_uncertainty_loss(
                        batch_mu, batch_logvar, y,
                        point_loss_name=config["training"].get("point_loss", "mse"),
                        point_loss_weight=float(plw),
                        low_rul_threshold=config["training"].get("low_rul_threshold"),
                        low_rul_weight=float(config["training"].get("low_rul_weight", 1.0)),
                        smooth_l1_beta=float(config["training"].get("smooth_l1_beta", 1.0)),
                    )
                else:
                    loss = gaussian_nll_loss(batch_mu, batch_logvar, y)

            if train:
                loss.backward()
                grad_clip_norm = config["training"].get("gradient_clip_norm")
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                optimizer.step()

        losses.append(loss.item())
        preds.append(batch_mu.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
        mus.append(batch_mu.detach().cpu().numpy())
        if batch_logvar is not None:
            logvars.append(batch_logvar.detach().cpu().numpy())

        progress.set_postfix(loss=f"{np.mean(losses):.4f}", batch=batch_idx)

    pred_arr = maybe_clip_predictions(np.concatenate(preds), config)
    true_arr = np.concatenate(targets)
    mu_arr = maybe_clip_predictions(np.concatenate(mus), config)
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
        sigma_arr = np.exp(0.5 * logvar_arr)  # σ = exp(0.5 * log(σ²))
        # 95% 置信区间：μ ± 1.96σ（正态分布双侧 95% 分位数）
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


def compute_engine_level_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    unit_ids: np.ndarray,
    cycles: np.ndarray,
) -> dict:
    """将窗口级预测聚合为发动机级指标（取每台发动机最后一个窗口的预测）。

    验证集使用滑动窗口（stride=1），每台发动机产生多个窗口预测；
    而测试集每台发动机只有一个窗口。为使 val / test 指标可比，
    val 也应取每台发动机最后一个周期的预测来计算 RMSE 和 PHM score。
    """
    engine_pred = []
    engine_true = []
    for uid in np.unique(unit_ids):
        mask = unit_ids == uid
        uid_cycles = cycles[mask]
        # 取该发动机最后一个周期（最大 cycle）对应的预测
        last_idx = np.argmax(uid_cycles)
        indices = np.where(mask)[0]
        engine_pred.append(pred[indices[last_idx]])
        engine_true.append(true[indices[last_idx]])
    engine_pred = np.asarray(engine_pred)
    engine_true = np.asarray(engine_true)
    return {
        "rmse": compute_rmse(engine_pred, engine_true),
        "phm_score": compute_phm_score(engine_pred, engine_true),
    }


def evaluate_on_test(model, loader, dataset, device, model_type: str, config: dict) -> dict:
    metrics = run_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        device=device,
        model_type=model_type,
        train=False,
        config=config,
        stage_name="Test",
    )
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


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_epoch: int,
    best_monitor_value: float,
    best_val_loss: float,
    best_val_rmse: float,
    epochs_without_improvement: int,
    input_dim: int,
    config: dict,
    history: list[dict],
) -> None:
    """保存完整的训练状态，支持断点续训。"""
    payload = {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_monitor_value": best_monitor_value,
        "best_val_loss": best_val_loss,
        "best_val_rmse": best_val_rmse,
        "epochs_without_improvement": epochs_without_improvement,
        "input_dim": input_dim,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "config": config,
    }
    torch.save(payload, path)
