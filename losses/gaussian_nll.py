from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_nll_loss(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """异方差高斯负对数似然损失。

    假设 target ~ N(μ, σ²)，其中 logvar = log(σ²)，则负对数似然为：
        NLL = 0.5 * (target - μ)² / σ² + 0.5 * log(σ²) + const
    用 precision = 1/σ² = exp(-logvar) 代替除法，提高数值稳定性。

    该损失会同时优化预测精度（第一项）和不确定性校准（第二项）：
    - 第一项鼓励 μ 接近 target
    - 第二项惩罚过大的 σ（防止模型通过放大方差来减小第一项）
    """
    precision = torch.exp(-logvar)          # 1/σ² = exp(-log(σ²))
    squared_error = (target - mu) ** 2
    loss = 0.5 * precision * squared_error + 0.5 * logvar
    return loss.mean()


def composite_uncertainty_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    point_loss_name: str = "mse",
    point_loss_weight: float = 0.1,
    low_rul_threshold: float | None = None,
    low_rul_weight: float = 1.0,
    smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    """复合不确定性损失：NLL + λ × 点预测损失。

    纯 NLL 损失可能导致模型通过放大 σ² 来降低损失，而不是提高 μ 精度。
    添加独立的点预测损失项为 μ 提供直接的梯度信号，使不确定性模型的
    预测精度逼近点模型水平，同时保留 σ 的校准能力。

    总损失 = NLL(μ, σ², y) + point_loss_weight × PointLoss(μ, y)
    """
    nll = gaussian_nll_loss(mu, logvar, target)
    point = weighted_point_loss(
        mu, target, point_loss_name,
        low_rul_threshold=low_rul_threshold,
        low_rul_weight=low_rul_weight,
        smooth_l1_beta=smooth_l1_beta,
    )
    return nll + point_loss_weight * point


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """标准均方误差损失，用于点预测模型。"""
    return F.mse_loss(pred, target)


def weighted_point_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    low_rul_threshold: float | None = None,
    low_rul_weight: float = 1.0,
    smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    """点预测任务的可配置损失。

    可选 Smooth L1，并对低 RUL 样本施加更高权重，
    让训练更关注临近失效区间。
    """
    if loss_name == "mse":
        base_loss = (pred - target) ** 2
    elif loss_name == "smooth_l1":
        base_loss = F.smooth_l1_loss(pred, target, beta=smooth_l1_beta, reduction="none")
    else:
        raise ValueError(f"Unsupported point loss: {loss_name}")

    if low_rul_threshold is None or low_rul_weight <= 1.0:
        return base_loss.mean()

    weights = torch.ones_like(target)
    weights = torch.where(target <= low_rul_threshold, low_rul_weight, weights)
    return (base_loss * weights).mean()
