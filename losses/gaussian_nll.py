from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_nll_loss(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    precision = torch.exp(-logvar)
    squared_error = (target - mu) ** 2
    loss = 0.5 * precision * squared_error + 0.5 * logvar
    return loss.mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)
