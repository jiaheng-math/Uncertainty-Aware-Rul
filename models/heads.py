from __future__ import annotations

import torch
import torch.nn as nn


class PointHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class GaussianHead(nn.Module):
    def __init__(self, hidden_dim: int, clamp_min: float = -10.0, clamp_max: float = 10.0) -> None:
        super().__init__()
        self.head_mu = nn.Linear(hidden_dim, 1)
        self.head_logvar = nn.Linear(hidden_dim, 1)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.head_mu(x).squeeze(-1)
        logvar = self.head_logvar(x).squeeze(-1).clamp(self.clamp_min, self.clamp_max)
        return mu, logvar
