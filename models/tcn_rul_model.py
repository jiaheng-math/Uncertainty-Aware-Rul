from __future__ import annotations

import torch
import torch.nn as nn

from models.heads import GaussianHead, PointHead
from models.tcn import TCN


class TCNPointModel(nn.Module):
    def __init__(self, n_features: int, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = TCN(n_features=n_features, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = PointHead(self.encoder.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = self.encoder(x)
        return self.head(features)


class TCNUncertaintyModel(nn.Module):
    def __init__(self, n_features: int, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = TCN(n_features=n_features, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = GaussianHead(self.encoder.hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        features = self.encoder(x)
        return self.head(features)
