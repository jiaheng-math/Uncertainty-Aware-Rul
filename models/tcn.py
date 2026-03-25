from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.conv = nn.utils.weight_norm(conv) if use_weight_norm else conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, use_weight_norm=True)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, use_weight_norm=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.relu(out + residual)


class TCN(nn.Module):
    def __init__(
        self,
        n_features: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = n_features
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.network = nn.ModuleList(layers)
        self.hidden_dim = num_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.network:
            x = layer(x)
        return x[:, :, -1]
