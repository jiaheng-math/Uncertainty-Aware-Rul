"""Model package for TCN-based RUL prediction."""

from __future__ import annotations

import torch.nn as nn

from models.tcn_rul_model import TCNPointModel, TCNUncertaintyModel


def build_model(config: dict, input_dim: int) -> nn.Module:
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


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
