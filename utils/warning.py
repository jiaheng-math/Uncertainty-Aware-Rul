from __future__ import annotations

import math


LEVELS = ["正常", "关注", "预警", "危险"]


def _resolve_warning_config(config: dict) -> dict:
    if "warning" in config:
        return config["warning"]
    return config


def get_warning_level(mu: float, logvar: float, config: dict) -> dict:
    warning_cfg = _resolve_warning_config(config)
    thresholds = warning_cfg["thresholds"]
    sigma_threshold = warning_cfg["sigma_threshold"]
    sigma_escalation = warning_cfg.get("sigma_escalation", True)

    sigma = math.exp(0.5 * float(logvar))
    lower = float(mu) - 1.96 * sigma

    if lower > thresholds["normal"]:
        level_idx = 0
    elif lower > thresholds["watch"]:
        level_idx = 1
    elif lower > thresholds["alert"]:
        level_idx = 2
    else:
        level_idx = 3

    escalated = False
    if sigma_escalation and sigma > sigma_threshold and level_idx < len(LEVELS) - 1:
        level_idx += 1
        escalated = True

    return {
        "level": LEVELS[level_idx],
        "escalated": escalated,
        "lower": lower,
        "sigma": sigma,
    }
