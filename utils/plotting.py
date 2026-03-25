from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_output(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss_curve(history: pd.DataFrame, best_epoch: int, output_path: str | Path) -> None:
    output_path = _prepare_output(output_path)
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2.0)
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", linewidth=2.0)
    plt.axvline(best_epoch, color="tab:red", linestyle="--", label=f"Best Epoch {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_test_predictions(
    unit_ids: np.ndarray,
    true_rul: np.ndarray,
    pred_mu: np.ndarray,
    output_path: str | Path,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> None:
    output_path = _prepare_output(output_path)
    order = np.argsort(unit_ids)
    x_axis = np.arange(len(order))

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, true_rul[order], label="True RUL", linewidth=2.0, color="black")
    plt.plot(x_axis, pred_mu[order], label="Predicted RUL", linewidth=2.0, color="tab:blue")
    if lower is not None and upper is not None:
        plt.fill_between(x_axis, lower[order], upper[order], color="gray", alpha=0.25, label="95% CI")
    plt.xlabel("Test Engine Index")
    plt.ylabel("RUL")
    plt.title("Test Prediction vs Ground Truth")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_engine_degradation(trajectories: list[dict], output_path: str | Path) -> None:
    output_path = _prepare_output(output_path)
    n_rows = len(trajectories)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    for ax, item in zip(axes, trajectories):
        ax.plot(item["cycles"], item["true_rul"], label="True RUL", color="black", linewidth=2.0)
        ax.plot(item["cycles"], item["pred_mu"], label="Predicted RUL", color="tab:blue", linewidth=2.0)
        if item.get("lower") is not None and item.get("upper") is not None:
            ax.fill_between(item["cycles"], item["lower"], item["upper"], color="gray", alpha=0.25, label="95% CI")
        ax.set_title(f"Validation Unit {item['unit_id']}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_warning_demo(trajectories: list[dict], output_path: str | Path) -> None:
    output_path = _prepare_output(output_path)
    n_rows = len(trajectories)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    level_map = {"正常": 0, "关注": 1, "预警": 2, "危险": 3}
    ytick_labels = ["正常", "关注", "预警", "危险"]

    for ax, item in zip(axes, trajectories):
        levels = np.array([level_map[level] for level in item["warning_levels"]], dtype=np.int64)
        ax.step(item["cycles"], levels, where="post", linewidth=2.0, color="tab:red", label="Warning Level")
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(ytick_labels)
        ax.set_ylim(-0.2, 3.2)
        ax.set_title(f"Warning Evolution - Unit {item['unit_id']}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Level")
        ax.grid(alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(item["cycles"], item["pred_mu"], color="tab:blue", linewidth=1.8, label="Predicted RUL")
        ax2.plot(item["cycles"], item["lower"], color="tab:green", linestyle="--", linewidth=1.5, label="Lower Bound")
        ax2.set_ylabel("RUL")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
