"""Utility helpers for GridWorld project."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that serializes numpy types into native Python types."""

    def default(self, obj):  # type: ignore[override]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_level_to_json(level: Dict, path: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(level, fp, indent=2, cls=NumpyEncoder)


def load_level_from_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def render_grid_heatmap(
    grid: np.ndarray,
    title: str,
    output_path: str,
    cmap: str = "coolwarm",
    show_colorbar: bool = False,
) -> None:
    ensure_dir(Path(output_path).parent)
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap=cmap)
    plt.title(title)
    if show_colorbar:
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_reward_curve(timesteps: List[int], rewards: List[float], output_path: str) -> None:
    ensure_dir(Path(output_path).parent)
    plt.figure(figsize=(7, 4))
    plt.plot(timesteps, rewards, label="Reward media 100 episodi")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward media")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_report(path: str, lines: Iterable[str]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fp:
        for line in lines:
            fp.write(f"{line}\n")


def aggregate_success_rates(results: Dict[str, List[float]]) -> Dict[str, float]:
    return {level: float(np.mean(values)) if values else 0.0 for level, values in results.items()}


def moving_average(values: List[float], window: int = 50) -> np.ndarray:
    if len(values) < window:
        return np.array(values, dtype=float)
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window
