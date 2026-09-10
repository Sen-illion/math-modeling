"""Q1 diagnostic figures."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ASSET_FIGURE_DIR, E0_KWH, FIGURE_DIR, REPO_ROOT


def _setup_font() -> None:
    font_path = REPO_ROOT / "paper" / "simsun.ttc"
    if font_path.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.sans-serif"] = ["SimSun", "NSimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_all(frame: pd.DataFrame, dispatch: dict, metrics: dict) -> list[Path]:
    _setup_font()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    hours = frame["end_min"].to_numpy() / 60.0
    paths = [
        _plot_price_load_pv(hours, frame),
        _plot_purchase_soc(hours, dispatch),
        _plot_charge_discharge(hours, dispatch),
        _plot_baseline_cost(metrics),
    ]
    copied = []
    for path in paths:
        dest = ASSET_FIGURE_DIR / path.name
        shutil.copy(path, dest)
        copied.append(dest)
    return paths


def _plot_price_load_pv(hours: np.ndarray, frame: pd.DataFrame) -> Path:
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(hours, frame["load_kw"], label="load kW", color="#1f4e79")
    ax1.plot(hours, frame["pv_kw"], label="PV kW", color="#2e7d32")
    ax1.set_xlabel("hour")
    ax1.set_ylabel("power (kW)")
    ax2 = ax1.twinx()
    ax2.plot(hours, frame["price"], label="price", color="#c62828", linestyle="--")
    ax2.set_ylabel("price (yuan/kWh)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Q1 price, load and PV")
    ax1.set_xlim(0, 24)
    fig.tight_layout()
    path = FIGURE_DIR / "Q1_price_load_pv.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_purchase_soc(hours: np.ndarray, dispatch: dict) -> Path:
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.step(hours, dispatch["purchase_kwh"], where="pre", label="purchase kWh", color="#1565c0")
    ax1.set_ylabel("purchase (kWh / 10 min)")
    ax2 = ax1.twinx()
    soc = np.concatenate(([E0_KWH], dispatch["soc_end_kwh"]))
    soc_hours = np.concatenate(([0.0], hours))
    ax2.plot(soc_hours, soc, label="SOC", color="#6a1b9a")
    ax2.set_ylabel("SOC (kWh)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Q1 purchase and SOC")
    ax1.set_xlim(0, 24)
    fig.tight_layout()
    path = FIGURE_DIR / "Q1_purchase_soc.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_charge_discharge(hours: np.ndarray, dispatch: dict) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(hours, dispatch["charge_kwh"], width=0.12, label="charge", color="#2e7d32")
    ax.bar(hours, -dispatch["discharge_kwh"], width=0.12, label="discharge", color="#c62828")
    ax.set_xlabel("hour")
    ax.set_ylabel("battery-side energy (kWh)")
    ax.set_title("Q1 charge and discharge")
    ax.set_xlim(0, 24)
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / "Q1_charge_discharge.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_baseline_cost(metrics: dict) -> Path:
    names = ["B0", "B1", "M0"]
    costs = [metrics[name]["cost"] for name in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, costs, color=["#90a4ae", "#fb8c00", "#1565c0"])
    ax.set_ylabel("daily purchase cost (yuan)")
    ax.set_title("Q1 baseline cost comparison")
    for i, cost in enumerate(costs):
        ax.text(i, cost, f"{cost:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    path = FIGURE_DIR / "Q1_baseline_cost.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
