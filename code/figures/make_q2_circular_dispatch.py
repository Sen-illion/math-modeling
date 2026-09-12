"""Q2 circular 24 h dispatch panel from official playback traces.

Source table: paper/assets/tables/Q2_circular_dispatch_20250320.csv
That CSV is the 2025-03-20 official day (policy D_pv7d_adaptive): plan purchase
taken from results/Q2/result2.xlsx, actual load/PV from the calendar-aligned
attachment-2 series, SOC0 from the specified-day table, and charge/discharge/SOC
from the frozen causal greedy playback. Block sums match the official
specified-day storage table.

Outputs PDF/PNG/SVG to paper/assets/figures and a PNG copy to paper/figures.
Artwork contains the polar rings and the legend only.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = REPO_ROOT / "paper" / "assets" / "tables" / "Q2_circular_dispatch_20250320.csv"
ASSET_DIR = REPO_ROOT / "paper" / "assets" / "figures"
PAPER_FIG_DIR = REPO_ROOT / "paper" / "figures"
STEM = "q2_circular_dispatch"

MM = 25.4
# Landscape manuscript slot: circle on the left, legend on the right.
FIG_MM = (168.0, 98.0)
E_MIN_KWH = 1200.0
E_MAX_KWH = 10800.0
N_SLOT = 144

# User gradient, in listed order: #2389d9 → #dda01e → #ffedcb → #897456
GRAD = ("#2389d9", "#dda01e", "#ffedcb", "#897456")
C = {
    "ink": "#1B2838",
    "muted": "#4A5C6A",
    "guide": "#2389d9",
    "hole_edge": "#897456",
    "track": "#ffedcb",
    "net": "#2389d9",
    "charge": "#2389d9",
    "charge_edge": "#897456",
    "discharge": "#dda01e",
    "soc": "#897456",
    "soc_fill": "#ffedcb",
    "center": "#1B2838",
}
CMAP_PRICE = LinearSegmentedColormap.from_list("q2_price_grad", list(GRAD))

R_HOLE = 0.27
R_SOC = (0.30, 0.445)
R_DISP_MID = 0.575
R_DISP_HALF = 0.100
R_NET = (0.695, 0.905)
R_PRICE = (0.915, 1.00)
R_LABEL = 1.145


def pick_font() -> str:
    bundled = REPO_ROOT / "paper" / "simsun.ttc"
    if bundled.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(bundled))
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
        try:
            mpl.font_manager.findfont(name, fallback_to_default=False)
            return name
        except (ValueError, OSError):
            continue
    return "DejaVu Sans"


def apply_style(font: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": 8.2,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 400,
            "savefig.bbox": "standard",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "mathtext.fontset": "custom",
            "mathtext.rm": font,
            "mathtext.it": f"{font}:italic",
            "mathtext.bf": f"{font}:bold",
        }
    )


def load_source() -> pd.DataFrame:
    if not TABLE_PATH.exists():
        raise FileNotFoundError(f"missing source table: {TABLE_PATH}")
    frame = pd.read_csv(TABLE_PATH)
    if len(frame) != N_SLOT:
        raise ValueError(f"expected {N_SLOT} slots, got {len(frame)}")
    required = (
        "price_yuan_per_kwh",
        "net_kw",
        "charge_kwh",
        "discharge_kwh",
        "soc_end_kwh",
        "soc0_kwh",
    )
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(f"source table missing {missing}")
    return frame


def _theta_centers() -> np.ndarray:
    slot = 2.0 * np.pi / N_SLOT
    return slot * (np.arange(N_SLOT) + 0.5)


def map_soc(soc: np.ndarray) -> np.ndarray:
    frac = np.clip((soc - E_MIN_KWH) / (E_MAX_KWH - E_MIN_KWH), 0.0, 1.0)
    return R_SOC[0] + frac * (R_SOC[1] - R_SOC[0])


def draw_rings(ax: plt.Axes, frame: pd.DataFrame) -> None:
    theta = _theta_centers()
    width = 2.0 * np.pi / N_SLOT
    price = frame["price_yuan_per_kwh"].to_numpy(float)
    net = frame["net_kw"].to_numpy(float)
    charge = frame["charge_kwh"].to_numpy(float)
    discharge = frame["discharge_kwh"].to_numpy(float)
    soc_end = frame["soc_end_kwh"].to_numpy(float)
    soc0 = float(frame["soc0_kwh"].iloc[0])

    norm = Normalize(vmin=float(price.min()), vmax=float(price.max()))
    price_colors = CMAP_PRICE(norm(price))

    ax.bar(
        theta,
        R_PRICE[1] - R_PRICE[0],
        width=width,
        bottom=R_PRICE[0],
        color=price_colors,
        align="center",
        linewidth=0,
        zorder=2,
    )

    net_span = max(float(net.max() - net.min()), 1e-9)
    net_frac = 0.10 + 0.90 * (net - net.min()) / net_span
    ax.bar(
        theta,
        net_frac * (R_NET[1] - R_NET[0]),
        width=width * 0.90,
        bottom=R_NET[0],
        color=C["net"],
        align="center",
        linewidth=0,
        zorder=3,
    )

    ax.bar(
        theta,
        2.0 * R_DISP_HALF,
        width=width,
        bottom=R_DISP_MID - R_DISP_HALF,
        color=C["track"],
        align="center",
        linewidth=0,
        zorder=4,
    )
    power = max(float(np.max(charge)), float(np.max(discharge)), 1e-9)
    disp_scale = R_DISP_HALF / power
    ax.bar(
        theta,
        charge * disp_scale,
        width=width * 0.80,
        bottom=R_DISP_MID,
        color=C["charge"],
        edgecolor=C["charge_edge"],
        align="center",
        linewidth=0.22,
        zorder=6,
    )
    ax.bar(
        theta,
        discharge * disp_scale,
        width=width * 0.80,
        bottom=R_DISP_MID - discharge * disp_scale,
        color=C["discharge"],
        align="center",
        linewidth=0,
        zorder=6,
    )

    guide = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(guide, np.full_like(guide, R_DISP_MID), color=GRAD[0], lw=0.65, zorder=5, alpha=0.45)
    ax.plot(guide, np.full_like(guide, R_NET[0]), color=C["guide"], lw=0.4, zorder=2)
    ax.plot(guide, np.full_like(guide, R_PRICE[0]), color="white", lw=0.55, zorder=4)

    soc = np.concatenate(([soc0], soc_end))
    theta_soc = 2.0 * np.pi * np.arange(N_SLOT + 1) / N_SLOT
    r_soc = map_soc(soc)
    ax.fill_between(theta_soc, R_SOC[0], r_soc, color=C["soc_fill"], zorder=5, linewidth=0)
    ax.plot(theta_soc, r_soc, color=C["soc"], lw=1.7, zorder=8, solid_capstyle="round")
    hour_idx = np.arange(0, N_SLOT, 6)
    ax.scatter(
        theta_soc[hour_idx],
        r_soc[hour_idx],
        s=11,
        color=C["soc"],
        edgecolors="white",
        linewidths=0.4,
        zorder=9,
    )

    ax.fill_between(guide, 0.0, R_HOLE, color="white", zorder=20)
    ax.plot(guide, np.full_like(guide, R_HOLE), color=C["hole_edge"], lw=0.9, zorder=21)
    ax.text(
        0.0,
        0.0,
        "24 h",
        ha="center",
        va="center",
        fontsize=13.5,
        color=C["center"],
        zorder=22,
        fontweight="bold",
    )

    for hour in range(24):
        ang = 2.0 * np.pi * hour / 24.0
        bold = hour in (0, 12)
        ax.text(
            ang,
            R_LABEL,
            f"{hour:02d}:00",
            ha="center",
            va="center",
            fontsize=8.1 if bold else 6.9,
            color=C["ink"] if bold else C["muted"],
            fontweight="bold" if bold else "normal",
            zorder=30,
        )


def style_polar(ax: plt.Axes) -> None:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.28)
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("white")
    ax.set_title("")


def draw_legend(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("")

    ax.text(0.0, 0.96, "电价色带", ha="left", va="top", fontsize=10.0, color=C["ink"], fontweight="bold")
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(
        grad,
        aspect="auto",
        cmap=CMAP_PRICE,
        extent=(0.0, 1.0, 0.84, 0.90),
        origin="lower",
        zorder=2,
    )
    ax.add_patch(
        Rectangle((0.0, 0.84), 1.0, 0.06, fill=False, edgecolor=GRAD[0], lw=0.5, zorder=3)
    )
    ax.text(0.0, 0.815, "低价", ha="left", va="top", fontsize=7.4, color=C["muted"])
    ax.text(0.5, 0.815, "中价", ha="center", va="top", fontsize=7.4, color=C["muted"])
    ax.text(1.0, 0.815, "高价", ha="right", va="top", fontsize=7.4, color=C["muted"])

    items = [
        (C["net"], "净负荷环", "负荷 − 光伏，越长净负荷越大", "patch"),
        (C["charge"], "调度环 · 充电", "蓝色柱向外：盈余时段补能", "patch"),
        (C["discharge"], "调度环 · 放电", "金色柱向内：缺口时段放电", "patch"),
        (C["soc"], "SOC 轨迹", "环线：1200–10800 kWh 荷电状态", "line"),
    ]
    y = 0.68
    for color, title, subtitle, kind in items:
        if kind == "line":
            ax.plot([0.02, 0.12], [y, y], color=color, lw=2.2, solid_capstyle="round")
            ax.scatter([0.07], [y], s=18, color=color, edgecolors="white", linewidths=0.4, zorder=4)
        else:
            ax.add_patch(
                FancyBboxPatch(
                    (0.02, y - 0.018),
                    0.10,
                    0.036,
                    boxstyle="round,pad=0.004,rounding_size=0.008",
                    linewidth=0,
                    facecolor=color,
                    mutation_aspect=0.4,
                )
            )
        ax.text(0.16, y, title, ha="left", va="center", fontsize=9.2, color=C["ink"], fontweight="bold")
        ax.text(0.16, y - 0.055, subtitle, ha="left", va="center", fontsize=7.2, color=C["muted"])
        y -= 0.168


def save(fig: mpl.figure.Figure) -> dict[str, str]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("pdf", "png", "svg"):
        dest = ASSET_DIR / f"{STEM}.{ext}"
        fig.savefig(dest)
        paths[ext] = dest.as_posix()
    shutil.copyfile(ASSET_DIR / f"{STEM}.png", PAPER_FIG_DIR / f"{STEM}.png")
    paths["paper_png"] = (PAPER_FIG_DIR / f"{STEM}.png").as_posix()
    return paths


def main() -> dict[str, str]:
    font = pick_font()
    apply_style(font)
    frame = load_source()

    w, h = FIG_MM
    fig = plt.figure(figsize=(w / MM, h / MM))
    fig.set_size_inches(w / MM, h / MM, forward=True)

    ax = fig.add_axes([0.015, 0.03, 0.58, 0.94], projection="polar")
    ax_leg = fig.add_axes([0.64, 0.10, 0.33, 0.82])
    style_polar(ax)
    draw_rings(ax, frame)
    draw_legend(ax_leg)
    paths = save(fig)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    out = main()
    for key, path in out.items():
        print(f"{key}: {path}")
