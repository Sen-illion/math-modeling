"""Q4 paper figures: intraday shape, weekday effect, and recent price level.

The price panel follows the same calendar alignment as the Q4 price model.
No optimization is run and no formal result files are modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import fontManager


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from Q4.load_data import load_att1_curve, load_year_prices  # noqa: E402


OUT_DIR = REPO_ROOT / "paper" / "figures"
COLORS = {
    "teal": "#4C78A8",
    "coral": "#E3A13A",
    "ink": "#333333",
    "muted": "#666666",
    "grid": "#D9DEE3",
    "light": "#EEF2F5",
}


def configure_style() -> str:
    bundled = REPO_ROOT / "paper" / "simsun.ttc"
    if bundled.exists():
        fontManager.addfont(str(bundled))
    font = "SimSun"
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": 9.2,
            "font.weight": "bold",
            "axes.labelsize": 10.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 11.0,
            "axes.titleweight": "bold",
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "legend.fontsize": 8.5,
            "text.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )
    return font


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick.set_fontweight("bold")


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_intraday(price: np.ndarray, correlations: np.ndarray) -> None:
    hours = np.arange(price.shape[1]) / 6.0
    mean = np.mean(price, axis=0)
    q10, q25, q75, q90 = np.quantile(price, [0.10, 0.25, 0.75, 0.90], axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    ax.fill_between(hours, q10, q90, color=COLORS["teal"], alpha=0.12, linewidth=0,
                    label="10%—90%分位区间")
    ax.fill_between(hours, q25, q75, color=COLORS["teal"], alpha=0.25, linewidth=0,
                    label="25%—75%分位区间")
    ax.plot(hours, mean, color=COLORS["teal"], linewidth=2.4, label="全年同刻平均电价")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xlabel("日内时刻")
    ax.set_ylabel("电价（元/kWh）")
    ax.set_title("附件4电价的日内形态与跨日波动区间")
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False, ncol=1)
    ax.text(
        0.985,
        0.965,
        f"单日曲线与平均曲线\n相关系数中位数：{np.median(correlations):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS["grid"], "alpha": 0.92},
    )
    fig.tight_layout(pad=0.8)
    save_figure(fig, "q4_price_intraday_pattern")


def plot_weekday(price: np.ndarray, dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    dow = dates.dt.weekday.to_numpy()
    overall = np.mean(price, axis=0)
    weekday_mean = np.vstack([np.mean(price[dow == k], axis=0) for k in range(7)])
    deviation = weekday_mean - overall
    vmax = float(np.max(np.abs(deviation)))
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_gold", [COLORS["teal"], "#FAFAFA", COLORS["coral"]]
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    im = ax.imshow(
        deviation,
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        extent=(0, 24, 6.5, -0.5),
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels(["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"])
    ax.set_xlabel("日内时刻")
    ax.set_ylabel("星期")
    ax.set_title("不同星期日期相对全年同刻均价的电价偏差")
    for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick.set_fontweight("bold")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("电价偏差（元/kWh）", fontweight="bold")
    for tick in cb.ax.get_yticklabels():
        tick.set_fontweight("bold")
    fig.tight_layout(pad=0.8)
    save_figure(fig, "q4_price_weekday_effect")
    return weekday_mean, deviation


def plot_recent(price: np.ndarray, dates: pd.Series, lag7_corr: float) -> None:
    daily_mean = pd.Series(np.mean(price, axis=1), index=pd.DatetimeIndex(dates))
    rolling7 = daily_mean.rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.scatter(
        daily_mean.index,
        daily_mean.values,
        s=16,
        color=COLORS["teal"],
        alpha=0.38,
        linewidths=0,
        label="日均电价",
        zorder=2,
    )
    ax.plot(
        rolling7.index,
        rolling7.values,
        color=COLORS["coral"],
        linewidth=2.1,
        label="近7日均值",
        zorder=3,
    )
    ax.set_xlim(daily_mean.index.min(), daily_mean.index.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    ax.set_xlabel("日期")
    ax.set_ylabel("日均电价（元/kWh）")
    ax.set_title("全年日均电价及近期价格水平变化")
    style_axis(ax)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.text(
        0.015,
        0.965,
        f"日均电价7日滞后相关系数：{lag7_corr:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": COLORS["grid"], "alpha": 0.92},
    )
    fig.tight_layout(pad=0.8)
    save_figure(fig, "q4_price_recent_level")


def main() -> int:
    font = configure_style()
    cold = load_att1_curve()
    year = load_year_prices(cold)
    dates = pd.to_datetime(year["dates"])
    price = np.asarray(year["price"], dtype=float)

    mean_curve = np.mean(price, axis=0)
    correlations = np.array([np.corrcoef(row, mean_curve)[0, 1] for row in price])
    daily_mean = np.mean(price, axis=1)
    lag7_corr = float(np.corrcoef(daily_mean[:-7], daily_mean[7:])[0, 1])

    plot_intraday(price, correlations)
    weekday_mean, deviation = plot_weekday(price, dates)
    plot_recent(price, dates, lag7_corr)

    index = {
        "source": year["source"],
        "alignment": year["alignment"],
        "n_days": int(price.shape[0]),
        "n_slots_per_day": int(price.shape[1]),
        "font": font,
        "figures": {
            "q4_price_intraday_pattern": {
                "claim": "附件4电价具有稳定的日内形态，同时存在跨日波动。",
                "median_daily_correlation_with_mean_curve": float(np.median(correlations)),
            },
            "q4_price_weekday_effect": {
                "claim": "相同日内时刻的价格随星期日期呈系统性偏差。",
                "weekday_daily_mean_yuan_per_kwh": [float(x) for x in weekday_mean.mean(axis=1)],
                "max_abs_weekday_slot_deviation_yuan_per_kwh": float(np.max(np.abs(deviation))),
            },
            "q4_price_recent_level": {
                "claim": "日均价格水平随日期变化，近7日均值可表征近期水平。",
                "daily_mean_lag7_correlation": lag7_corr,
            },
        },
        "outputs": [
            str(OUT_DIR / f"{stem}.{suffix}")
            for stem in (
                "q4_price_intraday_pattern",
                "q4_price_weekday_effect",
                "q4_price_recent_level",
            )
            for suffix in ("png", "pdf", "svg")
        ],
    }
    (OUT_DIR / "q4_price_structure_figure_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
