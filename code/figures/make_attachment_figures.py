"""Contest-paper figures from attachments 1-4.

Run from any cwd:

    python code/figures/make_attachment_figures.py

Outputs PNG/PDF/SVG to paper/assets/figures and PNG copies to paper/figures.
All numbers are computed from the official Excel attachments. Do not treat
this script as a model result generator.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "paper" / "assets" / "figures"
PAPER_FIG_DIR = REPO_ROOT / "paper" / "figures"
INDEX_PATH = ASSET_DIR / "attachment_figure_index.json"

MM = 25.4
DT_H = 1.0 / 6.0
N_SLOT = 144
N_DAY = 365
ISSUE_HOURS = (0, 6, 12, 18)
MONTH_START = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
MONTH_LAB = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

# Slot classes (physical millimetres). Keep identical within a class.
SLOT = {
    "stack2": (160.0, 108.0),
    "heat2": (160.0, 124.0),
    "split": (160.0, 96.0),
    "grid4": (160.0, 112.0),
}

C = {
    "load": "#2F5F8A",
    "pv": "#C4A35A",
    "net": "#A33B3B",
    "price": "#5E4B73",
    "ink": "#1F2933",
    "muted": "#5B6770",
    "grid": "#E4E7EB",
    "peak": "#B72230",
    "band": "#D9E3EE",
}

CMAP_LOAD = LinearSegmentedColormap.from_list(
    "load", ["#F7F4EE", "#E7C07A", "#C36B3A", "#7A1F1F"]
)
CMAP_PV = LinearSegmentedColormap.from_list(
    "pv", ["#F4F6F4", "#C9D7A8", "#6FA06A", "#2C5E3A"]
)
CMAP_PRICE = LinearSegmentedColormap.from_list(
    "price", ["#EAF1F6", "#8FA6C4", "#3F5F86", "#8B2E2E"]
)


def find_attachments() -> dict[str, Path]:
    att_root = REPO_ROOT / "problem_files"
    att_dir = next(p for p in att_root.iterdir() if p.is_dir())
    files = sorted(
        (p for p in att_dir.glob("*.xlsx") if not p.name.startswith("~$")),
        key=lambda p: p.stat().st_size,
    )
    if len(files) < 4:
        raise FileNotFoundError(f"expected 4 attachment workbooks in {att_dir}")
    return {"att1": files[0], "att3": files[1], "att4": files[2], "att2": files[3]}


def pick_font() -> str:
    bundled = REPO_ROOT / "paper" / "simsun.ttc"
    if bundled.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(bundled))
        return "SimSun"
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
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
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#334155",
            "axes.labelcolor": C["ink"],
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "axes.spines.top": False,
            "axes.spines.right": False,
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


def new_fig(slot: str) -> mpl.figure.Figure:
    w, h = SLOT[slot]
    fig = plt.figure(figsize=(w / MM, h / MM))
    fig.set_size_inches(w / MM, h / MM, forward=True)
    return fig


def polish(ax: plt.Axes, ygrid: bool = True) -> None:
    ax.tick_params(length=3.0, width=0.7)
    if ygrid:
        ax.grid(axis="y", color=C["grid"], linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.0,
        1.03,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=C["ink"],
        clip_on=False,
    )


def save(fig: mpl.figure.Figure, stem: str) -> dict[str, str]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("png", "pdf", "svg"):
        dest = ASSET_DIR / f"{stem}.{ext}"
        fig.savefig(dest)
        paths[ext] = dest.as_posix()
    shutil.copyfile(ASSET_DIR / f"{stem}.png", PAPER_FIG_DIR / f"{stem}.png")
    paths["paper_png"] = (PAPER_FIG_DIR / f"{stem}.png").as_posix()
    plt.close(fig)
    return paths


def hour_axis() -> np.ndarray:
    return (np.arange(N_SLOT) + 0.5) * DT_H


def contiguous_mask_spans(mask: np.ndarray, hours: np.ndarray) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = hours[i] - DT_H / 2
        if (not flag or i == len(mask) - 1) and start is not None:
            end = hours[i] + DT_H / 2 if flag else hours[i] - DT_H / 2
            spans.append((start, end))
            start = None
    return spans


def shade_peaks(ax: plt.Axes, hours: np.ndarray, peak: np.ndarray) -> None:
    for lo, hi in contiguous_mask_spans(peak, hours):
        ax.axvspan(lo, hi, color=C["peak"], alpha=0.07, lw=0, zorder=0)


def load_all(paths: dict[str, Path]) -> dict:
    att1 = pd.read_excel(paths["att1"])
    hours = hour_axis()
    price = pd.to_numeric(att1.iloc[:, 1], errors="coerce").to_numpy(float)
    load1 = pd.to_numeric(att1.iloc[:, 2], errors="coerce").to_numpy(float)
    pv1 = pd.to_numeric(att1.iloc[:, 3], errors="coerce").to_numpy(float)

    xl2 = pd.ExcelFile(paths["att2"])
    load = pd.read_excel(xl2, sheet_name=0, index_col=0).apply(pd.to_numeric, errors="coerce")
    pv = pd.read_excel(xl2, sheet_name=1, index_col=0).apply(pd.to_numeric, errors="coerce")
    load_a = load.to_numpy(float)
    pv_a = pv.to_numpy(float)
    dates = pd.to_datetime(load.index)

    att3 = pd.read_excel(paths["att3"])
    att3.iloc[:, 0] = att3.iloc[:, 0].ffill()
    fc = att3.iloc[:, 2:26].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if fc.shape != (N_DAY * 4, 24):
        raise ValueError(f"attachment 3 expected {(N_DAY * 4, 24)}, got {fc.shape}")
    forecast = fc.reshape(N_DAY, 4, 24)

    att4 = pd.read_excel(paths["att4"], index_col=0).apply(pd.to_numeric, errors="coerce")
    price_year = att4.to_numpy(float)

    if load_a.shape != (N_DAY, N_SLOT) or pv_a.shape != (N_DAY, N_SLOT):
        raise ValueError("attachment 2 is not 365 x 144")
    if price_year.shape != (N_DAY, N_SLOT):
        raise ValueError("attachment 4 is not 365 x 144")
    if np.isnan(load_a).any() or np.isnan(pv_a).any() or np.isnan(price_year).any():
        raise ValueError("attachments contain missing numeric cells")

    return {
        "hours": hours,
        "price_typical": price,
        "load_typical": load1,
        "pv_typical": pv1,
        "load": load_a,
        "pv": pv_a,
        "dates": dates,
        "forecast": forecast,
        "price_year": price_year,
        "paths": {k: v.as_posix() for k, v in paths.items()},
    }


def fig_typical_day(data: dict) -> tuple[str, dict]:
    hours = data["hours"]
    load = data["load_typical"]
    pv = data["pv_typical"]
    price = data["price_typical"]
    net = load - pv
    peak = price >= np.quantile(price, 0.85)

    fig = new_fig("stack2")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 0.9], hspace=0.28)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12)

    shade_peaks(ax0, hours, peak)
    ax0.fill_between(hours, 0.0, load, color=C["load"], alpha=0.16, lw=0, zorder=1)
    ax0.fill_between(hours, 0.0, pv, color=C["pv"], alpha=0.28, lw=0, zorder=2)
    ax0.plot(hours, load, color=C["load"], lw=1.45, label="负荷", zorder=3)
    ax0.plot(hours, pv, color=C["pv"], lw=1.45, label="光伏预测", zorder=3)
    ax0.plot(hours, net, color=C["net"], lw=1.2, label="净负荷", zorder=4)
    ax0.axhline(0.0, color="#94A3B8", lw=0.7, ls="--")
    ax0.set_ylabel("功率 (kW)")
    ax0.set_xlim(0.0, 24.0)
    ax0.set_xticks(np.arange(0, 25, 2))
    ax0.legend(loc="upper left", ncol=3, frameon=False, borderaxespad=0.2)
    polish(ax0)
    panel_label(ax0, "(a)")

    shade_peaks(ax1, hours, peak)
    ax1.fill_between(hours, 0.0, price, color=C["price"], alpha=0.18, lw=0, step="mid")
    ax1.step(hours, price, where="mid", color=C["price"], lw=1.35)
    ax1.set_ylabel("电价 (元/kWh)")
    ax1.set_xlabel("时刻 (h)")
    ax1.set_ylim(0.0, max(1.55, float(price.max()) * 1.08))
    polish(ax1)
    panel_label(ax1, "(b)")

    stem = "att1_typical_day_power_price"
    paths = save(fig, stem)
    stats = {
        "load_mwh": float(load.sum() * DT_H / 1000.0),
        "pv_mwh": float(pv.sum() * DT_H / 1000.0),
        "net_mwh": float(net.sum() * DT_H / 1000.0),
        "price_min": float(price.min()),
        "price_max": float(price.max()),
        "peak_quantile": 0.85,
        "peak_hours": float(peak.mean() * 24.0),
    }
    cap = (
        "附件1典型日：(a) 10 min 负荷、光伏预测功率与净负荷；浅红带为电价最高 15% 时段。"
        f"(b) 分时电价。负荷 {stats['load_mwh']:.2f} MWh，光伏 {stats['pv_mwh']:.2f} MWh，"
        f"净缺口 {stats['net_mwh']:.2f} MWh，电价 {stats['price_min']:.2f}–{stats['price_max']:.2f} 元/kWh。"
    )
    return stem, {"paths": paths, "question": "Q1", "source": "附件1", "caption": cap, "stats": stats}


def _month_ticks(ax: plt.Axes) -> None:
    ax.set_yticks(MONTH_START)
    ax.set_yticklabels(MONTH_LAB)


def fig_calendar(data: dict) -> tuple[str, dict]:
    load = data["load"]
    pv = data["pv"]
    fig = new_fig("heat2")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.035], wspace=0.04, hspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    cax0 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, 0])
    cax1 = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.10)

    im0 = ax0.imshow(load, aspect="auto", origin="upper", cmap=CMAP_LOAD, interpolation="nearest")
    im1 = ax1.imshow(pv, aspect="auto", origin="upper", cmap=CMAP_PV, interpolation="nearest")
    for ax, im, cax, label in (
        (ax0, im0, cax0, "负荷 (kW)"),
        (ax1, im1, cax1, "光伏 (kW)"),
    ):
        ax.set_xticks(np.linspace(0, N_SLOT, 9), [f"{h:g}" for h in np.linspace(0, 24, 9)])
        _month_ticks(ax)
        ax.set_ylabel("月份")
        ax.tick_params(length=2.5, width=0.6)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("#CBD5E1")
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=8)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=2, labelsize=7.5)
    ax1.set_xlabel("时刻 (h)")
    panel_label(ax0, "(a)")
    panel_label(ax1, "(b)")

    stem = "att2_load_pv_calendar"
    paths = save(fig, stem)
    stats = {
        "load_kw_min": float(load.min()),
        "load_kw_max": float(load.max()),
        "pv_kw_min": float(pv.min()),
        "pv_kw_max": float(pv.max()),
        "days": int(load.shape[0]),
    }
    cap = (
        "附件2全年日历热图：(a) 小区负荷功率；(b) 光伏实测功率。横轴为一日 144 个 10 min 时段，"
        f"纵轴为 2025-01-01 至 12-31。负荷 {stats['load_kw_min']:.0f}–{stats['load_kw_max']:.0f} kW，"
        f"光伏 0–{stats['pv_kw_max']:.0f} kW。"
    )
    return stem, {"paths": paths, "question": "Q2/Q3", "source": "附件2", "caption": cap, "stats": stats}


def fig_daily_and_duration(data: dict) -> tuple[str, dict]:
    load = data["load"]
    pv = data["pv"]
    load_mwh = load.sum(axis=1) * DT_H / 1000.0
    pv_mwh = pv.sum(axis=1) * DT_H / 1000.0
    net_mwh = load_mwh - pv_mwh
    day = np.arange(N_DAY)
    load_ma = pd.Series(load_mwh).rolling(7, min_periods=1, center=True).mean().to_numpy()
    pv_ma = pd.Series(pv_mwh).rolling(7, min_periods=1, center=True).mean().to_numpy()
    net_ma = pd.Series(net_mwh).rolling(7, min_periods=1, center=True).mean().to_numpy()
    net_kw = (load - pv).ravel()
    duration = np.sort(net_kw)[::-1]
    frac = np.linspace(0.0, 100.0, duration.size, endpoint=False)
    stride = max(1, duration.size // 720)
    surplus = float((net_kw < 0).mean())

    fig = new_fig("split")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.9], wspace=0.28)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.16)

    ax0.plot(day, load_mwh, color=C["load"], lw=0.4, alpha=0.22)
    ax0.plot(day, load_ma, color=C["load"], lw=1.45, label="负荷 7 日均值")
    ax0.plot(day, pv_ma, color=C["pv"], lw=1.45, label="光伏 7 日均值")
    ax0.plot(day, net_ma, color=C["net"], lw=1.35, label="净缺口 7 日均值")
    ax0.set_xlim(0, N_DAY - 1)
    ax0.set_ylim(0.0, float(np.max(load_mwh)) * 1.06)
    ax0.set_xticks(MONTH_START, MONTH_LAB)
    ax0.set_xlabel("月份")
    ax0.set_ylabel("日电量 (MWh)")
    ax0.legend(loc="lower left", frameon=False, fontsize=7.5)
    polish(ax0)
    panel_label(ax0, "(a)")

    ax1.plot(frac[::stride], duration[::stride], color=C["net"], lw=1.4)
    ax1.axhline(0.0, color="#94A3B8", lw=0.7, ls="--")
    ax1.fill_between(frac[::stride], duration[::stride], 0.0, where=duration[::stride] >= 0, color=C["net"], alpha=0.16, lw=0)
    ax1.fill_between(frac[::stride], duration[::stride], 0.0, where=duration[::stride] < 0, color=C["pv"], alpha=0.22, lw=0)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("时长占比 (%)")
    ax1.set_ylabel("净负荷 (kW)")
    polish(ax1, ygrid=True)
    ax1.grid(axis="x", color=C["grid"], linewidth=0.45)
    panel_label(ax1, "(b)")

    stem = "att2_daily_energy_duration"
    paths = save(fig, stem)
    stats = {
        "load_mwh_year": float(load_mwh.sum()),
        "pv_mwh_year": float(pv_mwh.sum()),
        "net_mwh_year": float(net_mwh.sum()),
        "surplus_time_frac": surplus,
        "net_kw_p95": float(np.quantile(net_kw, 0.95)),
        "net_kw_min": float(net_kw.min()),
        "net_kw_max": float(net_kw.max()),
    }
    cap = (
        "附件2能量结构：(a) 每日负荷电量（浅线）及负荷/光伏/净缺口的 7 日均值；"
        f"(b) 全年 10 min 净负荷持续曲线，光伏过剩时段占 {100 * surplus:.1f}%。"
        f"年负荷 {stats['load_mwh_year']:.0f} MWh，年光伏 {stats['pv_mwh_year']:.0f} MWh。"
    )
    return stem, {"paths": paths, "question": "Q2/Q3", "source": "附件2", "caption": cap, "stats": stats}


def fig_seasonal(data: dict) -> tuple[str, dict]:
    dates = data["dates"]
    load = data["load"]
    pv = data["pv"]
    hours = data["hours"]
    month = dates.month.to_numpy()
    seasons = [
        ("冬季", np.isin(month, [12, 1, 2])),
        ("春季", np.isin(month, [3, 4, 5])),
        ("夏季", np.isin(month, [6, 7, 8])),
        ("秋季", np.isin(month, [9, 10, 11])),
    ]
    fig = new_fig("grid4")
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12, wspace=0.12, hspace=0.32)
    letters = ["(a)", "(b)", "(c)", "(d)"]
    ymax = 0.0
    for ax, (name, mask), lab in zip(axes.ravel(), seasons, letters):
        sub_l = load[mask]
        sub_p = pv[mask]
        l_p = np.percentile(sub_l, [25, 50, 75], axis=0)
        p_p = np.percentile(sub_p, [25, 50, 75], axis=0)
        ax.fill_between(hours, l_p[0], l_p[2], color=C["load"], alpha=0.16, lw=0)
        ax.fill_between(hours, p_p[0], p_p[2], color=C["pv"], alpha=0.22, lw=0)
        ax.plot(hours, l_p[1], color=C["load"], lw=1.35, label="负荷中位")
        ax.plot(hours, p_p[1], color=C["pv"], lw=1.35, label="光伏中位")
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 6))
        polish(ax)
        ax.text(0.98, 0.94, name, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=C["muted"])
        panel_label(ax, lab)
        ymax = max(ymax, float(l_p[2].max()), float(p_p[2].max()))
        if ax is axes[0, 0]:
            ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("功率 (kW)")
    for ax in axes[1, :]:
        ax.set_xlabel("时刻 (h)")
    axes[0, 0].set_ylim(0, ymax * 1.05)

    stem = "att2_seasonal_diurnal"
    paths = save(fig, stem)
    cap = (
        "附件2四季日内曲线：各季 10 min 负荷（蓝）与光伏（金）的中位数及四分位带。"
        "夏季光伏显著抬升、夜间负荷仍高，表明储能需同时服务削峰与消纳。"
    )
    return stem, {"paths": paths, "question": "Q2/Q3", "source": "附件2", "caption": cap, "stats": {"ymax_kw": ymax}}


def hourly_actual_pv(pv: np.ndarray) -> np.ndarray:
    return pv.reshape(N_DAY, 24, 6).mean(axis=2)


def fig_forecast(data: dict) -> tuple[str, dict]:
    pv_h = hourly_actual_pv(data["pv"])
    fc = data["forecast"]
    mae_lead = np.full((4, 24), np.nan)
    mae_hour = np.full((4, 24), np.nan)
    hour_err = [[[] for _ in range(24)] for _ in range(4)]
    lead_err = [[[] for _ in range(24)] for _ in range(4)]
    day0_fc = []
    day0_ac = []
    for d in range(N_DAY):
        for j, issue in enumerate(ISSUE_HOURS):
            for k in range(24):
                abs_h = issue + k
                dd = d + abs_h // 24
                hh = abs_h % 24
                if dd >= N_DAY:
                    continue
                err = float(fc[d, j, k] - pv_h[dd, hh])
                lead_err[j][k].append(abs(err))
                hour_err[j][hh].append(abs(err))
                if j == 0:
                    day0_fc.append(float(fc[d, j, k]))
                    day0_ac.append(float(pv_h[dd, hh]))
    for j in range(4):
        for k in range(24):
            mae_lead[j, k] = float(np.mean(lead_err[j][k])) if lead_err[j][k] else np.nan
            mae_hour[j, k] = float(np.mean(hour_err[j][k])) if hour_err[j][k] else np.nan

    mean_fc0 = fc[:, 0, :].mean(axis=0)
    mean_ac = pv_h.mean(axis=0)

    fig = new_fig("split")
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.16, wspace=0.30)
    hod = np.arange(24) + 0.5
    colors = ["#2F5F8A", "#C4A35A", "#A33B3B", "#5E4B73"]
    labels = ["0:00 发布", "6:00 发布", "12:00 发布", "18:00 发布"]
    for row, color, lab in zip(mae_hour, colors, labels):
        ax0.plot(hod, row, color=color, lw=1.45, label=lab)
    ax0.set_xlim(0, 24)
    ax0.set_xticks(np.arange(0, 25, 4))
    ax0.set_xlabel("被预报时刻 (h)")
    ax0.set_ylabel("MAE (kW)")
    ax0.legend(loc="upper left", frameon=False, fontsize=7.5)
    polish(ax0)
    panel_label(ax0, "(a)")

    ax1.fill_between(hod, mean_ac, mean_fc0, color=C["net"], alpha=0.14, lw=0)
    ax1.plot(hod, mean_ac, color=C["pv"], lw=1.5, label="实测整点均值")
    ax1.plot(hod, mean_fc0, color=C["load"], lw=1.5, ls="--", label="0:00 预报均值")
    ax1.set_xlim(0, 24)
    ax1.set_xticks(np.arange(0, 25, 4))
    ax1.set_xlabel("时刻 (h)")
    ax1.set_ylabel("光伏功率 (kW)")
    ax1.legend(loc="upper left", frameon=False, fontsize=7.5)
    polish(ax1)
    panel_label(ax1, "(b)")

    stem = "att3_pv_forecast_error"
    paths = save(fig, stem)
    stats = {
        "mae_0utc_mean": float(np.nanmean(mae_lead[0])),
        "mae_6_mean": float(np.nanmean(mae_lead[1])),
        "mae_12_mean": float(np.nanmean(mae_lead[2])),
        "mae_18_mean": float(np.nanmean(mae_lead[3])),
        "mae_0utc_lead1": float(mae_lead[0, 0]),
        "mae_0utc_lead24": float(mae_lead[0, 23]),
    }
    cap = (
        "附件3光伏预报对照附件2实测整点均值：(a) 四个发布时刻按被预报钟点的 MAE，误差集中在日照时段；"
        "(b) 0:00 发布的全日平均预报曲线与实测均值。夜间 MAE 接近 0。"
        f"0:00 发布全预见期平均 MAE 为 {stats['mae_0utc_mean']:.0f} kW。"
    )
    return stem, {"paths": paths, "question": "Q3", "source": "附件3 vs 附件2", "caption": cap, "stats": stats}


def fig_price_year(data: dict) -> tuple[str, dict]:
    price_y = data["price_year"]
    hours = data["hours"]
    typical = data["price_typical"]
    p25, p50, p75 = np.percentile(price_y, [25, 50, 75], axis=0)

    fig = new_fig("heat2")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.035], height_ratios=[1.25, 0.9], wspace=0.04, hspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    cax0 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, :])
    fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.12)

    im = ax0.imshow(price_y, aspect="auto", origin="upper", cmap=CMAP_PRICE, interpolation="nearest")
    ax0.set_xticks(np.linspace(0, N_SLOT, 9), [f"{h:g}" for h in np.linspace(0, 24, 9)])
    _month_ticks(ax0)
    ax0.set_ylabel("月份")
    for spine in ax0.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#CBD5E1")
    cb = fig.colorbar(im, cax=cax0)
    cb.set_label("电价 (元/kWh)", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=7.5)
    panel_label(ax0, "(a)")

    ax1.fill_between(hours, p25, p75, color=C["price"], alpha=0.18, lw=0, label="全年四分位带")
    ax1.plot(hours, p50, color=C["price"], lw=1.4, label="全年中位电价")
    ax1.plot(hours, typical, color=C["peak"], lw=1.25, ls="--", label="附件1典型日")
    ax1.set_xlim(0, 24)
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.set_xlabel("时刻 (h)")
    ax1.set_ylabel("电价 (元/kWh)")
    ax1.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2)
    polish(ax1)
    panel_label(ax1, "(b)")

    cors = np.array([np.corrcoef(price_y[i], typical)[0, 1] for i in range(N_DAY)])
    best = int(np.argmax(cors))
    stem = "att4_price_calendar"
    paths = save(fig, stem)
    stats = {
        "price_min": float(price_y.min()),
        "price_max": float(price_y.max()),
        "price_mean": float(price_y.mean()),
        "best_match_day_index": best,
        "best_match_date": str(data["dates"][best].date()),
        "best_match_corr": float(cors[best]),
    }
    cap = (
        "附件4全年 10 min 电价：(a) 日历热图；(b) 日内中位数与四分位带，虚线为附件1典型日电价。"
        f"电价 {stats['price_min']:.3f}–{stats['price_max']:.3f} 元/kWh。"
        f"与附件1相关最高的日期为 {stats['best_match_date']}（相关系数 {stats['best_match_corr']:.3f}）。"
    )
    return stem, {"paths": paths, "question": "Q4", "source": "附件4", "caption": cap, "stats": stats}


def write_index(entries: dict[str, dict], font: str, attachments: dict[str, Path]) -> None:
    payload = {
        "mode": "manuscript",
        "font": font,
        "slot_mm": SLOT,
        "attachments": {k: v.name for k, v in attachments.items()},
        "figures": entries,
        "note": "Captions belong in LaTeX \\caption, not inside the artwork.",
    }
    INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    font = pick_font()
    apply_style(font)
    attachments = find_attachments()
    data = load_all(attachments)
    builders = (
        fig_typical_day,
        fig_calendar,
        fig_daily_and_duration,
        fig_seasonal,
        fig_forecast,
        fig_price_year,
    )
    entries = {}
    for fn in builders:
        stem, rec = fn(data)
        entries[stem] = rec
        print("wrote", stem)
    write_index(entries, font, attachments)
    print("index", INDEX_PATH)


if __name__ == "__main__":
    main()
