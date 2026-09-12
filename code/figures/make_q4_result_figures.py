"""Regenerate the Q4 paper figures from the current verified result artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results" / "Q4"
FIGURE = ROOT / "paper" / "figures"
ASSET = ROOT / "paper" / "assets" / "figures"

BLUE = "#4E79A7"
ORANGE = "#F28E2B"
GREEN = "#59A14F"
RED = "#E15759"
PURPLE = "#8E6C8A"
GRAY = "#8A8A8A"
LIGHT_GRAY = "#D9D9D9"
STRATEGY_COLORS = {"N0": BLUE, "LA": ORANGE, "M0L": GREEN}
STRATEGY_LABELS = {"N0": "策略 N0", "LA": "策略 LA", "M0L": "策略 M0L"}


def setup_style() -> None:
    font_path = ROOT / "paper" / "simsun.ttc"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        family = fm.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "SimSun"
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#B8B8B8",
            "grid.alpha": 0.28,
            "grid.linewidth": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
        }
    )


def finish(fig: plt.Figure, stem: str) -> list[str]:
    FIGURE.mkdir(parents=True, exist_ok=True)
    ASSET.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = FIGURE / f"{stem}.{suffix}"
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        shutil.copy2(path, ASSET / path.name)
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def month_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    ax.set_xlim(pd.Timestamp("2025-01-25"), pd.Timestamp("2026-01-05"))


def load_daily() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    q42 = pd.read_csv(RESULT / "q42_daily.csv", parse_dates=["date"])
    q43 = {
        name: pd.read_csv(RESULT / f"q43_{name}_daily.csv", parse_dates=["date"])
        for name in ("N0", "LA", "M0L")
    }
    return q42, q43


def official_dispatch(q42: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(q42.date, q42.soc0_actual, color=BLUE, lw=1.55, label="日初实际 SOC")
    ax.axhline(2400, color=RED, lw=1.65, ls="--", label="计划日末 SOC 目标：2400 kWh")
    ax.set_title("(a) 电池 SOC 日轨迹与计划目标")
    ax.set_ylabel("储能电量 SOC（kWh）")
    month_axis(ax)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

    ax = axes[0, 1]
    ax.plot(q42.date, q42.purchase_kwh / 1000, color=BLUE, lw=1.35, label="日购电量")
    ax.fill_between(q42.date, 0, q42.purchase_kwh / 1000, color=BLUE, alpha=0.14)
    ax.set_ylabel("日购电量（千 kWh）", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax2 = ax.twinx()
    ax2.plot(q42.date, q42.curtail_kwh / 1000, color=RED, lw=1.25, alpha=0.92, label="日弃电量")
    ax2.fill_between(q42.date, 0, q42.curtail_kwh / 1000, color=RED, alpha=0.10)
    ax2.set_ylabel("日弃电量（千 kWh）", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax.set_title("(b) 日购电量与弃电量")
    month_axis(ax)

    ax = axes[1, 0]
    ax.bar(q42.date, q42.plan_cost / 10000, width=0.9, color=BLUE, alpha=0.88, label="计划购电成本")
    ax.bar(
        q42.date,
        q42.emergency_cost / 10000,
        width=0.9,
        bottom=q42.plan_cost / 10000,
        color=RED,
        alpha=0.90,
        label="紧急购电成本",
    )
    ax.set_title("(c) 日成本构成：计划购电与紧急购电")
    ax.set_ylabel("日成本（万元）")
    ax.set_xlabel("日期（2025 年）")
    month_axis(ax)
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95)

    ax = axes[1, 1]
    no_emergency = q42.emergency_slots <= 0
    ax.scatter(
        q42.loc[no_emergency, "emergency_slots"],
        q42.loc[no_emergency, "emergency_cost"],
        s=16,
        color=GRAY,
        alpha=0.45,
        label="无应急购电日",
    )
    active = q42.loc[~no_emergency]
    ax.scatter(active.emergency_slots, active.emergency_cost, s=23, color=RED, alpha=0.70, label="发生应急购电")
    for _, row in active.nlargest(3, "emergency_cost").iterrows():
        ax.annotate(row.date.strftime("%m-%d"), (row.emergency_slots, row.emergency_cost), xytext=(5, 0), textcoords="offset points", color=RED, fontsize=9)
    ax.set_title("(d) 应急时段数与当日应急购电成本")
    ax.set_xlabel("当日应急购电时段数（个）")
    ax.set_ylabel("当日应急购电成本（元）")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    return finish(fig, "q4_official_dispatch_details")


def cost_structure(q43: dict[str, pd.DataFrame]) -> list[str]:
    fig = plt.figure(figsize=(13.0, 12.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.9, 1.0])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    months = pd.period_range("2025-02", "2025-12", freq="M")
    x = np.arange(len(months))
    width = 0.25

    ax = axes[0]
    for idx, name in enumerate(("N0", "LA", "M0L")):
        d = q43[name].copy()
        d["month"] = d.date.dt.to_period("M")
        monthly = d.groupby("month").agg(total=("total_cost", "sum"), emergency=("emergency_cost", "sum")).reindex(months, fill_value=0)
        base = (monthly.total - monthly.emergency) / 10000
        emergency = monthly.emergency / 10000
        pos = x + (idx - 1) * width
        ax.bar(pos, base, width, color=STRATEGY_COLORS[name], alpha=0.90, label=STRATEGY_LABELS[name] if idx < 3 else None)
        ax.bar(pos, emergency, width, bottom=base, color=STRATEGY_COLORS[name], alpha=0.32, hatch="///", edgecolor=STRATEGY_COLORS[name])
    ax.set_title("(a) 各策略月度成本构成（实色为非应急成本，斜线为紧急购电成本）")
    ax.set_ylabel("月度成本（万元）")
    ax.set_xticks(x, [f"{p.month:02d}月" for p in months])
    ax.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.95)

    ax = axes[1]
    shares = []
    for name in ("N0", "LA", "M0L"):
        d = q43[name].copy()
        d["month"] = d.date.dt.to_period("M")
        monthly = d.groupby("month").agg(total=("total_cost", "sum"), emergency=("emergency_cost", "sum")).reindex(months, fill_value=0)
        share = 100 * monthly.emergency / monthly.total
        shares.extend(share.tolist())
        ax.plot(x, share, marker="o", ms=4.6, lw=1.8, color=STRATEGY_COLORS[name], label=STRATEGY_LABELS[name])
    overall = 100 * sum(d.emergency_cost.sum() for d in q43.values()) / sum(d.total_cost.sum() for d in q43.values())
    ax.axhline(overall, color=GRAY, ls="--", lw=1.1, label=f"三策略总体均值：{overall:.1f}%")
    ax.set_title("(b) 紧急购电成本占比的月度演变")
    ax.set_ylabel("紧急购电成本占比（%）")
    ax.set_xticks(x, [f"{p.month:02d}月" for p in months])
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95)

    ax = axes[2]
    for name in ("N0", "LA", "M0L"):
        d = q43[name]
        cumulative = d.total_cost.cumsum() / 10000
        ax.plot(d.date, cumulative, lw=2.1, color=STRATEGY_COLORS[name], label=STRATEGY_LABELS[name])
        ax.annotate(f"{cumulative.iloc[-1]:,.1f} 万元", (d.date.iloc[-1], cumulative.iloc[-1]), xytext=(7, 0), textcoords="offset points", color=STRATEGY_COLORS[name], fontsize=10)
    ax.set_title("(c) 累计购电成本曲线（334 天）")
    ax.set_ylabel("累计总成本（万元）")
    ax.set_xlabel("日期（2025 年）")
    month_axis(ax)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    return finish(fig, "q4_cost_structure")


def daily_distribution(q43: dict[str, pd.DataFrame]) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.7), constrained_layout=True)
    names = ("N0", "LA", "M0L")
    values = [q43[n].total_cost.to_numpy() / 10000 for n in names]
    rng = np.random.default_rng(42)

    ax = axes[0]
    bp = ax.boxplot(values, patch_artist=True, widths=0.50, showfliers=False, medianprops={"color": "#222222", "lw": 1.6})
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(STRATEGY_COLORS[name])
        patch.set_alpha(0.28)
        patch.set_edgecolor(STRATEGY_COLORS[name])
        patch.set_linewidth(1.4)
    for i, (name, vals) in enumerate(zip(names, values), start=1):
        jitter = rng.normal(i, 0.055, len(vals))
        ax.scatter(jitter, vals, s=9, color=STRATEGY_COLORS[name], alpha=0.34, edgecolors="none")
        ax.text(i, max(v.max() for v in values) * 0.985, f"均值 {vals.mean():.2f}\n中位 {np.median(vals):.2f}", ha="center", va="top", color=STRATEGY_COLORS[name], fontsize=10)
    ax.set_title("(a) 日总成本分布")
    ax.set_ylabel("日总成本（万元）")
    ax.set_xticks([1, 2, 3], [STRATEGY_LABELS[n] for n in names])

    ax = axes[1]
    for name, vals in zip(names, values):
        ordered = np.sort(vals)
        y = np.arange(1, len(vals) + 1) / len(vals)
        p90 = np.quantile(vals, 0.9)
        ax.step(ordered, y, where="post", lw=2.0, color=STRATEGY_COLORS[name], label=f"{STRATEGY_LABELS[name]}（$P_{{90}}={p90:.2f}$ 万元）")
        ax.vlines(p90, 0, 0.9, color=STRATEGY_COLORS[name], ls=":", lw=1.3)
    ax.axhline(0.9, color=GRAY, ls="--", lw=1.0)
    ax.set_title("(b) 日总成本经验累积分布（ECDF）")
    ax.set_xlabel("日总成本（万元）")
    ax.set_ylabel("累积概率 $F(x)$")
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    return finish(fig, "q4_daily_cost_distribution")


def error_decomposition() -> tuple[list[str], list[dict]]:
    q42_oracle = json.loads((RESULT / "diagnostics" / "oracle_current_q42" / "q42_netrho_summary.json").read_text(encoding="utf-8"))
    q43_oracle = json.loads((RESULT / "diagnostics" / "oracle_current_q43" / "q43_LA_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((RESULT / "metrics.json").read_text(encoding="utf-8"))
    q2 = json.loads((ROOT / "results" / "Q2" / "opt" / "full_year_summary.json").read_text(encoding="utf-8"))
    q3 = json.loads((ROOT / "results" / "Q3" / "run_manifest.json").read_text(encoding="utf-8"))
    rows = [
        {
            "name": "Q4-2  净负载分位—回放策略",
            "base": float(q2["total_cost"]),
            "oracle": float(q42_oracle["total_cost"]),
            "actual": float(metrics["q4_2"]["total_cost"]),
        },
        {
            "name": "Q4-3  日内滚动策略 LA",
            "base": float(q3["metrics"]["LA"]["total_cost"]),
            "oracle": float(q43_oracle["total_cost"]),
            "actual": float(metrics["q4_3"]["LA"]["total_cost"]),
        },
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)
    for ax, row in zip(axes, rows):
        env = row["oracle"] - row["base"]
        forecast = row["actual"] - row["oracle"]
        vals = np.array([row["base"], env, row["oracle"], forecast, row["actual"]]) / 10000
        xpos = np.arange(5)
        ax.bar(xpos[[0, 2, 4]], vals[[0, 2, 4]], width=0.60, color=["#9BB9D3", "#6BAED6", BLUE])
        ax.bar(xpos[1], vals[1], width=0.60, bottom=vals[0], color=ORANGE)
        ax.bar(xpos[3], vals[3], width=0.60, bottom=vals[2], color=RED)
        ax.hlines(vals[2], 1.3, 2.7, color=GRAY, ls=":", lw=1.2)
        ax.hlines(vals[4], 3.3, 4.0, color=GRAY, ls=":", lw=1.2)
        labels = ["原问题\n固定电价", "实时电价\n环境差额", "Oracle\n总成本", "电价预测\n误差代价", "实际策略\n总成本"]
        ax.set_xticks(xpos, labels)
        ax.set_ylabel("全年总成本（万元）")
        env_pct = 100 * env / row["base"]
        forecast_pct = 100 * forecast / row["oracle"]
        total_pct = 100 * (row["actual"] - row["base"]) / row["base"]
        ax.set_title(f"{row['name']}\n环境差额 {env_pct:+.2f}%，预测误差 {forecast_pct:+.2f}%，合计 {total_pct:+.2f}%")
        bottoms = [0, vals[0], 0, vals[2], 0]
        for i, (v, bottom) in enumerate(zip(vals, bottoms)):
            text_value = f"{v:,.1f}" if i in (0, 2, 4) else f"{v:+,.1f}"
            ax.text(i, bottom + v + max(vals[[0, 2, 4]]) * 0.012, text_value, ha="center", va="bottom", fontsize=9.5)
        ax.set_ylim(0, max(vals[[0, 2, 4]]) * 1.16)
    return finish(fig, "q4_error_decomposition"), rows


def main() -> None:
    setup_style()
    q42, q43 = load_daily()
    index = {
        "source": {
            "official_metrics": "results/Q4/metrics.json",
            "q42_daily": "results/Q4/q42_daily.csv",
            "q43_daily": [f"results/Q4/q43_{n}_daily.csv" for n in ("N0", "LA", "M0L")],
            "oracle": [
                "results/Q4/diagnostics/oracle_current_q42/q42_netrho_summary.json",
                "results/Q4/diagnostics/oracle_current_q43/q43_LA_summary.json",
            ],
        },
        "figures": {},
    }
    index["figures"]["official_dispatch"] = official_dispatch(q42)
    index["figures"]["cost_structure"] = cost_structure(q43)
    index["figures"]["daily_distribution"] = daily_distribution(q43)
    index["figures"]["error_decomposition"], index["oracle_rows"] = error_decomposition()
    (FIGURE / "q4_result_figure_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(FIGURE / "q4_result_figure_index.json", ASSET / "q4_result_figure_index.json")
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
