"""Four-question relationship diagram for the contest paper.

Artwork only: no internal title. Caption belongs in LaTeX.
Layout follows CUMCM C-paper route maps: shared banner, expanding
data row, four method columns, inheritance arrows with relation pills.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "paper" / "assets" / "figures"
PAPER_FIG_DIR = REPO_ROOT / "paper" / "figures"
STEM = "problem_relationship_framework"

MM = 25.4
FIG_MM = (168.0, 126.0)

C = {
    "ink": "#1F2933",
    "muted": "#52606D",
    "line": "#7B8A96",
    "shared_fill": "#E7F1FA",
    "shared_edge": "#2389D9",
    "data_fill": "#FFF6E0",
    "data_edge": "#DDA01E",
    "q1_fill": "#EAF4FC",
    "q1_head": "#2389D9",
    "q2_fill": "#FFF8EA",
    "q2_head": "#DDA01E",
    "q3_fill": "#F4EEE8",
    "q3_head": "#897456",
    "q4_fill": "#EEF6EC",
    "q4_head": "#3E7A52",
    "white": "#FFFFFF",
    "band": "#F6F8FA",
    "band_edge": "#C5D0D8",
}


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
            "font.size": 8.0,
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 400,
            "savefig.bbox": "standard",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _axxy(x_mm: float, y_mm: float) -> tuple[float, float]:
    w, h = FIG_MM
    return x_mm / w, y_mm / h


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    fc: str,
    ec: str,
    lw: float = 0.85,
    rad: float = 0.011,
) -> None:
    ww, hh = FIG_MM
    ax.add_patch(
        FancyBboxPatch(
            _axxy(x, y),
            w / ww,
            h / hh,
            boxstyle=f"round,pad=0.0035,rounding_size={rad}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            transform=ax.transAxes,
            clip_on=False,
            mutation_aspect=hh / ww,
        )
    )


def text(
    ax: plt.Axes,
    x: float,
    y: float,
    s: str,
    *,
    size: float = 8.0,
    color: str = C["ink"],
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    linespacing: float = 1.22,
) -> None:
    ax.text(
        *_axxy(x, y),
        s,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        transform=ax.transAxes,
        clip_on=False,
        linespacing=linespacing,
    )


def arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = C["line"],
    lw: float = 1.05,
    scale: float = 10,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            _axxy(x0, y0),
            _axxy(x1, y1),
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            transform=ax.transAxes,
            clip_on=False,
            shrinkA=0,
            shrinkB=0,
        )
    )


def plus_node(ax: plt.Axes, x: float, y: float, color: str) -> None:
    add_box(ax, x - 1.85, y - 1.85, 3.7, 3.7, C["white"], color, lw=0.8, rad=0.02)
    ax.text(
        *_axxy(x, y - 0.12),
        "+",
        fontsize=8.2,
        color=color,
        fontweight="bold",
        ha="center",
        va="center",
        transform=ax.transAxes,
        clip_on=False,
        zorder=5,
    )


def question_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    head: str,
    fill: str,
    qid: str,
    subtitle: str,
    lines: list[str],
) -> None:
    head_h = 12.4
    add_box(ax, x, y, w, h, fill, head, lw=1.0, rad=0.01)
    add_box(ax, x, y + h - head_h, w, head_h, head, head, lw=0.0, rad=0.008)
    text(ax, x + w / 2, y + h - 4.15, qid, size=9.0, color="white", weight="bold")
    text(ax, x + w / 2, y + h - 9.35, subtitle, size=7.15, color="white")
    body_top = y + h - head_h - 3.1
    for i, line in enumerate(lines):
        text(
            ax,
            x + 2.6,
            body_top - i * 5.55,
            line,
            size=7.0,
            color=C["ink"],
            ha="left",
            va="top",
        )


def draw(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, 5.5, 108.4, 157.0, 14.6, C["shared_fill"], C["shared_edge"], lw=1.1)
    text(ax, 84.0, 119.4, "公共约束与统一口径", size=9.2, color=C["shared_edge"], weight="bold")
    text(
        ax,
        84.0,
        113.2,
        r"容量 12000 kWh　　运行窗 $[1200,\ 10800]$ kWh　　$\eta=0.9$　　接口 5000 kW"
        "\n不得倒送电、允许弃光　　$144\\times 10$ min　　决策量 $G,\\,c,\\,d,\\,E$　　同一套电量平衡与 SOC 方程",
        size=7.15,
        color=C["ink"],
    )

    text(ax, 5.8, 104.6, "数据逐问扩展", size=8.0, color=C["muted"], ha="left", weight="bold")
    data = [
        (7.2, "附件 1  电价", "贯穿四问"),
        (47.0, "附件 2  负荷 / 光伏", "问题二、三、四"),
        (86.8, "附件 3  光伏预报", "问题三、四"),
        (126.6, "附件 4  波动电价", "问题四"),
    ]
    dw, dh, dy = 34.2, 11.2, 91.6
    for i, (x, title, use) in enumerate(data):
        add_box(ax, x, dy, dw, dh, C["data_fill"], C["data_edge"], lw=0.85)
        text(ax, x + dw / 2, dy + 7.45, title, size=7.55, color=C["ink"], weight="bold")
        text(ax, x + dw / 2, dy + 3.25, use, size=6.6, color=C["muted"])
        if i < 3:
            x0 = x + dw + 0.15
            x1 = data[i + 1][0] - 0.15
            mid = 0.5 * (x0 + x1)
            arrow(ax, x0, dy + dh / 2, x1, dy + dh / 2, C["data_edge"], lw=0.95, scale=9)
            plus_node(ax, mid, dy + dh / 2, C["data_edge"])

    qy, qw, qh = 20.4, 34.8, 57.2
    gap = 6.4
    qx = [6.2 + i * (qw + gap) for i in range(4)]
    cards = [
        (
            C["q1_head"],
            C["q1_fill"],
            "问题一",
            "典型日确定性计划",
            [
                r"输入：附件 1 + 典型日曲线",
                r"确定性日前 LP",
                r"已知全天负荷、光伏与电价",
                r"日闭环  $E_0=E_{144}=6000$",
                r"目标  $\min\sum\pi_t G_t$",
                r"无紧急购电",
            ],
        ),
        (
            C["q2_head"],
            C["q2_fill"],
            "问题二",
            "全年日前预测执行",
            [
                r"输入：附件 1、2",
                r"XGB 负荷 + 7 日光伏均值",
                r"自适应分位  D_pv7d_adaptive",
                r"0:00 日前 LP，锁死 $G_t$",
                r"贪心回放，不用当日真值排计划",
                r"缺口按 $5\pi$ 紧急外购",
            ],
        ),
        (
            C["q3_head"],
            C["q3_fill"],
            "问题三",
            "日内滚动调整",
            [
                r"输入：附件 1、2、3",
                r"继承储能、回放与紧急口径",
                r"6 / 12 / 18 点调整剩余合同",
                r"LA：48 h 展望，$q_L=0.65$",
                r"光伏插值用实测起点",
                r"结算  $0.5\pi$ / $1.5\pi$ / $5\pi$",
            ],
        ),
        (
            C["q4_head"],
            C["q4_fill"],
            "问题四",
            "波动电价环境",
            [
                r"输入：附件 1–4",
                r"调度结构同问题二、三",
                r"Q4-2 沿用日前自适应",
                r"Q4-3 沿用日内 LA",
                r"决策电价 $\hat{p}_0$，结算附件 4",
                r"日内不改电价、不重选裕度",
            ],
        ),
    ]
    for x, spec in zip(qx, cards):
        question_card(ax, x, qy, qw, qh, *spec)

    rel = [
        "单日特例 → 全年",
        "继承 + 日内调整",
        "同构 + 更换电价",
    ]
    band_y = qy + qh + 2.6
    for i, lab in enumerate(rel):
        x0 = qx[i] + qw * 0.72
        x1 = qx[i + 1] + qw * 0.28
        mid = 0.5 * (x0 + x1)
        arrow(ax, x0, band_y, x1, band_y, C["ink"], lw=1.05, scale=9)
        pill_w = 24.8
        add_box(
            ax,
            mid - pill_w / 2,
            band_y + 1.7,
            pill_w,
            6.6,
            C["white"],
            C["ink"],
            lw=0.65,
            rad=0.014,
        )
        text(ax, mid, band_y + 5.0, lab, size=6.55, color=C["ink"])

    add_box(ax, 6.4, 4.0, 155.2, 13.6, C["band"], C["band_edge"], lw=0.7)
    text(
        ax,
        84.0,
        12.4,
        "结构关系：问题一是后三问的单日特例；问题二、三为“继承 + 新增机制”；问题四与二、三为“同构 + 更换电价环境”。",
        size=6.95,
        color=C["ink"],
    )
    text(
        ax,
        84.0,
        7.15,
        "正式窗口 2025-02-01 至 12-31（334 天）；四问可并行建模，但储能参数、时刻口径与充放电回放规则必须统一。",
        size=6.75,
        color=C["muted"],
    )


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
    apply_style(pick_font())
    w, h = FIG_MM
    fig = plt.figure(figsize=(w / MM, h / MM))
    fig.set_size_inches(w / MM, h / MM, forward=True)
    ax = fig.add_axes([0, 0, 1, 1])
    draw(ax)
    paths = save(fig)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for key, path in main().items():
        print(f"{key}: {path}")
