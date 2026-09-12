"""Simpler four-question route map, extra version (does not replace the swimlane figure).

Layout follows the dashed-group / stage-column style in
reference-figures/excellent (pipeline columns, short bullets, chevrons).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "paper" / "assets" / "figures"
PAPER_FIG_DIR = REPO_ROOT / "paper" / "figures"
STEM = "problem_relationship_framework_simple"

MM = 25.4
FIG_MM = (168.0, 114.0)

BLUE = "#2389d9"
SKY = "#70beff"
ICE = "#dcf2ff"
GOLD = "#dda01e"
INK = "#15202B"
MUTED = "#3A4A56"


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def mix(a: str, b: str, t: float) -> str:
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    return _hex(
        (
            int(ra + (rb - ra) * t),
            int(ga + (gb - ga) * t),
            int(ba + (bb - ba) * t),
        )
    )


def tint(h: str, t: float = 0.82) -> str:
    return mix(h, "#ffffff", t)


def _register_bundled() -> None:
    for path in (REPO_ROOT / "paper" / "simsun.ttc", REPO_ROOT / "paper" / "simkai.ttf"):
        if path.exists():
            try:
                fontManager.addfont(str(path))
            except (OSError, ValueError, RuntimeError):
                continue


def _first_font(candidates: tuple[str, ...]) -> str:
    for name in candidates:
        try:
            found = mpl.font_manager.findfont(
                FontProperties(family=name),
                fallback_to_default=False,
            )
        except (ValueError, OSError):
            continue
        if "dejavu" in Path(found).name.lower() and "dejavu" not in name.lower():
            continue
        return name
    return "DejaVu Sans"


FONTS: dict[str, object] = {}


def pick_fonts() -> dict[str, object]:
    _register_bundled()
    hei = _first_font(("SimHei", "STXihei", "Noto Sans SC", "Microsoft YaHei"))
    song = _first_font(("STSong", "SimSun", "NSimSun", "Source Han Serif SC", "Noto Serif SC"))
    return {
        "hei": FontProperties(family=hei),
        "song": FontProperties(family=song),
        "hei_name": hei,
        "song_name": song,
    }


def apply_style(fonts: dict[str, object]) -> None:
    FONTS.clear()
    FONTS.update(fonts)
    mpl.rcParams.update(
        {
            "font.family": fonts["song_name"],
            "font.sans-serif": [fonts["hei_name"], "SimHei"],
            "font.serif": [fonts["song_name"], "STSong", "SimSun", "Times New Roman"],
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
    lw: float = 0.9,
    rad: float = 0.012,
    ls: str = "-",
) -> None:
    ww, hh = FIG_MM
    ax.add_patch(
        FancyBboxPatch(
            _axxy(x, y),
            w / ww,
            h / hh,
            boxstyle=f"round,pad=0.003,rounding_size={rad}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
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
    color: str = "#1F2933",
    face: str = "song",
    ha: str = "center",
    va: str = "center",
    linespacing: float = 1.25,
) -> None:
    base = FONTS.get(face, FONTS.get("song"))
    prop = base.copy() if isinstance(base, FontProperties) else FontProperties(size=size)
    prop.set_size(size)
    ax.text(
        *_axxy(x, y),
        s,
        color=color,
        fontproperties=prop,
        ha=ha,
        va=va,
        transform=ax.transAxes,
        clip_on=False,
        linespacing=linespacing,
    )


def chevron(ax: plt.Axes, x: float, y: float, w: float, h: float, color: str) -> None:
    pts = [
        _axxy(x, y),
        _axxy(x + w * 0.68, y),
        _axxy(x + w, y + h / 2),
        _axxy(x + w * 0.68, y + h),
        _axxy(x, y + h),
        _axxy(x + w * 0.32, y + h / 2),
    ]
    ax.add_patch(
        Polygon(
            pts,
            closed=True,
            facecolor=color,
            edgecolor="none",
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            _axxy(x0, y0),
            _axxy(x1, y1),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.95,
            color=color,
            transform=ax.transAxes,
            clip_on=False,
            shrinkA=0,
            shrinkB=0,
        )
    )


def stage_column(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    qid: str,
    subtitle: str,
    bullets: list[str],
    head_fg: str = "#ffffff",
) -> None:
    add_box(ax, x + 0.7, y - 0.7, w, h, "#D7E3EE", "#D7E3EE", lw=0.0, rad=0.012)
    add_box(ax, x, y, w, h, ICE, color, lw=1.2, rad=0.012, ls=(0, (1.8, 1.15)))
    head_h = 15.2
    add_box(ax, x + 0.85, y + h - head_h - 0.7, w - 1.7, head_h, color, color, lw=0.0, rad=0.009)
    text(ax, x + w / 2, y + h - 5.4, qid, size=11.0, face="hei", color=head_fg)
    text(ax, x + w / 2, y + h - 12.4, subtitle, size=8.0, face="hei", color=head_fg)
    bh, gap = 11.6, 2.3
    by = y + 3.4 + (len(bullets) - 1) * (bh + gap)
    inset = 2.4
    for line in bullets:
        add_box(ax, x + inset, by, w - 2 * inset, bh, "#ffffff", mix(color, "#ffffff", 0.55), lw=0.55, rad=0.009)
        add_box(ax, x + inset + 0.5, by + 2.4, 0.95, bh - 4.8, color, color, lw=0.0, rad=0.005)
        text(
            ax,
            x + inset + 2.55,
            by + bh / 2,
            line,
            size=6.85,
            face="song",
            color=INK,
            ha="left",
        )
        by -= bh + gap


def draw(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, 5.0, 99.6, 158.0, 11.8, ICE, BLUE, lw=1.1)
    text(ax, 84.0, 108.4, "公共约束与统一口径", size=10.8, face="hei", color=BLUE)
    text(
        ax,
        84.0,
        102.8,
        "12000 kWh　[1200, 10800]　η = 0.9　5000 kW　不得倒送、允许弃光　144 × 10 min　决策量 G, c, d, E",
        size=7.1,
        face="song",
        color=INK,
    )

    cols = [
        (BLUE, "问题一", "典型日确定性计划", ["日前 LP，已知全天曲线", "日闭环 E$_0$=E$_{144}$=6000", "无紧急购电"], "#ffffff"),
        (SKY, "问题二", "全年预测 + 回放", ["XGB 负荷 + 7 日光伏", "自适应分位 D_pv7d_adaptive", "锁 G$_t$ 回放，缺口 5π"], INK),
        (BLUE, "问题三", "日内滚动调整", ["继承储能与回放口径", "6/12/18 点 LA，48 h，q$_L$=0.65", "结算 0.5π / 1.5π / 5π"], "#ffffff"),
        (GOLD, "问题四", "波动电价环境", ["决策电价改 $\\hat{p}_0$", "沿用问题二 / 问题三", "结算附件 4，不重选参数"], INK),
    ]
    cw, ch, cy = 35.0, 64.0, 21.6
    gap = 7.0
    xs = [5.5 + i * (cw + gap) for i in range(4)]
    for x, (color, qid, subtitle, bullets, fg) in zip(xs, cols):
        stage_column(ax, x, cy, cw, ch, color, qid, subtitle, bullets, fg)

    rel = ["单日特例 → 全年", "继承 + 日内调整", "同构 + 更换电价"]
    pw, ph = 29.6, 7.4
    for i, lab in enumerate(rel):
        x0 = xs[i] + cw
        x1 = xs[i + 1]
        mid = 0.5 * (x0 + x1)
        chevron(ax, mid - 2.7, cy + ch / 2 - 3.4, 5.4, 6.8, SKY)
        add_box(ax, mid - pw / 2, cy + ch + 3.4, pw, ph, tint(GOLD, 0.86), GOLD, lw=0.95, rad=0.014)
        text(ax, mid, cy + ch + 3.4 + ph / 2, lab, size=7.3, face="hei", color=mix(GOLD, INK, 0.28))

    chips = [
        (BLUE, "附件 1 电价 · 贯穿四问"),
        (SKY, "附件 2 负荷/光伏 · 问题二起"),
        (BLUE, "附件 3 光伏预报 · 问题三起"),
        (GOLD, "附件 4 波动电价 · 问题四"),
    ]
    th, ty = 10.0, 6.2
    for i, (col, lab) in enumerate(chips):
        x = xs[i]
        add_box(ax, x, ty, cw, th, ICE if col != GOLD else tint(GOLD, 0.86), col, lw=0.95)
        text(ax, x + cw / 2, ty + th / 2, lab, size=7.0, face="hei", color=INK)
        if i < 3:
            arrow(ax, x + cw + 0.2, ty + th / 2, xs[i + 1] - 0.2, ty + th / 2, SKY)


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
    fonts = pick_fonts()
    apply_style(fonts)
    print(f"hei: {fonts['hei_name']}; song: {fonts['song_name']}")
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
