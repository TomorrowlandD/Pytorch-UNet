"""Compact 100% stacked bar chart for per-image Dice advantages."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Figure contract
# Core conclusion: SAM-3 has the higher per-image Dice on 313 of 508
# validation images (61.6%), while U-Net is higher on 195 images (38.4%).
# Archetype: quantitative grid (single compact composition panel).
# Output target: half-page area in a scientific presentation.

# Exact values supplied by the user; no inferred or simulated results.
TOTAL_IMAGES = 508
SAM3_COUNT = 313
UNET_COUNT = 195
SAM3_PERCENT = 61.6
UNET_PERCENT = 38.4

assert SAM3_COUNT + UNET_COUNT == TOTAL_IMAGES
assert abs(SAM3_PERCENT + UNET_PERCENT - 100.0) < 1e-9


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft YaHei",
            "Arial",
            "DejaVu Sans",
            "Liberation Sans",
        ],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9.5,
        "axes.unicode_minus": False,
    }
)


# Keep these model colors consistent with the mean Dice/IoU comparison chart.
COLORS = {
    "SAM-3": "#E58B3E",
    "U-Net": "#4C78A8",
}


def make_figure() -> plt.Figure:
    """Build the compact horizontal 100% stacked bar chart."""
    fig, ax = plt.subplots(figsize=(7.2, 3.15), facecolor="white")
    ax.set_facecolor("white")

    bar_y = 0.0
    bar_height = 0.22

    ax.barh(
        bar_y,
        SAM3_PERCENT,
        left=0,
        height=bar_height,
        color=COLORS["SAM-3"],
        edgecolor="none",
        zorder=3,
    )
    ax.barh(
        bar_y,
        UNET_PERCENT,
        left=SAM3_PERCENT,
        height=bar_height,
        color=COLORS["U-Net"],
        edgecolor="none",
        zorder=3,
    )

    # A single subtle divider separates the two model regions.
    ax.vlines(
        SAM3_PERCENT,
        bar_y - bar_height / 2,
        bar_y + bar_height / 2,
        color="white",
        linewidth=1.1,
        zorder=4,
    )

    ax.text(
        SAM3_PERCENT / 2,
        bar_y,
        f"SAM-3  {SAM3_COUNT}（{SAM3_PERCENT:.1f}%）",
        ha="center",
        va="center",
        color="white",
        fontsize=10.8,
        fontweight="semibold",
        zorder=5,
    )
    ax.text(
        SAM3_PERCENT + UNET_PERCENT / 2,
        bar_y,
        f"U-Net  {UNET_COUNT}（{UNET_PERCENT:.1f}%）",
        ha="center",
        va="center",
        color="white",
        fontsize=10.8,
        fontweight="semibold",
        zorder=5,
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.34, 0.34)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.tick_params(axis="x", labelsize=9.5, length=0, pad=7, colors="#555555")
    ax.grid(axis="x", color="#E4E4E4", linewidth=0.55, alpha=0.75, zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        "逐图 Dice 优势样本分布",
        x=0.5,
        y=0.925,
        fontsize=16,
        fontweight="semibold",
        color="#222222",
    )
    fig.text(
        0.5,
        0.825,
        "508 张验证图像，按逐图 Dice 比较",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#666666",
    )

    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.20, top=0.70)
    return fig


def save_figure(fig: plt.Figure) -> None:
    """Export editable SVG, 300 dpi PNG, and an archival PDF."""
    output_base = Path(__file__).with_name("逐图分割性能胜负统计")
    save_kwargs = {
        "facecolor": "white",
        "bbox_inches": "tight",
        "pad_inches": 0.08,
    }
    fig.savefig(output_base.with_suffix(".svg"), **save_kwargs)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, **save_kwargs)
    fig.savefig(output_base.with_suffix(".pdf"), **save_kwargs)


if __name__ == "__main__":
    figure = make_figure()
    save_figure(figure)
    plt.close(figure)
