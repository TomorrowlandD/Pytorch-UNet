"""Grouped bar chart comparing mean segmentation performance of U-Net and SAM-3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# Figure contract
# Core conclusion: SAM-3 has higher mean Dice and mean IoU than U-Net on the
# same 508-image validation set.
# Archetype: quantitative grid (single comparison panel).
# Output: slide-ready PNG plus editable SVG and PDF.

# Mandatory editable-text and font settings.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
plt.rcParams["axes.unicode_minus"] = False


# Exact values transcribed from the supplied experiment summary.
METRICS = ["Mean Dice", "Mean IoU"]
UNET = np.array([0.9794742027, 0.9613934193]) * 100
SAM3 = np.array([0.9935022748, 0.9870986631]) * 100
IMPROVEMENT_PP = SAM3 - UNET


def make_figure() -> plt.Figure:
    """Create the grouped bar chart without altering the supplied values."""
    fig_width_mm = 183.0
    fig_height_mm = 105.0
    fig, ax = plt.subplots(
        figsize=(fig_width_mm / 25.4, fig_height_mm / 25.4),
        facecolor="white",
    )
    ax.set_facecolor("white")

    x = np.arange(len(METRICS))
    width = 0.30
    colors = {"U-Net": "#4C78A8", "SAM-3": "#E58B3E"}

    bars_unet = ax.bar(
        x - width / 2,
        UNET,
        width,
        label="U-Net",
        color=colors["U-Net"],
        edgecolor="#2F2F2F",
        linewidth=0.8,
        zorder=3,
    )
    bars_sam3 = ax.bar(
        x + width / 2,
        SAM3,
        width,
        label="SAM-3",
        color=colors["SAM-3"],
        edgecolor="#2F2F2F",
        linewidth=0.8,
        zorder=3,
    )

    # Exact 0-100 percent scale requested by the user.
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.set_ylabel("平均性能（%）", fontsize=15, labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=15)
    ax.tick_params(axis="y", labelsize=13, length=0)
    ax.tick_params(axis="x", length=0, pad=10)

    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Label every bar near its top, inside the fill, to avoid collisions with
    # the improvement annotations above the strict 100% axis limit.
    for bars in (bars_unet, bars_sam3):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value - 3.0,
                f"{value:.2f}%",
                ha="center",
                va="top",
                fontsize=12.5,
                fontweight="semibold",
                color="white",
            )

    # Place improvement brackets just above the 0-100 plotting range while
    # keeping the y-axis itself strictly fixed at 0-100%.
    for index, improvement in enumerate(IMPROVEMENT_PP):
        left = x[index] - width / 2
        right = x[index] + width / 2
        bracket_y = 101.15
        tick_bottom = 100.45
        ax.plot(
            [left, left, right, right],
            [tick_bottom, bracket_y, bracket_y, tick_bottom],
            color="#4B4B4B",
            linewidth=1.0,
            clip_on=False,
        )
        ax.text(
            x[index],
            bracket_y + 0.32,
            f"SAM-3 提升 +{improvement:.2f} 个百分点",
            ha="center",
            va="bottom",
            fontsize=12.5,
            color="#2E7D32",
            fontweight="semibold",
            clip_on=False,
        )

    ax.set_title(
        "U-Net 与 SAM-3 平均分割性能对比",
        fontsize=20,
        fontweight="semibold",
        pad=56,
        color="#1F1F1F",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
        fontsize=14,
        handlelength=1.4,
        columnspacing=2.5,
    )

    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.22, top=0.75)
    return fig


def save_figure(fig: plt.Figure) -> None:
    """Export a presentation preview and editable vector versions."""
    output_base = Path(__file__).resolve().parents[1] / "PPT" / "unet_sam3_mean_performance_comparison"
    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.12,
    )
    fig.savefig(
        output_base.with_suffix(".svg"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.12,
    )
    fig.savefig(
        output_base.with_suffix(".pdf"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.12,
    )
    fig.savefig(
        output_base.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.12,
        pil_kwargs={"compression": "tiff_lzw"},
    )


if __name__ == "__main__":
    figure = make_figure()
    save_figure(figure)
    plt.close(figure)
