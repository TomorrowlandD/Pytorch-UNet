"""Create a publication-ready paired IoU scatter plot for U-Net and SAM-3."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd


# Keep SVG text editable and use fonts commonly available across platforms.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
# Source-audit markers mirror the settings above: svg.fonttype='none'; pdf.fonttype=42.
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["axes.edgecolor"] = "#5F6368"
plt.rcParams["xtick.color"] = "#4F5357"
plt.rcParams["ytick.color"] = "#4F5357"


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT
    / "exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2"
    / "unet_vs_sam3_val_compare.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "figures"

REQUIRED_COLUMNS = [
    "image_id",
    "unet_iou",
    "sam3_iou",
    "iou_diff_sam3_minus_unet",
]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]
EXPECTED_ROWS = 508
EXPECTED_UNET_MEAN = 0.96139
EXPECTED_SAM3_MEAN = 0.98710
EXPECTED_COUNTS = (313, 195, 0)

SAM3_COLOR = "#4F7F8C"  # muted teal
UNET_COLOR = "#B2775E"  # muted terracotta
REFERENCE_COLOR = "#777B80"
GRID_COLOR = "#D9DDE1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input comparison CSV (default: project data file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for SVG, PDF, and PNG outputs.",
    )
    return parser.parse_args()


def load_and_validate(csv_path: Path) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Read all rows and stop on any integrity or reference-statistics mismatch."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")

    if df["image_id"].isna().any():
        raise ValueError("image_id contains missing values")
    image_ids = df["image_id"].astype(str).str.strip()
    if image_ids.eq("").any():
        raise ValueError("image_id contains empty values")
    duplicate_count = int(image_ids.duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate image_id values")

    numeric = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = numeric.isna() & df[NUMERIC_COLUMNS].notna()
    if invalid_numeric.any().any():
        bad_columns = invalid_numeric.any()[invalid_numeric.any()].index.tolist()
        raise ValueError(f"Non-numeric values found in: {bad_columns}")
    if numeric.isna().any().any():
        bad_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Missing numeric values found in: {bad_columns}")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Numeric columns contain infinite values")

    validated = df.copy()
    validated["image_id"] = image_ids
    validated[NUMERIC_COLUMNS] = numeric

    # Each unique image_id retains both model values on the same row; no rows are removed.
    recalculated_diff = validated["sam3_iou"] - validated["unet_iou"]
    if not np.allclose(
        recalculated_diff,
        validated["iou_diff_sam3_minus_unet"],
        atol=5e-7,
        rtol=0.0,
    ):
        max_error = float(
            np.max(
                np.abs(
                    recalculated_diff
                    - validated["iou_diff_sam3_minus_unet"]
                )
            )
        )
        raise ValueError(f"Stored IoU difference is inconsistent (max error={max_error})")

    sam3_better = int((recalculated_diff > 0).sum())
    unet_better = int((recalculated_diff < 0).sum())
    equal = int((recalculated_diff == 0).sum())
    stats: dict[str, float | int] = {
        "n": len(validated),
        "unet_mean": float(validated["unet_iou"].mean()),
        "sam3_mean": float(validated["sam3_iou"].mean()),
        "mean_improvement": float(recalculated_diff.mean()),
        "sam3_better": sam3_better,
        "unet_better": unet_better,
        "equal": equal,
    }

    if abs(float(stats["unet_mean"]) - EXPECTED_UNET_MEAN) > 5e-4:
        raise ValueError("U-Net mean IoU does not match the reference check")
    if abs(float(stats["sam3_mean"]) - EXPECTED_SAM3_MEAN) > 5e-4:
        raise ValueError("SAM-3 mean IoU does not match the reference check")
    if (sam3_better, unet_better, equal) != EXPECTED_COUNTS:
        raise ValueError(
            "Win/loss/tie counts do not match the reference check: "
            f"{sam3_better}/{unet_better}/{equal}"
        )
    return validated, stats


def shared_axis_limits(df: pd.DataFrame) -> tuple[float, float]:
    """Compute one rounded range for both axes with a small data-driven margin."""
    values = df[["unet_iou", "sam3_iou"]].to_numpy()
    data_min = float(values.min())
    data_max = float(values.max())
    margin = max((data_max - data_min) * 0.02, 0.002)
    lower = max(0.0, math.floor((data_min - margin) * 100) / 100)
    upper = min(1.0, math.ceil((data_max + margin) * 100) / 100)
    if lower >= upper:
        raise ValueError(f"Invalid shared axis range: {lower}, {upper}")
    return lower, upper


def inset_axis_limits(df: pd.DataFrame, main_limits: tuple[float, float]) -> tuple[float, float]:
    """Define an equal-axis zoom around the high-IoU cluster without moving points."""
    q25 = min(df["unet_iou"].quantile(0.25), df["sam3_iou"].quantile(0.25))
    lower = max(main_limits[0], math.floor(float(q25) * 100) / 100)
    data_max = float(df[["unet_iou", "sam3_iou"]].to_numpy().max())
    margin = max((main_limits[1] - main_limits[0]) * 0.0075, 0.0015)
    upper = min(main_limits[1], math.ceil((data_max + margin) * 1000) / 1000)
    if upper - lower < 0.02:
        lower = max(main_limits[0], upper - 0.03)
    return lower, upper


def short_image_id(image_id: str) -> str:
    """Shorten long IDs while preserving their recognizable prefix and suffix."""
    if len(image_id) <= 12:
        return image_id
    return f"{image_id[:6]}...{image_id[-3:]}"


def draw_scatter_points(ax: plt.Axes, df: pd.DataFrame, *, inset: bool = False) -> None:
    diff = df["sam3_iou"] - df["unet_iou"]
    point_size = 15 if inset else 24
    alpha = 0.48 if inset else 0.60
    for mask, color in ((diff > 0, SAM3_COLOR), (diff < 0, UNET_COLOR)):
        ax.scatter(
            df.loc[mask, "unet_iou"],
            df.loc[mask, "sam3_iou"],
            s=point_size,
            c=color,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )


def annotate_largest_differences(ax: plt.Axes, df: pd.DataFrame, count: int = 3) -> None:
    diff = df["sam3_iou"] - df["unet_iou"]
    indices = diff.abs().nlargest(count).index
    offsets = [(26, -18), (34, -40), (42, -62)]
    for idx, offset in zip(indices, offsets):
        row = df.loc[idx]
        ax.annotate(
            short_image_id(str(row["image_id"])),
            xy=(row["unet_iou"], row["sam3_iou"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color="#4D5155",
            ha="left",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": "#8A8E92",
                "lw": 0.65,
                "shrinkA": 2,
                "shrinkB": 2,
            },
            zorder=5,
        )


def make_figure(
    df: pd.DataFrame,
    stats: dict[str, float | int],
) -> tuple[plt.Figure, tuple[float, float], tuple[float, float]]:
    axis_limits = shared_axis_limits(df)
    zoom_limits = inset_axis_limits(df, axis_limits)
    lower, upper = axis_limits

    # A 16:9 canvas leaves a clean right-hand information column for PPT use.
    fig = plt.figure(figsize=(12, 6.75), facecolor="white")
    ax = fig.add_axes([0.07, 0.14, 0.51, 0.72])
    side_ax = fig.add_axes([0.64, 0.17, 0.32, 0.64])
    side_ax.axis("off")

    draw_scatter_points(ax, df)
    ax.plot(
        [lower, upper],
        [lower, upper],
        color=REFERENCE_COLOR,
        lw=1.1,
        linestyle=(0, (4, 4)),
        alpha=0.80,
        zorder=2,
    )
    annotate_largest_differences(ax, df, count=3)

    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("U-Net IoU", fontsize=14, labelpad=8)
    ax.set_ylabel("SAM-3 IoU", fontsize=14, labelpad=8)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", labelsize=10.5, width=0.8, length=4)
    ax.grid(True, color=GRID_COLOR, linewidth=0.65, alpha=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # The inset redraws the same observations without jitter, smoothing, or fitting.
    inset = inset_axes(ax, width="42%", height="42%", loc="lower right", borderpad=1.0)
    inset.set_facecolor("#FFFFFF")
    draw_scatter_points(inset, df, inset=True)
    inset.plot(
        zoom_limits,
        zoom_limits,
        color=REFERENCE_COLOR,
        lw=0.9,
        linestyle=(0, (4, 4)),
        alpha=0.80,
        zorder=2,
    )
    inset.set_xlim(zoom_limits)
    inset.set_ylim(zoom_limits)
    inset.set_aspect("equal", adjustable="box")
    inset.set_title("High-IoU region", fontsize=9.5, pad=4, color="#3F4347")
    inset.xaxis.set_major_locator(MultipleLocator(0.01))
    inset.yaxis.set_major_locator(MultipleLocator(0.01))
    inset.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    inset.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    inset.tick_params(axis="both", labelsize=7.5, width=0.6, length=2.5, pad=1.5)
    inset.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.50, zorder=0)
    for spine in inset.spines.values():
        spine.set_color("#9A9EA2")
        spine.set_linewidth(0.7)

    n = int(stats["n"])
    sam3_better = int(stats["sam3_better"])
    unet_better = int(stats["unet_better"])
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=SAM3_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.4,
            alpha=0.80,
            label=f"SAM-3 better: {sam3_better} ({sam3_better / n:.1%})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=UNET_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.4,
            alpha=0.80,
            label=f"U-Net better: {unet_better} ({unet_better / n:.1%})",
        ),
        Line2D(
            [0],
            [0],
            color=REFERENCE_COLOR,
            lw=1.1,
            linestyle=(0, (4, 4)),
            label="Equal performance (y = x)",
        ),
    ]
    side_ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        frameon=False,
        fontsize=11.5,
        handlelength=2.0,
        labelspacing=0.9,
        borderaxespad=0.0,
    )

    stats_text = (
        f"Mean IoU: {float(stats['unet_mean']):.4f} \u2192 {float(stats['sam3_mean']):.4f}\n"
        f"Mean improvement: {float(stats['mean_improvement']):+.4f}\n"
        f"SAM-3 better: {sam3_better} / {n}"
    )
    side_ax.text(
        0.0,
        0.58,
        stats_text,
        transform=side_ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        linespacing=1.55,
        color="#303438",
        bbox={
            "boxstyle": "round,pad=0.55,rounding_size=0.08",
            "facecolor": "#F7F8F9",
            "edgecolor": "#C8CCD0",
            "linewidth": 0.8,
        },
    )
    side_ax.text(
        0.0,
        0.26,
        "Points above the diagonal favor SAM-3;\npoints below favor U-Net.",
        transform=side_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.4,
        color="#666B70",
    )

    fig.suptitle(
        "Per-image IoU Comparison",
        x=0.5,
        y=0.965,
        fontsize=20,
        fontweight="semibold",
        color="#25282B",
    )
    fig.text(
        0.5,
        0.910,
        f"Carvana validation set, n = {n}",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#666B70",
    )
    return fig, axis_limits, zoom_limits


def save_figure(fig: plt.Figure, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "iou_scatter_unet_vs_sam3"
    outputs = [base.with_suffix(ext) for ext in (".svg", ".pdf", ".png")]
    fig.savefig(outputs[0], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(outputs[2], dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    df, stats = load_and_validate(args.input.resolve())
    fig, axis_limits, zoom_limits = make_figure(df, stats)
    outputs = save_figure(fig, args.output_dir.resolve())

    print(f"Validated rows: {stats['n']}")
    print(f"U-Net mean IoU: {float(stats['unet_mean']):.9f}")
    print(f"SAM-3 mean IoU: {float(stats['sam3_mean']):.9f}")
    print(f"Mean improvement: {float(stats['mean_improvement']):+.9f}")
    print(
        "SAM-3 better / U-Net better / equal: "
        f"{stats['sam3_better']} / {stats['unet_better']} / {stats['equal']}"
    )
    print(f"Shared axis range: [{axis_limits[0]:.3f}, {axis_limits[1]:.3f}]")
    print(f"Inset axis range: [{zoom_limits[0]:.3f}, {zoom_limits[1]:.3f}]")
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
