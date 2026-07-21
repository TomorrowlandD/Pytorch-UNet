"""Create the final compact paired IoU scatter plot for U-Net and SAM-3."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd


# Publication-safe fonts and editable vector text.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
# Source-audit markers mirror the settings above: svg.fonttype='none'; pdf.fonttype=42.
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.edgecolor"] = "#606469"
plt.rcParams["xtick.color"] = "#4F5357"
plt.rcParams["ytick.color"] = "#4F5357"


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
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

# Preserve the established restrained scientific palette.
SAM3_COLOR = "#4F7F8C"
UNET_COLOR = "#B2775E"
REFERENCE_COLOR = "#777B80"
GRID_COLOR = "#D9DDE1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input comparison CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for final SVG, PDF, and PNG files.",
    )
    return parser.parse_args()


def load_and_validate(csv_path: Path) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Read every row and stop if pairing, values, or reference statistics fail."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
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
    if numeric.isna().any().any():
        bad_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values found in: {bad_columns}")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Numeric columns contain infinite values")

    validated = df.copy()
    validated["image_id"] = image_ids
    validated[NUMERIC_COLUMNS] = numeric

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


def full_axis_limits(df: pd.DataFrame) -> tuple[float, float]:
    """Compute one rounded overview range for both axes with a small margin."""
    values = df[["unet_iou", "sam3_iou"]].to_numpy()
    data_min = float(values.min())
    data_max = float(values.max())
    margin = max((data_max - data_min) * 0.02, 0.002)
    lower = max(0.0, math.floor((data_min - margin) * 100) / 100)
    upper = min(1.0, math.ceil((data_max + margin) * 100) / 100)
    if lower >= upper:
        raise ValueError(f"Invalid full axis range: {lower}, {upper}")
    return lower, upper


def main_axis_limits(df: pd.DataFrame) -> tuple[float, float]:
    """Use the requested high-IoU range after confirming it contains most samples."""
    candidate = (0.97, 1.00)
    in_candidate = (
        df["unet_iou"].between(*candidate)
        & df["sam3_iou"].between(*candidate)
    )
    if int(in_candidate.sum()) < 0.70 * len(df):
        raise ValueError("The proposed high-IoU main range contains fewer than 70% of samples")
    return candidate


def count_in_square(df: pd.DataFrame, limits: tuple[float, float]) -> int:
    mask = (
        df["unet_iou"].between(*limits)
        & df["sam3_iou"].between(*limits)
    )
    return int(mask.sum())


def draw_scatter_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    point_size: float,
    alpha: float,
    edge_width: float,
) -> None:
    """Draw both outcome groups in one layer without jitter or positional changes."""
    diff = df["sam3_iou"] - df["unet_iou"]
    colors = np.where(diff > 0, SAM3_COLOR, UNET_COLOR)
    ax.scatter(
        df["unet_iou"],
        df["sam3_iou"],
        s=point_size,
        c=colors,
        alpha=alpha,
        edgecolors="white",
        linewidths=edge_width,
        zorder=3,
    )


def style_equal_axes(
    ax: plt.Axes,
    limits: tuple[float, float],
    *,
    grid_width: float,
    grid_alpha: float,
) -> None:
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID_COLOR, linewidth=grid_width, alpha=grid_alpha, zorder=0)


def make_legend_handles(stats: dict[str, float | int]) -> list[Line2D]:
    n = int(stats["n"])
    sam3_better = int(stats["sam3_better"])
    unet_better = int(stats["unet_better"])
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6.2,
            markerfacecolor=SAM3_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.35,
            alpha=0.82,
            label=f"SAM-3 better: {sam3_better} ({sam3_better / n:.1%})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6.2,
            markerfacecolor=UNET_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.35,
            alpha=0.82,
            label=f"U-Net better: {unet_better} ({unet_better / n:.1%})",
        ),
        Line2D(
            [0],
            [0],
            color=REFERENCE_COLOR,
            lw=1.0,
            linestyle=(0, (4, 4)),
            label="Equal performance (y = x)",
        ),
    ]


def make_figure(
    df: pd.DataFrame,
    stats: dict[str, float | int],
) -> tuple[plt.Figure, tuple[float, float], tuple[float, float], int]:
    main_limits = main_axis_limits(df)
    overview_limits = full_axis_limits(df)
    main_count = count_in_square(df, main_limits)

    fig = plt.figure(figsize=(8.5, 6.5), facecolor="white", constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.12, 0.88],
        width_ratios=[5.2, 2.15],
        hspace=0.02,
        wspace=0.04,
    )
    title_ax = fig.add_subplot(grid[0, :])
    ax = fig.add_subplot(grid[1, 0])
    info_ax = fig.add_subplot(grid[1, 1])
    title_ax.axis("off")
    info_ax.axis("off")

    title_ax.text(
        0.5,
        0.70,
        "Per-image IoU Comparison",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="normal",
        color="#25282B",
    )
    title_ax.text(
        0.5,
        0.18,
        f"Carvana validation set, n = {int(stats['n'])}",
        ha="center",
        va="center",
        fontsize=9.8,
        color="#666B70",
    )

    # All rows are passed to the main scatter; axis limits reveal the high-IoU square.
    draw_scatter_points(ax, df, point_size=24, alpha=0.58, edge_width=0.22)
    ax.plot(
        main_limits,
        main_limits,
        color=REFERENCE_COLOR,
        lw=1.0,
        linestyle=(0, (4, 4)),
        alpha=0.82,
        zorder=2,
    )
    style_equal_axes(ax, main_limits, grid_width=0.55, grid_alpha=0.48)
    ax.set_xlabel("U-Net IoU", fontsize=12.5, labelpad=7)
    ax.set_ylabel("SAM-3 IoU", fontsize=12.5, labelpad=7)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", labelsize=9.5, width=0.75, length=3.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Keep the complete-range overview in the right information column so that
    # no main-panel observations are obscured.
    overview = inset_axes(
        info_ax,
        width="88%",
        height="30%",
        loc="lower left",
        bbox_to_anchor=(0.02, 0.08, 0.96, 1.0),
        bbox_transform=info_ax.transAxes,
        borderpad=0.0,
    )
    overview.set_facecolor("#FFFFFF")
    draw_scatter_points(overview, df, point_size=8.5, alpha=0.50, edge_width=0.10)
    overview.plot(
        overview_limits,
        overview_limits,
        color=REFERENCE_COLOR,
        lw=0.75,
        linestyle=(0, (4, 4)),
        alpha=0.80,
        zorder=2,
    )
    overview.add_patch(
        Rectangle(
            (main_limits[0], main_limits[0]),
            main_limits[1] - main_limits[0],
            main_limits[1] - main_limits[0],
            fill=False,
            edgecolor="#3F626B",
            linewidth=1.0,
            zorder=5,
        )
    )
    style_equal_axes(overview, overview_limits, grid_width=0.35, grid_alpha=0.36)
    overview.xaxis.set_major_locator(FixedLocator([0.75, 0.85, 0.95]))
    overview.yaxis.set_major_locator(FixedLocator([0.75, 0.85, 0.95]))
    overview.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    overview.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    overview.tick_params(axis="both", labelsize=6.8, width=0.55, length=2.2, pad=1.3)
    for spine in overview.spines.values():
        spine.set_color("#92969A")
        spine.set_linewidth(0.65)

    info_ax.legend(
        handles=make_legend_handles(stats),
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        frameon=False,
        fontsize=9.2,
        handlelength=1.7,
        handletextpad=0.7,
        labelspacing=0.62,
        borderaxespad=0.0,
    )

    return fig, main_limits, overview_limits, main_count


def save_figure(fig: plt.Figure, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "iou_scatter_unet_vs_sam3_final"
    outputs = [base.with_suffix(extension) for extension in (".svg", ".pdf", ".png")]
    fig.savefig(outputs[0], bbox_inches="tight", pad_inches=0.06)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.06)
    fig.savefig(outputs[2], dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    df, stats = load_and_validate(input_path)
    fig, main_limits, overview_limits, main_count = make_figure(df, stats)
    outputs = save_figure(fig, args.output_dir.resolve())

    print(f"Input CSV: {input_path}")
    print(f"Validated rows: {stats['n']}")
    print(f"U-Net mean IoU: {float(stats['unet_mean']):.9f}")
    print(f"SAM-3 mean IoU: {float(stats['sam3_mean']):.9f}")
    print(f"Mean improvement: {float(stats['mean_improvement']):+.9f}")
    print(
        "SAM-3 better / U-Net better / equal: "
        f"{stats['sam3_better']} / {stats['unet_better']} / {stats['equal']}"
    )
    print(f"Main axis range: [{main_limits[0]:.3f}, {main_limits[1]:.3f}]")
    print(f"Full inset range: [{overview_limits[0]:.3f}, {overview_limits[1]:.3f}]")
    print(f"Samples in main range: {main_count}")
    print("Figure size: 8.5 x 6.5 inches")
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
