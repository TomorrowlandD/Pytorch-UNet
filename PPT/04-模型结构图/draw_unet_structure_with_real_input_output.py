from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image, ImageOps


# Slide-native 16:9 canvas. At 300 dpi the raster export is exactly 3840 x 2160 px.
CANVAS_W = 3840
CANVAS_H = 2160
EXPORT_DPI = 300
FONT_SCALE = 0.60
LINE_SCALE = 0.33

OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OUTPUT_DIR.parents[1]
SOURCE_DIR = (
    PROJECT_ROOT
    / "exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2"
    / "selected_original_data"
)
INPUT_PATH = SOURCE_DIR / "imgs" / "near_tie_04_917f262f1608_15.jpg"
MASK_PATH = SOURCE_DIR / "masks" / "near_tie_04_917f262f1608_15_mask.gif"

SVG_PATH = OUTPUT_DIR / "unet_structure_with_real_input_output.svg"
PNG_PATH = OUTPUT_DIR / "unet_structure_with_real_input_output.png"
PDF_PATH = OUTPUT_DIR / "unet_structure_with_real_input_output.pdf"
TIFF_PATH = OUTPUT_DIR / "unet_structure_with_real_input_output.tiff"
MASK_FIRST_FRAME_PATH = OUTPUT_DIR / "near_tie_04_917f262f1608_15_mask_first_frame.png"


COLORS = {
    "navy": "#0B3B8F",
    "navy_dark": "#082B6F",
    "blue": "#1F63B5",
    "upsample_fill": "#DCEEFF",
    "upsample_edge": "#4C8FD5",
    "red": "#D62728",
    "orange": "#F28E1C",
    "gray": "#7A818B",
    "gray_dark": "#4E5968",
    "border": "#AAB4C3",
    "light_border": "#DCE2EA",
    "text": "#172033",
    "muted": "#667085",
    "white": "#FFFFFF",
    "panel": "#FAFBFD",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 16,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "image.composite_image": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def load_and_validate_sources() -> tuple[Image.Image, Image.Image]:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Missing input image: {INPUT_PATH}")
    if not MASK_PATH.is_file():
        raise FileNotFoundError(f"Missing mask: {MASK_PATH}")

    input_image = Image.open(INPUT_PATH).convert("RGB")

    with Image.open(MASK_PATH) as gif:
        gif.seek(0)
        mask = gif.convert("L")

    # The source is already binary. Thresholding is deterministic and prevents
    # palette conversion from introducing any non-binary values.
    mask = mask.point(lambda value: 255 if value >= 128 else 0, mode="L")
    values = set(np.unique(np.asarray(mask)).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"Mask is not binary after conversion: {sorted(values)}")

    mask.save(MASK_FIRST_FRAME_PATH, format="PNG", optimize=True)
    return input_image, mask


def contain_image(
    image: Image.Image,
    size: tuple[int, int],
    *,
    resample: Image.Resampling,
    background: int | tuple[int, int, int],
) -> Image.Image:
    contained = ImageOps.contain(image, size, method=resample)
    if image.mode == "L":
        panel = Image.new("L", size, color=int(background))
    else:
        panel = Image.new("RGB", size, color=background)
    left = (size[0] - contained.width) // 2
    top = (size[1] - contained.height) // 2
    panel.paste(contained, (left, top))
    return panel


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    value: str,
    *,
    size: float,
    color: str = COLORS["text"],
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    linespacing: float = 1.15,
    gid: str | None = None,
    zorder: int = 10,
) -> None:
    artist = ax.text(
        x,
        y,
        value,
        fontsize=size * FONT_SCALE,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=linespacing,
        zorder=zorder,
    )
    if gid:
        artist.set_gid(gid)


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    width: float,
    head: float,
    gid: str,
    zorder: int = 5,
) -> FancyArrowPatch:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=head * LINE_SCALE,
        linewidth=width * LINE_SCALE,
        color=color,
        shrinkA=0,
        shrinkB=0,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    arrow.set_gid(gid)
    ax.add_patch(arrow)
    return arrow


def add_feature_stage(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    gid: str,
    fill: str = COLORS["navy"],
) -> tuple[float, float, float, float]:
    block_w = 32
    block_h = 120
    gap = 20
    total_w = block_w * 2 + gap

    for index, block_x in enumerate((x, x + block_w + gap), start=1):
        rect = FancyBboxPatch(
            (block_x, y),
            block_w,
            block_h,
            boxstyle="round,pad=0,rounding_size=2",
            linewidth=1.5 * LINE_SCALE,
            edgecolor=COLORS["navy_dark"],
            facecolor=fill,
            zorder=7,
        )
        rect.set_gid(f"{gid}-conv-{index}")
        ax.add_patch(rect)

    add_arrow(
        ax,
        (x + block_w + 3, y + block_h / 2),
        (x + block_w + gap - 3, y + block_h / 2),
        color=COLORS["blue"],
        width=2.0,
        head=12,
        gid=f"{gid}-internal-flow",
        zorder=8,
    )
    return x, y, total_w, block_h


def add_upsampling_module(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    gid: str,
) -> tuple[float, float, float, float]:
    width = 34
    height = 120
    rect = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0,rounding_size=2",
        linewidth=2 * LINE_SCALE,
        edgecolor=COLORS["upsample_edge"],
        facecolor=COLORS["upsample_fill"],
        zorder=7,
    )
    rect.set_gid(gid)
    ax.add_patch(rect)
    return x, y, width, height


def draw_legend(ax: plt.Axes) -> None:
    panel_x, panel_y = 2980, 1450
    panel_w, panel_h = 760, 570
    panel = FancyBboxPatch(
        (panel_x, panel_y),
        panel_w,
        panel_h,
        boxstyle="round,pad=0,rounding_size=8",
        linewidth=1.2 * LINE_SCALE,
        edgecolor=COLORS["light_border"],
        facecolor=COLORS["white"],
        zorder=3,
    )
    panel.set_gid("legend-panel")
    ax.add_patch(panel)

    add_text(
        ax,
        panel_x + panel_w / 2,
        1500,
        "Legend",
        size=19,
        color=COLORS["gray_dark"],
        weight="bold",
        gid="legend-title",
    )

    icon_x = 3045
    label_x = 3110
    rows = [1575, 1665, 1755, 1845, 1935]

    # Conv + ReLU
    y = rows[0]
    swatch = Rectangle((icon_x, y - 20), 18, 40, facecolor=COLORS["navy"], edgecolor=COLORS["navy_dark"], lw=1.5 * LINE_SCALE, zorder=8)
    swatch.set_gid("legend-conv-swatch")
    ax.add_patch(swatch)
    add_text(ax, label_x, y, "Conv + ReLU", size=16, ha="left", gid="legend-conv-label")

    # Max Pool
    y = rows[1]
    add_arrow(
        ax,
        (icon_x + 9, y - 20),
        (icon_x + 9, y + 20),
        color=COLORS["red"],
        width=3.0,
        head=15,
        gid="legend-max-pool-arrow",
        zorder=8,
    )
    add_text(ax, label_x, y, "Max Pool", size=16, color=COLORS["red"], ha="left", gid="legend-max-pool-label")

    # Skip Connection
    y = rows[2]
    add_arrow(
        ax,
        (icon_x - 18, y),
        (icon_x + 48, y),
        color=COLORS["gray"],
        width=2.0,
        head=14,
        gid="legend-skip-arrow",
        zorder=8,
    )
    add_text(ax, label_x, y, "Skip Connection", size=16, color=COLORS["gray_dark"], ha="left", gid="legend-skip-label")

    # Bilinear Upsample + Conv
    y = rows[3]
    up_rect = Rectangle(
        (icon_x, y - 20),
        18,
        40,
        facecolor=COLORS["upsample_fill"],
        edgecolor=COLORS["upsample_edge"],
        lw=1.8 * LINE_SCALE,
        zorder=8,
    )
    up_rect.set_gid("legend-bilinear-swatch")
    ax.add_patch(up_rect)
    add_arrow(
        ax,
        (icon_x - 17, y + 18),
        (icon_x - 17, y - 18),
        color=COLORS["orange"],
        width=2.8,
        head=14,
        gid="legend-bilinear-arrow",
        zorder=8,
    )
    add_text(
        ax,
        label_x,
        y,
        "Bilinear Upsample + Conv",
        size=16,
        color=COLORS["orange"],
        ha="left",
        gid="legend-bilinear-label",
    )

    # 1x1 head
    y = rows[4]
    head = Rectangle((icon_x, y - 20), 18, 40, facecolor=COLORS["white"], edgecolor=COLORS["navy"], lw=2.2 * LINE_SCALE, zorder=8)
    head.set_gid("legend-1x1-swatch")
    ax.add_patch(head)
    add_text(ax, label_x, y, "1×1 Conv", size=16, ha="left", gid="legend-1x1-label")


def build_figure(input_image: Image.Image, mask: Image.Image) -> plt.Figure:
    figure = plt.figure(
        figsize=(CANVAS_W / EXPORT_DPI, CANVAS_H / EXPORT_DPI),
        dpi=EXPORT_DPI,
        facecolor=COLORS["white"],
    )
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Input and output panels use identical containers.
    panel_w, panel_h = 600, 400
    panel_y = 170
    input_x = 70
    output_x = 3170

    prepared_input = contain_image(
        input_image,
        (panel_w, panel_h),
        resample=Image.Resampling.LANCZOS,
        background=(255, 255, 255),
    )
    prepared_mask = contain_image(
        mask,
        (panel_w, panel_h),
        resample=Image.Resampling.NEAREST,
        background=0,
    )
    if not set(np.unique(np.asarray(prepared_mask)).tolist()).issubset({0, 255}):
        raise ValueError("Nearest-neighbor panel preparation introduced non-binary mask values")

    heading_y = 105
    add_text(ax, input_x + panel_w / 2, heading_y, "Input Image", size=30, weight="bold", gid="input-title")
    add_text(ax, output_x + panel_w / 2, heading_y, "U-Net Prediction", size=30, weight="bold", gid="output-title")

    ax.imshow(
        np.asarray(prepared_input),
        extent=(input_x, input_x + panel_w, panel_y + panel_h, panel_y),
        origin="upper",
        interpolation="nearest",
        aspect="auto",
        zorder=2,
    ).set_gid("real-input-image")
    ax.imshow(
        np.asarray(prepared_mask),
        extent=(output_x, output_x + panel_w, panel_y + panel_h, panel_y),
        origin="upper",
        interpolation="nearest",
        cmap="gray",
        vmin=0,
        vmax=255,
        aspect="auto",
        zorder=2,
    ).set_gid("real-predicted-mask")

    for gid, x in (("input-frame", input_x), ("output-frame", output_x)):
        frame = Rectangle(
            (x, panel_y),
            panel_w,
            panel_h,
            facecolor="none",
            edgecolor=COLORS["border"],
            linewidth=3 * LINE_SCALE,
            zorder=3,
        )
        frame.set_gid(gid)
        ax.add_patch(frame)

    # Section headings.
    add_text(ax, 1340, heading_y, "Encoder Path", size=30, color=COLORS["navy_dark"], weight="bold", gid="encoder-title")
    add_text(ax, 2540, heading_y, "Decoder Path", size=30, color=COLORS["navy_dark"], weight="bold", gid="decoder-title")

    # Stage coordinates define the classic U-shaped topology.
    encoder_x = [850, 1110, 1370, 1630]
    stage_y = [310, 590, 870, 1150]
    bottleneck_x, bottleneck_y = 1890, 1430
    upsample_x = [2150, 2385, 2620, 2855]
    decoder_x = [2200, 2435, 2670, 2905]
    decoder_y = list(reversed(stage_y))

    encoder_boxes = [
        add_feature_stage(ax, x, y, gid=f"encoder-stage-{index}")
        for index, (x, y) in enumerate(zip(encoder_x, stage_y), start=1)
    ]
    bottleneck_box = add_feature_stage(ax, bottleneck_x, bottleneck_y, gid="bottleneck")
    add_text(
        ax,
        bottleneck_x + bottleneck_box[2] / 2,
        bottleneck_y + 155,
        "Bottleneck",
        size=20,
        color=COLORS["gray_dark"],
        weight="bold",
        gid="bottleneck-label",
    )
    add_text(
        ax,
        bottleneck_x + bottleneck_box[2] / 2,
        bottleneck_y + 188,
        "Deep Semantic Features",
        size=14,
        color=COLORS["muted"],
        gid="bottleneck-semantic-label",
    )

    up_boxes = [
        add_upsampling_module(ax, x, y, gid=f"bilinear-upsampling-stage-{index}")
        for index, (x, y) in enumerate(zip(upsample_x, decoder_y), start=4)
    ]
    decoder_boxes = [
        add_feature_stage(ax, x, y, gid=f"decoder-stage-{index}")
        for index, (x, y) in enumerate(zip(decoder_x, decoder_y), start=4)
    ]

    # Encoder flow: four max-pooling transitions.
    encoder_targets = encoder_boxes[1:] + [bottleneck_box]
    for index, (source, target) in enumerate(zip(encoder_boxes, encoder_targets), start=1):
        source_center = (source[0] + source[2] / 2, source[1] + source[3])
        target_center = (target[0] + target[2] / 2, target[1])
        add_arrow(
            ax,
            source_center,
            target_center,
            color=COLORS["red"],
            width=3.8,
            head=22,
            gid=f"max-pool-{index}",
            zorder=6,
        )
        mid_x = (source_center[0] + target_center[0]) / 2
        mid_y = (source_center[1] + target_center[1]) / 2 - 70
        add_text(
            ax,
            mid_x,
            mid_y,
            "Max Pool",
            size=15,
            color=COLORS["red"],
            weight="bold",
            gid=f"max-pool-label-{index}",
        )

    # Skip connections link matching encoder and decoder resolutions.
    matching_up_boxes = list(reversed(up_boxes))
    for index, (source, target) in enumerate(zip(encoder_boxes, matching_up_boxes), start=1):
        y = source[1] + source[3] / 2
        add_arrow(
            ax,
            (source[0] + source[2] + 10, y),
            (target[0] - 10, y),
            color=COLORS["gray"],
            width=2.1,
            head=17,
            gid=f"skip-connection-{index}",
            zorder=4,
        )
        if index == 1:
            add_text(
                ax,
                (source[0] + source[2] + target[0]) / 2,
                y - 26,
                "Skip Connection",
                size=19,
                color=COLORS["gray_dark"],
                weight="bold",
                gid="skip-connection-callout",
                zorder=6,
            )

    # Decoder flow: bottom-up bilinear upsampling followed by convolution.
    decoder_sources = [bottleneck_box] + decoder_boxes[:-1]
    for index, (source, up_box, decoder_box) in enumerate(zip(decoder_sources, up_boxes, decoder_boxes), start=1):
        source_point = (source[0] + source[2] + 10, source[1] + source[3] / 2)
        up_point = (up_box[0] + up_box[2] / 2, up_box[1] + up_box[3] / 2)
        add_arrow(
            ax,
            source_point,
            up_point,
            color=COLORS["orange"],
            width=4.0,
            head=23,
            gid=f"bilinear-upsampling-flow-{index}",
            zorder=6,
        )
        add_arrow(
            ax,
            (up_box[0] + up_box[2] + 4, up_point[1]),
            (decoder_box[0] - 7, up_point[1]),
            color=COLORS["blue"],
            width=2.2,
            head=14,
            gid=f"upsampling-to-decoder-{index}",
            zorder=8,
        )

    add_text(
        ax,
        2520,
        515,
        "Bilinear Upsample + Conv",
        size=15,
        color=COLORS["orange"],
        weight="bold",
        gid="bilinear-upsample-callout",
    )

    # Network input and segmentation head.
    add_arrow(
        ax,
        (input_x + panel_w, stage_y[0] + 60),
        (encoder_x[0] - 25, stage_y[0] + 60),
        color=COLORS["blue"],
        width=4.5,
        head=26,
        gid="input-to-network",
        zorder=6,
    )

    top_decoder = decoder_boxes[-1]
    head_x, head_y, head_w, head_h = 3020, stage_y[0], 44, 120
    add_arrow(
        ax,
        (top_decoder[0] + top_decoder[2] + 5, stage_y[0] + 60),
        (head_x - 8, stage_y[0] + 60),
        color=COLORS["blue"],
        width=2.4,
        head=15,
        gid="decoder-to-1x1",
        zorder=8,
    )
    head = Rectangle(
        (head_x, head_y),
        head_w,
        head_h,
        facecolor=COLORS["white"],
        edgecolor=COLORS["navy"],
        linewidth=3 * LINE_SCALE,
        zorder=7,
    )
    head.set_gid("final-1x1-conv")
    ax.add_patch(head)
    add_text(ax, head_x + head_w / 2, head_y + head_h + 30, "1×1 Conv", size=19, weight="bold", gid="final-1x1-label")

    add_arrow(
        ax,
        (head_x + head_w + 5, stage_y[0] + 60),
        (output_x - 20, stage_y[0] + 60),
        color=COLORS["blue"],
        width=4.5,
        head=26,
        gid="network-to-predicted-mask",
        zorder=6,
    )
    draw_legend(ax)
    return figure


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_image, mask = load_and_validate_sources()
    figure = build_figure(input_image, mask)

    metadata = {
        "Title": "Simplified U-Net Segmentation Pipeline with Real Input and Prediction",
        "Description": "Editable simplified U-Net encoder-decoder schematic with real experimental input and binary prediction mask.",
        "Creator": "Deterministic Python/Matplotlib workflow",
    }
    figure.savefig(
        SVG_PATH,
        format="svg",
        dpi=EXPORT_DPI,
        facecolor=COLORS["white"],
        bbox_inches=None,
        pad_inches=0,
        metadata=metadata,
    )
    figure.savefig(
        PNG_PATH,
        format="png",
        dpi=EXPORT_DPI,
        facecolor=COLORS["white"],
        bbox_inches=None,
        pad_inches=0,
        metadata={"Title": metadata["Title"], "Description": metadata["Description"]},
    )
    figure.savefig(
        PDF_PATH,
        format="pdf",
        dpi=EXPORT_DPI,
        facecolor=COLORS["white"],
        bbox_inches=None,
        pad_inches=0,
        metadata={"Title": metadata["Title"], "Subject": metadata["Description"]},
    )
    figure.savefig(
        TIFF_PATH,
        format="tiff",
        dpi=EXPORT_DPI,
        facecolor=COLORS["white"],
        bbox_inches=None,
        pad_inches=0,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)

    with Image.open(PNG_PATH) as rendered:
        if rendered.size != (CANVAS_W, CANVAS_H):
            raise ValueError(f"Unexpected PNG size: {rendered.size}")
        # Check the exact output-image interior, excluding the vector border.
        mask_crop = rendered.convert("RGB").crop((3185, 185, 3755, 555))
        colors = set(mask_crop.getdata())
        if not colors.issubset({(0, 0, 0), (255, 255, 255)}):
            raise ValueError(f"Rendered mask contains non-binary colors: {len(colors)} colors")

    print(f"SVG: {SVG_PATH}")
    print(f"PNG: {PNG_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"TIFF: {TIFF_PATH}")
    print(f"Mask first frame: {MASK_FIRST_FRAME_PATH}")


if __name__ == "__main__":
    main()
