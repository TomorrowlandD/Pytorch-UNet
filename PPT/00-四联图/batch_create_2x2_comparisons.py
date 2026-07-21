from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import argparse
import csv
import shutil
import zipfile


CANVAS_WIDTH = 2400
CANVAS_HEIGHT = 1800
OUTER_MARGIN_X = 70
OUTER_MARGIN_Y = 55
COLUMN_GAP = 55
ROW_GAP = 45
TITLE_HEIGHT = 74
BORDER_WIDTH = 3

PANEL_WIDTH = (CANVAS_WIDTH - 2 * OUTER_MARGIN_X - COLUMN_GAP) // 2
PANEL_IMAGE_HEIGHT = 736
PANEL_HEIGHT = TITLE_HEIGHT + PANEL_IMAGE_HEIGHT

TITLE_BG = (243, 245, 247)
BORDER_COLOR = (190, 196, 202)
TEXT_COLOR = (35, 39, 43)

POSITIONS = [
    (OUTER_MARGIN_X, OUTER_MARGIN_Y),
    (OUTER_MARGIN_X + PANEL_WIDTH + COLUMN_GAP, OUTER_MARGIN_Y),
    (OUTER_MARGIN_X, OUTER_MARGIN_Y + PANEL_HEIGHT + ROW_GAP),
    (
        OUTER_MARGIN_X + PANEL_WIDTH + COLUMN_GAP,
        OUTER_MARGIN_Y + PANEL_HEIGHT + ROW_GAP,
    ),
]


def find_font() -> str:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError("未找到可用字体，无法绘制中文标题。")


def category_of(sample_key: str) -> str:
    if sample_key.startswith("near_tie_"):
        return "near_tie"
    if sample_key.startswith("sam3_strong_win_"):
        return "sam3_strong_win"
    if sample_key.startswith("unet_strong_win_"):
        return "unet_strong_win"
    return "other"


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.seek(0)
        return image.convert("RGB").copy()


def fit_image(
    image: Image.Image,
    width: int,
    height: int,
    *,
    is_mask: bool,
) -> Image.Image:
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    background = (0, 0, 0) if is_mask else (255, 255, 255)
    fitted = ImageOps.contain(image, (width, height), method=resample)
    panel = Image.new("RGB", (width, height), background)
    x = (width - fitted.width) // 2
    y = (height - fitted.height) // 2
    panel.paste(fitted, (x, y))
    return panel


def draw_panel(
    canvas: Image.Image,
    title: str,
    image_path: Path,
    position: tuple[int, int],
    *,
    is_mask: bool,
    title_font: ImageFont.FreeTypeFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = position

    image = fit_image(
        load_rgb(image_path),
        PANEL_WIDTH,
        PANEL_IMAGE_HEIGHT,
        is_mask=is_mask,
    )

    draw.rectangle(
        [x, y, x + PANEL_WIDTH - 1, y + TITLE_HEIGHT - 1],
        fill=TITLE_BG,
    )

    bbox = draw.textbbox((0, 0), title, font=title_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = x + (PANEL_WIDTH - text_w) // 2
    text_y = y + (TITLE_HEIGHT - text_h) // 2 - bbox[1]
    draw.text((text_x, text_y), title, font=title_font, fill=TEXT_COLOR)

    canvas.paste(image, (x, y + TITLE_HEIGHT))

    draw.rectangle(
        [x, y, x + PANEL_WIDTH - 1, y + PANEL_HEIGHT - 1],
        outline=BORDER_COLOR,
        width=BORDER_WIDTH,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="批量生成 2×2 车辆分割对比图")
    parser.add_argument("source_zip", type=Path, help="包含原图、GT、U-Net、SAM-3 的 ZIP")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("25份车辆分割对比图_2x2"),
        help="输出目录",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path("25份车辆分割对比图_2x2.zip"),
        help="最终压缩包路径",
    )
    args = parser.parse_args()

    source_zip = args.source_zip.resolve()
    output_root = args.output_dir.resolve()
    output_zip = args.output_zip.resolve()
    extract_dir = output_root.parent / f"{output_root.name}_source_tmp"

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    if output_root.exists():
        shutil.rmtree(output_root)
    if output_zip.exists():
        output_zip.unlink()

    extract_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source_zip, "r") as zf:
        zf.extractall(extract_dir)

    all_files = [p for p in extract_dir.rglob("*") if p.is_file()]

    sam3_files = {
        p.stem.removesuffix("_sam3_pred"): p
        for p in all_files
        if p.suffix.lower() == ".png" and p.stem.endswith("_sam3_pred")
    }
    unet_files = {
        p.stem.removesuffix("_unet_pred"): p
        for p in all_files
        if p.suffix.lower() == ".png" and p.stem.endswith("_unet_pred")
    }
    input_files = {
        p.stem: p
        for p in all_files
        if p.suffix.lower() in {".jpg", ".jpeg"}
    }
    gt_files = {
        p.stem.removesuffix("_mask"): p
        for p in all_files
        if p.suffix.lower() == ".gif" and p.stem.endswith("_mask")
    }

    sample_keys = sorted(set(sam3_files) & set(unet_files) & set(input_files) & set(gt_files))
    if not sample_keys:
        raise RuntimeError("没有匹配到完整样本组。")

    title_font = ImageFont.truetype(find_font(), 42)
    manifest_rows = []

    for sample_key in sample_keys:
        category = category_of(sample_key)
        category_dir = output_root / category
        category_dir.mkdir(parents=True, exist_ok=True)

        canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
        panels = [
            ("输入图像", input_files[sample_key], False),
            ("真实标注（GT）", gt_files[sample_key], True),
            ("U-Net 预测", unet_files[sample_key], True),
            ("SAM-3 预测", sam3_files[sample_key], True),
        ]

        for (title, path, is_mask), position in zip(panels, POSITIONS):
            draw_panel(
                canvas,
                title,
                path,
                position,
                is_mask=is_mask,
                title_font=title_font,
            )

        output_name = f"{sample_key}_comparison_2x2.png"
        output_path = category_dir / output_name
        canvas.save(output_path, "PNG", optimize=True)

        manifest_rows.append(
            {
                "sample_key": sample_key,
                "category": category,
                "output_file": str(output_path.relative_to(output_root)),
            }
        )

    manifest_path = output_root / "generated_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_key", "category", "output_file"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    shutil.copy2(Path(__file__), output_root / Path(__file__).name)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(output_root.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(output_root.parent))

    shutil.rmtree(extract_dir)
    print(f"已生成 {len(sample_keys)} 张对比图：{output_zip}")


if __name__ == "__main__":
    main()
