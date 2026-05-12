from pathlib import Path
import shutil

src_img_dir = Path("data_full/imgs")
src_mask_dir = Path("data_full/masks")
dst_img_dir = Path("data/imgs")
dst_mask_dir = Path("data/masks")

dst_img_dir.mkdir(parents=True, exist_ok=True)
dst_mask_dir.mkdir(parents=True, exist_ok=True)

imgs = sorted([p for p in src_img_dir.iterdir() if p.is_file()])[:50]

count = 0
for img_path in imgs:
    stem = img_path.stem
    candidates = list(src_mask_dir.glob(stem + "_mask.*"))

    if not candidates:
        print(f"未找到对应 mask: {img_path.name}")
        continue

    mask_path = candidates[0]

    shutil.copy2(img_path, dst_img_dir / img_path.name)
    shutil.copy2(mask_path, dst_mask_dir / mask_path.name)

    count += 1

print(f"已复制 {count} 对 image/mask")
