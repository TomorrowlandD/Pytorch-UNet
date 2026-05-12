from PIL import Image, ImageDraw
from pathlib import Path
import random

img_dir = Path("data/imgs")
mask_dir = Path("data/masks")
img_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)

W, H = 256, 256

for i in range(20):
    img = Image.new("RGB", (W, H), (30, 30, 30))
    mask = Image.new("L", (W, H), 0)

    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    x1 = random.randint(40, 90)
    y1 = random.randint(40, 90)
    x2 = random.randint(150, 220)
    y2 = random.randint(150, 220)

    draw_img.ellipse((x1, y1, x2, y2), fill=(200, 200, 200))
    draw_mask.ellipse((x1, y1, x2, y2), fill=255)

    img.save(img_dir / f"sample_{i}.png")
    mask.save(mask_dir / f"sample_{i}.png")

print("Toy dataset created.")