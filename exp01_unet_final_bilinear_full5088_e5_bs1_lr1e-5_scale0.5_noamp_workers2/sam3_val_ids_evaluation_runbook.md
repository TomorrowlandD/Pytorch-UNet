# SAM3 固定 U-Net 验证集评估流程记录

本文记录上次已经跑通的 SAM3 smoke test，以及后续用于读取 U-Net `val_ids.txt`、评估同一批验证集图像的操作流程。

## 1. 已确认状态

SAM3 smoke test 已经成功：

```text
valid images: 1
mean Dice: 0.9924821948
mean IoU: 0.9850765810
saved metrics: outputs/smoke_test_val/metrics.csv
```

`pkg_resources`、`timm` 相关输出只是 warning，不是报错。

当时剩余问题是：原有脚本只能跑前 N 张或随机 N 张，还不能直接读取 U-Net 的 `val_ids.txt`。因此需要新增一个专门脚本，让 SAM3 跑 U-Net 同一批验证图像。

## 2. 进入 SAM3 项目

```bash
cd /root/autodl-tmp/projects/sam3
```

## 3. 设置路径变量

作用：告诉 SAM3 去哪里读 U-Net 的验证集、原图、mask，以及结果保存到哪里。

```bash
export UNET_PROJECT=/root/autodl-tmp/projects/Pytorch-UNet
export EXP_NAME=exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2
export VAL_IDS=${UNET_PROJECT}/results/${EXP_NAME}/val_ids.txt
export IMG_DIR=${UNET_PROJECT}/data/imgs
export MASK_DIR=${UNET_PROJECT}/data/masks
export SAM3_OUT=${UNET_PROJECT}/results/${EXP_NAME}/sam3_val_metrics.csv
```

检查：

```bash
wc -l ${VAL_IDS}
ls -lh checkpoint/sam3.pt
```

## 4. 新建 SAM3 固定验证集评估脚本

作用：读取 `val_ids.txt`，只跑 U-Net 那批固定验证图像。

```bash
cat > eval_sam3_carvana_val_ids.py <<'PY'
import argparse
import csv
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def calc_dice_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    dice = 2 * inter / (pred_sum + gt_sum + 1e-8)
    iou = inter / (union + 1e-8)
    return dice, iou, pred_sum, gt_sum, inter, union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--ckpt", default="checkpoint/sam3.pt")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--prompt", default="car")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    ids_file = Path(args.ids_file)
    ckpt_path = Path(args.ckpt)
    out_csv = Path(args.out_csv)

    assert img_dir.exists(), f"img_dir not found: {img_dir}"
    assert mask_dir.exists(), f"mask_dir not found: {mask_dir}"
    assert ids_file.exists(), f"ids_file not found: {ids_file}"
    assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"

    with ids_file.open("r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    print("image count:", len(image_ids))
    print("prompt:", args.prompt)
    print("threshold:", args.threshold)
    print("out csv:", out_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = build_sam3_image_model(
        checkpoint_path=str(ckpt_path),
        load_from_HF=False,
        device=device,
        eval_mode=True,
    )

    processor = Sam3Processor(
        model,
        confidence_threshold=args.threshold,
        device=device,
    )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )

    rows = []

    for image_id in tqdm(image_ids):
        img_files = list(img_dir.glob(image_id + ".*"))
        gt_path = mask_dir / f"{image_id}_mask.gif"

        if len(img_files) != 1:
            raise RuntimeError(f"bad image match for {image_id}: {img_files}")
        if not gt_path.exists():
            raise RuntimeError(f"GT not found: {gt_path}")

        image = Image.open(img_files[0]).convert("RGB")
        gt_img = Image.open(gt_path).convert("L")
        gt = np.array(gt_img) > 0

        with torch.inference_mode(), autocast_ctx:
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=args.prompt)

        masks = output["masks"]
        scores = output["scores"]

        if len(masks) == 0:
            pred = np.zeros(gt.shape, dtype=bool)
            score = 0.0
        else:
            best_idx = int(torch.argmax(scores).item())
            score = float(scores[best_idx].detach().float().cpu().item())
            pred_arr = masks[best_idx, 0].detach().float().cpu().numpy()
            pred_img = Image.fromarray((pred_arr > 0).astype(np.uint8) * 255)

            if pred_img.size != gt_img.size:
                pred_img = pred_img.resize(gt_img.size, Image.NEAREST)

            pred = np.array(pred_img) > 0

        dice, iou, pred_sum, gt_sum, inter, union = calc_dice_iou(pred, gt)

        rows.append({
            "image_id": image_id,
            "sam3_score": score,
            "sam3_dice": dice,
            "sam3_iou": iou,
            "pred_pixels": int(pred_sum),
            "gt_pixels": int(gt_sum),
            "intersection": int(inter),
            "union": int(union),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id", "sam3_score", "sam3_dice", "sam3_iou",
                "pred_pixels", "gt_pixels", "intersection", "union"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    dices = np.array([r["sam3_dice"] for r in rows])
    ious = np.array([r["sam3_iou"] for r in rows])
    scores = np.array([r["sam3_score"] for r in rows])

    print("\n===== Summary =====")
    print("valid images:", len(rows))
    print("mean score:", scores.mean())
    print("mean Dice:", dices.mean())
    print("mean IoU:", ious.mean())
    print("min Dice:", dices.min())
    print("min IoU:", ious.min())
    print("max Dice:", dices.max())
    print("max IoU:", ious.max())
    print("saved metrics:", out_csv)


if __name__ == "__main__":
    main()
PY
```

## 5. 跑 SAM3 的固定验证集

作用：正式生成 SAM3 在 U-Net 同一验证集上的结果。

```bash
python eval_sam3_carvana_val_ids.py \
  --img-dir ${IMG_DIR} \
  --mask-dir ${MASK_DIR} \
  --ids-file ${VAL_IDS} \
  --ckpt checkpoint/sam3.pt \
  --prompt car \
  --threshold 0.3 \
  --out-csv ${SAM3_OUT}
```

## 6. 检查结果

```bash
ls -lh ${SAM3_OUT}
head ${SAM3_OUT}
tail ${SAM3_OUT}
```

跑完后得到两张核心表：

```text
U-Net:
${UNET_PROJECT}/results/${EXP_NAME}/unet_epoch4_val_metrics.csv

SAM3:
${UNET_PROJECT}/results/${EXP_NAME}/sam3_val_metrics.csv
```

下一步是按 `image_id` 合并两张表，得到最终对比表。
