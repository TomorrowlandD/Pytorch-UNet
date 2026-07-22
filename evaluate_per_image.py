"""Evaluate a trained U-Net checkpoint on a fixed list of validation images."""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from unet import UNet
from utils.data_loading import BasicDataset, load_image
from utils.dice_score import (
    dice_coeff,
    iou_score,
    multiclass_dice_coeff,
    multiclass_iou_score,
)
from utils.model_loading import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a U-Net checkpoint per image on a fixed validation set'
    )
    parser.add_argument('--model', required=True, help='Checkpoint to evaluate')
    parser.add_argument('--images-dir', default='data/imgs')
    parser.add_argument('--masks-dir', default='data/masks')
    parser.add_argument(
        '--val-ids',
        required=True,
        help='Text file with one image ID per line, or a CSV containing image_id',
    )
    parser.add_argument('--output', required=True, help='Per-image metrics CSV')
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--bilinear', action='store_true', default=False)
    parser.add_argument('--attention', choices=['none', 'lite_sr_mhsa'], default='none')
    parser.add_argument('--attention-dim', type=int, default=128)
    parser.add_argument('--attention-heads', type=int, default=4)
    parser.add_argument('--attention-sr-ratio', type=int, default=2)
    parser.add_argument('--attention-max-scale', type=float, default=1e-2)
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    parser.add_argument('--device', default=None, help='For example cuda, cpu, or cuda:0')
    return parser.parse_args()


def _normalise_image_id(value):
    image_id = str(value).strip()
    if not image_id:
        raise ValueError('Validation image IDs must not be empty')
    return Path(image_id).stem


def read_validation_ids(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Validation ID file not found: {path}')

    if path.suffix.lower() == '.csv':
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or 'image_id' not in reader.fieldnames:
                raise ValueError(f'CSV must contain an image_id column: {path}')
            image_ids = [_normalise_image_id(row['image_id']) for row in reader]
    else:
        with path.open('r', encoding='utf-8-sig') as handle:
            image_ids = [
                _normalise_image_id(line)
                for line in handle
                if line.strip()
            ]

    if not image_ids:
        raise ValueError(f'No validation image IDs found in {path}')
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f'Duplicate validation image IDs found in {path}')
    return image_ids


def _find_single_file(directory, pattern, kind, image_id):
    matches = list(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f'Expected exactly one {kind} for {image_id}, found {len(matches)}: {matches}'
        )
    return matches[0]


def load_sample(image_id, images_dir, masks_dir, mask_values, scale):
    image_path = _find_single_file(images_dir, f'{image_id}.*', 'image', image_id)
    mask_path = _find_single_file(masks_dir, f'{image_id}_mask.*', 'mask', image_id)

    image = load_image(image_path)
    mask = load_image(mask_path)
    if image.size != mask.size:
        raise ValueError(
            f'Image and mask sizes differ for {image_id}: {image.size} versus {mask.size}'
        )

    image_array = BasicDataset.preprocess(
        mask_values,
        image,
        scale=scale,
        is_mask=False,
    )
    # The reference U-Net/SAM-3 comparison scores predictions at original
    # image resolution, so retain the unscaled ground-truth mask here.
    mask_array = BasicDataset.preprocess(
        mask_values,
        mask,
        scale=1.0,
        is_mask=True,
    )
    return (
        torch.as_tensor(image_array.copy()).float().unsqueeze(0),
        torch.as_tensor(mask_array.copy()).long().unsqueeze(0),
    )


def per_image_scores(logits, true_mask, n_classes, mask_threshold):
    logits = F.interpolate(
        logits,
        size=true_mask.shape[-2:],
        mode='bilinear',
        align_corners=False,
    )

    if n_classes == 1:
        predicted = (torch.sigmoid(logits).squeeze(1) > mask_threshold).float()
        target = true_mask.float()
        dice = dice_coeff(predicted, target, reduce_batch_first=False)
        iou = iou_score(predicted, target, reduce_batch_first=False)
    else:
        predicted = F.one_hot(
            logits.argmax(dim=1),
            n_classes,
        ).permute(0, 3, 1, 2).float()
        target = F.one_hot(
            true_mask,
            n_classes,
        ).permute(0, 3, 1, 2).float()
        dice = multiclass_dice_coeff(
            predicted[:, 1:],
            target[:, 1:],
            reduce_batch_first=False,
        )
        iou = multiclass_iou_score(
            predicted[:, 1:],
            target[:, 1:],
            reduce_batch_first=False,
        )

    return float(dice.item()), float(iou.item())


@torch.inference_mode()
def evaluate_checkpoint(model, image_ids, images_dir, masks_dir, mask_values, scale,
                        mask_threshold, device):
    model.eval()
    rows = []
    for image_id in tqdm(image_ids, desc='Per-image evaluation', unit='image'):
        image, true_mask = load_sample(
            image_id,
            images_dir,
            masks_dir,
            mask_values,
            scale,
        )
        image = image.to(device=device, dtype=torch.float32)
        true_mask = true_mask.to(device=device, dtype=torch.long)
        logits = model(image)
        dice, iou = per_image_scores(
            logits,
            true_mask,
            model.n_classes,
            mask_threshold,
        )
        rows.append({'image_id': image_id, 'dice': dice, 'iou': iou})
    return rows


def write_metrics(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image_id', 'dice', 'iou'])
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows):
    dice = np.asarray([row['dice'] for row in rows], dtype=np.float64)
    iou = np.asarray([row['iou'] for row in rows], dtype=np.float64)
    return {
        'images': len(rows),
        'mean_dice': float(dice.mean()),
        'mean_iou': float(iou.mean()),
        'min_dice': float(dice.min()),
        'min_iou': float(iou.min()),
        'max_dice': float(dice.max()),
        'max_iou': float(iou.max()),
    }


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not 0 < args.scale <= 1:
        raise ValueError('--scale must be in (0, 1]')
    if args.classes < 1:
        raise ValueError('--classes must be positive')

    image_ids = read_validation_ids(args.val_ids)
    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model = UNet(
        n_channels=3,
        n_classes=args.classes,
        bilinear=args.bilinear,
        attention=args.attention,
        attention_dim=args.attention_dim,
        attention_heads=args.attention_heads,
        attention_sr_ratio=args.attention_sr_ratio,
        attention_max_scale=args.attention_max_scale,
    ).to(device=device)
    mask_values = load_checkpoint(
        model,
        args.model,
        map_location=device,
        load_mode='strict',
    )
    logging.info('Using device: %s', device)
    logging.info('Loaded %d fixed validation image IDs', len(image_ids))

    rows = evaluate_checkpoint(
        model=model,
        image_ids=image_ids,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        mask_values=mask_values,
        scale=args.scale,
        mask_threshold=args.mask_threshold,
        device=device,
    )
    write_metrics(rows, args.output)
    summary = summarise(rows)

    logging.info('Per-image metrics saved to %s', args.output)
    print(f"images: {summary['images']}")
    print(f"mean Dice: {summary['mean_dice']:.10f}")
    print(f"mean IoU: {summary['mean_iou']:.10f}")
    print(f"min Dice: {summary['min_dice']:.10f}")
    print(f"min IoU: {summary['min_iou']:.10f}")
    print(f"max Dice: {summary['max_dice']:.10f}")
    print(f"max IoU: {summary['max_iou']:.10f}")


if __name__ == '__main__':
    main()
