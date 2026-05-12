import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from predict import predict_img
from unet import UNet


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}


def get_args():
    parser = argparse.ArgumentParser(description='Save input, ground-truth, prediction, and overlay images')
    parser.add_argument('--model', '-m', required=True, metavar='FILE', help='Path to a trained checkpoint')
    parser.add_argument('--input', '-i', nargs='*', default=None,
                        help='Input image files or one image directory. Defaults to data/imgs')
    parser.add_argument('--mask-dir', type=str, default='data/masks', help='Directory containing ground-truth masks')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Directory where visualization images will be saved')
    parser.add_argument('--exp-name', type=str, default='default',
                        help='Experiment name used when --output-dir is not set')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Scale factor for model input')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of output classes')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value for foreground in single-class models')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of images to visualize')
    return parser.parse_args()


def list_images(path: Path):
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith('.')
    )


def collect_input_files(input_args, limit: int):
    if not input_args:
        files = list_images(Path('data/imgs'))
    elif len(input_args) == 1 and Path(input_args[0]).is_dir():
        files = list_images(Path(input_args[0]))
    else:
        files = [Path(p) for p in input_args]

    if limit > 0:
        files = files[:limit]
    if not files:
        raise RuntimeError('No input images found')
    return files


def find_mask_file(image_file: Path, mask_dir: Path) -> Path:
    candidates = list(mask_dir.glob(image_file.stem + '.*'))
    if not candidates:
        candidates = list(mask_dir.glob(image_file.stem + '_mask.*'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one mask for {image_file.name}, found {len(candidates)}')
    return candidates[0]


def mask_to_uint8(mask):
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    return (array > 0).astype(np.uint8) * 255


def save_overlay(image: Image.Image, pred_mask: np.ndarray, output_file: Path):
    base = image.convert('RGB')
    overlay = Image.new('RGBA', base.size, (255, 0, 0, 0))
    alpha = (pred_mask > 0).astype(np.uint8) * 120
    overlay.putalpha(Image.fromarray(alpha))
    result = Image.alpha_composite(base.convert('RGBA'), overlay)
    result.convert('RGB').save(output_file)


def load_model(args, device):
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    state_dict.pop('mask_values', None)
    net.load_state_dict(state_dict)
    return net


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir) if args.output_dir else Path('results') / args.exp_name / 'predictions'
    output_dir.mkdir(parents=True, exist_ok=True)

    net = load_model(args, device)
    input_files = collect_input_files(args.input, args.limit)
    mask_dir = Path(args.mask_dir)

    logging.info(f'Using device {device}')
    logging.info(f'Saving visualizations to {output_dir}')

    for image_file in input_files:
        mask_file = find_mask_file(image_file, mask_dir)
        image = Image.open(image_file).convert('RGB')
        gt_mask = mask_to_uint8(Image.open(mask_file))
        pred_mask = predict_img(
            net=net,
            full_img=image,
            scale_factor=args.scale,
            out_threshold=args.mask_threshold,
            device=device,
        )
        pred_mask = mask_to_uint8(pred_mask)

        stem = image_file.stem
        image.save(output_dir / f'{stem}_input.png')
        Image.fromarray(gt_mask).save(output_dir / f'{stem}_gt.png')
        Image.fromarray(pred_mask).save(output_dir / f'{stem}_pred.png')
        save_overlay(image, pred_mask, output_dir / f'{stem}_overlay.png')
        logging.info(f'Saved visualization group for {image_file.name}')


if __name__ == '__main__':
    main()
