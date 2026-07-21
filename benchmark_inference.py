import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from predict import mask_to_image
from unet import UNet
from utils.data_loading import BasicDataset


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}


def get_args():
    parser = argparse.ArgumentParser(
        description='Benchmark end-to-end U-Net latency on a fixed validation image list'
    )
    parser.add_argument('--model', '-m', type=Path, required=True, help='Path to the trained checkpoint')
    parser.add_argument('--images-dir', type=Path, default=Path('data/imgs'), help='Directory containing input images')
    parser.add_argument(
        '--val-ids',
        type=Path,
        required=True,
        help='Validation IDs as one ID per line, or a CSV containing an image_id column',
    )
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Input image scale factor')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of output classes')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Foreground threshold for a single-class model')
    parser.add_argument('--warmup', type=int, default=25, help='Warmup images before each timed run (20-30)')
    parser.add_argument('--runs', type=int, default=3, help='Number of complete timed runs')
    parser.add_argument('--expected-images', type=int, default=508,
                        help='Required number of unique validation images')
    return parser.parse_args()


def validate_args(args):
    if not args.model.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {args.model}')
    if not args.images_dir.is_dir():
        raise NotADirectoryError(f'Image directory not found: {args.images_dir}')
    if not args.val_ids.is_file():
        raise FileNotFoundError(f'Validation ID file not found: {args.val_ids}')
    if not 0 < args.scale <= 1:
        raise ValueError('--scale must be in the interval (0, 1]')
    if args.classes < 1:
        raise ValueError('--classes must be at least 1')
    if not 20 <= args.warmup <= 30:
        raise ValueError('--warmup must be between 20 and 30 images')
    if args.runs < 1:
        raise ValueError('--runs must be at least 1')
    if args.expected_images < 1:
        raise ValueError('--expected-images must be at least 1')


def load_validation_ids(path: Path):
    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or 'image_id' not in reader.fieldnames:
                raise ValueError(f'CSV must contain an image_id column: {path}')
            image_ids = [row['image_id'].strip() for row in reader if row['image_id'].strip()]
    else:
        with path.open(encoding='utf-8-sig') as file:
            image_ids = [line.strip() for line in file if line.strip()]

    return image_ids


def resolve_validation_images(images_dir: Path, image_ids, expected_images: int):
    if len(image_ids) != expected_images:
        raise ValueError(
            f'Expected {expected_images} validation IDs, but found {len(image_ids)} in the list'
        )
    if len(set(image_ids)) != len(image_ids):
        raise ValueError('Validation ID list contains duplicate IDs')

    image_index = {}
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith('.'):
            image_index.setdefault(path.stem, []).append(path)

    validation_images = []
    for image_id in image_ids:
        matches = image_index.get(image_id, [])
        if len(matches) != 1:
            raise RuntimeError(
                f'Expected exactly one image for ID {image_id!r}, found {len(matches)} in {images_dir}'
            )
        validation_images.append(matches[0])

    return validation_images


def load_rgb_image(path: Path):
    """Read and fully decode an image before the timed region starts."""
    with Image.open(path) as source:
        return source.convert('RGB').copy()


def load_model(args, device):
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model.to(device=device)

    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    model.load_state_dict(state_dict)
    model.eval()
    return model, mask_values


@torch.inference_mode()
def infer_complete_mask(model, full_image, device, scale, mask_values, mask_threshold):
    """Run preprocessing, forward inference, and final mask postprocessing."""
    image = BasicDataset.preprocess(None, full_image, scale, is_mask=False)
    image = torch.from_numpy(image).unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)

    output = model(image).cpu()
    output = F.interpolate(
        output,
        (full_image.size[1], full_image.size[0]),
        mode='bilinear',
    )

    if model.n_classes > 1:
        mask = output.argmax(dim=1)
    else:
        mask = torch.sigmoid(output) > mask_threshold

    mask = mask[0].long().squeeze().numpy()
    return mask_to_image(mask, mask_values)


def warm_up(model, image_paths, device, args, mask_values):
    for image_path in image_paths[:args.warmup]:
        image = load_rgb_image(image_path)
        infer_complete_mask(
            model=model,
            full_image=image,
            device=device,
            scale=args.scale,
            mask_values=mask_values,
            mask_threshold=args.mask_threshold,
        )
    torch.cuda.synchronize(device)


def benchmark_run(model, image_paths, device, args, mask_values):
    elapsed_ns = []

    for image_path in image_paths:
        # Disk reading and image decoding are deliberately outside the timed region.
        image = load_rgb_image(image_path)

        torch.cuda.synchronize(device)
        start_ns = time.perf_counter_ns()

        final_mask = infer_complete_mask(
            model=model,
            full_image=image,
            device=device,
            scale=args.scale,
            mask_values=mask_values,
            mask_threshold=args.mask_threshold,
        )

        torch.cuda.synchronize(device)
        end_ns = time.perf_counter_ns()
        elapsed_ns.append(end_ns - start_ns)

        # Keep the final mask alive through the end timestamp, then release references.
        del final_mask, image

    return statistics.fmean(elapsed_ns) / 1_000_000


def main():
    args = get_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required because this benchmark uses synchronized GPU timing')

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(device)
    image_ids = load_validation_ids(args.val_ids)
    image_paths = resolve_validation_images(args.images_dir, image_ids, args.expected_images)
    model, mask_values = load_model(args, device)

    print(f'GPU: {gpu_name}')
    print(f'Checkpoint: {args.model}')
    print(f'Images: {len(image_paths)}')
    print('Batch size: 1')
    print(f'Scale: {args.scale}')
    print(f'Upsampling: {"bilinear" if args.bilinear else "transposed_conv"}')
    print(f'Warmup: {args.warmup} images before each run')
    print(f'Runs: {args.runs}')

    run_means_ms = []
    for run_index in range(1, args.runs + 1):
        warm_up(model, image_paths, device, args, mask_values)
        mean_ms = benchmark_run(model, image_paths, device, args, mask_values)
        run_means_ms.append(mean_ms)
        print(f'Run {run_index}: {mean_ms:.3f} ms/image, {1000.0 / mean_ms:.3f} FPS')

    final_mean_ms = statistics.fmean(run_means_ms)
    print(f'Mean ({args.runs} runs): {final_mean_ms:.3f} ms/image, {1000.0 / final_mean_ms:.3f} FPS')


if __name__ == '__main__':
    main()
