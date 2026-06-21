import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def collect_files(directory: Path):
    files = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith('.'):
            files[path.stem] = path
    return files


def binarize_mask(mask_path: Path, threshold: int):
    mask = Image.open(mask_path).convert('L')
    mask_array = np.asarray(mask)
    binary = (mask_array > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary, mode='L')


def resize_pair(image: Image.Image, mask: Image.Image, size: int | None):
    if size is None:
        return image, mask

    target_size = (size, size)
    image = image.resize(target_size, resample=Image.BICUBIC)
    mask = mask.resize(target_size, resample=Image.NEAREST)
    return image, mask


def prepare_kvasir(
        src_images: Path,
        src_masks: Path,
        out_images: Path,
        out_masks: Path,
        threshold: int,
        limit: int,
        size: int | None,
):
    image_files = collect_files(src_images)
    mask_files = collect_files(src_masks)
    shared_ids = sorted(set(image_files) & set(mask_files))

    if limit is not None:
        shared_ids = shared_ids[:limit]

    if not shared_ids:
        raise RuntimeError(f'No matched image/mask pairs found in {src_images} and {src_masks}')

    missing_masks = sorted(set(image_files) - set(mask_files))
    missing_images = sorted(set(mask_files) - set(image_files))
    if missing_masks:
        print(f'Warning: {len(missing_masks)} images have no matching mask. Example: {missing_masks[:3]}')
    if missing_images:
        print(f'Warning: {len(missing_images)} masks have no matching image. Example: {missing_images[:3]}')

    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    for sample_id in tqdm(shared_ids, desc='Preparing Kvasir-SEG pairs', unit='pair'):
        image = Image.open(image_files[sample_id]).convert('RGB')
        mask = binarize_mask(mask_files[sample_id], threshold)

        if image.size != mask.size:
            raise RuntimeError(
                f'Image and mask size mismatch for {sample_id}: image={image.size}, mask={mask.size}'
            )

        image, mask = resize_pair(image, mask, size)

        image.save(out_images / f'{sample_id}.png')
        mask.save(out_masks / f'{sample_id}.png')

    print(f'Prepared {len(shared_ids)} image/mask pairs')
    print(f'Images: {out_images}')
    print(f'Masks:  {out_masks}')


def get_args():
    parser = argparse.ArgumentParser(description='Prepare Kvasir-SEG for this U-Net project')
    parser.add_argument('--src-images', type=Path, required=True, help='Directory containing raw Kvasir images')
    parser.add_argument('--src-masks', type=Path, required=True, help='Directory containing raw Kvasir masks')
    parser.add_argument('--out-images', type=Path, default=Path('data/kvasir/imgs'),
                        help='Output directory for prepared images')
    parser.add_argument('--out-masks', type=Path, default=Path('data/kvasir/masks'),
                        help='Output directory for prepared binary masks')
    parser.add_argument('--threshold', type=int, default=127,
                        help='Mask binarization threshold: pixels above it become foreground')
    parser.add_argument('--size', type=int, default=None,
                        help='Optional fixed square output size, e.g. 384 writes 384x384 images and masks')
    parser.add_argument('--limit', type=int, default=None,
                        help='Optional maximum number of matched pairs to prepare for smoke tests')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    prepare_kvasir(
        src_images=args.src_images,
        src_masks=args.src_masks,
        out_images=args.out_images,
        out_masks=args.out_masks,
        threshold=args.threshold,
        limit=args.limit,
        size=args.size,
    )
