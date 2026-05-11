import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    # 统一读取不同格式的数据文件。
    # 普通图片会通过 PIL.Image.open 读取；.npy/.pt/.pth 会先转成 numpy 数组再包装成 PIL Image。
    # 后续 preprocess 会统一把 PIL Image 转成 numpy，再转成 torch Tensor。
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


# 这个函数是统计mask中出现的所有像素值
def unique_mask_values(idx, mask_dir, mask_suffix):
    # 找到当前 image id 对应的 mask 文件，并统计 mask 中出现过的像素值。
    # 语义分割中，mask 的像素值代表类别；例如 0 表示背景，255 可能表示目标。
    # 这里先扫描所有可能的原始像素值，后面会把它们映射成连续的类别编号 0, 1, 2...
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        # 单通道 mask：直接统计所有唯一灰度值。
        return np.unique(mask)
    elif mask.ndim == 3:
        # 彩色 mask：把每个像素的 RGB 值当成一个类别标识，再统计唯一颜色。
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    # 图片如果既不是二维，也不是三维，那么就会直接报错。
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')



# 继承了Dataset,负责读取出一对 image与mask
class BasicDataset(Dataset):
    # 这里的scale是resize的缩放比例,images_dir 代表的是image的存放路径,mask_dir代表的是mask的存放路径
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # 从 images_dir 中收集样本 id。id 是去掉扩展名后的文件名。
        # 例如 data/imgs/sample_0.png 会得到 id: sample_0。
        # 后续 __getitem__ 会用这个 id 去 images_dir 和 mask_dir 中分别找 image 与 mask。
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        # 如果images目录下读不到文件，则会直接报错。
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        # 扫描全部 mask，得到数据集中所有可能的 mask 原始像素值。
        # 原版代码使用 multiprocessing.Pool 加速；这里保留串行版本，便于小数据集调试和阅读。
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        # mask_values 保存“原始 mask 像素值 -> 类别编号”的映射基础。
        # 例如 mask_values = [0, 255] 时，后续 preprocess 会把 0 映射为类别 0，把 255 映射为类别 1。
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        # image 和 mask 都会按相同 scale 缩放，确保二者空间尺寸仍然对应。
        # 注意：image 和 mask 的 resize 插值方式不同。
        # - image 使用 BICUBIC，让图像缩放更平滑；
        # - mask 使用 NEAREST，避免类别编号被插值成不存在的中间值。
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            # mask 是监督标签，不是连续图像值。
            # 这里创建 H x W 的整数矩阵，每个位置存放该像素所属的类别编号。
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    # 单通道 mask：原始像素值等于 v 的位置，标记为类别 i。
                    mask[img == v] = i
                else:
                    # 彩色 mask：RGB 三个通道都等于 v 的位置，标记为类别 i。
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            # image 是模型输入，需要转成 PyTorch 常用的 C x H x W。
            # PIL/NumPy 读取彩色图通常是 H x W x C，模型期望通道维在前。
            if img.ndim == 2:
                # 灰度图没有显式通道维，这里补成 1 x H x W。
                img = img[np.newaxis, ...]
            else:
                # 彩色图从 H x W x C 转成 C x H x W。
                img = img.transpose((2, 0, 1))

            # 如果像素值仍在 0-255 范围，就归一化到 0-1。
            # mask 不能这样处理，因为 mask 的值是类别编号，不是图像强度。
            if (img > 1).any():
                img = img / 255.0

            return img


    
    def __getitem__(self, idx):
        # DataLoader 每次取样本时会调用这里。
        # idx -> id -> 找到 image 文件和 mask 文件 -> 读取 -> 预处理 -> 返回 Tensor。
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # image 和 mask 必须逐像素对应，所以二者原始尺寸必须一致。
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        # 返回给 DataLoader 的单个样本：
        # - image: float Tensor，形状通常是 C x H x W，作为模型输入；
        # - mask: long Tensor，形状通常是 H x W，作为每个像素的类别标签。
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        # Carvana 原始数据的 mask 文件名通常带 _mask 后缀。
        # 例如 image id 是 0001，mask 文件可能是 0001_mask.gif。
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
