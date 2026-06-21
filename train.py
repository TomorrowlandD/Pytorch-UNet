import argparse
import csv
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

dir_img = Path('./data/imgs/')      # 原始输入图片目录，Dataset 会从这里读取 image。
dir_mask = Path('./data/masks/')    # 标注 mask 目录，Dataset 会从这里读取每张 image 对应的监督标签。
dir_checkpoint = Path('./checkpoints/')

LOSS_ALIASES = {
    'cross_entropy': 'ce',
    'cross_entropy+dice': 'ce-dice',
    'bce+dice': 'bce-dice',
}


def tensor_to_float(value):
    return value.item() if hasattr(value, 'item') else float(value)


def init_metrics_file(results_dir: Path, exp_name: str) -> Path:
    exp_dir = results_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = exp_dir / 'metrics.csv'
    if not metrics_file.exists():
        with metrics_file.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_dice', 'val_iou'])
    return metrics_file


def append_metrics_row(metrics_file: Path, epoch: int, train_loss: float, val_dice, val_iou):
    with metrics_file.open('a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, tensor_to_float(val_dice), tensor_to_float(val_iou)])


def resolve_loss_name(loss_name: str, n_classes: int) -> str:
    loss_name = LOSS_ALIASES.get(loss_name, loss_name)
    if loss_name == 'auto':
        return 'ce-dice' if n_classes > 1 else 'bce-dice'
    if n_classes == 1 and loss_name in {'ce', 'ce-dice'}:
        raise ValueError('CE loss requires --classes > 1. Use --loss bce, dice, bce-dice, or auto.')
    if n_classes > 1 and loss_name in {'bce', 'bce-dice'}:
        raise ValueError('BCE loss requires --classes 1. Use --loss ce, dice, ce-dice, or auto.')
    return loss_name


def compute_segmentation_loss(masks_pred, true_masks, n_classes: int, loss_name: str, criterion):
    if n_classes == 1:
        logits = masks_pred.squeeze(1)
        target = true_masks.float()
        dice = dice_loss(F.sigmoid(logits), target, multiclass=False)
        if loss_name == 'dice':
            return dice
        bce = criterion(logits, target)
        if loss_name == 'bce':
            return bce
        return bce + dice

    target_one_hot = F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float()
    dice = dice_loss(F.softmax(masks_pred, dim=1).float(), target_one_hot, multiclass=True)
    if loss_name == 'dice':
        return dice
    ce = criterion(masks_pred, true_masks)
    if loss_name == 'ce':
        return ce
    return ce + dice


# 定义训练参数
def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        num_workers: int = 4,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        loss_name: str = 'auto',
        exp_name: str = 'default',
        results_dir: Path = Path('./results'),
):
    loss_name = resolve_loss_name(loss_name, model.n_classes)

    # 1. 创建 Dataset：负责把磁盘上的 image/mask 文件读取并预处理成 Tensor。
    try:
        # Carvana 数据集的 mask 文件名通常带 _mask 后缀。
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        # 当前项目的小样本数据中 image 和 mask 文件名一致，因此回退到 BasicDataset。
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. 将完整 Dataset 划分为训练集和验证集。
    # val_set是验证集,train_set是训练集
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. 创建 DataLoader：负责反复调用 Dataset.__getitem__，并把多个样本组成 batch。
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    metrics_file = init_metrics_file(results_dir, exp_name)

    # 记录训练配置和训练过程
    # (Initialize logging)
    # wandb 和 logging 用来记录实验，不负责训练模型。
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp,
             num_workers=num_workers, weight_decay=weight_decay, momentum=momentum,
             gradient_clipping=gradient_clipping, loss=loss_name,
             exp_name=exp_name, results_dir=str(results_dir))
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Data workers:    {num_workers}
        Loss:            {loss_name}
    ''')



    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    # 优化器=>根据梯度更新模型参数
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # 学习率调度器=>根据验证 Dice 调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # 自动混合精度缩放器 (GradScaler)=>AMP 混合精度训练时使用
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # 如果多分类，使用 CrossEntropyLoss；如果是 classes=1 的二分类，使用 BCEWithLogitsLoss。
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    # 记录训练了多少个 batch
    global_step = 0


    # 5. Begin training
    # 进入epochs轮训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        last_val_metrics = None
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            # 读取数据
            for batch in train_loader:
                # batch 来自 DataLoader，包含两部分：
                # - image: 模型输入，形状通常是 N x C x H x W；
                # - mask: 像素级标准答案，形状通常是 N x H x W。
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # TODO image 是连续输入值，使用 float32；mask 是类别编号，使用 long。
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)


                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # 只把 image 送入模型；true_masks 不进入模型，它用于和预测结果计算监督信号。
                    # 预测的masks结果
                    masks_pred = model(images)
                    loss = compute_segmentation_loss(
                        masks_pred,
                        true_masks,
                        model.n_classes,
                        loss_name,
                        criterion,
                    )


                # 清空旧的梯度()
                optimizer.zero_grad(set_to_none=True)
                # 反向传播、计算梯度
                grad_scaler.scale(loss).backward()
                # AMP 下反缩放梯度
                grad_scaler.unscale_(optimizer)
                # 梯度裁剪、防止梯度过大
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                # 更新参数
                grad_scaler.step(optimizer)
                # 更新AMP的scaler状态
                grad_scaler.update()

                # 更新进度条
                pbar.update(images.shape[0])
                # 训练步数加 1
                global_step += 1
                # 累加 loss
                epoch_loss += loss.item()
                # 记录到 W&B
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                # 在进度条显示当前 loss
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                # division_step大致表示把一个 epoch 分成 5 段，每段验证一次。
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # 把权重和梯度记录到 W&B。
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        last_val_metrics = evaluate(model, val_loader, device, amp)
                        val_score = last_val_metrics['dice']
                        val_iou = last_val_metrics['iou']
                        # 根据验证 Dice 调整学习率。
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        logging.info('Validation IoU score: {}'.format(val_iou))
                        try:
                            # 把验证结果记录到 W&B，包括：学习率;验证 Dice;输入图片;真实 mask;预测 mask;权重和梯度直方图。
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'validation IoU': val_iou,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass


        # 保存checkpoint
        # 每个 epoch 结束后保存模型参数。
        # checkpoint 保存的是模型训练成果,用于后续:加载模型继续训练;用训练好的模型做预测;保留不同 epoch 的模型参数
        if last_val_metrics is None:
            last_val_metrics = evaluate(model, val_loader, device, amp)
            val_score = last_val_metrics['dice']
            val_iou = last_val_metrics['iou']
            scheduler.step(val_score)
            logging.info('Validation Dice score: {}'.format(val_score))
            logging.info('Validation IoU score: {}'.format(val_iou))

        avg_epoch_loss = epoch_loss / max(len(train_loader), 1)
        append_metrics_row(metrics_file, epoch, avg_epoch_loss, last_val_metrics['dice'], last_val_metrics['iou'])
        logging.info(f'Metrics saved to {metrics_file}')

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True) #创建 checkpoint 文件夹
            state_dict = model.state_dict()         # 取出模型参数
            # 保存 mask_values，预测阶段需要用它把类别编号还原成原始 mask 像素值。
            state_dict['mask_values'] = dataset.mask_values     # 保存 mask 类别值信息
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


# 让别人可以在命令行中设置训练参数。
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader worker processes')
    parser.add_argument('--exp-name', type=str, default='default', help='Experiment name for results/<exp-name>/metrics.csv')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory for per-experiment result files')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--loss', type=str, default='auto',
                        choices=['auto', 'ce', 'cross_entropy', 'dice', 'ce-dice', 'cross_entropy+dice',
                                 'bce', 'bce-dice', 'bce+dice'],
                        help='Loss function: auto keeps the old default, ce-dice for classes>1 and bce-dice for classes=1')

    return parser.parse_args()

# 可以做:读取命令行参数;选择 CPU/GPU;创建 U-Net 模型;可选：加载已有模型;调用 train_model() 开始训练
if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    # 如果训练过程中 GPU 显存不够，程序会：1. 清空 CUDA 缓存；2. 开启模型 checkpointing；3. 重新调用 `train_model()`。
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            num_workers=args.num_workers,
            loss_name=args.loss,
            exp_name=args.exp_name,
            results_dir=Path(args.results_dir)
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            num_workers=args.num_workers,
            loss_name=args.loss,
            exp_name=args.exp_name,
            results_dir=Path(args.results_dir)
        )
