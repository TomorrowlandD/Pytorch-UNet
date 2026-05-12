import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff

# 验证 / 推理阶段不需要计算梯度(也就是关闭梯度计算),与with no_grad()代码块差不多意思
@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    # 将模型切换到评估模式(验证集模式)
    net.eval()
    # 验证集一共有多少个 batch
    num_val_batches = len(dataloader)
    # 用于累加每个 batch 的 Dice 分数
    dice_score = 0
    # AMP 混合精度环境，属于性能优化,torch.autocast 是 AMP 混合精度相关，不影响验证主线。
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):

        # 遍历验证集中的每个 batch
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

            # image 是输入图片，mask_true 是真实 mask
            image, mask_true = batch['image'], batch['mask']

            # 图片转为 float32 并移动到指定设备
            # channels_last 是性能优化，不改变逻辑 shape
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            # mask_true 是类别编号图，所以使用 long 类型
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # 前向传播，得到模型输出 logits
            mask_pred = net(image)

            # 二分类分割
            if net.n_classes == 1:
                # 检查真实 mask 是否只包含 0 和 1
                assert mask_true.min() >= 0 and mask_true.max() <= 1, \
                    'True mask indices should be in [0, 1]'

                # logits → sigmoid 概率 → 阈值化二值 mask
                # 就是把预测的分数先转化为概率,然后设置一个阈值,高于这个阈值的概率,答案就是1.0,否则就是0.0
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

                # 计算二分类 Dice
                # 注意：这里要求 mask_pred 和 mask_true 的 shape 一致
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

            # 多分类分割
            else:
                # 检查真实 mask 中类别编号是否在合法范围内
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, \
                    'True mask indices should be in [0, n_classes['

                # 真实 mask: [N, H, W]
                # one_hot 后: [N, H, W, C]
                # permute 后: [N, C, H, W]
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

                # 模型输出 logits: [N, C, H, W]
                # argmax 后得到类别编号图: [N, H, W]
                # one_hot + permute 后: [N, C, H, W]
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                # 计算多分类 Dice，忽略第 0 类背景,因为背景通常是维度类的第一个
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:],
                    mask_true[:, 1:],
                    reduce_batch_first=False
                )

    # 验证结束后，把模型切回训练模式
    net.train()

    # 返回验证集平均 Dice
    return dice_score / max(num_val_batches, 1)
