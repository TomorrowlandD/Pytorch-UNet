import torch
from torch import Tensor


# 计算单类 / 二分类 Dice 系数
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # input 和 target 的 shape 必须一致，才能逐像素比较
    assert input.size() == target.size()

    # 如果要把 batch 维度一起计算，则 input 应该是 [N, H, W]
    assert input.dim() == 3 or not reduce_batch_first

    # 决定在哪些维度上求和
    # 对 [H, W]：在 H、W 上求和
    # 对 [N, H, W]：
    #   reduce_batch_first=False 时，每张图单独算
    #   reduce_batch_first=True 时，整个 batch 一起算
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # 计算 Dice 分子：2 × 交集
    # input * target 只有预测和真实都为 1 的位置才为 1
    inter = 2 * (input * target).sum(dim=sum_dim)

    # 计算 Dice 分母：预测区域大小 + 真实区域大小
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)

    # 处理预测和真实都为空的特殊情况，避免除以 0
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    # Dice = (2 × 交集) / (预测区域 + 真实区域)
    # epsilon 用于数值稳定，防止除零
    dice = (inter + epsilon) / (sets_sum + epsilon)

    # 如果得到多个 Dice，则取平均
    return dice.mean()


# 计算多分类 Dice 系数
def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # 多分类输入通常是 [N, C, H, W]
    # flatten(0, 1) 将 N 和 C 合并成 N*C
    # 即把每张图的每个类别 mask 都当成一个单独的二值 mask 来计算 Dice
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)



def iou_score(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    intersection = (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean()


def multiclass_iou_score(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    return iou_score(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


# 计算 Dice Loss
def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # 如果是多分类，使用 multiclass_dice_coeff
    # 如果是二分类 / 单类，使用 dice_coeff
    fn = multiclass_dice_coeff if multiclass else dice_coeff

    # Dice 越大越好，但 loss 需要越小越好
    # 所以 Dice Loss = 1 - Dice
    return 1 - fn(input, target, reduce_batch_first=True)
