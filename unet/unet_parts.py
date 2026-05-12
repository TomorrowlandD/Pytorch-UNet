""" Parts of the U-Net model.

这个文件负责定义 U-Net 中反复使用的基础模块。
unet_model.py 会导入这些模块，再把它们组装成完整的 UNet。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2.

    DoubleConv 是 U-Net 中最基础的特征提取模块。
    它连续做两次:
        Conv2d -> BatchNorm2d -> ReLU

    直觉理解:
    - Conv2d 负责从局部区域提取特征；
    - BatchNorm2d 让中间特征更稳定；
    - ReLU 引入非线性，让网络能表达更复杂的模式。
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        # in_channels: 输入特征图的通道数。
        # out_channels: 输出特征图的通道数。
        # mid_channels: 中间通道数；如果不指定，就默认等于 out_channels。
        if not mid_channels:
            mid_channels = out_channels

        # nn.Sequential 表示这些层会按顺序执行。
        # 假设输入形状是 N x in_channels x H x W，
        # 第一层卷积后大致变成 N x mid_channels x H x W，
        # 第二层卷积后大致变成 N x out_channels x H x W。
        #
        # kernel_size=3 表示使用 3x3 卷积核。
        # padding=1 表示在特征图边缘补 1 圈像素。
        # 对 3x3 卷积来说，padding=1 通常可以让 H 和 W 保持不变。
        # bias=False 是因为后面接了 BatchNorm2d，通常不再需要卷积层自己的 bias。
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 把输入 x 依次送入上面定义的两组 Conv-BN-ReLU。
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv.
    Down 是 U-Net 编码器中的一个下采样模块。
    它先缩小特征图的空间尺寸，再用 DoubleConv 提取特征。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # MaxPool2d(2) 会把 H 和 W 大致缩小为原来的一半。
        # 例如: N x 64 x 256 x 256 -> N x 64 x 128 x 128。
        #
        # DoubleConv 再把通道数从 in_channels 变成 out_channels。
        # 例如: N x 64 x 128 x 128 -> N x 128 x 128 x 128。
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        # 编码器中每经过一个 Down，空间尺寸变小，通道数通常变多。
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv.

    Up 是 U-Net 解码器中的一个上采样模块。
    它做三件事:
    1. 把深层小尺寸特征图放大；
    2. 和编码器对应层传来的特征图拼接；
    3. 用 DoubleConv 融合拼接后的特征。
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # bilinear=True 时，使用双线性插值上采样。
        # 它只放大 H 和 W，不主动学习上采样参数。
        #
        # bilinear=False 时，使用 ConvTranspose2d，也就是转置卷积。
        # 它也能放大 H 和 W，但带有可学习参数。
        if bilinear:
            # scale_factor=2 表示高度和宽度都放大 2 倍。
            # mode='bilinear' 是图像任务中常见的插值方式。
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            # 使用双线性插值时，通道数不会在 self.up 中减少。
            # 这里通过 DoubleConv 的 mid_channels=in_channels//2 来控制中间通道规模。
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # ConvTranspose2d 这里同时完成两件事:
            # 1. kernel_size=2, stride=2 让 H 和 W 放大 2 倍；
            # 2. 通道数从 in_channels 变成 in_channels//2。
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        # x1: 来自更深层的特征图，空间尺寸较小，语义信息更强。
        # x2: 来自编码器的 skip connection，空间尺寸较大，细节信息更多。

        # 先把 x1 上采样，让它的 H 和 W 接近 x2。
        x1 = self.up(x1)

        # PyTorch 图像张量常见格式是 NCHW:
        # N 是 batch size，C 是通道数，H 是高度，W 是宽度。
        # size()[2] 取 H，size()[3] 取 W。
        #
        # 因为输入尺寸不一定能被多次 2 整除，或者上采样存在取整问题，
        # x1 上采样后可能和 x2 差 1 个像素左右。
        # 这里计算二者在高度和宽度上的差值，后面用 pad 对齐。
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # F.pad 的参数顺序是 [左, 右, 上, 下]。
        # 这里把差值尽量平均补到左右、上下两边，
        # 让 x1 的空间尺寸和 x2 对齐，方便后面拼接。
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        # 在通道维 dim=1 上拼接。
        # 不是把图片变高或变宽，而是把两份特征“叠到通道上”。
        #
        # 例如:
        # x2 是 N x 512 x 32 x 32
        # x1 是 N x 512 x 32 x 32
        # 拼接后就是 N x 1024 x 32 x 32。
        #
        # 这一步就是 U-Net 的 skip connection:
        # 把编码器的细节特征 x2 传给解码器，和深层语义特征 x1 一起使用。
        x = torch.cat([x2, x1], dim=1)

        # 拼接后通道数变多，所以再用 DoubleConv 融合两路特征，
        # 同时把通道数整理成 out_channels。
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution.

    OutConv 是 U-Net 的输出层。
    它负责把最后的特征通道数转换成类别数。
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        # kernel_size=1 表示 1x1 卷积。
        # 它不会融合周围像素的空间邻域，主要用于调整通道数。
        #
        # 对语义分割来说，out_channels 通常等于类别数 n_classes。
        # 输出的每个像素位置都会有 out_channels 个值。
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 把最后一层特征映射到类别通道。
        return self.conv(x)
