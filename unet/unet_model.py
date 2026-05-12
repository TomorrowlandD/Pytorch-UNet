""" Full assembly of the parts to form the complete network.

这个文件负责把 unet_parts.py 里定义的基础模块组装成完整的 U-Net。
可以把它理解成“总装图”：这里决定网络有几层、每层怎么连接、数据怎么流动。
"""

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()

        # n_channels: 输入图像的通道数。
        #   RGB 彩色图通常是 3，灰度图通常是 1。
        # n_classes: 输出分割类别数。
        #   二分类分割可以是 1 或 2，多分类分割就是类别数量。
        # bilinear: 解码器上采样时是否使用双线性插值。
        #   False 时使用 ConvTranspose2d，也就是转置卷积。
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # inc 是网络入口层。
        # 输入形状大致是: N x n_channels x H x W
        # 输出形状大致是: N x 64 x H x W
        # 这里 DoubleConv 使用 padding=1 的 3x3 卷积，所以通常不改变 H 和 W。
        self.inc = (DoubleConv(n_channels, 64))

        # down1 到 down4 是编码器部分，也就是 U-Net 左半边。
        # 每个 Down 内部都是: MaxPool2d(2) + DoubleConv。
        # MaxPool2d(2) 会让 H 和 W 大致减半，DoubleConv 会提取更深层特征。
        #
        # 如果输入是 N x 3 x 256 x 256，那么形状大致变化为:
        # x1: N x 64  x 256 x 256
        # x2: N x 128 x 128 x 128
        # x3: N x 256 x 64  x 64
        # x4: N x 512 x 32  x 32
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))

        # 如果使用 bilinear 上采样，Up 模块本身不会通过转置卷积减少通道数。
        # 因此这里用 factor=2 让最深层和解码器的通道数减半，控制模型规模。
        factor = 2 if bilinear else 1

        # down4 是编码器最底部，也可以理解为 U-Net 的瓶颈层。
        # 继续把空间尺寸缩小一半，同时把语义特征提得更抽象。
        # bilinear=False 时，输出通道是 1024；bilinear=True 时，输出通道是 512。
        self.down4 = (Down(512, 1024 // factor))

        # up1 到 up4 是解码器部分，也就是 U-Net 右半边。
        # 每个 Up 都接收两个输入:
        #   1. 来自更深层的特征，例如 x5；
        #   2. 来自编码器对应层的 skip connection，例如 x4。
        #
        # 这些 skip connection 让解码器在恢复图像尺寸时，还能拿到浅层的边界和位置细节。
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        # 最后一层输出卷积。
        # 它用 1x1 卷积把通道数从 64 变成 n_classes。
        # 输出形状大致是: N x n_classes x H x W。
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # forward 定义输入数据在网络里的真实流动路线。
        # x 是输入图像张量，形状通常是 N x C x H x W。

        # 编码器路径: 尺寸逐步变小，通道数逐步变多。
        # x1 到 x4 会被保留下来，后面作为 skip connection 传给解码器。
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器路径: 尺寸逐步恢复。
        # up1(x5, x4) 表示:
        #   x5 是更深、更抽象的特征；
        #   x4 是编码器同层级保留下来的细节特征。
        # 后面的 up2/up3/up4 也是同样的模式。
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # logits 是模型的原始输出。
        # 对语义分割来说，它保留了空间结构:
        # 每个像素位置都会有 n_classes 个输出值，用来表示该像素属于各类别的倾向。
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        # checkpointing 是一种节省显存的技术。
        # 它不会改变网络结构，只是在训练时少保存一些中间结果，
        # 需要反向传播时再重新计算，从而用更多计算换更少显存。
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
