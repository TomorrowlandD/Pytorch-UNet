# U-Net vs SAM3 同验证集对比实验总结

## 1. 当前项目主线

本项目已经从单纯复现 U-Net，收束为：

```text
复现并训练 U-Net 车辆分割模型
-> 引入 SAM3 零样本基础分割模型
-> 在同一批 Carvana 验证集上比较 Dice / IoU
-> 分析监督式专用模型与基础分割模型的差异
```

当前不再继续扩展 Kvasir-SEG、Attention U-Net、scSE 等支线，最终包装重点放在 U-Net 与 SAM3 的公平对比和结果分析。

## 2. 实验设置

### 数据集

```text
DATASET=Carvana-full5088
image_dir=/root/autodl-tmp/projects/Pytorch-UNet/data/imgs
mask_dir=/root/autodl-tmp/projects/Pytorch-UNet/data/masks
```

### 验证集

使用 U-Net 训练时的固定验证集划分：

```text
validation=10%
random_split seed=0
valid images=508
```

验证集列表：

```text
results/exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2/val_ids.txt
```

### U-Net 设置

```text
EXP_NAME=exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2
MODEL=U-Net
EPOCHS=5
BATCH_SIZE=1
LR=1e-5
SCALE=0.5
AMP=no
LOSS=cross_entropy+dice
UPSAMPLING=bilinear
CLASSES=2
```

本次训练中第 4 轮验证集效果最好，因此采用 epoch 4 作为最终 U-Net 对比 checkpoint：

```text
checkpoints/exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2_best_epoch4.pth
```

U-Net 逐图结果：

```text
results/exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2/unet_epoch4_val_metrics.csv
```

### SAM3 设置

```text
SAM3 checkpoint=/root/autodl-tmp/projects/sam3/checkpoint/sam3.pt
prompt=car
threshold=0.3
mode=zero-shot inference
```

SAM3 逐图结果：

```text
results/exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2/sam3_val_metrics.csv
```

合并对比结果：

```text
results/exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2/unet_vs_sam3_val_compare.csv
```

## 3. 关键结果

### U-Net 结果

```text
valid images: 508
mean Dice: 0.9794742027
mean IoU: 0.9613934193
min Dice: 0.8512483239
min IoU: 0.7410203815
max Dice: 0.9972764254
max IoU: 0.9945676327
```

### SAM3 结果

```text
valid images: 508
mean score: 0.9607145054
mean Dice: 0.9935022748
mean IoU: 0.9870986631
min Dice: 0.9769007499
min IoU: 0.9548445566
max Dice: 0.9973435181
max IoU: 0.9947011126
```

### 同批逐图对比

```text
common images: 508
SAM3 better Dice images: 313
U-Net better Dice images: 195
same Dice images: 0
mean Dice diff: +0.0140280721
mean IoU diff: +0.0257052438
max Dice gain: +0.1436794331
```

## 4. 阶段性结论

在同一批 U-Net 验证集 508 张图片上，SAM3 零样本分割的平均 Dice 和平均 IoU 均高于本次 U-Net best checkpoint。

```text
U-Net mean Dice: 0.9795
SAM3 mean Dice: 0.9935
Dice improvement: +0.0140

U-Net mean IoU: 0.9614
SAM3 mean IoU: 0.9871
IoU improvement: +0.0257
```

从最差样本看，SAM3 的稳定性也更强：

```text
U-Net min Dice: 0.8512
SAM3 min Dice: 0.9769

U-Net min IoU: 0.7410
SAM3 min IoU: 0.9548
```

因此，当前可以较稳妥地表述为：

```text
在 Carvana 车辆主体分割任务中，SAM3 在无需针对该数据集训练的情况下，仅使用文本提示 car，就在同一验证集上取得了高于 U-Net best checkpoint 的平均 Dice / IoU，并且最差样本表现更稳定。
```

## 5. 不能过度下的结论

当前结果不能直接写成：

```text
SAM3 全面碾压 U-Net
SAM3 在所有样本上都优于 U-Net
U-Net 没有价值
```

原因：

```text
U-Net 仍有 195 张样本 Dice 高于 SAM3。
很多样本二者差距很小，只是 0.001 量级。
SAM3 依赖大模型、文本 prompt 和更高推理成本。
U-Net 训练完成后模型更轻量，部署与推理成本更低。
```

更合适的分析角度是：

```text
SAM3 在平均指标和稳定性上更强，体现了视觉基础模型对车辆语义目标的零样本分割能力；
U-Net 虽然需要标注数据和训练过程，但在相当一部分验证样本上仍能达到接近甚至略优的结果，并且模型结构更轻量。
```

## 6. 后续写报告的分析框架

建议按下面结构写：

```text
1. 实验目的
   比较传统监督式 U-Net 与 SAM3 零样本基础分割模型在同一车辆分割任务上的表现。

2. 实验设置
   使用同一 Carvana 验证集、同一 GT mask、同一 Dice / IoU 指标。

3. 总体结果
   对比 mean Dice / mean IoU / min Dice / min IoU。

4. 逐图胜负统计
   SAM3 在 313 张上 Dice 更高，U-Net 在 195 张上 Dice 更高。

5. 典型样本分析
   选择 SAM3 明显更优、U-Net 略优、二者接近的样本做可视化。

6. 原因解释
   SAM3 具有大规模预训练带来的语义识别能力；
   U-Net 是针对固定训练集优化的轻量监督式模型。

7. 局限性
   只在 Carvana 数据集上验证；
   SAM3 使用固定 prompt=car；
   未系统比较推理速度、显存占用和部署成本。
```

## 7. 推荐可视化样本

从合并结果中优先选择 SAM3 优势明显的样本：

```text
0495dcf27283_01
0495dcf27283_16
fce0ba5b8ed7_01
fce0ba5b8ed7_16
fd9da5d0bb6f_14
fdc2c87853ce_06
```

再选择若干 U-Net 略优或二者非常接近的样本，用于说明 U-Net 并非完全失效，而是与 SAM3 在多数常规样本上都能取得较高分割质量。

## 8. 当前最简汇报表述

```text
本项目在复现 U-Net 车辆图像分割流程的基础上，引入 SAM3 作为零样本基础分割模型进行对比实验。实验固定 U-Net 的验证集划分，在同一批 508 张 Carvana 验证图像上分别评估 U-Net best checkpoint 与 SAM3 prompt=car 的分割结果，并统一采用 Dice 和 IoU 作为指标。结果显示，U-Net 的 mean Dice / IoU 为 0.9795 / 0.9614，SAM3 的 mean Dice / IoU 为 0.9935 / 0.9871；逐图对比中，SAM3 在 313 张样本上 Dice 更高，U-Net 在 195 张样本上 Dice 更高。该结果表明，SAM3 在该车辆分割任务上具有较强的零样本分割能力和稳定性，同时也体现了传统监督式 U-Net 在轻量化和部署成本方面的价值。
```
