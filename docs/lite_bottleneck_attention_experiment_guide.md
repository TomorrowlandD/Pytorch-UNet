# U-Net 轻量 Bottleneck Attention 完整设计与实验教程

## 1. 文档目的

本文档记录本项目下一阶段的 Attention U-Net 探索方案，覆盖以下内容：

1. 当前正式 baseline 的唯一口径；
2. 轻量注意力模块的结构、尺寸和设计理由；
3. 如何在不破坏原 U-Net 的前提下接入模块；
4. 如何兼容加载原 U-Net checkpoint；
5. 后续需要修改的代码和命令行参数；
6. 单元测试、显存测试和 smoke test；
7. 服务器 tmux 训练流程；
8. 固定 508 张验证集的逐图评估方法；
9. 如何判断注意力是否真正改善了 U-Net；
10. 实验结论可以和不可以如何表述。

本文档是实施方案，不代表代码已经完成。后续实现时应以本文档为设计依据；如果实际代码需要偏离本文档，必须在实验记录中注明原因。

---

## 2. 当前正式 baseline

### 2.1 Baseline 身份

本项目用于和 SAM3 对比的正式 baseline 不是 `experiment_results.csv` 中其他实验的约 0.99 Dice 结果，而是下面这组固定实验：

```text
EXP_NAME=exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2
MODEL=U-Net
DATASET=Carvana-full5088
EPOCHS=5
BATCH_SIZE=1
LR=1e-5
SCALE=0.5
AMP=no
LOSS=cross_entropy+dice
UPSAMPLING=bilinear
CLASSES=2
validation=10%
random_split seed=0
```

数据划分为：

```text
训练集：4580 张
验证集：508 张
```

固定验证集列表：

```text
exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2/val_ids.txt
```

### 2.2 Baseline checkpoint

5 个 epoch 中第 4 个 epoch 的验证表现最好，因此正式对比使用：

```text
checkpoints/exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2_best_epoch4.pth
```

### 2.3 Baseline 指标口径

正式 baseline 不是训练进度条中的某个临时数值，而是：

1. 使用 epoch 4 best checkpoint；
2. 对固定的 508 张验证图像逐图推理；
3. 对每张图的前景类别计算 Dice 和 IoU；
4. 对 508 个逐图指标做宏平均。

逐图结果文件：

```text
exp01_unet_final_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2/unet_epoch4_val_metrics.csv
```

正式 baseline：

```text
valid images: 508
mean Dice: 0.9794742027
mean IoU: 0.9613934193
min Dice: 0.8512483239
min IoU: 0.7410203815
max Dice: 0.9972764254
max IoU: 0.9945676327
```

与 SAM3 的同验证集结果：

```text
SAM3 mean Dice: 0.9935022748
SAM3 mean IoU: 0.9870986631
Dice gap: +0.0140280721
IoU gap: +0.0257052438
```

Attention 实验的目标是检验显式全局特征交互能否缩小上述差距，不是以其他实验中的约 0.99 U-Net 结果为 baseline。

### 2.4 两种指标文件不得混用

后续必须区分：

```text
训练过程 metrics.csv
    用途：观察每个 epoch 的 train_loss / val_dice / val_iou，选择 checkpoint。

逐图 attention_val_metrics.csv
    用途：在固定 508 张验证集上计算正式宏平均，并和当前 baseline、SAM3 比较。
```

`experiment_results.csv` 继续遵循项目既有流水线记录训练实验结果；正式 Attention 机制对比必须额外生成逐图 CSV 和对比总结，不能只看训练日志最后一行。

---

## 3. 研究问题与实验假设

### 3.1 研究问题

```text
在保持监督数据、训练配置、损失函数和 U-Net 编解码结构不变的情况下，
在 Bottleneck 中加入轻量全局多头注意力，是否能提高 U-Net 在固定 Carvana 验证集上的分割性能和稳定性？
```

### 3.2 机制假设

原 U-Net 主要依赖卷积逐层扩大感受野。Bottleneck 中不同空间位置仍缺少一次显式的全局特征交互。

轻量注意力可能改善：

- 车身局部预测断裂；
- 大范围形状不一致；
- 阴影、反光和复杂背景干扰；
- 非典型车辆姿态；
- 当前 baseline 中 Dice 较低的困难样本。

### 3.3 不预设一定提升

Attention 也可能没有明显收益，因为：

- Carvana 汽车通常位于画面中央，任务相对规则；
- U-Net 深层卷积已经具有较大感受野；
- 当前只训练 5 epoch；
- SAM3 的优势还来自预训练、模型规模和文本提示；
- 过强的全局平滑可能损害边缘细节。

因此，本实验是机制探索，而不是预先证明 Attention 一定有效。

---

## 4. 推荐模块：Lite Bottleneck SR-MHSA

模块全称：

```text
Lite Bottleneck Spatial-Reduction Multi-Head Self-Attention
```

建议代码名称：

```text
LiteSpatialReductionMHSA
```

建议配置名称：

```text
lite_sr_mhsa
```

### 4.1 接入位置

注意力只插在 `down4` 和 `up1` 之间：

```python
x1 = self.inc(x)
x2 = self.down1(x1)
x3 = self.down2(x2)
x4 = self.down3(x3)
x5 = self.down4(x4)

x5 = self.bottleneck_attention(x5)

x = self.up1(x5, x4)
x = self.up2(x, x3)
x = self.up3(x, x2)
x = self.up4(x, x1)
logits = self.outc(x)
```

注意力开启和关闭时，`x5` 的输入输出形状都必须相同。

### 4.2 当前特征尺寸

Carvana 原图约为：

```text
1918 × 1280
```

`scale=0.5` 后约为：

```text
959 × 640
```

bilinear U-Net 经过四次下采样后的 Bottleneck：

```text
B × 512 × 40 × 59
```

完整空间 token 数：

```text
Nq = 40 × 59 = 2360
```

### 4.3 固定设计参数

第一轮只测试一个注意力配置，避免同时改变多个变量：

```text
input_channels=512
attention_dim=128
num_heads=4
head_dim=32
spatial_reduction_ratio=2
attention_dropout=0.0
layer_scale_init=1e-3
max_layer_scale=1e-2
```

第一轮不要同时测试 `attention_dim=64/256`、2/8 heads 或多个插入位置。只有确认该模块有效后，才考虑进一步消融。

### 4.4 模块数据流

```text
x5: B×512×40×59
 │
 ├──────────────────────────────────────── residual ───────────────┐
 │                                                                │
 └─ 1×1 Conv: 512 → 128                                           │
      │                                                            │
      ├─ 3×3 Depthwise Conv，作为轻量二维位置编码                  │
      │                                                            │
      ├─ Q: 40×59 = 2360 tokens                                   │
      │                                                            │
      └─ AvgPool2d(2)                                              │
           └─ K/V: 20×29 = 580 tokens                             │
                │                                                  │
                └─ 4-head attention                               │
                     │                                             │
                     └─ reshape 回 B×128×40×59                    │
                          │                                        │
                          └─ 1×1 Conv: 128 → 512                   │
                               │                                   │
                               └─ LayerScale γ                     │
                                    └───────────────────────────────┘
```

### 4.5 为什么对 K/V 做空间降采样

完整 MHSA 会构造 `2360 × 2360` 的注意力关系。这里保留完整分辨率的 Q，但把 K/V 池化为：

```text
20 × 29 = 580 tokens
```

每个完整分辨率位置仍能访问整幅 Bottleneck 特征，只是全局上下文使用较低空间分辨率表达。

这仍然属于全局注意力，但显存和计算量明显降低。

### 4.6 为什么需要位置编码

纯 Self-Attention 本身不理解二维空间顺序。使用一个分组数等于通道数的 `3×3 Depthwise Conv`：

```python
nn.Conv2d(
    attention_dim,
    attention_dim,
    kernel_size=3,
    padding=1,
    groups=attention_dim,
)
```

可以用极少参数向特征中注入局部二维位置信息。

### 4.7 为什么使用外层残差和 LayerScale

模块输出：

```text
output = x + gamma * attention_delta
```

其中有效 `gamma` 初始值为 `1e-3`，并通过 `tanh` 门控严格限制在
`[-1e-2, +1e-2]`。

作用：

- 保持初始网络接近原 U-Net；
- 避免随机初始化 Attention 立即破坏旧特征；
- 提高加载原 checkpoint 后微调的稳定性；
- 即使 Attention 暂时没有学好，主干仍保留原始信息。
- 防止 RMSprop 在大量 batch 更新后将残差系数放大到 1 以上。

2026-07-22 的首次 smoke test 中，未约束 LayerScale 从 `0.001` 增长到
`abs mean=0.300`、`abs max=1.647`，同时 Attention 权重明显膨胀，导致验证
Dice 从约 `0.57` 下降到 `0.096`。因此后续实现必须使用有界门控，并为
Attention 参数设置独立的低学习率和较低 momentum。

### 4.8 理论开销

按当前 bilinear U-Net 和固定配置估算：

```text
新增参数：约 19.9 万
相对 baseline 参数增量：约 1.15%
新增计算量：约 0.759 GMAC
相对 baseline 理论计算增量：约 0.20%
FP32 attention score 理论大小：约 20.9 MiB
FP16 attention score 理论大小：约 10.4 MiB
```

实际训练时间增长可能高于理论 MAC 增量，因为 Attention kernel 和卷积 kernel 的硬件效率不同。预计先以 3%～10% 额外训练时间作为工程估计，最终以服务器实测为准。

---

## 5. 计划中的模块代码

以下代码是实施草案。正式实现时需要结合当前 PyTorch 版本进行测试。

建议在 `unet/unet_parts.py` 中新增：

```python
class LiteSpatialReductionMHSA(nn.Module):
    def __init__(
            self,
            channels: int,
            attention_dim: int = 128,
            num_heads: int = 4,
            sr_ratio: int = 2,
            layer_scale_init: float = 1e-3,
            max_layer_scale: float = 1e-2,
    ):
        super().__init__()

        if attention_dim % num_heads != 0:
            raise ValueError('attention_dim must be divisible by num_heads')
        if sr_ratio < 1:
            raise ValueError('sr_ratio must be at least 1')

        self.channels = channels
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio

        self.reduce = nn.Conv2d(
            channels,
            attention_dim,
            kernel_size=1,
            bias=False,
        )
        self.position = nn.Conv2d(
            attention_dim,
            attention_dim,
            kernel_size=3,
            padding=1,
            groups=attention_dim,
        )

        self.q_norm = nn.LayerNorm(attention_dim)
        self.kv_norm = nn.LayerNorm(attention_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(attention_dim)

        self.expand = nn.Conv2d(
            attention_dim,
            channels,
            kernel_size=1,
            bias=False,
        )
        initial_ratio = layer_scale_init / max_layer_scale
        initial_logit = math.atanh(initial_ratio)
        self.layer_scale_logits = nn.Parameter(
            torch.full((channels,), initial_logit)
        )
        self.register_buffer(
            'max_layer_scale',
            torch.tensor(float(max_layer_scale)),
        )

    def effective_layer_scale(self):
        return self.max_layer_scale * torch.tanh(self.layer_scale_logits)

    def forward(self, x):
        batch_size, _, height, width = x.shape

        features = self.reduce(x)
        features = features + self.position(features)

        query = features.flatten(2).transpose(1, 2)

        if self.sr_ratio > 1:
            kv_features = F.avg_pool2d(
                features,
                kernel_size=self.sr_ratio,
                stride=self.sr_ratio,
            )
        else:
            kv_features = features

        key_value = kv_features.flatten(2).transpose(1, 2)

        query = self.q_norm(query)
        key_value = self.kv_norm(key_value)

        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            need_weights=False,
        )
        attended = self.output_norm(attended)

        attended = attended.transpose(1, 2).reshape(
            batch_size,
            self.attention_dim,
            height,
            width,
        )
        delta = self.expand(attended)

        scale = self.effective_layer_scale().view(1, -1, 1, 1)
        return x + scale * delta
```

实现注意事项：

- 必须使用 `reshape`，不要假设 transpose 后张量连续；
- 调用 MHA 时使用 `need_weights=False`，避免保存不需要的注意力权重；
- 第一版不输出 attention map，正式训练只输出分割结果；
- 不要在这一模块中改变空间尺寸；
- `sr_ratio=2` 时，`40×59` 的 K/V 尺寸为 `20×29`；
- `LayerNorm` 操作 token 的最后一维，即 attention channel；
- 位置 Depthwise Conv 使用 padding=1，保持 `40×59` 不变。

---

## 6. 将模块接入 U-Net

### 6.1 构造函数参数

建议扩展 `UNet` 构造函数：

```python
class UNet(nn.Module):
    def __init__(
            self,
            n_channels,
            n_classes,
            bilinear=False,
            attention='none',
            attention_dim=128,
            attention_heads=4,
            attention_sr_ratio=2,
    ):
```

在创建 `down4` 后确定 Bottleneck 通道数：

```python
factor = 2 if bilinear else 1
bottleneck_channels = 1024 // factor

self.down4 = Down(512, bottleneck_channels)
```

然后创建可插拔模块：

```python
if attention == 'none':
    self.bottleneck_attention = nn.Identity()
elif attention == 'lite_sr_mhsa':
    self.bottleneck_attention = LiteSpatialReductionMHSA(
        channels=bottleneck_channels,
        attention_dim=attention_dim,
        num_heads=attention_heads,
        sr_ratio=attention_sr_ratio,
    )
else:
    raise ValueError(f'Unsupported attention type: {attention}')
```

### 6.2 forward 接入

只增加一行：

```python
x5 = self.down4(x4)
x5 = self.bottleneck_attention(x5)
x = self.up1(x5, x4)
```

### 6.3 为什么不破坏原 U-Net

当：

```text
attention=none
```

时：

```python
self.bottleneck_attention = nn.Identity()
```

因此模型严格执行原 U-Net 数据流。

Attention 开启时：

- 编码器不变；
- 解码器不变；
- skip connections 不变；
- 上采样方式不变；
- 输出尺寸和类别数不变；
- 只在 Bottleneck 增加一个残差插件。

正式模型名称应写为：

```text
U-Net + Lite Bottleneck SR-MHSA
```

不应把它写成 TransUNet 或完整 Transformer U-Net。

---

## 7. 命令行参数设计

建议在 `train.py` 增加：

```python
parser.add_argument(
    '--attention',
    choices=['none', 'lite_sr_mhsa'],
    default='none',
)
parser.add_argument('--attention-dim', type=int, default=128)
parser.add_argument('--attention-heads', type=int, default=4)
parser.add_argument('--attention-sr-ratio', type=int, default=2)
parser.add_argument('--attention-max-scale', type=float, default=1e-2)
parser.add_argument('--attention-lr-scale', type=float, default=0.1)
parser.add_argument('--attention-momentum', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=0)
```

创建模型：

```python
model = UNet(
    n_channels=3,
    n_classes=args.classes,
    bilinear=args.bilinear,
    attention=args.attention,
    attention_dim=args.attention_dim,
    attention_heads=args.attention_heads,
    attention_sr_ratio=args.attention_sr_ratio,
)
```

日志和 W&B 配置中必须记录：

```text
attention
attention_dim
attention_heads
attention_sr_ratio
attention_max_scale
attention_lr_scale
attention_momentum
seed
```

不记录这些配置会导致 checkpoint 和实验结果无法追溯。

---

## 8. 随机种子与公平对比

当前数据集划分已经使用 `random_split seed=0`，但正式结构消融还应固定：

- Python random；
- NumPy；
- PyTorch CPU；
- PyTorch CUDA；
- DataLoader shuffle generator。

建议新增：

```python
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

如果启用完全确定性算法导致性能明显下降或某些 CUDA 算子报错，可以不强制 `torch.use_deterministic_algorithms(True)`，但必须至少固定上述随机源并在记录中说明。

最严谨的对比不是“旧 baseline checkpoint 对新随机初始化 Attention”，而是：

```text
同一 seed 的新 baseline
vs
同一 seed 的 Attention U-Net
```

两者之间唯一结构变量是 Attention。

---

## 9. 原 checkpoint 兼容策略

### 9.1 可以加载哪些权重

原 U-Net checkpoint 可以加载到 Attention U-Net 中：

```text
Encoder     ← 原 checkpoint
Bottleneck  ← 原 down4 权重
Decoder     ← 原 checkpoint
Output head ← 原 checkpoint
Attention   ← 新初始化
```

### 9.2 为什么默认严格加载会失败

新增模块后，旧 checkpoint 中不存在：

```text
bottleneck_attention.*
```

因此默认 `strict=True` 会报告 missing keys。

### 9.3 建议显式设计加载模式

不要在所有场景下无条件使用 `strict=False`，否则可能掩盖真正的漏载错误。

建议增加：

```text
--load-mode strict
--load-mode backbone
```

其中：

```text
strict：checkpoint 必须和模型完全一致；
backbone：允许旧 U-Net checkpoint 缺少 bottleneck_attention.*。
```

兼容加载伪代码：

```python
state_dict = torch.load(checkpoint_path, map_location=device)
state_dict.pop('mask_values', None)

incompatible = model.load_state_dict(state_dict, strict=False)

allowed_missing_prefixes = ('bottleneck_attention.',)
invalid_missing = [
    key for key in incompatible.missing_keys
    if not key.startswith(allowed_missing_prefixes)
]

if invalid_missing or incompatible.unexpected_keys:
    raise RuntimeError(
        f'Invalid checkpoint mismatch: '
        f'missing={invalid_missing}, '
        f'unexpected={incompatible.unexpected_keys}'
    )
```

### 9.4 三种训练含义

#### A. 直接加载旧权重并推理

不用于判断 Attention 效果，因为 Attention 尚未训练。

#### B. 加载旧 baseline 后微调 Attention U-Net

可以作为迁移学习或快速微调实验，但必须记录：

```text
initialized from baseline epoch4 checkpoint
```

它不能和从头训练的 baseline 构成严格结构消融。

#### C. 从头训练 baseline 和 Attention

这是分析 Attention 是否有效的正式方案。

---

## 10. 代码修改清单

正式实现预计修改或新增：

```text
unet/unet_parts.py
    新增 LiteSpatialReductionMHSA。

unet/unet_model.py
    新增 attention 配置和 bottleneck_attention 插入点。

train.py
    新增 attention、seed、兼容加载参数；记录实验配置。

predict.py
    支持创建 Attention U-Net，并严格识别 checkpoint 类型。

benchmark_inference.py
    支持 Attention 参数；记录参数量、时间和显存。

visualize_prediction.py
    支持加载 Attention checkpoint。

scripts/evaluate_unet_val_ids.py
    建议新增；统一在固定 val_ids 上逐图评估。

tests/test_attention_module.py
    建议新增；检查形状、梯度、Identity 和 checkpoint 兼容性。
```

第一轮不要同时修改损失函数、数据增强、epoch、学习率或 upsampling。

---

## 11. 测试教程

### 11.1 模块形状测试

测试输入：

```python
x = torch.randn(1, 512, 40, 59)
module = LiteSpatialReductionMHSA(
    channels=512,
    attention_dim=128,
    num_heads=4,
    sr_ratio=2,
)
y = module(x)
assert y.shape == x.shape
```

### 11.2 反向传播测试

```python
y.mean().backward()
assert module.reduce.weight.grad is not None
assert module.expand.weight.grad is not None
assert module.layer_scale_logits.grad is not None
assert module.effective_layer_scale().abs().max() <= 1e-2
```

### 11.3 完整模型测试

使用小尺寸输入，避免本地 CPU 过慢：

```python
model = UNet(
    n_channels=3,
    n_classes=2,
    bilinear=True,
    attention='lite_sr_mhsa',
)

x = torch.randn(1, 3, 128, 192)
y = model(x)
assert y.shape == (1, 2, 128, 192)
```

### 11.4 Identity 回归测试

```text
UNet(..., attention='none')
```

必须和改造前原 U-Net 使用相同 checkpoint 得到一致输出，允许的差异只应来自浮点误差。

### 11.5 旧 checkpoint 兼容测试

检查：

- 旧 checkpoint 能以 backbone 模式载入 Attention U-Net；
- missing keys 只有 `bottleneck_attention.*`；
- 其他 U-Net 权重全部匹配；
- strict 模式仍能发现真正的不兼容。

### 11.6 参数量检查

分别打印：

```text
baseline total parameters
attention total parameters
attention-only parameters
```

验收目标：

```text
attention-only parameters ≈ 0.20M
relative increase ≈ 1.15%
```

如果实际新增参数显著超过约 0.25M，需要检查是否误用了未降维的 512 通道完整注意力。

### 11.7 服务器显存测试

在正式训练前记录：

```bash
nvidia-smi
```

分别观察：

```text
模型初始化后显存
第一个训练 batch 的峰值显存
第一次验证时的峰值显存
```

---

## 12. 实验阶段划分

### 阶段 1：工程 smoke test

目的：验证不会报错，不用于正式结论。

建议：

```text
完整数据集
1 epoch
batch size 1
scale 0.5
bilinear
no AMP
attention=lite_sr_mhsa
```

验收：

- forward/backward 正常；
- 没有 OOM；
- 日志包含 Dice 和 IoU；
- `metrics.csv` 正常生成；
- checkpoint 可以重新加载；
- 单 epoch 时间在合理范围内。

### 阶段 2：最小正式实验

只新增一个 Attention 训练：

```text
现有 exp01 baseline
vs
新 Attention U-Net
```

优点是成本最低；缺点是模型初始化随机性没有严格配对，因此只能作为探索性证据。

### 阶段 3：推荐的配对结构消融

使用新加入的 `--seed 0` 分别训练：

```text
Baseline rerun, seed=0
Attention U-Net, seed=0
```

这两个实验才是 Attention 是否有效的主要结构消融。

### 阶段 4：必要时多 seed

只有在第一轮结果较小或不稳定时，再运行：

```text
seed=1
seed=2
```

不要一开始同时搜索大量 heads、dimension 和 sr_ratio。

---

## 13. 正式训练参数

Attention 对比只允许改变：

```text
ATTENTION=none
```

与：

```text
ATTENTION=lite_sr_mhsa
```

固定参数：

```text
MODEL=U-Net
DATASET=Carvana-full5088
EPOCHS=5
BATCH_SIZE=1
LR=1e-5
SCALE=0.5
AMP=no
LOSS=cross_entropy+dice
UPSAMPLING=bilinear
VALIDATION=10
NUM_WORKERS=2
CLASSES=2
ATTENTION_DIM=128
ATTENTION_HEADS=4
ATTENTION_SR_RATIO=2
ATTENTION_MAX_SCALE=1e-2
ATTENTION_LR_SCALE=0.1
ATTENTION_MOMENTUM=0.9
SEED=0
```

注意：正式 baseline 是 bilinear，因此 Attention 实验必须添加 `--bilinear`。不能使用 transposed convolution 的 Attention 模型去和当前 baseline 比较。

---

## 14. 本地开发与 Git 流程

本地修改前：

```powershell
cd "D:\DeepLearning Workplace\Project\04-Unet\code\Pytorch-UNet"
git status
git pull --ff-only origin master
```

如果工作区已有未提交改动，先确认这些改动的所有者和用途，不要覆盖、删除或重置。

完成代码和测试后，仅暂存本次相关文件，例如：

```powershell
git add unet/unet_parts.py
git add unet/unet_model.py
git add train.py
git add predict.py
git add benchmark_inference.py
git add visualize_prediction.py
git add scripts/evaluate_unet_val_ids.py
git add tests/test_attention_module.py
git add docs/lite_bottleneck_attention_experiment_guide.md
```

不要提交：

```text
data/
logs/
results/
checkpoints/
checkpoints_archive/
wandb/
*.pth
*.pt
*.ckpt
*.zip
```

提交消息示例：

```text
Add lightweight bottleneck attention experiment support
```

---

## 15. 服务器 tmux 训练教程

### 15.1 同步代码

```bash
cd /root/autodl-tmp/projects/Pytorch-UNet
git status
git pull --ff-only origin master
git log --oneline -5
```

### 15.2 创建 tmux

```bash
tmux new -s unet_attention
```

如果会话已经存在：

```bash
tmux attach -t unet_attention
```

长期训练必须放在 tmux 中。

### 15.3 Attention smoke test 参数

实验名在正式运行前根据当前编号确认，下面使用占位名称：

```bash
EXP_NAME=exp_attention_smoke_e1_seed0
EPOCHS=1
BATCH_SIZE=1
LR=1e-5
SCALE=0.5
LOSS=cross_entropy+dice
ATTENTION=lite_sr_mhsa
ATTENTION_DIM=128
ATTENTION_HEADS=4
ATTENTION_SR_RATIO=2
SEED=0
```

### 15.4 Attention smoke test 命令

```bash
cd /root/autodl-tmp/projects/Pytorch-UNet
mkdir -p logs
export WANDB_MODE=offline

python train.py \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --learning-rate ${LR} \
  --scale ${SCALE} \
  --validation 10 \
  --num-workers 2 \
  --classes 2 \
  --loss ${LOSS} \
  --bilinear \
  --attention ${ATTENTION} \
  --attention-dim ${ATTENTION_DIM} \
  --attention-heads ${ATTENTION_HEADS} \
  --attention-sr-ratio ${ATTENTION_SR_RATIO} \
  --attention-max-scale ${ATTENTION_MAX_SCALE} \
  --attention-lr-scale ${ATTENTION_LR_SCALE} \
  --attention-momentum ${ATTENTION_MOMENTUM} \
  --seed ${SEED} \
  --exp-name ${EXP_NAME} \
  2>&1 | tee logs/${EXP_NAME}.log
```

当前固定 `AMP=no`，不要添加 `--amp`。

### 15.5 正式 Attention 实验参数

正式实验名应明确包含模型差异：

```bash
EXP_NAME=exp_attention_litesr_bilinear_full5088_e5_bs1_lr1e-5_scale0.5_noamp_workers2_seed0
MODEL=U-Net+LiteSRMHSA
DATASET=Carvana-full5088
EPOCHS=5
BATCH_SIZE=1
LR=1e-5
SCALE=0.5
AMP=no
LOSS=cross_entropy+dice
UPSAMPLING=bilinear
ATTENTION=lite_sr_mhsa
ATTENTION_DIM=128
ATTENTION_HEADS=4
ATTENTION_SR_RATIO=2
ATTENTION_MAX_SCALE=1e-2
ATTENTION_LR_SCALE=0.1
ATTENTION_MOMENTUM=0.9
SEED=0
REMARK="lightweight bottleneck spatial-reduction MHSA"
```

训练命令与 smoke test 相同，只把 `EPOCHS=5` 和正式 `EXP_NAME` 代入。

### 15.6 训练中检查

```bash
echo ${EXP_NAME}
tail -n 50 logs/${EXP_NAME}.log
nvidia-smi
```

不要在训练尚未结束时写入最终 CSV。

---

## 16. Checkpoint 选择

当前实现按实验名隔离 checkpoint：

```text
checkpoints/<EXP_NAME>/checkpoint_epoch1.pth
...
checkpoints/<EXP_NAME>/checkpoint_epoch5.pth
```

不同实验不再共同写入 `checkpoints/checkpoint_epochN.pth`，从而避免 baseline、
smoke test 和 Attention 实验相互覆盖或被错误归档。

当前 baseline 使用 5 个 epoch 中表现最好的 epoch 4 checkpoint。Attention 模型必须使用相同选择规则：

```text
在 epoch 1～5 中，按训练流程记录的验证 Dice 选择最佳 checkpoint。
```

需要记录：

```text
best epoch
best checkpoint path
best training val Dice
best training val IoU
```

不要：

- baseline 用 best epoch、Attention 用 final epoch；
- 根据逐图对比结果反向挑 checkpoint；
- 只挑对 SAM3 最有利的 checkpoint。

---

## 17. 固定 508 张验证集逐图评估

### 17.1 建议新增统一评估脚本

建议新增：

```text
scripts/evaluate_unet_val_ids.py
```

输入参数建议包括：

```text
--checkpoint
--image-dir
--mask-dir
--val-ids
--scale
--classes
--bilinear
--attention
--attention-dim
--attention-heads
--attention-sr-ratio
--output-csv
```

### 17.2 评估前校准

在使用新脚本评估 Attention 之前，必须先用它重新评估现有 baseline epoch4 checkpoint。

验收条件：

```text
rows = 508
mean Dice ≈ 0.9794742027
mean IoU ≈ 0.9613934193
```

如果不能复现，先检查：

- 是否使用同一个 `val_ids.txt`；
- 是否添加 `--bilinear`；
- 是否使用 `scale=0.5`；
- mask 后缀和 mask value 映射是否一致；
- 两类输出是否使用 `argmax`；
- 是否只计算前景类别；
- Dice/IoU epsilon 是否一致；
- 图像和 mask resize 方法是否一致。

在 baseline 复现通过前，不得用新脚本宣布 Attention 提升或下降。

### 17.3 Attention 逐图结果格式

建议输出：

```csv
image_id,attention_dice,attention_iou,pred_pixels,gt_pixels
```

文件路径示例：

```text
results/${EXP_NAME}/attention_best_val_metrics.csv
```

### 17.4 合并对比格式

建议生成：

```text
results/${EXP_NAME}/baseline_vs_attention_val_compare.csv
```

字段：

```csv
image_id,baseline_dice,baseline_iou,attention_dice,attention_iou,dice_diff_attention_minus_baseline,iou_diff_attention_minus_baseline
```

如果同时和 SAM3 合并：

```csv
image_id,baseline_dice,attention_dice,sam3_dice,baseline_iou,attention_iou,sam3_iou
```

---

## 18. 正式统计指标

### 18.1 Primary endpoint

```text
固定 508 张验证图像的 mean per-image foreground Dice
```

即：

```text
mean(attention_dice - baseline_dice)
```

### 18.2 Secondary endpoints

```text
mean per-image foreground IoU
Attention 胜出图片数
Baseline 胜出图片数
Dice 差值的中位数
Dice 第 5 百分位数
IoU 第 5 百分位数
原 baseline 低 Dice 样本的平均改善
最差样本变化
参数量
峰值显存
单 epoch 时间
```

最小值容易被单张异常图主导，因此不能只依赖 min Dice；建议同时报告第 5 百分位数。

### 18.3 配对统计

因为两种模型在完全相同的 508 张图片上评估，应使用逐图配对差值，而不是把两组样本当作独立样本。

建议：

- 对 `attention_dice - baseline_dice` 做 bootstrap 95% CI；
- bootstrap 固定随机种子；
- 建议 10,000 次重采样；
- 同时报告胜负数量和差值分布。

### 18.4 SAM3 差距缩小比例

当前 Dice gap：

```text
0.0140280721
```

定义：

```text
gap_closure_ratio = attention_dice_gain / 0.0140280721
```

例子：

| Attention Dice 增量 | 新 Dice | 缩小 SAM3 Dice 差距 |
| ---: | ---: | ---: |
| +0.001 | 0.980474 | 约 7% |
| +0.002 | 0.981474 | 约 14% |
| +0.004 | 0.983474 | 约 29% |
| +0.007 | 0.986474 | 约 50% |

---

## 19. 结果判定规则

以下是本项目预先采用的探索性判定标准，不是所有分割研究的通用标准：

### 明确支持

```text
mean Dice 提升 ≥ 0.002
且 mean IoU 同方向改善
且困难样本或第 5 百分位数不明显恶化
```

如果配对 bootstrap 95% CI 下界也大于 0，证据更强。

### 弱提升

```text
mean Dice 提升在 +0.0005 到 +0.002 之间
```

需要运行更多 seed，不能只凭一次训练下结论。

### 无充分证据

```text
mean Dice 变化绝对值 < 0.0005
```

说明本次模块在当前训练设置下与 baseline 基本持平。

### 负面结果

```text
mean Dice 或 IoU 明显下降
或平均值略升但困难样本显著恶化
```

需要检查收敛、学习率、LayerScale 和模块实现，但不能自动把负面结果解释成“全局上下文无效”。

---

## 20. 结果记录到 CSV

正式实验表头保持不变：

```csv
exp_name,model,dataset,epochs,batch_size,lr,scale,amp,loss,upsampling,final_dice,final_iou,remark
```

训练结束后仍按项目规定从日志提取最终训练指标：

```bash
FINAL_DICE=$(grep "Validation Dice score" logs/${EXP_NAME}.log | tail -1 | awk -F': ' '{print $NF}')
FINAL_IOU=$(grep "Validation IoU score" logs/${EXP_NAME}.log | tail -1 | awk -F': ' '{print $NF}')
```

如果为空，不写 CSV。

Attention 行建议：

```bash
MODEL="U-Net+LiteSRMHSA"
REMARK="bottleneck attn_dim=128 heads=4 sr_ratio=2 seed=0; formal comparison uses best-checkpoint per-image macro metrics"

echo "${EXP_NAME},${MODEL},${DATASET},${EPOCHS},${BATCH_SIZE},${LR},${SCALE},${AMP},${LOSS},${UPSAMPLING},${FINAL_DICE},${FINAL_IOU},${REMARK}" >> experiment_results.csv
```

注意：

```text
experiment_results.csv 中的 final 指标
和
best checkpoint 的 508 张逐图宏平均
```

必须在文档中分别标注，不能把二者当作同一个数值。

---

## 21. 提交服务器实验结果

训练和评估完成后检查：

```bash
tail -n 50 logs/${EXP_NAME}.log
cat results/${EXP_NAME}/metrics.csv
head results/${EXP_NAME}/attention_best_val_metrics.csv
wc -l results/${EXP_NAME}/attention_best_val_metrics.csv
tail -n 5 experiment_results.csv
```

逐图 CSV 应为：

```text
1 行表头 + 508 行图片 = 509 行
```

根据项目 Git 规则，服务器通常只提交：

```bash
git add experiment_results.csv
git commit -m "Record ${EXP_NAME} result"
git push origin master
```

如果决定将逐图对比结果纳入版本控制，需要先确认文件范围和仓库策略；不要提交 checkpoint、日志目录或大型结果压缩包。

---

## 22. 推荐的最终对比表

| 模型 | 监督数据 | Mean Dice | Mean IoU | Dice P5 | IoU P5 | 胜出图片数 | 参数量 | 单 epoch 时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net baseline | 4580 | 0.979474 | 0.961393 | 待统计 | 待统计 | 基准 | 约 17.26M | 待统一实测 |
| U-Net + Lite SR-MHSA | 4580 | 待实验 | 待实验 | 待实验 | 待实验 | 待实验 | 约 17.46M | 待实验 |
| SAM3 zero-shot | 0 | 0.993502 | 0.987099 | 待统计 | 待统计 | 313/508 相对原 U-Net | 大模型 | 推理成本另计 |

SAM3 的“监督数据 0”只表示没有使用 Carvana 标注进行针对性训练，不表示 SAM3 没有大规模预训练数据。

---

## 23. 结论写作模板

### 23.1 如果 Attention 明显提升

可以写：

```text
在保持数据划分、训练配置、损失函数和 U-Net 编解码结构不变的情况下，
在 Bottleneck 中加入轻量空间降采样多头注意力后，模型在固定 508 张验证图像上的
平均 Dice/IoU 有所提高，并在困难样本上表现出更好的稳定性。
该结果说明，显式全局特征交互可能有助于改善传统 U-Net 的车辆整体形状建模。
```

不可以写：

```text
已经证明 SAM3 的优势完全来自 Transformer 或 Attention。
```

### 23.2 如果提升很小

```text
轻量 Bottleneck Attention 在当前 5 epoch 训练设置下仅带来有限变化。
结果表明，单个低成本全局注意力模块不足以复现 SAM3 的明显性能优势，
SAM3 的优势可能还与大规模预训练、模型容量和提示机制有关。
```

### 23.3 如果没有提升或下降

```text
在当前模块配置和训练预算下，引入轻量 Bottleneck Attention 未改善 U-Net 的整体指标。
该结果只说明本次实现和训练设置未获得收益，不能据此否定全局上下文建模本身。
```

---

## 24. 第一轮实验明确不做的内容

为了保持实验可解释性，第一轮不做：

- Attention Gate；
- SE、CBAM、scSE；
- 多层 Attention；
- Encoder 高分辨率全局 Attention；
- Decoder Attention；
- 完整 Transformer Encoder FFN；
- 同时修改 loss；
- 同时修改学习率；
- 同时修改 scale；
- 同时修改 AMP；
- 同时修改 upsampling；
- 大规模超参数搜索。

第一轮唯一问题是：

```text
增加一个轻量 Bottleneck 全局注意力，是否优于相同训练条件下的 bilinear U-Net？
```

---

## 25. 完整执行检查清单

### 代码阶段

- [ ] 实现 `LiteSpatialReductionMHSA`；
- [ ] `attention=none` 使用 `nn.Identity()`；
- [ ] Bottleneck 输入输出形状相同；
- [ ] 命令行记录所有 Attention 参数；
- [ ] 添加全局 seed；
- [ ] 兼容加载只允许缺少 Attention keys；
- [ ] predict/benchmark/visualization 支持新模型；
- [ ] 新增固定 val_ids 评估脚本。

### 测试阶段

- [ ] 模块 shape test；
- [ ] backward test；
- [ ] 完整模型 forward test；
- [ ] Identity 回归测试；
- [ ] 旧 checkpoint 兼容测试；
- [ ] 参数增量约 0.20M；
- [ ] baseline epoch4 逐图指标可复现。

### 训练阶段

- [ ] 服务器已拉取正确 commit；
- [ ] 训练运行在 tmux；
- [ ] `echo ${EXP_NAME}` 正确；
- [ ] 只有 Attention 一个结构变量；
- [ ] 使用 `--bilinear`；
- [ ] 没有使用 `--amp`；
- [ ] 先完成 1 epoch smoke test；
- [ ] 记录峰值显存和单 epoch 时间；
- [ ] 正式训练 5 epoch；
- [ ] 按同一规则选择 best checkpoint。

### 评估阶段

- [ ] 使用同一个 508 张 `val_ids.txt`；
- [ ] 输出 508 行逐图指标；
- [ ] 计算 mean Dice / IoU；
- [ ] 生成 baseline vs Attention 配对 CSV；
- [ ] 统计逐图胜负；
- [ ] 统计第 5 百分位数；
- [ ] 检查 baseline 困难样本；
- [ ] 必要时计算 bootstrap 95% CI；
- [ ] 计算 SAM3 gap closure ratio。

### 记录阶段

- [ ] `experiment_results.csv` 写入完整正式表头；
- [ ] model 写为 `U-Net+LiteSRMHSA`；
- [ ] remark 写明 dim/heads/sr_ratio/seed；
- [ ] 训练 final 指标和逐图 best-checkpoint 宏平均分开标注；
- [ ] 服务器提交 CSV；
- [ ] 本地拉取结果；
- [ ] 最后再同步到 Word/PPT。

---

## 26. 当前推荐决策

第一轮正式采用：

```text
位置：Bottleneck，down4 之后、up1 之前
形式：Spatial-Reduction Multi-Head Attention
输入通道：512
注意力维度：128
头数：4
K/V 空间降采样：2
位置编码：3×3 Depthwise Conv
输出：512 通道
融合：LayerScale residual，gamma=1e-3
开关：attention=none / lite_sr_mhsa
```

这是当前项目在可解释性、全局建模能力、参数量、显存和训练成本之间最平衡的方案。
