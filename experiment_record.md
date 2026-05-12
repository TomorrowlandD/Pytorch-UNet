## 实验记录

### 说明

本文件用于记录 U-Net 复现、baseline 和对比实验的关键配置与结果。


### exp012_baseline_carvana5088_original_e5_bs1_lr1e-5_scale0.5_noamp_workers2

- 数据集：Carvana full 5088
- 配置：epochs=5, batch_size=1, lr=1e-5, scale=0.5, validation=10
- AMP：no
- num_workers：2
- final Dice：0.9883347749710083
- 作者 README 参考 Dice：0.988423
- 差值：约 0.000088
- 结论：full Carvana baseline 基本复现成功。

### exp013_speedtest_carvana5088_e1_bs1_lr1e-5_scale0.5_noamp_workers4

- 目的：测试 num_workers=4 是否比 workers=2 更快
- 配置：epochs=1, batch_size=1, lr=1e-5, scale=0.5, validation=10, AMP=no
- 结果：1 epoch 约 7m53s，速度约 11.57 img/s
- 对比：workers=2 baseline 第 1 个 epoch 约 8m05s，速度约 11.59 img/s
- 结论：workers=4 提升很小；后续默认继续使用 workers=2 更稳。
