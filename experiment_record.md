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
