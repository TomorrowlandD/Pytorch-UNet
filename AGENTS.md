## 项目说明

本项目基于开源项目复现：https://github.com/milesial/Pytorch-UNet

## 文件删除规则

禁止批量删除文件或目录。

不要使用：

- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

需要删除文件时，只能一次删除一个明确路径的文件：

```powershell
Remove-Item "C:\path\to\file.txt"
```

如果需要批量删除文件，应停止操作，并让用户手动删除。

## 固定实验流水线

以后以这条线为准：

```text
本地 Codex 改代码
→ 本地提交到 GitHub
→ 服务器拉取 GitHub 最新代码
→ 服务器在 tmux 中跑训练
→ 提取 final_dice / final_iou
→ 写入 experiment_results.csv
→ 服务器提交 CSV 到 GitHub
→ 本地拉取 CSV
→ 手动复制 CSV 数据到 Word
```

服务器上的长期训练必须放在 `tmux` 中执行，避免 SSH 或网页终端断开后训练中断。

分工：

| 位置 | 任务 |
| --- | --- |
| 本地 Windows | Codex 改代码、提交代码、同步 Word |
| GitHub | 同步代码和实验记录 |
| 服务器 AutoDL | 拉代码、跑训练、写结果 |
| Word | 只做最终展示 |

## Git 和提交规则

本地改代码前：

```powershell
cd "D:\DeepLearning Workplace\Project\04-Unet\code\Pytorch-UNet"
git status
git pull --ff-only origin master
```

服务器跑实验前：

```bash
cd /root/autodl-tmp/projects/Pytorch-UNet
git status
git pull --ff-only origin master
git log --oneline -5
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

实验结束后服务器通常只提交：

```bash
git add experiment_results.csv
git commit -m "Record ${EXP_NAME} result"
git push origin master
```

如果同时写了实验说明，可一起提交 `experiment_record.md`。

## CSV 结果表

正式实验使用这个表头：

```csv
exp_name,model,dataset,epochs,batch_size,lr,scale,amp,loss,upsampling,final_dice,final_iou,remark
```

旧表头不再作为正式格式，例如：

```csv
exp_name,model,dataset,epochs,batch_size,lr,scale,amp,final_dice,log_file,remark
exp_name,model,dataset,epochs,batch_size,lr,scale,amp,final_dice,final_iou,remark
```

旧表不要和新表混用。需要时先备份再整理：

```bash
cp experiment_results.csv experiment_results_backup_old.csv
mv experiment_results.csv experiment_results_old.csv
echo "exp_name,model,dataset,epochs,batch_size,lr,scale,amp,loss,upsampling,final_dice,final_iou,remark" > experiment_results.csv
```

写入 CSV 时必须包含 `LOSS` 和 `UPSAMPLING`：

```bash
echo "${EXP_NAME},${MODEL},${DATASET},${EPOCHS},${BATCH_SIZE},${LR},${SCALE},${AMP},${LOSS},${UPSAMPLING},${FINAL_DICE},${FINAL_IOU},${REMARK}" >> experiment_results.csv
```

## 对比实验总原则

一次只改一个变量。

做学习率对比时，只改 `LR`。

做 loss 对比时，只改 `LOSS`。

做 upsampling 对比时，只改 `UPSAMPLING` 和对应命令参数 `--bilinear`。

其他参数保持一致：

```text
MODEL=U-Net
DATASET=Carvana-full5088
EPOCHS=5
BATCH_SIZE=1
SCALE=0.5
AMP=no
validation=10
num_workers=2
classes=2
```

## 学习率对比实验

实验顺序：

| 实验 | EXP_NAME | LR | 备注 |
| --- | --- | --- | --- |
| exp022 | `exp022_lr1e-5_full5088_e5_bs1_scale0.5_noamp_workers2` | `1e-5` | baseline rerun with IoU |
| exp020 | `exp020_lr1e-4_full5088_e5_bs1_scale0.5_noamp_workers2` | `1e-4` | learning rate comparison lr=1e-4 |
| exp021 | `exp021_lr5e-5_full5088_e5_bs1_scale0.5_noamp_workers2` | `5e-5` | learning rate comparison lr=5e-5 |
| exp023 | `exp023_lr5e-6_full5088_e5_bs1_scale0.5_noamp_workers2` | `5e-6` | learning rate comparison lr=5e-6 |

先跑 `exp022`，因为旧的 `exp012` 没有 `final_iou`。

## Loss 对比实验

当前 baseline loss：

```text
LOSS=cross_entropy+dice
```

如果后续去掉 Dice Loss，则记录：

```text
LOSS=cross_entropy
```

Loss 对比只允许改变 `LOSS`，其他参数保持一致。

## Upsampling 对比实验

当前代码中：

```text
不加 --bilinear  => UPSAMPLING=transposed_conv
加 --bilinear    => UPSAMPLING=bilinear
```

上采样方式对比必须成对实验：

| 实验 | 命令参数 | CSV 记录 |
| --- | --- | --- |
| baseline | 不加 `--bilinear` | `UPSAMPLING=transposed_conv` |
| bilinear 对比 | 加 `--bilinear` | `UPSAMPLING=bilinear` |

Upsampling 对比只允许改变 `UPSAMPLING` 和是否添加 `--bilinear`，其他参数保持一致。

## 服务器训练模板

进入或创建 tmux：

```bash
tmux new -s unet_exp
tmux attach -t unet_exp
```

在 tmux 中进入项目：

```bash
cd /root/autodl-tmp/projects/Pytorch-UNet
```

设置实验参数示例：

```bash
EXP_NAME=exp022_lr1e-5_full5088_e5_bs1_scale0.5_noamp_workers2
MODEL=U-Net
DATASET=Carvana-full5088
EPOCHS=5
BATCH_SIZE=1
LR=1e-5
SCALE=0.5
AMP=no
LOSS=cross_entropy+dice
UPSAMPLING=transposed_conv
REMARK="baseline rerun with IoU"
```

默认 transposed convolution 训练命令：

```bash
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
  --exp-name ${EXP_NAME} \
  2>&1 | tee logs/${EXP_NAME}.log
```

bilinear 对比训练命令：

```bash
mkdir -p logs
export WANDB_MODE=offline
UPSAMPLING=bilinear

python train.py \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --learning-rate ${LR} \
  --scale ${SCALE} \
  --validation 10 \
  --num-workers 2 \
  --classes 2 \
  --bilinear \
  --exp-name ${EXP_NAME} \
  2>&1 | tee logs/${EXP_NAME}.log
```

当前固定 `AMP=no` 时，不要加 `--amp`。只有专门做 AMP 对比时才改变它。

## 训练结束后记录结果

确认训练结束后再提取结果：

```bash
tail -n 50 logs/${EXP_NAME}.log
cat results/${EXP_NAME}/metrics.csv
```

提取最终指标：

```bash
FINAL_DICE=$(grep "Validation Dice score" logs/${EXP_NAME}.log | tail -1 | awk -F': ' '{print $NF}')
FINAL_IOU=$(grep "Validation IoU score" logs/${EXP_NAME}.log | tail -1 | awk -F': ' '{print $NF}')

echo "FINAL_DICE=${FINAL_DICE}"
echo "FINAL_IOU=${FINAL_IOU}"
```

如果 `FINAL_DICE` 或 `FINAL_IOU` 为空，不要写入 CSV，先检查日志和代码。

写入 CSV：

```bash
if [ ! -f experiment_results.csv ]; then
  echo "exp_name,model,dataset,epochs,batch_size,lr,scale,amp,loss,upsampling,final_dice,final_iou,remark" > experiment_results.csv
fi

echo "${EXP_NAME},${MODEL},${DATASET},${EPOCHS},${BATCH_SIZE},${LR},${SCALE},${AMP},${LOSS},${UPSAMPLING},${FINAL_DICE},${FINAL_IOU},${REMARK}" >> experiment_results.csv

tail -n 5 experiment_results.csv
```

提交结果：

```bash
git status
git add experiment_results.csv
git commit -m "Record ${EXP_NAME} result"
git push origin master
```

## 本地同步到 Word

服务器推送后，本地执行：

```powershell
cd "D:\DeepLearning Workplace\Project\04-Unet\code\Pytorch-UNet"
git pull --ff-only origin master
start experiment_results.csv
```

然后手动把 CSV 对应实验行复制到 Word。

## 每次实验检查清单

1. `echo ${EXP_NAME}`，确认实验名前后一致。
2. 确认本次实验只改了一个变量。
3. 日志里必须有 `Validation Dice score` 和 `Validation IoU score`。
4. `tail -n 5 experiment_results.csv`，确认 CSV 写入新行。
5. 服务器 `git push origin master`，本地 `git pull --ff-only origin master`。
