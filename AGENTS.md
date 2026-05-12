## 项目说明

本项目是基于以下开源项目复现的项目：https://github.com/milesial/Pytorch-UNet

## 文件删除规则

禁止批量删除文件或目录。

不要使用：

- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

需要删除文件时，只能一次删除一个明确路径的文件。

正确示例：

```powershell
Remove-Item "C:\path\to\file.txt"
```

如果需要批量删除文件，应停止操作，并向用户请求，让用户手动删除。

## 固定工作流核心结论

长期工作流固定为：

```text
本地代码 + Codex 修改代码
        ↓
本地 git push 到 GitHub
        ↓
服务器 git pull 最新代码
        ↓
服务器跑训练，用 tee 保存日志
        ↓
服务器提取 Dice / 记录实验结果
        ↓
服务器 git push 实验结果表到 GitHub
        ↓
本地 git pull，同步实验记录
```

GitHub 是代码与实验记录的同步中心；服务器是训练环境；本地 + Codex 是代码修改环境。

## 推荐项目结构

服务器项目目录：

```bash
cd /root/autodl-tmp/projects/Pytorch-UNet
```

建议形成下面结构：

```text
Pytorch-UNet/
├── train.py
├── evaluate.py
├── unet/
├── utils/
├── data/                         # 本地数据，不提交 Git
├── checkpoints/                  # 当前训练权重，不提交 Git
├── checkpoints_archive/          # 权重备份，不提交 Git
├── logs/                         # 训练日志，不提交 Git
├── wandb/                        # wandb 离线日志，不提交 Git
├── experiment_results.csv        # 实验结果总表，建议提交 Git
├── experiment_record.md          # 实验说明记录，建议提交 Git
└── show_results.sh               # 查询结果脚本，建议提交 Git
```

## `.gitignore` 规则

本地和服务器应保持同一份 `.gitignore`，确保数据集、权重、日志不进入 Git：

```gitignore
# datasets
data/
data_full/
data_small/
datasets/

# training outputs
checkpoints/
checkpoints_archive/
logs/
runs/
wandb/
results/

# model weights
*.pth
*.pt
*.ckpt

# compressed files
*.zip
*.tar
*.tar.gz

# python cache
__pycache__/
*.pyc
.ipynb_checkpoints/

# system
.DS_Store
```

可以提交到 GitHub 的主要是：

```text
代码文件
.gitignore
experiment_results.csv
experiment_record.md
show_results.sh
README.md
```

不要提交：

```text
data/
logs/
wandb/
checkpoints/
*.pth
*.zip
```

## 实验记录体系

### `experiment_results.csv`

用于快速查询每次实验的关键结果。首次初始化：

```bash
cd /root/autodl-tmp/projects/Pytorch-UNet

if [ ! -f experiment_results.csv ]; then
cat > experiment_results.csv << 'EOF'
exp_name,model,dataset,epochs,batch_size,lr,scale,amp,final_dice,log_file,remark
EOF
fi
```

### `experiment_record.md`

用于记录实验说明、观察和后续报告素材。首次初始化：

```bash
if [ ! -f experiment_record.md ]; then
cat > experiment_record.md << 'EOF'
## 实验记录

### 说明

本文件用于记录 U-Net 复现、baseline 和对比实验的关键配置与结果。

EOF
fi
```

`experiment_results.csv` 适合机器查询；`experiment_record.md` 适合写实验说明和总结。

## 训练命名规则

每次实验先设置实验名：

```bash
EXP_NAME=exp010_baseline_small49_e1_bs1_lr1e-5_scale0.5_amp
```

命名规则建议：

```text
实验编号_实验类型_数据集规模_epoch_batchsize_lr_scale_amp
```

示例：

```text
exp010_baseline_carvana500_e5_bs2_lr1e-5_scale0.5_amp
exp020_lr1e-4_carvana500_e5_bs2_scale0.5_amp
exp030_bs4_carvana500_e5_lr1e-5_scale0.5_amp
```

## 标准训练命令模板

正式训练建议使用下面模板：

```bash
cd /root/autodl-tmp/projects/Pytorch-UNet

EXP_NAME=exp010_baseline_small49_e1_bs1_lr1e-5_scale0.5_amp

mkdir -p logs
mkdir -p checkpoints_archive/${EXP_NAME}

export WANDB_MODE=offline

python train.py \
  --epochs 1 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --scale 0.5 \
  --amp \
  2>&1 | tee logs/${EXP_NAME}.log

FINAL_DICE=$(grep "Validation Dice score" logs/${EXP_NAME}.log | tail -1 | awk -F': ' '{print $NF}')

echo "${EXP_NAME},U-Net,Carvana-small49,1,1,1e-5,0.5,yes,${FINAL_DICE},logs/${EXP_NAME}.log,small test" >> experiment_results.csv

cp checkpoints/*.pth checkpoints_archive/${EXP_NAME}/ 2>/dev/null || true
```

这段命令完成：

| 步骤 | 作用 |
| --- | --- |
| `tee logs/${EXP_NAME}.log` | 保存完整训练日志 |
| `grep ... tail -1` | 提取最后一次 Dice |
| `echo ... >> experiment_results.csv` | 写入实验结果总表 |
| `cp checkpoints/*.pth` | 备份本次 checkpoint |

## 快速查看训练结果

查看结果总表：

```bash
column -s, -t experiment_results.csv
```

如果没有 `column`：

```bash
cat experiment_results.csv
```

查看某次实验完整日志：

```bash
less logs/实验名.log
```

查看某次实验最后 Dice：

```bash
grep "Validation Dice score" logs/实验名.log | tail -1
```

创建查询脚本：

```bash
cat > show_results.sh << 'EOF'
#!/bin/bash

echo "===== experiment_results.csv ====="
if [ -f experiment_results.csv ]; then
    column -s, -t experiment_results.csv 2>/dev/null || cat experiment_results.csv
else
    echo "未找到 experiment_results.csv"
fi

echo
echo "===== logs final dice ====="
for f in logs/*.log; do
    [ -e "$f" ] || continue
    final_dice=$(grep "Validation Dice score" "$f" | tail -1 | awk -F': ' '{print $NF}')
    echo "$(basename "$f") : ${final_dice}"
done
EOF

chmod +x show_results.sh
```

以后查看：

```bash
./show_results.sh
```

## 训练记录提交规则

建议提交：

```text
experiment_results.csv
experiment_record.md
show_results.sh
```

不建议提交：

```text
logs/
checkpoints/
checkpoints_archive/
wandb/
data/
```

## Git 工作流总原则

长期原则：

```text
本地负责改代码
服务器负责跑实验
GitHub 负责同步
```

完整链路：

```text
本地 + Codex 修改代码
        ↓
本地 git push
        ↓
服务器 git pull
        ↓
服务器训练
        ↓
服务器更新 experiment_results.csv / experiment_record.md
        ↓
服务器 git push
        ↓
本地 git pull
```

## 本地端流程：Codex 修改代码后

修改代码前，先拉取最新版本：

```bash
git pull --ff-only
```

如果刚刚清空过历史或本地第一次同步，确认不需要保留本地改动后再使用：

```bash
git fetch origin
git reset --hard origin/master
```

以后正常使用：

```bash
git pull --ff-only
```

Codex 修改本地文件，例如：

```text
train.py
evaluate.py
utils/
unet/
```

修改完成后，本地检查：

```bash
git status
git diff
```

本地提交代码：

```bash
git add .
git commit -m "修改训练代码：补充某某功能"
git push
```

注意：本地不要提交数据、日志、权重。

## 服务器端流程：拉取最新代码并训练

服务器拉取本地推送的新代码：

```bash
cd /root/autodl-tmp/projects/Pytorch-UNet
git status
git pull --ff-only
```

如果 `git status` 显示有未提交修改，不要直接 pull，先判断这些修改是什么。

每次代码改动后，先跑小样本回归测试：

```bash
EXP_NAME=exp_test_after_code_update_e1_bs1_lr1e-5_scale0.5_amp

mkdir -p logs
mkdir -p checkpoints_archive/${EXP_NAME}

export WANDB_MODE=offline

python train.py \
  --epochs 1 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --scale 0.5 \
  --amp \
  2>&1 | tee logs/${EXP_NAME}.log

FINAL_DICE=$(grep "Validation Dice score" logs/${EXP_NAME}.log | tail -1 | awk -F': ' '{print $NF}')

echo "${EXP_NAME},U-Net,Carvana-small49,1,1,1e-5,0.5,yes,${FINAL_DICE},logs/${EXP_NAME}.log,code update regression test" >> experiment_results.csv

cp checkpoints/*.pth checkpoints_archive/${EXP_NAME}/ 2>/dev/null || true
```

如果能看到下面信息，说明代码还能跑：

```text
INFO: Using device cuda
INFO: Creating dataset with 49 examples
INFO: Checkpoint 1 saved!
```

训练完成后，服务器只提交结果表和记录文件：

```bash
git status
git add experiment_results.csv experiment_record.md show_results.sh .gitignore
git commit -m "记录代码更新后的训练测试结果"
git push
```

如果没有修改 `experiment_record.md` 或 `show_results.sh`，只提交：

```bash
git add experiment_results.csv
git commit -m "记录训练实验结果"
git push
```

## 本地同步服务器实验结果

服务器推送结果后，本地执行：

```bash
git pull --ff-only
```

这样本地就能看到最新的：

```text
experiment_results.csv
experiment_record.md
```

Codex 下次读取本地代码时，也能看到实验结果记录。

## 完整流程示例

### 阶段 A：本地 Codex 改代码

本地执行：

```bash
cd 本地/Pytorch-UNet
git pull --ff-only
```

Codex 修改代码后：

```bash
git status
git add .
git commit -m "新增 IoU 指标计算"
git push
```

### 阶段 B：服务器拉取并测试

服务器执行：

```bash
cd /root/autodl-tmp/projects/Pytorch-UNet
git pull --ff-only
```

跑小样本测试：

```bash
EXP_NAME=exp011_iou_update_small49_e1_bs1_lr1e-5_scale0.5_amp

mkdir -p logs
mkdir -p checkpoints_archive/${EXP_NAME}

export WANDB_MODE=offline

python train.py \
  --epochs 1 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --scale 0.5 \
  --amp \
  2>&1 | tee logs/${EXP_NAME}.log

FINAL_DICE=$(grep "Validation Dice score" logs/${EXP_NAME}.log | tail -1 | awk -F': ' '{print $NF}')

echo "${EXP_NAME},U-Net,Carvana-small49,1,1,1e-5,0.5,yes,${FINAL_DICE},logs/${EXP_NAME}.log,IoU code update small test" >> experiment_results.csv
```

提交实验结果：

```bash
git add experiment_results.csv
git commit -m "记录 IoU 修改后的小样本测试结果"
git push
```

### 阶段 C：本地同步结果

本地执行：

```bash
git pull --ff-only
```

然后查看：

```bash
column -s, -t experiment_results.csv
```

## 避免 Git 冲突的规则

同一时间只在一个地方改代码：

| 任务 | 修改位置 |
| --- | --- |
| 改代码 | 本地 + Codex |
| 跑训练 | 服务器 |
| 记录实验结果 | 服务器 |
| 查看和整理 | 本地同步后查看 |

本地改代码前先 pull：

```bash
git pull --ff-only
```

服务器跑实验前先 pull：

```bash
git pull --ff-only
```

不要同时在本地和服务器改同一个文件，尤其是：

```text
train.py
evaluate.py
experiment_results.csv
experiment_record.md
```

如果服务器刚刚更新了实验记录，本地一定要先 pull，再让 Codex 修改。

## 初始化步骤

服务器创建记录文件和查询脚本：

```bash
cd /root/autodl-tmp/projects/Pytorch-UNet

if [ ! -f experiment_results.csv ]; then
cat > experiment_results.csv << 'EOF'
exp_name,model,dataset,epochs,batch_size,lr,scale,amp,final_dice,log_file,remark
EOF
fi

if [ ! -f experiment_record.md ]; then
cat > experiment_record.md << 'EOF'
## 实验记录

### 说明

本文件用于记录 U-Net 复现、baseline 和对比实验的关键配置与结果。

EOF
fi

cat > show_results.sh << 'EOF'
#!/bin/bash

echo "===== experiment_results.csv ====="
if [ -f experiment_results.csv ]; then
    column -s, -t experiment_results.csv 2>/dev/null || cat experiment_results.csv
else
    echo "未找到 experiment_results.csv"
fi

echo
echo "===== logs final dice ====="
for f in logs/*.log; do
    [ -e "$f" ] || continue
    final_dice=$(grep "Validation Dice score" "$f" | tail -1 | awk -F': ' '{print $NF}')
    echo "$(basename "$f") : ${final_dice}"
done
EOF

chmod +x show_results.sh
```

提交初始化文件：

```bash
git status
git add .gitignore experiment_results.csv experiment_record.md show_results.sh
git commit -m "初始化实验记录与结果查询工具"
git push
```

本地同步：

```bash
git pull --ff-only
```

如果本地历史刚清理过或不一致，确认不需要保留本地改动后再使用：

```bash
git fetch origin
git reset --hard origin/master
```

## 最终固定工作流

以后固定使用这条线：

```text
本地 Codex 改代码
    ↓
本地 git add / commit / push
    ↓
服务器 git pull --ff-only
    ↓
服务器小样本测试
    ↓
服务器正式训练
    ↓
服务器写 experiment_results.csv
    ↓
服务器 git commit / push
    ↓
本地 git pull --ff-only
    ↓
Codex 继续基于最新代码修改
```

这套流程打通后，后续做 baseline、学习率对比、batch size 对比、loss 对比、指标扩展，都应围绕这套记录体系推进。
