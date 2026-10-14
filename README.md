# YOLOv8n 八类野生动物检测

输入动物图片，输出水牛、大象、犀牛、斑马、熊、长颈鹿、狮子和鹿的类别、检测框与置信度。基于COCO预训练的YOLOv8n微调，比较尺寸、Mosaic及模糊增强。Demo支持图片检测。

## 本地运行

在PowerShell中设置项目目录和Python解释器：

```powershell
Set-Location -LiteralPath '你的项目路径'
$py = '你的python环境'
```

```powershell
# 已有模型预测演示图片，不训练
& $py -X utf8 scripts/demo_wildlife8.py

# 仅检查实验三环境，不启动训练
& $py -X utf8 run.py experiment3_mosaic --check-only
```

演示默认使用 `wildlife8/runs/train/weights/best.pt` 和 `wildlife8/demo_images`，在 `wildlife8/runs/demo*` 保存带框图片及 `counts.csv`。实际位置以终端输出为准。自定义图片、模型选择和演示步骤见[图片Demo](docs/DEMO.md)。仅预测图片不需要完整训练数据。

## 重新训练五项实验

五项实验均包含训练步骤，按下表逐项运行。添加 `--check-only` 可检查环境和输入。

| 实验 | scripts中的主脚本 | 新训练模型数 | 命令 |
|---|---|---:|---|
| 1 推理尺寸对照 | `experiment1_inference_sizes.py` | `1` | `& $py -X utf8 run.py experiment1_inference_sizes` |
| 2 训练尺寸对照 | `experiment2_training_sizes.py` | `2` | `& $py -X utf8 run.py experiment2_training_sizes` |
| 3 Mosaic增强对照 | `experiment3_mosaic.py` | `2` | `& $py -X utf8 run.py experiment3_mosaic` |
| 4 鲁棒性与模糊增强 | `experiment4_robustness.py` | `3` | `& $py -X utf8 run.py experiment4_robustness` |
| 5 模糊比例对照 | `experiment5_blur_ratios.py` | `6` | `& $py -X utf8 run.py experiment5_blur_ratios` |

实验四复用已有seed42基线，另训练三个模型。所有模型均从COCO预训练权重开始微调。新结果进入 `wildlife8/对应实验名/时间戳/`。

[五项实验设计](docs/EXPERIMENTS.md) · [脚本与正式结果位置](docs/EXPERIMENT_NAMES.md) · [配置和恢复](docs/统一配置与运行.md)

## 新电脑或重新下载项目

需要Python 3.12、兼容的NVIDIA GPU与驱动。参考环境是PyTorch 2.5.1+cu121、Ultralytics 8.4.147、RTX 3060 Laptop。当前训练要求CUDA，Demo固定GPU 0。

新电脑的环境安装步骤见[安装与复现](docs/REPRODUCE.md)，解释器路径以实际安装位置为准。已有模型可直接用于图片Demo；训练或完整测试集评估需要完整数据，获取方式和旧附件路径兼容见[数据附件](DATA_ASSET.md)。

## 实验结果
五项实验均已完成。实验五包含0%、25%、50%三个比例、两个种子，共六个30轮模型。

| 实验 | 对照 | mAP50–95 | 变化（百分点） |
|---|---|---|---|
| 1 同模型推理尺寸 | 416 → 640 | 75.21% → 57.66% | -17.54 |
| 2 训练测试尺寸 | 416/416 → 640/640 | 75.21% → 75.25% | +0.05 |
| 3 Mosaic | 关闭 → 开启 | 73.61% → 75.21% | +1.60 |
| 4 模糊增强迁移 | 基线 → 50%  模糊测试 | 64.80% → 71.62% | +6.82 |
| 5 模糊增强比例 | 0% → 25%  模糊测试 | 64.80% → 71.49% | +6.69 |

实验四、五表内为模糊测试条件；完整结果见[结果汇总](docs/RESULTS.md)。差值由未舍入数据计算。前3项为单种子，后2项为种子42、43的均值。

## 数据与结论边界

固定训练1753张、验证383张、测试360张，来源为African Wildlife、COCO和Open Images。验证集选择权重及预设方案，测试集报告结果。mAP50–95不是图片分类准确率。

同一416模型改640推理后测试mAP下降；匹配640训练和测试的总体表现与416接近；Mosaic有类别差异；50%模糊增强提高模糊表现，但降低清晰和JPEG表现。实验五按验证规则选中50%；25%的测试表现用于分析性能取舍。

## 文档导航

- [脚本用途与可选旧工具](docs/SCRIPTS.md)
- [六分钟讲稿](presentation/TALK.md)、[PPT](presentation/五项实验汇报.pptx)、[Word报告](docs/五项实验报告.docx)
- [成员贡献](CONTRIBUTIONS.md)
- [来源与贡献边界](CODE_ORIGIN.md)

核心网络、训练器、损失和AP评估来自[Ultralytics](https://github.com/ultralytics/ultralytics)。代码来源与开发方式见[来源说明](CODE_ORIGIN.md)。上游许可见[许可证](UPSTREAM_LICENSE.txt)。

