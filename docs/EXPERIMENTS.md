# 五项实验设计

环境与资源配置见[复现步骤](REPRODUCE.md)。所有实验均已完成，以下命令用于新建运行。

## 1 推理尺寸对照

固定416训练模型，以416和640分别评估完整测试集。

结果目录：`wildlife8/experiment1_inference_sizes/20261006_162734_758962`。

```powershell
& $py run.py inference_sizes
```

## 2 训练尺寸对照

从相同COCO初始权重分别训练416和640模型，并按对应尺寸评估。

结果目录：`wildlife8/experiment2_training_sizes/20261006_165841_934483`。

```powershell
& $py run.py training_sizes
```

## 3 Mosaic增强对照

保持416、30轮、batch 8、seed 42，比较Mosaic开启和关闭。

结果目录：`wildlife8/mosaic_experiments/20261006_180013_955548`。

```powershell
& $py run.py mosaic
```

## 4 鲁棒性与模糊增强

验证集选择模糊作为目标退化，训练集一半图像施加固定高斯模糊；比较两个种子下清晰及九种退化测试条件。seed42基线为已有模型，另训练三个模型。

结果目录：`wildlife8/robustness_rerun_20261006_210326_635336`。

```powershell
& $py run.py robustness
```

## 5 模糊比例对照

比较0%、25%、50%固定模糊比例，每组种子42和43，各30轮。25%子集包含在50%子集内，共享样本模糊强度一致。所有模型采用独立解释器训练。验证集按清晰和模糊等权规则选中50%。

结果目录：`wildlife8/experiment3_blur_ratios/20261007_104229_335871`。

```powershell
& $py run.py blur_ratios
```

训练集用于更新参数，原清晰验证集用于选择best.pt。正式指标采用完整测试集；实验五比例选择不使用测试结果。
