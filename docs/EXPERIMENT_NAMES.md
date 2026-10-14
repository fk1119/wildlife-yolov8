# 五项实验名称与运行入口

以下路径均相对项目根目录。主脚本位于scripts中。

## 实验1 推理尺寸对照

- 主脚本：[ experiment1_inference_sizes.py ](../scripts/experiment1_inference_sizes.py)
- 正式结论：[打开结论](../wildlife8/experiment1_inference_sizes/20261006_162734_758962/实验对照.md)
- 已有结果：`wildlife8/experiment1_inference_sizes/20261006_162734_758962`
- 新建运行：`& $py -X utf8 run.py experiment1_inference_sizes`

## 实验2 训练尺寸对照

- 主脚本：[ experiment2_training_sizes.py ](../scripts/experiment2_training_sizes.py)
- 正式结论：[打开结论](../wildlife8/experiment2_training_sizes/20261006_165841_934483/实验对照.md)
- 已有结果：`wildlife8/experiment2_training_sizes/20261006_165841_934483`
- 新建运行：`& $py -X utf8 run.py experiment2_training_sizes`

## 实验3 Mosaic增强对照

- 主脚本：[ experiment3_mosaic.py ](../scripts/experiment3_mosaic.py)
- 正式结论：[打开结论](../wildlife8/experiment3_mosaic/20261006_180013_955548/comparison/实验分析.md)
- 已有结果：`wildlife8/experiment3_mosaic/20261006_180013_955548`
- 新建运行：`& $py -X utf8 run.py experiment3_mosaic`

## 实验4 鲁棒性与模糊增强

- 主脚本：[ experiment4_robustness.py ](../scripts/experiment4_robustness.py)
- 正式结论：[打开结论](../wildlife8/experiment4_robustness/实验报告.md)
- 已有结果：`wildlife8/experiment4_robustness`
- 新建运行：`& $py -X utf8 run.py experiment4_robustness`

## 实验5 模糊比例对照

- 主脚本：[ experiment5_blur_ratios.py ](../scripts/experiment5_blur_ratios.py)
- 正式结论：[打开结论](../wildlife8/experiment5_blur_ratios/20261007_104229_335871/实验对照.md)
- 已有结果：`wildlife8/experiment5_blur_ratios/20261007_104229_335871`
- 新建运行：`& $py -X utf8 run.py experiment5_blur_ratios`

`$py` 设置见[安装与复现](REPRODUCE.md)。新建命令会训练；只检查加 `--check-only`。五项新运行统一保存到对应实验目录的时间戳子目录；实验四已有正式结果直接保留在experiment4_robustness根目录。

实验四报告生成器为 `scripts/summarize_experiment4_robustness.py`。`run.py` 会自动传入新运行目录；手动汇总新结果须指定 `--output`，不指定则读取原正式结果。

历史JSON、YAML和模型内保存的旧绝对路径用于追溯，不是当前操作命令。
