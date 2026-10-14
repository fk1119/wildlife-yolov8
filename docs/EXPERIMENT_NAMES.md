# 五项实验名称与运行入口

| 实验 | 主脚本 | 已完成的正式结果目录 | 新建实验命令 |
|---|---|---|---|
| 1 推理尺寸对照 | `scripts/experiment1_inference_sizes.py` | `wildlife8/experiment1_inference_sizes/20261006_162734_758962` | `python run.py experiment1_inference_sizes` |
| 2 训练尺寸对照 | `scripts/experiment2_training_sizes.py` | `wildlife8/experiment2_training_sizes/20261006_165841_934483` | `python run.py experiment2_training_sizes` |
| 3 Mosaic增强对照 | `scripts/experiment3_mosaic.py` | `wildlife8/experiment3_mosaic/20261006_180013_955548` | `python run.py experiment3_mosaic` |
| 4 鲁棒性与模糊增强 | `scripts/experiment4_robustness.py` | `wildlife8/experiment4_robustness` | `python run.py experiment4_robustness` |
| 5 模糊比例对照 | `scripts/experiment5_blur_ratios.py` | `wildlife8/experiment5_blur_ratios/20261007_104229_335871` | `python run.py experiment5_blur_ratios` |

运行命令会开展实验；只检查环境时添加 `--check-only`。新运行保存到对应实验目录的时间戳子目录。实验四已有正式结果保留在 experiment4_robustness 根目录。旧短命令仍兼容。

实验四结果汇总脚本：`scripts/summarize_experiment4_robustness.py`，汇总新运行时传入 `--output` 指定其时间戳目录。

历史 JSON、CSV、YAML、日志及模型内部保存的旧路径代表当时运行位置，保留原样，不是当前运行入口。已有模型、指标和训练参数未修改。旧 ZIP 附件通过资源清单的 archive_path 字段兼容。
