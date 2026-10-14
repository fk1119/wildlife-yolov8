# 安装与复现

运行要求：Python环境、模型权重；训练与完整评估另需数据集。

## 本地环境

```powershell
Set-Location -LiteralPath '你的项目路径'
$py = '你的python环境'
```

数据检查、实验预检和图片演示：

```powershell
& $py -X utf8 scripts/check_wildlife8.py --data-only
& $py -X utf8 run.py experiment3_mosaic --check-only
& $py -X utf8 scripts/demo_wildlife8.py
```

## 新电脑

解压或克隆后，在项目根目录打开PowerShell。以下依赖面向Windows、Python 3.12和NVIDIA CUDA环境：

```powershell
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements-course.txt
```

`environment-lock.txt` 是原机器环境记录，不是通用安装文件。安装依赖不会自动下载数据。当前训练不支持CPU，Demo也固定使用GPU 0。

预测自定义图片时，已有参考权重即可，见[Demo](DEMO.md)。若要训练或重算完整测试指标，先取得[数据附件](../DATA_ASSET.md)，将实际附件路径传入：

```powershell
& $py -X utf8 scripts/import_resources.py --archive '实际附件ZIP路径'
& $py -X utf8 scripts/check_wildlife8.py --data-only
```

导入器按 `resources_manifest.json` 的目标路径和SHA256校验，支持 `archive_path` 对应的旧包路径。相同文件跳过，不同文件拒绝覆盖。生成的数据配置位于 `wildlife8/dataset.yaml`。

## 五个入口预检

先定义当前窗口的 `$py`。以下只检查输入、数据和环境，不训练、不重新评估：

```powershell
& $py -X utf8 run.py experiment1_inference_sizes --check-only
& $py -X utf8 run.py experiment2_training_sizes --check-only
& $py -X utf8 run.py experiment3_mosaic --check-only
& $py -X utf8 run.py experiment4_robustness --check-only
& $py -X utf8 run.py experiment5_blur_ratios --check-only
```

删去 `--check-only` 就会新建运行并开展训练。模型数、时间戳输出及已有结果位置见[实验设计](EXPERIMENTS.md)。训练任务串行执行。

## 移动目录与中断处理

默认配置使用相对路径，主数据配置由正式入口生成。移动后也可执行 `& $py -X utf8 scripts/configure_package_paths.py`。历史JSON/YAML中的绝对路径是原始证据，不会批量改写。移动电脑或数据后新建实验，不按旧路径续训。

统一入口仅实验三提供 `--resume-run`；它可复用已完成训练并补评估，不等于从任意中断epoch继续训练。详见[统一配置](统一配置与运行.md)。

## 查看与复现的区别

查看报告不需运行环境；图片Demo不计算mAP；五项实验入口会训练。若仅重算已有模型完整测试mAP，可使用安装的Ultralytics评估API，指定已有权重和本项目数据配置，Demo仅输出预测框及数量。五项统一入口均包含训练步骤。
