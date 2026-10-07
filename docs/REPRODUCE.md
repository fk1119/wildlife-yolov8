当前推荐使用[统一配置与运行](统一配置与运行.md)：无需另行维护dataset.yaml，统一入口会自动按config.json生成。下面保留旧脚本调用方式供兼容参考。

# 环境、数据与复现

## 环境

参考环境为Python 3.12、PyTorch 2.5.1+cu121、Ultralytics 8.4.147、RTX 3060 Laptop GPU。requirements-course.txt固定主要依赖；environment-lock.txt为原机环境记录，并非其他系统的通用锁文件。

换电脑先按README安装环境。每名成员在自己的项目根目录使用本项目虚拟环境：

```powershell
$py = '.\.venv\Scripts\python.exe'
```

完整步骤见 [小组成员独立运行](小组成员独立运行.md)。移动项目后重新配置主数据路径并创建新实验；历史实验配置保留原路径，不支持自动跨电脑续训。

## 获取资源

附件从仓库发布者提供的GitHub Release或数据下载地址获取；当前下载链接尚待发布者提供，见 [DATA_ASSET.md](../DATA_ASSET.md)。本地也可从此前完整备份ZIP导入，不需重新下载。

```powershell
& $py -X utf8 scripts/import_resources.py --archive '实际附件ZIP路径'
& $py -X utf8 scripts/check_wildlife8.py --data-only
& $py -X utf8 scripts/experiment1_inference_sizes.py --check-only
& $py -X utf8 scripts/experiment2_training_sizes.py --check-only
& $py -X utf8 scripts/run_mosaic_experiment.py --check-only
& $py -X utf8 scripts/experiment3_blur_ratios.py --check-only
```

导入器按resources_manifest.json逐文件SHA256验证，仅导入清单内文件；已有且相同的文件跳过，已有但不同的文件拒绝覆盖。它不解压任意路径、不修改历史指标或原数据来源清单。导入完成会生成本目录的dataset.yaml。移动整个项目后执行 `scripts/configure_package_paths.py` 更新路径。

## 保存与复现的区别

仓库包含已有小体积结果证据，附件包含原图、标注和best.pt等权重。数GB增强图不重复分发，重新运行时由原图与固定协议生成。五项已完成结果由experiments.json索引，新运行写入独立目录。

原鲁棒性实验复用已有seed42基线，另训三个模型；新推理实验会重训一个416模型；模糊比例会独立训练六个模型。详见 [EXPERIMENTS.md](EXPERIMENTS.md)。所有训练逐项执行，避免同时争用GPU。中断的运行保留状态与日志。

## 如何评价

训练集更新参数，验证集选择checkpoint和规定方案，测试集报告结果。P表示预测中正确的比例，R表示标注目标被找到的比例；mAP50–95综合多个IoU门槛下的AP。mAP不是逐图答对率。演示conf=0.25与正式AP评估阈值不同，不混用演示框数与测试指标。
