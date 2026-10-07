# YOLOv8n 八类野生动物检测

基于Ultralytics YOLOv8n与COCO预训练权重，检测水牛、大象、犀牛、斑马、熊、长颈鹿、狮子和鹿，输出类别、边界框和置信度。项目比较输入尺寸、Mosaic和模糊增强对检测性能的影响。

## 实验结果

五项实验均已完成。实验五采用2026年10月7日修复后完成的运行 `20261007_104229_335871`，包含0%、25%、50%三个比例、两个种子，共六个30轮模型。

| 实验 | 对照 | mAP50–95 | 变化（百分点） |
|---|---|---|---|
| 1 同模型推理尺寸 | 416 → 640 | 75.21% → 57.66% | -17.54 |
| 2 训练测试尺寸 | 416/416 → 640/640 | 75.21% → 75.25% | +0.05 |
| 3 Mosaic | 关闭 → 开启 | 73.61% → 75.21% | +1.60 |
| 4 模糊增强迁移 | 基线 → 50%  模糊测试 | 64.80% → 71.62% | +6.82 |
| 5 模糊增强比例 | 0% → 25%  模糊测试 | 64.80% → 71.49% | +6.69 |

实验四、五表内为模糊测试条件；完整结果见[结果汇总](docs/RESULTS.md)。差值由未舍入数据计算。前3项为单种子，后2项为种子42、43的均值。

## 数据与方法

固定划分为训练1753张、验证383张、测试360张，数据来自African Wildlife、COCO和Open Images。训练集更新模型参数，验证集选择权重及增强方案，测试集报告结果。主指标mAP50–95综合类别和定位表现。

## 运行

参考环境：Windows、Python 3.12、PyTorch 2.5.1+cu121、Ultralytics 8.4.147、NVIDIA GPU（原实验为RTX 3060 Laptop）。训练与当前Demo默认使用GPU 0。先从仓库 **Releases** 下载本次数据权重附件，详情见 [附件说明](DATA_ASSET.md)。仅阅读结果不需要安装环境。


Windows PowerShell，在项目根目录执行：

```powershell
python -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install -r requirements-course.txt
& $py -X utf8 scripts/import_resources.py --archive '..\wildlife8-data-models.zip'
& $py run.py mosaic --check-only
& $py -X utf8 scripts/demo_wildlife8.py
```

各实验入口统一为 `python run.py <实验名>`，参数与路径在[config.json](config.json)配置。详细命令见[实验设计](docs/EXPERIMENTS.md)和[复现步骤](docs/REPRODUCE.md)。

## 文档

- [实验状态与证据索引](experiments.json)
- [实验结果](docs/RESULTS.md)与[失败案例](docs/failures/03_真实失败分析.md)
- [汇报PPT](presentation/五项实验汇报.pptx)与[实验报告](docs/五项实验报告.docx)
- [讲稿](docs/TALK.md)与[Demo](docs/DEMO.md)
- [数据附件](DATA_ASSET.md)、[来源](CODE_ORIGIN.md)
- [运行检查](docs/VALIDATION_CURRENT.md)

## 结论与限制

同一416模型直接改为640推理后，测试mAP下降；匹配训练和测试的640方案与416方案总体接近。Mosaic对不同类别的影响不一致。模糊增强提高模糊图表现，但50%组降低清晰和JPEG压缩图表现。实验五按预设验证规则选择50%比例。

测试集此前已查看，人工退化不代表全部真实环境；有限种子、数据来源差异、潜在漏标和预训练重叠限制了结论范围。

上游实现：[Ultralytics](https://github.com/ultralytics/ultralytics)。网络、训练器、损失及标准AP评估来自上游，许可证见[UPSTREAM_LICENSE.txt](UPSTREAM_LICENSE.txt)。
