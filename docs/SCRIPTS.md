# 脚本用途与依赖

日常从根目录run.py选择实验，或使用图片Demo。主入口见[五项实验](EXPERIMENT_NAMES.md)。

## 正式实验必需

| scripts中的脚本 | 用途 |
|---|---|
| experiment1_inference_sizes.py | 实验一入口，先训练一个模型再比较两种推理尺寸 |
| experiment2_training_sizes.py | 实验二入口，分别训练两种尺寸模型 |
| experiment3_mosaic.py | 实验三开关Mosaic训练、评估和对照 |
| experiment4_robustness.py | 实验四三次新训练、清晰与退化评估 |
| experiment5_blur_ratios.py | 实验五三比例乘两个种子的完整流程 |
| project_config.py | 配置加载、路径解析与主dataset.yaml生成 |
| size_experiment_common.py | 实验一、二的公共实现及实验五使用的预检工具 |
| isolated_training.py | 独立进程训练，隔离各组初始化和随机状态 |
| live_process.py | 实时输出、日志和子进程错误处理 |
| compare_mosaic_predictions.py | 实验三调用的逐图比较模块 |
| summarize_experiment4_robustness.py | 实验四结果汇总；统一入口自动传入新目录 |

## 演示、资源与质量检查

| 脚本 | 用途 |
|---|---|
| demo_wildlife8.py | 已有权重图片预测；仅图片、GPU 0，不计算mAP |
| import_resources.py | 校验并导入附件，兼容旧目录名 |
| check_wildlife8.py | 数据及参考基线检查；--data-only只检查数据 |
| configure_package_paths.py | 移动项目后按当前配置生成主数据路径 |
| audit_submission.py | 语法、文档链接、实验索引检查，GitHub工作流会调用 |
| test_blur_ratio_workflow.py | 实验五模拟流程检查 |
| test_isolated_training.py | 独立训练进程及错误传播检查 |
| test_live_process.py | 实时输出与日志检查 |
| test_mosaic_recovery.py | 实验三已完成训练复用与恢复检查 |
| test_package_paths.py | 移动目录检查 |
| test_project_config.py | 配置与参数传递检查 |
| test_resource_import.py | 资源导入安全及路径兼容检查，GitHub工作流会调用 |
| test_size_experiment_workflow.py | 两项尺寸流程及模型对应关系检查 |

test_开头的是软件测试，用于软件验证。检查命令：

```powershell
& $py -X utf8 scripts/audit_submission.py
& $py -X utf8 -m unittest discover -s scripts -p 'test_*.py'
```

## 可选工具与历史工具

| 脚本 | 用途与当前限制 |
|---|---|
| compare_inference_sizes.py | 只评估指定已有权重的416/640推理，不训练；与实验一完整流程不同 |
| reproduce_myself.py | 个人单模型微调，输出my_runs，不是五项统一入口 |
| run_wildlife8_controls.py | 旧640与关闭Mosaic对照；正式实验已由二、三替代 |
| summarize_wildlife8_controls.py | 汇总旧controls，使用旧输出材料路径 |
| analyze_wildlife8_failures.py | 重新生成旧基线失败说明，使用旧输出材料路径 |
| summarize_generalization.py | 汇总额外图片和阈值实验；依赖generalization数据，提交版未包含这些输入 |

历史工具独立于五项正式实验流程。公共模块和实验三、四的分析脚本属于运行依赖；文件增删后需同步文档及submission_manifest.json。
