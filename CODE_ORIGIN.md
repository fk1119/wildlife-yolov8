# 来源、许可与贡献边界

模型与框架来自 [Ultralytics](https://github.com/ultralytics/ultralytics)，版本8.4.147。网络结构、损失函数、训练器、预训练加载、Mosaic、NMS与标准AP评估来自上游；初始yolov8n.pt为COCO预训练权重。本项目是课程复现和对照分析。新增import_resources.py负责按清单导入本地资源，发布检查脚本负责文件与链接审查；上游许可原文：[UPSTREAM_LICENSE.txt](UPSTREAM_LICENSE.txt)。保留上游来源。

数据来源为 [African Wildlife](https://docs.ultralytics.com/datasets/detect/african-wildlife/)、[COCO](https://cocodataset.org/)、[Open Images](https://storage.googleapis.com/openimages/web/download_v7.html)。图片及标注不是本组自行采集；逐文件来源见wildlife8/source_metadata.json和manifest.json。Open Images原池被重新划分为课程子集。各图片按来源许可使用，提供附件时同时保留来源信息。

相关研究入口（背景参考，并非本项目逐项复现）：

- Vasiljevic等，2016，[Examining the Impact of Blur on Recognition by Convolutional Networks](https://arxiv.org/abs/1611.05760)。
- 2023，[Does training with blurred images bring convolutional neural networks closer to humans with respect to robust object recognition and internal representations?](https://pmc.ncbi.nlm.nih.gov/articles/PMC9975555/)。分类任务的清晰/模糊混合训练与本项目固定选图目标检测设置不同。
- CVPR 2021，[Improved Handling of Motion Blur in Online Object Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Sayed_Improved_Handling_of_Motion_Blur_in_Online_Object_Detection_CVPR_2021_paper.pdf)。其运动模糊不同于本项目高斯模糊。

这些研究支持问题背景，不证明本项目的25%或50%必然最优。成员贡献见CONTRIBUTIONS.md。
