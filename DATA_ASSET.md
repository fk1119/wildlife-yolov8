# 数据与权重附件

下载方法：进入本仓库的 **Releases** 页面，下载 `wildlife8-data-models.zip`，放在项目文件夹上一级。发布者需先上传本次生成的同名附件；普通 Code → Download ZIP 只包含源码与材料。

本次附件：605,011,167 字节；SHA256：`b386378592918c07f69a0320d36067ed1282719ab2a87d0bb884be789d8e48fa`。

包含固定2496张原图及标签、COCO初始权重、原参考权重，以及本次五项正式实验的best.pt（新增14份）。逐文件路径、大小和SHA256见 [resources_manifest.json](resources_manifest.json)。

```powershell
python -X utf8 scripts/import_resources.py --archive '..\wildlife8-data-models.zip'
python -X utf8 scripts/check_wildlife8.py --data-only
```

数据来源与许可说明见 [CODE_ORIGIN.md](CODE_ORIGIN.md)，原图来源记录见 wildlife8/source_metadata.json。


当前源码仓库已包含参考及五项正式实验的测试权重。仅预测自己的图片无需下载本附件；完整数据集评估或重新训练时才需要其中的数据。附件仍保留相同权重，导入器会校验并跳过已有相同文件。
