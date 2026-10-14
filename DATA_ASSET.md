# 数据与权重附件

## 什么时候需要附件

- 本机已经有 `wildlife8/data` 和模型时，不需要为了运行而重复下载。
- 仅预测自己的图片：安装环境并使用已有 `best.pt`，不需要完整数据集。
- 新电脑训练或完整测试集评估：需要2496张原图及对应标注。源码下载不保证包含被Git忽略的数据。

## 获取和导入

可检查[仓库Releases页面](https://github.com/fk1119/wildlife-yolov8/releases)是否有发布者提供的 `wildlife8-data-models.zip`。附件发布状态待核验；缺少附件时由仓库维护者提供。导入命令使用资源包的实际存放路径。

历史资源包记录为605,011,167字节，SHA256为 `b386378592918c07f69a0320d36067ed1282719ab2a87d0bb884be789d8e48fa`。该哈希用于识别历史数据包，源码ZIP不包含完整数据。

定义当前窗口的 `$py` 后，在项目根目录执行：

```powershell
& $py -X utf8 scripts/import_resources.py --archive '实际资源包ZIP路径'
& $py -X utf8 scripts/check_wildlife8.py --data-only
```

[资源清单](resources_manifest.json)中的 `path` 为当前安装位置，`archive_path`（如有）为兼容旧附件的内部路径。导入器先查新路径，未找到再查旧路径，按大小和SHA256验证；已有相同文件跳过，已有不同文件拒绝覆盖。当前清单不再导入已清理的旧 `robustness` 模型目录。

重命名后的五项结果位置见[实验入口说明](docs/EXPERIMENT_NAMES.md)。人工退化和增强数据在新运行时生成，不要求预先保存数GB中间图。来源和许可见[来源说明](CODE_ORIGIN.md)。
