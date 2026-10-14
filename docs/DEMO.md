# 图片检测与演示

`scripts/demo_wildlife8.py` 接受本地单张图片或图片文件夹。视频、摄像头和目标跟踪尚未实现。固定使用GPU 0，不读取config.json中的设备字段，也没有 `--device` 参数。

## 本机快速演示

```powershell
Set-Location -LiteralPath '你的项目路径'
$py = '你的python环境'
```

```powershell
& $py -X utf8 scripts/demo_wildlife8.py
```

默认读取 `wildlife8/demo_images`，使用 `wildlife8/runs/train/weights/best.pt`。无需先导入完整数据集。输出是 `wildlife8/runs/demo*` 中的带框图和 `counts.csv`，以终端路径为准。CSV为每图预测框数量，不是去重个体数量。

## 自己的图片和其他模型

```powershell
& $py -X utf8 scripts/demo_wildlife8.py --source '图片或图片文件夹路径' --weights wildlife8/runs/train/weights/best.pt --imgsz 416 --conf 0.25 --iou 0.7
```

支持jpg、jpeg、png、bmp，文件夹只扫描第一层。`--conf` 和 `--iou` 必须在0至1之间。640训练模型使用 `--imgsz 640`；其他模型位置见[实验入口](EXPERIMENT_NAMES.md)。本脚本不训练，也不输出测试集mAP。
