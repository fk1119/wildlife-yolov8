# 四分钟演示

## 环境与输入

资源导入后，在仓库根目录使用已安装的解释器：

```powershell
$py = '.\.venv\Scripts\python.exe'
$image = 'wildlife8/data/images/test/coco_val2017_000000286849.jpg'
& $py -X utf8 scripts/demo_wildlife8.py --weights wildlife8/runs/train/weights/best.pt --source $image --imgsz 416 --conf 0.25 --iou 0.7
```

演示使用附件中原416基线，输出位置以终端打印路径为准。该演示不训练模型，也不重新计算完整测试指标。

| 时间 | 内容 |
|---|---|
| 0:00–0:30 | 展示原图、类别和模型权重 |
| 0:30–2:00 | 执行推理，查看终端结果 |
| 2:00–3:00 | 打开预测图片，说明类别、检测框和置信度 |
| 3:00–4:00 | 对照失败案例与完整测试指标 |

长颈鹿案例有4个标注框和4个预测框，但只有1个满足同类IoU≥0.5。预测数量相同不代表定位正确。离线预测与标注图位于[失败案例](failures/03_真实失败分析.md)。
