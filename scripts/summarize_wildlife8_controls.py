"""Collect recorded results; never create substitute or estimated metrics."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'wildlife8/controls'
DELIVERY = ROOT/'八类野生动物输出材料/补充对照实验'
NAMES = ['基线 416', '分辨率 640', '关闭 Mosaic']
KEYS = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
CN = ['水牛', '大象', '犀牛', '斑马', '熊', '长颈鹿', '狮子', '鹿']

def load(p):
    return json.loads(p.read_text(encoding='utf8'))

def main():
    import yaml
    experiments = [('baseline416', ROOT/'wildlife8/runs/train', ROOT/'wildlife8/test_results.json')]
    for name in ['resolution640', 'no_mosaic416']:
        record = load(OUT/name/'record.json')
        assert record['state'] == 'complete', name
        assert record['manifest_sha256'] == hashlib.sha256((ROOT/'wildlife8/manifest.json').read_bytes()).hexdigest()
        experiments.append((name, OUT/name/'train', OUT/name/'test_results.json'))
    results = []
    classes = load(ROOT/'wildlife8/manifest.json')['classes']
    expected_settings = [(416, 1.0), (640, 1.0), (416, 0.0)]
    base_args = yaml.safe_load((experiments[0][1]/'args.yaml').read_text(encoding='utf8'))
    ignore = {'project', 'name', 'save_dir', 'imgsz', 'mosaic'}
    for i, (name, train, test_path) in enumerate(experiments):
        args = yaml.safe_load((train/'args.yaml').read_text(encoding='utf8'))
        assert (args['imgsz'], args['mosaic']) == expected_settings[i]
        differences = {k: [base_args.get(k), args.get(k)] for k in base_args.keys()|args.keys()
                       if k not in ignore and base_args.get(k) != args.get(k)}
        assert not differences, (name, differences)
        with (train/'results.csv').open() as f:
            rows = [{k.strip(): float(v) for k,v in row.items()} for row in csv.DictReader(f)]
        assert [int(row['epoch']) for row in rows] == list(range(1,31))
        test = load(test_path)
        assert len(test['per_class']) == 8
        assert [r['Class'] for r in test['per_class']] == classes
        status = load(ROOT/'wildlife8/training_status.json') if i == 0 else load(OUT/name/'record.json')
        if i:
            assert hashlib.sha256((train/'weights/best.pt').read_bytes()).hexdigest() == status['best_sha256']
        result = dict(id=name, name=NAMES[i], imgsz=args['imgsz'], mosaic=args['mosaic'],
                      test=test, curves=rows, train_path=str(train.relative_to(ROOT)),
                      training_seconds=status.get('training_seconds', status.get('this_call_seconds')),
                      peak_allocated_MiB=status.get('peak_allocated_MiB'),
                      peak_reserved_MiB=status.get('peak_reserved_MiB'))
        results.append(result)
    report = dict(experiments=results, configuration_check='All other saved args match baseline',
                  limitations=['One seed per configuration', 'Previously inspected test subset',
                               'No baseline peak VRAM record', 'No causal claim from individual class differences'])
    OUT.mkdir(exist_ok=True)
    (OUT/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    DELIVERY.mkdir(parents=True,exist_ok=True)
    lines = ['# 八类野生动物检测：补充对照结果', '',
             '三组在同一冻结数据上各训练 30 轮；YOLOv8n、batch=8、seed=42，均从同一 COCO 预训练权重开始。按验证集选择 best.pt，再在相同 360 张测试图上评价。', '',
             '| 组别 | P | R | mAP@50 | mAP@50–95 | 训练调用时间 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in results:
        vals = [f'{r["test"]["metrics"][k]*100:.2f}%' for k in KEYS]
        lines.append('| '+r['name']+' | '+' | '.join(vals)+f' | {r["training_seconds"]/60:.2f} 分钟 |')
    lines += ['', '## 对照回答的问题', '']
    baseline = results[0]['test']['metrics']
    for r in results[1:]:
        delta = (r['test']['metrics'][KEYS[-1]]-baseline[KEYS[-1]])*100
        recall = (r['test']['metrics'][KEYS[1]]-baseline[KEYS[1]])*100
        ratio = r['training_seconds']/results[0]['training_seconds']
        lines.append(f'- {r["name"]}相对基线：mAP@50–95 变化 {delta:+.2f} 个百分点，Recall 变化 {recall:+.2f} 个百分点，训练调用耗时约为基线的 {ratio:.2f} 倍。')
    lines += ['', '分辨率对照同时改变训练和测试输入尺寸；Mosaic 对照只关闭拼接增强，未关闭其他增强。时间来自不同时间段的真实运行，包含初始化/验证等开销，受温度和后台负载影响，不是严格性能基准。', '',
              '## 完整逐类结果', '', '| 类别 | 基线 416 | 分辨率 640 | 关闭 Mosaic |', '|---|---:|---:|---:|']
    for i, name in enumerate(CN):
        vals = [f'{r["test"]["per_class"][i]["mAP50-95"]*100:.2f}%' for r in results]
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines += ['', '表中均为 mAP@50–95。类别差异需结合样本量、目标尺寸、遮挡和标注复核解释；不能由分数直接推断原因。', '',
              '## 显存记录', '', 'PyTorch 峰值分配显存仅涵盖该进程的张量分配，不等于整张显卡占用。原基线没有同口径记录，不补估算值。', '']
    for r in results:
        value = '未记录' if r['peak_allocated_MiB'] is None else f'{r["peak_allocated_MiB"]:.1f} MiB（分配）；{r["peak_reserved_MiB"]:.1f} MiB（保留）'
        lines.append(f'- {r["name"]}：{value}')
    lines += ['', '## 结论边界', '',
              '- 每组只有 seed=42 一次训练。观察到的差异不是多次重复的均值，也不能声称统计显著或稳定提升。',
              '- 测试集在前期已查看，本轮是固定配置的补充对照，不是全新盲测，也不依据测试结果继续调参。',
              '- 原四类与八类任务不同，不把分数相减作为改进证据。',
              '- 混合来源数据包含预训练重叠、潜在近重复和漏标等限制，成绩不是官方完整数据集基准。',
              '- 不把最高测试分数自动当作部署选择；目前保留原基线作为已验证演示入口。', '',
              '## 文件与复现', '',
              '- scripts/run_wildlife8_controls.py：顺序运行新增两组对照。',
              '- scripts/summarize_wildlife8_controls.py：从真实 JSON、CSV、args.yaml 重新汇总，并核验其他训练参数一致。',
              '- wildlife8/controls/各组/train：30 轮 results.csv、args.yaml、best.pt、last.pt。',
              '- wildlife8/controls/各组/test_results.json 与 record.json：测试指标、环境、时间、哈希和运行状态。',
              '- wildlife8/controls/*.log：原始控制台日志。',
              '- CODE_ORIGIN.md：上游算法与 AI 辅助配套脚本的来源说明。', '',
              '个人复现命令（会创建独立新目录，不覆盖参考基线）：', '', '```powershell',
              '& ".\\.venv\\Scripts\\python.exe" -X utf8 scripts/reproduce_myself.py', '```']
    (DELIVERY/'01_补充实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('Verified three runs, 90 epochs, same non-study parameters. Wrote comparison.json and report.')

if __name__ == '__main__':
    main()
