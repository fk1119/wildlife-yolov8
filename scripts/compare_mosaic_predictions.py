"""Compare two fresh Mosaic runs on all test images and retain a fixed sample."""
import csv
import datetime
import hashlib
import html
import json
import os
import argparse
from pathlib import Path
from project_config import CONFIG, DATA, INITIAL, OUTPUT, BASELINE, DEVICE, WORKERS, TRAINING, dataset_config, snapshot

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.yolo')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['MPLBACKEND'] = 'Agg'
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Results

NAMES = ['buffalo', 'elephant', 'rhino', 'zebra', 'bear', 'giraffe', 'lion', 'deer']
CN = ['水牛', '大象', '犀牛', '斑马', '熊', '长颈鹿', '狮子', '鹿']


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def match(boxes, gt):
    used = set()
    for box in sorted(boxes, key=lambda b: -b[4]):
        candidates = []
        for i, target in enumerate(gt):
            if i in used or int(box[5]) != int(target[5]):
                continue
            inter = max(0, min(box[2], target[2])-max(box[0], target[0])) * max(0, min(box[3], target[3])-max(box[1], target[1]))
            area = (box[2]-box[0])*(box[3]-box[1])+(target[2]-target[0])*(target[3]-target[1])-inter
            candidates.append((inter / area if area > 0 else 0, i))
        if candidates:
            overlap, i = max(candidates)
            if overlap >= .5:
                used.add(i)
    return dict(tp=len(used), fp=len(boxes)-len(used), fn=len(gt)-len(used))


def compare_runs(on_run, off_run, out):
    runs = {'on': Path(on_run).resolve(), 'off': Path(off_run).resolve()}
    records = {key: json.loads((path/'run_record.json').read_text(encoding='utf-8')) for key,path in runs.items()}
    assert records['on']['initial_sha256'] == records['off']['initial_sha256']
    assert records['on']['dataset_manifest_sha256'] == records['off']['dataset_manifest_sha256']
    assert records['on']['configuration']['mosaic'] == 1 and records['off']['configuration']['mosaic'] == 0
    args = {key: yaml.safe_load((path/'train/args.yaml').read_text(encoding='utf-8')) for key,path in runs.items()}
    differences = {k: [args['on'].get(k), args['off'].get(k)] for k in set(args['on']) | set(args['off']) if args['on'].get(k) != args['off'].get(k)}
    assert set(differences) <= {'mosaic', 'project', 'name', 'save_dir'}, differences
    for path in runs.values():
        with (path/'train/results.csv').open() as f:
            epochs = list(csv.DictReader(f))
        expected = records['on']['configuration']['epochs']
        assert len(epochs) == expected and int(epochs[-1]['epoch']) == expected
    image_size = records['on']['configuration']['imgsz']
    seed = records['on']['configuration']['seed']
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'images').mkdir()
    files = sorted((DATA/'images/test').glob('*.jpg'))
    assert len(files) == 360
    info = {}
    for f in files:
        im = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
        h,w = im.shape[:2]
        gt = []
        for line in (DATA/'labels/test'/f.with_suffix('.txt').name).read_text().splitlines():
            c,x,y,bw,bh = map(float, line.split())
            gt.append([(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h, 1., c])
        info[f.name] = dict(gt=gt, classes=sorted({int(b[5]) for b in gt}), predictions={})
    # Freeze illustrative selection BEFORE inference. Deterministic hash order,
    # three distinct images per class; not a population-representative sample.
    selected = []
    for cid in range(8):
        candidates = [f.name for f in files if cid in info[f.name]['classes'] and f.name not in selected]
        candidates.sort(key=lambda name: hashlib.sha256(('mosaic-fixed-20261005:'+name).encode()).hexdigest())
        selected.extend(candidates[:3])
    write_json(out/'protocol.json', dict(runs={k:str(v) for k,v in runs.items()}, parameter_differences=differences,
        weights_sha256={k:hashlib.sha256((v/'train/weights/best.pt').read_bytes()).hexdigest() for k,v in runs.items()},
        fixed_sample=selected, sample_rule='Before prediction: three unique images per class by salted filename SHA256 order.',
        imgsz=image_size, confidence=.25, nms_iou=.7, diagnostic_matching='confidence-ordered same-class greedy IoU>=0.5; not official AP',
        test_images=360, caveat='Previously inspected test set; one seed; diagnostic comparisons do not establish population generalization.'))
    for key,path in runs.items():
        model = YOLO(str(path/'train/weights/best.pt'))
        for result in model.predict(source=[str(f) for f in files], stream=True, imgsz=image_size, batch=1, rect=True, conf=.25, iou=.7, device=DEVICE, half=False, verbose=False):
            name = Path(result.path).name
            boxes = result.boxes.data.cpu().numpy().tolist()
            info[name]['predictions'][key] = boxes
            info[name][key] = match(boxes, info[name]['gt'])
        print('Completed predictions:', key, flush=True)
    groups = {'improved': [], 'worse': [], 'equal': []}
    totals = {key:dict(tp=0, fp=0, fn=0) for key in runs}
    for name, item in info.items():
        item['error_delta'] = item['on']['fp']+item['on']['fn']-item['off']['fp']-item['off']['fn']
        group = 'improved' if item['error_delta'] < 0 else 'worse' if item['error_delta'] > 0 else 'equal'
        groups[group].append(name)
        for key in runs:
            for metric in totals[key]: totals[key][metric] += item[key][metric]
    extras = []
    for group in groups:
        candidates = sorted(groups[group], key=lambda name:(-abs(info[name]['error_delta']),name))
        extras.extend((name,group) for name in candidates[:3])
    write_json(out/'all_predictions.json', info)
    write_json(out/'diagnostics.json', dict(totals=totals, groups={k:len(v) for k,v in groups.items()},
        extra_examples=extras, grouping='FP+FN difference at fixed threshold; equal error counts do not imply identical boxes'))
    source_results = {key:json.loads((path/'test_results.json').read_text(encoding='utf-8')) for key,path in runs.items()}
    cls_rows = []
    for cid,name in enumerate(NAMES):
        a = next(r for r in source_results['on']['per_class'] if r['Class']==name)
        b = next(r for r in source_results['off']['per_class'] if r['Class']==name)
        cls_rows.append(dict(class_name=name, chinese=CN[cid], instances=a['Instances'], on_AP=a['mAP50-95'], off_AP=b['mAP50-95'], change_pp=100*(a['mAP50-95']-b['mAP50-95'])))
    with (out/'per_class.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(cls_rows[0]));writer.writeheader();writer.writerows(cls_rows)
    cards = []
    display = list(dict.fromkeys(selected+[n for n,g in extras]))
    for index,name in enumerate(display):
        item=info[name];im=cv2.imdecode(np.fromfile(DATA/'images/test'/name,dtype=np.uint8),cv2.IMREAD_COLOR)
        figures=[]
        for key,label in [('gt','标注'),('on','开启Mosaic'),('off','关闭Mosaic')]:
            boxes=item['gt'] if key=='gt' else item['predictions'][key]
            result=Results(im,path=name,names=dict(enumerate(NAMES)),boxes=np.array(boxes,dtype=np.float32).reshape(-1,6))
            dest=f'images/{index:02d}_{key}.jpg'
            result.save(filename=str(out/dest),conf=key!='gt')
            caption=label if key=='gt' else f"{label}：匹配{item[key]['tp']} / 多余{item[key]['fp']} / 未匹配标注{item[key]['fn']}"
            figures.append(f'<figure><img loading="lazy" src="{dest}"><figcaption>{caption}</figcaption></figure>')
        tags=['预先固定样本'] if name in selected else []
        tags += [{'improved':'按错误数筛选的改善案例','worse':'按错误数筛选的退步案例','equal':'按错误数筛选的持平案例'}[g] for n,g in extras if n==name]
        cards.append('<section><h2>'+html.escape(name)+'</h2><p>'+ '；'.join(tags)+'</p><div class="row">'+''.join(figures)+'</div></section>')
    table=''.join(f"<tr><td>{r['chinese']}</td><td>{r['on_AP']*100:.2f}%</td><td>{r['off_AP']*100:.2f}%</td><td>{r['change_pp']:+.2f}</td></tr>" for r in cls_rows)
    summary=f"全部360张图的固定阈值诊断：开启相对关闭，FP+FN减少{len(groups['improved'])}张、增加{len(groups['worse'])}张、相同{len(groups['equal'])}张。错误数相同不表示框相同。"
    page='<!doctype html><meta charset="utf-8"><title>Mosaic新训练对照</title><style>body{font-family:Microsoft YaHei,sans-serif;margin:32px;color:#183b32;background:#f7f8f5}table{border-collapse:collapse}td,th{padding:10px 22px;border:1px solid #bbb}section{margin:35px 0;border-top:1px solid #bbb}h2{font-size:17px}.row{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}img{width:100%}figcaption{font-size:14px}p{line-height:1.7}@media(max-width:800px){.row{display:block}figure{margin-bottom:20px}}</style>'
    page+='<h1>Mosaic开关：两次新训练的完整诊断</h1><p>开启：'+html.escape(str(runs['on']))+'；关闭：'+html.escape(str(runs['off']))+f'。所有预测固定{image_size}、conf 0.25、NMS IoU 0.7。</p><p>下表来自两次已完成的360图标准AP评估。逐图诊断采用另一套固定阈值匹配规则，不代替标准AP。</p><table><tr><th>类别</th><th>开启AP50–95</th><th>关闭AP50–95</th><th>差值/百分点</th></tr>'+table+'</table><p>'+summary+'</p><p>先展示预测前固定的24张覆盖八类样本，再补充按结果挑选的改善、退步和错误数持平案例。样本仅用于解释，不代表独立泛化评估。未匹配预测和标注可能涉及漏标及定位误差，需结合原图复核。</p>'+''.join(cards)
    (out/'图片对照.html').write_text(page,encoding='utf-8')
    report='# Mosaic两次新训练对照\n\n'+summary+'\n\n| 类别 | 开启AP50–95 | 关闭AP50–95 | 变化/百分点 |\n|---|---:|---:|---:|\n'
    report+=''.join(f"| {r['chinese']} | {r['on_AP']*100:.2f}% | {r['off_AP']*100:.2f}% | {r['change_pp']:+.2f} |\n" for r in cls_rows)
    report+=f'\n[打开图片对照](图片对照.html)。24张样本在推理前按固定规则选择，另附结果导向案例，均有明确标注。\n\n逐图统计为conf=0.25、同类IoU>=0.5贪心匹配，不等同于官方AP或框架汇总P/R。两次均为seed{seed}，不作统计显著性或普适性结论。原始参数、权重哈希、样本清单和全部360张的预测坐标均保存于同目录。\n'
    (out/'实验分析.md').write_text(report,encoding='utf-8')
    print(summary, flush=True)
    print(json.dumps(cls_rows,ensure_ascii=False), flush=True)
    print('DONE:', out, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description='Compare two runs without relying on historical paths.')
    parser.add_argument('--on-run', required=True, type=Path)
    parser.add_argument('--off-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    compare_runs(args.on_run, args.off_run, args.output)


if __name__=='__main__':
    main()
