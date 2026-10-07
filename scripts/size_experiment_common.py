"""Shared implementation for the two independently runnable size experiments."""
import argparse
import csv
import datetime
import gc
import hashlib
import json
import os
import traceback
from pathlib import Path
from project_config import CONFIG, DATA, INITIAL, OUTPUT, BASELINE, DEVICE, WORKERS, TRAINING, dataset_config, snapshot

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.yolo')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['MPLBACKEND'] = 'Agg'
INITIAL_SHA = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
METRICS = {'precision': 'metrics/precision(B)', 'recall': 'metrics/recall(B)',
           'mAP50': 'metrics/mAP50(B)', 'mAP50-95': 'metrics/mAP50-95(B)'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=lambda v: v.item()), encoding='utf-8')


def preflight():
    import torch
    import yaml
    import ultralytics
    cfg = dataset_config()
    initial = INITIAL
    if digest(initial) != INITIAL_SHA:
        raise RuntimeError('Initial yolov8n.pt checksum mismatch.')
    data = yaml.safe_load(cfg.read_text(encoding='utf-8'))
    base = Path(data['path'])
    if not base.is_absolute():
        base = (cfg.parent / base).resolve()
    if len(data['names']) != 8:
        raise RuntimeError('Expected eight classes.')
    # Hash actual images AND annotations so comparisons document the data used.
    fingerprints, counts = {}, {}
    for split in ('train', 'val', 'test'):
        folder = base / data[split]
        if folder.parent.name != 'images':
            raise RuntimeError(f'Expected images/{split} layout: {folder}')
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in
                        {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
        if not images:
            raise RuntimeError(f'Empty split: {split}')
        h = hashlib.sha256()
        for image in images:
            label = folder.parent.parent / 'labels' / folder.name / (image.stem + '.txt')
            if not label.is_file():
                raise FileNotFoundError(f'Missing annotation: {label}')
            for file in (image, label):
                h.update(str(file.relative_to(base)).encode('utf-8'))
                h.update(digest(file).encode('ascii'))
        fingerprints[split] = h.hexdigest()
        counts[split] = len(images)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Use the project Python environment.')
    environment = {'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                   'gpu': torch.cuda.get_device_name(DEVICE)}
    return cfg, initial, {'data_sha256': fingerprints, 'image_counts': counts,
                          'dataset_yaml_sha256': digest(cfg), 'environment': environment}


def release_gpu():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def run_experiment(mode, out, cfg, initial, args, evidence, factory, cleanup, train_worker=None):
    """factory is YOLO in production; injectable for a no-training workflow test."""
    if mode not in ('inference', 'training'):
        raise ValueError(mode)
    if train_worker is None:
        from isolated_training import train_isolated
        train_worker = train_isolated
    out.mkdir(parents=True, exist_ok=False)
    sizes = CONFIG['experiments']['inference_sizes' if mode == 'inference' else 'training_sizes']['sizes']
    train_sizes = (sizes[0],) if mode == 'inference' else tuple(sizes)
    snapshot(out)
    shared = dict(epochs=args.epochs, batch=args.batch, device=DEVICE, workers=WORKERS,
                  seed=args.seed, deterministic=True, optimizer='AdamW', lr0=.001,
                  weight_decay=.0005, amp=False, cache=False, patience=args.epochs + 1,
                  close_mosaic=0, mosaic=1.0, resume=False, plots=True, verbose=False)
    evaluation = dict(split='test', batch=1, device=DEVICE, workers=WORKERS, conf=.001,
                      iou=.7, max_det=300, half=False, augment=False, rect=True, plots=True)
    save_json(out / 'protocol.json', dict(mode=mode, initial_weights=str(initial),
              initial_sha256=digest(initial), dataset=str(cfg), shared_train=shared,
              shared_test=evaluation, train_sizes=train_sizes, evidence=evidence))
    (out / 'dataset.yaml').write_bytes(cfg.read_bytes())
    rows, class_rows = [], []
    stage = 'initialization'
    try:
        for train_size in train_sizes:
            stage = f'train_{train_size}'
            save_json(out / 'status.json', {'status': 'running', 'stage': stage})
            print(f'\n=== Training {train_size}, from initial pretrained weights ===', flush=True)
            train_worker(initial, out / 'training_jobs' / stage,
                         data=str(cfg), imgsz=train_size, project=str(out), name=stage, **shared)
            best = out / stage / 'weights/best.pt'
            if not best.is_file():
                raise FileNotFoundError(f'Training did not produce {best}')
            cleanup()
            # Experiment 1 uses this EXACT checkpoint for both evaluations.
            test_sizes = tuple(sizes) if mode == 'inference' else (train_size,)
            for test_size in test_sizes:
                stage = f'test_train{train_size}_test{test_size}'
                save_json(out / 'status.json', {'status': 'running', 'stage': stage})
                print(f'\n=== Testing: train={train_size}, test={test_size} ===', flush=True)
                model = factory(str(best))
                metrics = model.val(data=str(cfg), imgsz=test_size, project=str(out),
                                    name=stage, **evaluation)
                per_class = metrics.summary()
                result = dict(train_size=train_size, test_size=test_size,
                              weights=str(best.relative_to(out)), weights_sha256=digest(best),
                              metrics=metrics.results_dict, per_class=per_class,
                              speed_ms=metrics.speed)
                save_json(out / (stage + '.json'), result)
                row = dict(train_size=train_size, test_size=test_size)
                row.update({label: float(metrics.results_dict[key]) for label, key in METRICS.items()})
                row['inference_ms'] = float(metrics.speed['inference'])
                rows.append(row)
                class_rows.extend(dict(train_size=train_size, test_size=test_size, **c) for c in per_class)
                del model, metrics
                cleanup()
        for name, entries in [('comparison.csv', rows), ('per_class.csv', class_rows)]:
            if entries:
                with (out / name).open('w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(entries[0]))
                    writer.writeheader()
                    writer.writerows(entries)
        changes = {key: (rows[1][key] - rows[0][key]) * 100 for key in METRICS}
        save_json(out / 'comparison.json', {'results': rows, 'second_minus_first_pp': changes})
        title = f'同一{sizes[0]}模型：{sizes[0]}与{sizes[1]}测试' if mode == 'inference' else f'{sizes[0]}训练/测试与{sizes[1]}训练/测试'
        report = f'# {title}\n\n两组均在完整测试集评估，best.pt由训练期间的验证集表现选择。\n\n'
        report += '| 训练尺寸 | 测试尺寸 | Precision | Recall | mAP50 | mAP50–95 | 推理毫秒/图 |\n|---|---|---|---|---|---|---|\n'
        for row in rows:
            report += f"| {row['train_size']} | {row['test_size']} | "
            report += ' | '.join(f'{row[k]*100:.2f}%' for k in METRICS)
            report += f" | {row['inference_ms']:.2f} |\n"
        report += '\n第二组减第一组：\n\n' + '\n'.join(f'- {k}：{v:+.2f} 个百分点' for k, v in changes.items())
        report += '\n\n这是单个随机种子、当前测试集上的结果；不能据此断言任何数据上都更好。速度是本次评估记录，不是独立速度基准。\n'
        report += ('\n本实验只改变测试输入尺寸，两组权重相同。\n' if mode == 'inference'
                   else '\n本实验同时比较两套训练和测试尺寸配置，不能把差异单独归因于测试尺寸。两个模型均从同一预训练权重重新训练。\n')
        (out / '实验对照.md').write_text(report, encoding='utf-8')
        save_json(out / 'status.json', {'status': 'complete'})
        print(report, flush=True)
        print(f'\nDONE: {out}', flush=True)
    except BaseException:
        save_json(out / 'status.json', {'status': 'failed', 'stage': stage})
        (out / 'FAILED.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    return rows


def main(mode):
    parser = argparse.ArgumentParser(description='Fresh training, full-test evaluation and size comparison.')
    parser.add_argument('--check-only', action='store_true', help='Verify inputs without training')
    parser.add_argument('--epochs', type=int, default=TRAINING['epochs'])
    parser.add_argument('--batch', type=int, default=TRAINING['batch'])
    parser.add_argument('--seed', type=int, default=TRAINING['seed'])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error('epochs and batch must be positive')
    print('Checking weights, actual images, labels and GPU...', flush=True)
    cfg, initial, evidence = preflight()
    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        print('CHECK PASSED. No training or testing started.', flush=True)
        return
    from ultralytics import YOLO
    category = 'experiment1_inference_sizes' if mode == 'inference' else 'experiment2_training_sizes'
    out = OUTPUT / category / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    print('NEW EXPERIMENT:', out, flush=True)
    run_experiment(mode, out, cfg, initial, args, evidence, YOLO, release_gpu)
