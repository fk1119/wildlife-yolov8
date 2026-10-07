"""Course-specific controlled experiments; model/trainer are upstream Ultralytics.

AI-assisted orchestration, not a new detection algorithm. Never overwrites runs.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'wildlife8/controls'
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.yolo')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['MPLBACKEND'] = 'Agg'
CONFIGS = {'resolution640': {'imgsz': 640, 'mosaic': 1.0},
           'no_mosaic416': {'imgsz': 416, 'mosaic': 0.0}}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=lambda x: x.item()), encoding='utf8')

def worker(name):
    import torch
    import ultralytics
    from ultralytics import YOLO
    target = OUT / name
    target.mkdir(parents=True, exist_ok=False)
    initial = ROOT / 'assets/yolov8n.pt'
    assert sha(initial) == 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
    settings = dict(data=str(ROOT/'wildlife8/dataset.yaml'), epochs=30, batch=8,
                    device=0, workers=0, seed=42, deterministic=True, optimizer='AdamW',
                    lr0=.001, weight_decay=.0005, amp=False, cache=False, patience=31,
                    close_mosaic=0, project=str(target), name='train', plots=True,
                    verbose=False, **CONFIGS[name])
    label_hash = hashlib.sha256()
    for p in sorted((ROOT/'wildlife8/data/labels').rglob('*.txt')):
        label_hash.update(p.relative_to(ROOT/'wildlife8').as_posix().encode())
        label_hash.update(p.read_bytes())
    record = dict(name=name, started_at=datetime.datetime.now().isoformat(),
                  initial_sha256=sha(initial), manifest_sha256=sha(ROOT/'wildlife8/manifest.json'),
                  labels_sha256=label_hash.hexdigest(), torch=torch.__version__,
                  ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0),
                  settings=settings, state='training')
    write(target/'record.json', record)
    try:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = YOLO(str(initial))
        model.train(**settings)
        record.update(training_seconds=time.perf_counter()-started,
                      peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                      state='evaluating')
        write(target/'record.json', record)
        best = YOLO(str(target/'train/weights/best.pt'))
        metrics = best.val(data=settings['data'], split='test', imgsz=settings['imgsz'],
                           batch=1, device=0, workers=0, project=str(target), name='test', plots=True)
        write(target/'test_results.json', dict(split='test', imgsz=settings['imgsz'],
              metrics=metrics.results_dict, per_class=metrics.summary(), speed_ms=metrics.speed,
              classes=best.names, note='Single seed, frozen course subset. Test already inspected before supplementary experiments.'))
        record.update(state='complete', best_sha256=sha(target/'train/weights/best.pt'),
                      ended_at=datetime.datetime.now().isoformat())
        write(target/'record.json', record)
    except Exception as exc:
        record.update(state='failed', error=repr(exc))
        write(target/'record.json', record)
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=CONFIGS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    OUT.mkdir(exist_ok=True)
    subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/check_wildlife8.py'), '--data-only'], check=True)
    plan = dict(created_at=datetime.datetime.now().isoformat(), configs=CONFIGS,
                baseline='wildlife8/runs/train', epochs=30, seed=42,
                rule='Fixed in advance. No hyperparameter search or selecting best test result. Each best checkpoint selected by validation fitness.',
                limitation='Test set previously inspected. Supplementary comparisons, not a fresh untouched holdout. Baseline peak VRAM unavailable.')
    if not (OUT/'plan.json').exists():
        write(OUT/'plan.json', plan)
    for name in CONFIGS:
        record = OUT/name/'record.json'
        if record.exists() and json.loads(record.read_text(encoding='utf8')).get('state') == 'complete':
            print('Already complete:', name, flush=True)
            continue
        if (OUT/name).exists():
            raise RuntimeError(f'Incomplete run exists: {name}; inspect before retrying')
        print('START', name, datetime.datetime.now().isoformat(), flush=True)
        with (OUT/(name+'.log')).open('w', encoding='utf8') as log:
            subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--worker', name],
                           stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        print('DONE', name, flush=True)

if __name__ == '__main__':
    main()
