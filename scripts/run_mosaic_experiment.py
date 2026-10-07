"""Run a clean Mosaic ON/OFF experiment from the same initial checkpoint.

This is intentionally independent of every previous run.  It creates one new
timestamped experiment directory, trains both conditions, evaluates both on
the test split, then calls the parameterized image comparison script.
"""
import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from project_config import CONFIG, DATA, INITIAL, OUTPUT, BASELINE, DEVICE, WORKERS, TRAINING, dataset_config, snapshot
from isolated_training import train_isolated

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.yolo')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['MPLBACKEND'] = 'Agg'
INITIAL_SHA = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
MANIFEST_SHA = '2fa6c7cbf957271447d8506f7ab6f251c1634cc14d1c5c9a2845e9425eafb0ed'


def json_default(value):
    # Metrics.summary() includes NumPy integer counts, including nested values.
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Unsupported JSON type: {type(value).__name__}')


def completed_training(group_dir, expected_record):
    """Reuse only a fully completed, matching run; never restart over its files."""
    record = json.loads((group_dir / 'run_record.json').read_text(encoding='utf-8'))
    for key in ('condition', 'initial_sha256', 'dataset_manifest_sha256', 'configuration', 'dataset'):
        if record[key] != expected_record[key]:
            raise RuntimeError(f'Recovery configuration mismatch: {key}')
    with (group_dir / 'train/results.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    epochs = expected_record['configuration']['epochs']
    if [int(row['epoch']) for row in rows] != list(range(1, epochs + 1)):
        raise RuntimeError('Training is incomplete; this recovery mode only reuses completed training.')
    if not (group_dir / 'train/weights/best.pt').is_file():
        raise FileNotFoundError('Completed training checkpoint is missing.')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_inputs():
    import yaml
    import torch
    cfg = dataset_config()
    initial = INITIAL
    manifest = ROOT / 'wildlife8' / 'manifest.json'
    if not cfg.exists() or not initial.exists() or not manifest.exists():
        raise FileNotFoundError('dataset.yaml, assets/yolov8n.pt, and wildlife8/manifest.json are required')
    if sha256(initial) != INITIAL_SHA:
        raise RuntimeError('Initial checkpoint hash changed; refusing an untracked experiment.')
    if sha256(manifest) != MANIFEST_SHA:
        raise RuntimeError('Dataset manifest hash changed; refusing an untracked experiment.')
    data = yaml.safe_load(cfg.read_text(encoding='utf-8'))
    if len(data.get('names', [])) != 8:
        raise RuntimeError('Expected the frozen eight-class dataset.')
    for split in ('train', 'val', 'test'):
        images = DATA / 'images' / split
        labels = DATA / 'labels' / split
        if not images.exists() or not labels.exists() or not list(images.glob('*.jpg')):
            raise RuntimeError(f'Missing or empty {split} images/labels directory.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this configured experiment.')
    return cfg, initial


def train_and_test(initial, cfg, group_dir, mosaic, args):
    import torch
    from ultralytics import YOLO
    reuse = group_dir.exists() and bool(args.resume_run)
    if not reuse:
        group_dir.mkdir(parents=True, exist_ok=False)
    record = {
        'condition': 'mosaic_on' if mosaic else 'mosaic_off',
        'mosaic': mosaic,
        'initial_weights': str(initial),
        'initial_sha256': sha256(initial),
        'dataset': str(cfg),
        'dataset_manifest_sha256': MANIFEST_SHA,
        'configuration': {'epochs': args.epochs, 'imgsz': args.imgsz, 'batch': args.batch, 'mosaic': mosaic,
                          'seed': args.seed, 'optimizer': 'AdamW', 'lr0': 0.001,
                          'weight_decay': 0.0005, 'workers': WORKERS, 'amp': False,
                          'close_mosaic': 0},
        'torch': torch.__version__,
    }
    if reuse:
        completed_training(group_dir, record)
        saved = group_dir / 'test_results.json'
        if saved.exists():
            result = json.loads(saved.read_text(encoding='utf-8'))
            if (result['configuration'] != record['configuration'] or
                    result['weights_sha256'] != sha256(group_dir / 'train/weights/best.pt')):
                raise RuntimeError('Saved evaluation does not match training.')
            print(f'Reusing completed training and evaluation: {group_dir}', flush=True)
            return
        print(f'Reusing completed training; rerunning evaluation only: {group_dir}', flush=True)
    else:
        (group_dir / 'run_record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        train_isolated(initial, group_dir / 'training_job',
                data=str(cfg), epochs=args.epochs, imgsz=args.imgsz, mosaic=mosaic,
                batch=args.batch, device=DEVICE, workers=WORKERS, seed=args.seed, deterministic=True,
                optimizer='AdamW', lr0=0.001, weight_decay=0.0005, amp=False, cache=False,
                patience=args.epochs + 1, close_mosaic=0, project=str(group_dir),
                    name='train', plots=True, verbose=False)
        gc.collect()
        torch.cuda.empty_cache()
    best_path = group_dir / 'train' / 'weights' / 'best.pt'
    if not best_path.exists():
        raise RuntimeError(f'No best.pt produced for {group_dir.name}.')
    best = YOLO(str(best_path))
    metrics = best.val(data=str(cfg), split='test', imgsz=args.imgsz, batch=1, device=DEVICE,
                       workers=WORKERS, project=str(group_dir), name='test', plots=True)
    result = {
        'condition': record['condition'], 'configuration': record['configuration'],
        'weights_sha256': sha256(best_path),
        'metrics': {k: (v.item() if hasattr(v, 'item') else v) for k, v in metrics.results_dict.items()},
        'per_class': metrics.summary(), 'speed_ms': metrics.speed,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, default=json_default)
    pending = group_dir / 'test_results.json.tmp'
    pending.write_text(payload, encoding='utf-8')
    pending.replace(group_dir / 'test_results.json')
    del best, metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Fresh, reproducible Mosaic ON/OFF training, testing, and comparison.')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume-run', type=Path, help='Reuse completed training in an interrupted experiment')
    parser.add_argument('--epochs', type=int, default=TRAINING['epochs'])
    parser.add_argument('--imgsz', type=int, default=CONFIG['experiments']['mosaic']['imgsz'])
    parser.add_argument('--batch', type=int, default=TRAINING['batch'])
    parser.add_argument('--seed', type=int, default=TRAINING['seed'])
    args = parser.parse_args()
    if args.resume_run:
        out = args.resume_run.resolve()
        if out.parent != (OUTPUT / 'mosaic_experiments').resolve():
            parser.error('Recovery directory must be directly inside wildlife8/mosaic_experiments.')
        protocol = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
        if protocol['initial_sha256'] != INITIAL_SHA or protocol['dataset_manifest_sha256'] != MANIFEST_SHA:
            parser.error('Recovery protocol hashes mismatch.')
        for key in ('epochs', 'imgsz', 'batch', 'seed'):
            setattr(args, key, protocol['shared_configuration'][key])
    cfg, initial = check_inputs()
    print('Inputs verified. Initial checkpoint and frozen dataset are unchanged.', flush=True)
    if args.check_only:
        return
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not args.resume_run:
        out = OUTPUT / 'mosaic_experiments' / stamp
        out.mkdir(parents=True, exist_ok=False)
        snapshot(out)
        protocol = {'created_at': stamp, 'initial_sha256': INITIAL_SHA, 'dataset_manifest_sha256': MANIFEST_SHA,
                    'shared_configuration': vars(args), 'conditions': {'on': 1, 'off': 0}}
        (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        train_and_test(initial, cfg, out / 'on', 1, args)
        train_and_test(initial, cfg, out / 'off', 0, args)
        comparison = out / 'comparison'
        if comparison.exists():
            comparison = out / ('comparison_recovery_' + stamp)
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'compare_mosaic_predictions.py'),
                        '--on-run', str(out / 'on'), '--off-run', str(out / 'off'), '--output', str(comparison)],
                       check=True)
        (out / 'DONE').write_text('Training, test evaluation, and fixed-test-image comparison completed.\n', encoding='utf-8')
        print(f'DONE: {out}', flush=True)
    except Exception as exc:
        (out / 'FAILED.txt').write_text(repr(exc), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
