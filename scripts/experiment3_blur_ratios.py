"""Fresh 0/25/50 percent blur study: data -> train -> validation -> test -> report.

Uses size_experiment_common only for input checking, hashes and GPU cleanup.
Never reads previous experiment weights, results or generated datasets.
"""
import argparse
import csv
import datetime
import json
import random
import shutil
import statistics
import time
import traceback
from pathlib import Path
from project_config import CONFIG, DATA, INITIAL, OUTPUT, BASELINE, DEVICE, WORKERS, TRAINING, dataset_config, snapshot

from size_experiment_common import ROOT, METRICS, digest, preflight, release_gpu, save_json
from isolated_training import train_isolated

RATIOS = (0, 25, 50)
CONDITIONS = ('clean', 'dark1', 'dark2', 'dark3', 'blur1', 'blur2', 'blur3', 'jpeg1', 'jpeg2', 'jpeg3')
VALIDATION = ('clean', 'blur1', 'blur2', 'blur3')
ASSIGNMENT_SEED = 20261004


def assignments(count):
    order = list(range(count))
    random.Random(ASSIGNMENT_SEED).shuffle(order)
    return {ratio: {idx: 1 + rank % 3 for rank, idx in enumerate(order[:count * ratio // 100])}
            for ratio in RATIOS}


def corrupt(image, condition):
    import cv2
    import numpy as np
    if condition == 'clean':
        return image.copy()
    level = int(condition[-1]) - 1
    if condition.startswith('blur'):
        sigma = min(image.shape[:2]) * (.0015, .004, .008)[level]
        return cv2.GaussianBlur(image, (0, 0), sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    if condition.startswith('dark'):
        return np.clip(image.astype(np.float32) * (.6, .3, .12)[level], 0, 255).astype(np.uint8)
    if condition.startswith('jpeg'):
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, (60, 25, 8)[level]])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    raise ValueError(condition)


def read_image(path):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Invalid image: {path}')
    return image


def prepare_dataset(dest, split, sources, conditions, names, original_splits):
    import cv2
    import numpy as np
    import yaml
    images = dest / 'images' / split
    labels = dest / 'labels' / split
    images.mkdir(parents=True, exist_ok=False)
    labels.mkdir(parents=True, exist_ok=False)
    records = []
    total = len(sources)
    started = last_print = time.monotonic()
    print(f'PREPARE {dest.name}/{split}: 0/{total} images', flush=True)
    for index, (source, condition) in enumerate(zip(sources, conditions, strict=True), 1):
        image = read_image(source)
        transformed = corrupt(image, condition)
        if transformed.shape != image.shape:
            raise RuntimeError('Corruption changed image geometry')
        target = images / (source.stem + '.png')
        if target.exists():
            raise RuntimeError(f'Duplicate image stem: {source.stem}')
        ok, encoded = cv2.imencode('.png', transformed, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError(f'PNG encoding failed: {target}')
        encoded.tofile(target)
        if not np.array_equal(read_image(target), transformed):
            raise RuntimeError('PNG pixel verification failed')
        label = source.parent.parent.parent / 'labels' / source.parent.name / (source.stem + '.txt')
        target_label = labels / label.name
        shutil.copyfile(label, target_label)
        if digest(label) != digest(target_label):
            raise RuntimeError('Label copy verification failed')
        records.append(dict(source=str(source), condition=condition, file=target.name,
                            image_sha256=digest(target), label_sha256=digest(target_label)))
        now = time.monotonic()
        if index == total or index % 25 == 0 or now - last_print >= 5:
            elapsed = now - started
            remaining = elapsed / index * (total - index)
            print(f'PREPARE {dest.name}/{split}: {index}/{total} '
                  f'({index / total:.1%}), elapsed {elapsed:.0f}s, ETA {remaining:.0f}s', flush=True)
            last_print = now
    config = dict(path=str(dest), names=names, **{s: str(p) for s, p in original_splits.items()})
    config[split] = f'images/{split}'
    cfg = dest / 'dataset.yaml'
    cfg.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    save_json(dest / 'manifest.json', records)
    return cfg


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selection(rows):
    # Equal weight to clean performance and mean blur performance, fixed in advance.
    scores = {}
    for ratio in RATIOS:
        clean = statistics.mean(r['mAP50-95'] for r in rows if r['ratio'] == ratio and r['condition'] == 'clean')
        blur = statistics.mean(r['mAP50-95'] for r in rows if r['ratio'] == ratio and r['condition'].startswith('blur'))
        scores[ratio] = (clean + blur) / 2
    return min(RATIOS, key=lambda ratio: (-scores[ratio], ratio)), scores


def run(out, cfg, initial, evidence, args):
    import yaml
    from ultralytics import YOLO
    data = yaml.safe_load(cfg.read_text(encoding='utf-8'))
    base = Path(data['path'])
    if not base.is_absolute():
        base = (cfg.parent / base).resolve()
    splits = {s: base / data[s] for s in ('train', 'val', 'test')}
    files = {s: sorted(p for p in folder.iterdir() if p.suffix.lower() in
                      {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}) for s, folder in splits.items()}
    allocation = assignments(len(files['train']))
    out.mkdir(parents=True, exist_ok=False)
    snapshot(out)
    shared = dict(epochs=args.epochs, imgsz=CONFIG['experiments']['blur_ratios']['imgsz'], batch=args.batch, device=DEVICE, workers=WORKERS,
                  deterministic=True, optimizer='AdamW', lr0=.001, weight_decay=.0005,
                  amp=False, cache=False, patience=args.epochs + 1, mosaic=1., close_mosaic=0,
                  resume=False, plots=True, verbose=True)
    evaluation = dict(imgsz=CONFIG['experiments']['blur_ratios']['imgsz'], batch=1, device=DEVICE, workers=WORKERS, conf=.001, iou=.7,
                      max_det=300, half=False, augment=False, rect=True, plots=False, verbose=True)
    save_json(out / 'protocol.json', dict(evidence=evidence, initial_sha256=digest(initial),
              initial_weights=str(initial), train=shared, evaluation=evaluation, seeds=args.seeds,
              training_isolation='fresh interpreter per ratio/seed, matching robustness study',
              ratios=RATIOS, counts={r: len(a) for r, a in allocation.items()},
              assignment_seed=ASSIGNMENT_SEED, conditions=CONDITIONS,
              blur_sigma_fraction=[.0015, .004, .008], dark_multiplier=[.6, .3, .12], jpeg_quality=[60, 25, 8],
              selection='Mean over seeds of (clean val AP + mean three blur val AP)/2; ties prefer lower ratio.',
              design='Blur chosen before this run, informed by prior experiments. Nested fixed subsets; identical severity for shared samples. All arms use PNG, including 0%. Best checkpoint selected on original clean validation data.',
              limitations='Previously inspected test data; synthetic corruptions; two seeds do not establish statistical significance.'))
    stage = 'prepare'
    save_json(out / 'status.json', dict(status='running', stage=stage))
    try:
        train_cfg, eval_cfg = {}, {}
        for ratio in RATIOS:
            stage = f'prepare_train_blur{ratio}'
            save_json(out / 'status.json', dict(status='running', stage=stage))
            print(f'Preparing training ratio={ratio}%...', flush=True)
            conditions = [f'blur{allocation[ratio][i]}' if i in allocation[ratio] else 'clean'
                          for i in range(len(files['train']))]
            train_cfg[ratio] = prepare_dataset(out / 'data' / f'train_blur{ratio}', 'train',
                                               files['train'], conditions, data['names'], splits)
        for split, conditions in (('val', VALIDATION), ('test', CONDITIONS)):
            for condition in conditions:
                stage = f'prepare_{split}_{condition}'
                save_json(out / 'status.json', dict(status='running', stage=stage))
                print(f'Preparing {split}/{condition}...', flush=True)
                eval_cfg[split, condition] = prepare_dataset(out / 'data' / f'{split}_{condition}', split,
                    files[split], [condition] * len(files[split]), data['names'], splits)
        weights = {}
        for seed in args.seeds:
            for ratio in RATIOS:
                stage = f'train_blur{ratio}_seed{seed}'
                save_json(out / 'status.json', dict(status='running', stage=stage))
                print(f'\nTRAIN {len(weights)+1}/{len(args.seeds)*len(RATIOS)}: '
                      f'{stage}, {args.epochs} epochs', flush=True)
                train_isolated(initial, out / 'training_jobs' / stage,
                               data=str(train_cfg[ratio]), seed=seed,
                               project=str(out / 'training'), name=stage, **shared)
                best = out / 'training' / stage / 'weights/best.pt'
                if not best.is_file():
                    raise FileNotFoundError(best)
                weights[ratio, seed] = best
                release_gpu()
        rows, class_rows = [], []
        for split, conditions in (('val', VALIDATION), ('test', CONDITIONS)):
            for (ratio, seed), best in weights.items():
                model = YOLO(str(best))
                weight_hash = digest(best)
                for condition in conditions:
                    stage = f'{split}_blur{ratio}_seed{seed}_{condition}'
                    save_json(out / 'status.json', dict(status='running', stage=stage))
                    print(f'EVALUATE {len(rows)+1}/{len(weights)*(len(VALIDATION)+len(CONDITIONS))}: '
                          f'{stage}', flush=True)
                    m = model.val(data=str(eval_cfg[split, condition]), split=split,
                                  project=str(out / 'evaluations'), name=stage, **evaluation)
                    per_class = m.summary()
                    result = dict(ratio=ratio, seed=seed, split=split, condition=condition,
                                  weights_sha256=weight_hash, metrics=m.results_dict,
                                  per_class=per_class, speed_ms=m.speed)
                    dest = out / 'metrics'
                    dest.mkdir(exist_ok=True)
                    save_json(dest / (stage + '.json'), result)
                    row = dict(ratio=ratio, seed=seed, split=split, condition=condition)
                    row.update({label: float(m.results_dict[key]) for label, key in METRICS.items()})
                    rows.append(row)
                    class_rows.extend(dict(ratio=ratio, seed=seed, split=split, condition=condition, **c) for c in per_class)
                    del m
                del model
                release_gpu()
            if split == 'val':
                chosen, scores = selection(rows)
                save_json(out / 'validation_selection.json', dict(selected_ratio=chosen, scores=scores,
                          rule='Equal weight clean AP and three-level average blur AP; then mean over seeds.'))
                print(f'Validation-selected ratio: {chosen}%. Starting test evaluation.', flush=True)
        write_csv(out / 'all_metrics.csv', rows)
        write_csv(out / 'per_class.csv', class_rows)
        summary = []
        for ratio in RATIOS:
            for group in ('clean', 'blur', 'dark', 'jpeg'):
                values = [statistics.mean(r['mAP50-95'] for r in rows if r['ratio'] == ratio
                          and r['seed'] == seed and r['split'] == 'test' and r['condition'].startswith(group))
                          for seed in args.seeds]
                baseline = next((r['mean_mAP50-95'] for r in summary if r['ratio'] == 0 and r['condition'] == group), statistics.mean(values))
                summary.append(dict(ratio=ratio, condition=group, **{f'seed{seed}': v for seed, v in zip(args.seeds, values)},
                                    **{'mean_mAP50-95': statistics.mean(values), 'change_vs_0_pp': (statistics.mean(values) - baseline) * 100}))
        write_csv(out / 'comparison.csv', summary)
        report = '# 模糊增强比例对照\n\n'
        report += f'验证集按预设清晰/模糊等权规则选择：{chosen}%。测试结果仅用于报告，不重新选择比例。\n\n'
        report += '| 模糊比例 | 测试条件 | 平均mAP50–95 | 相对0%变化/百分点 |\n|---|---|---|---|\n'
        report += ''.join(f"| {r['ratio']}% | {r['condition']} | {r['mean_mAP50-95']*100:.2f}% | {r['change_vs_0_pp']:+.2f} |\n" for r in summary)
        report += '\nblur/dark/jpeg分别先对三个强度平均，再对种子平均；clean只对种子平均。各个种子的结果见comparison.csv。\n\n'
        report += '所有模型从同一预训练权重开始。25%模糊样本包含在50%组内，重合样本的模糊强度相同；原数据不修改。\n\n'
        report += '这是人工退化和有限随机种子的结果，不能证明真实环境下普遍有效或统计显著。测试集此前已被查看，本实验不是新的盲测。\n'
        (out / '实验对照.md').write_text(report, encoding='utf-8')
        save_json(out / 'status.json', dict(status='complete'))
        print(report, flush=True)
        print(f'DONE: {out}', flush=True)
    except BaseException:
        save_json(out / 'status.json', dict(status='failed', stage=stage))
        (out / 'FAILED.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=TRAINING['epochs'])
    parser.add_argument('--batch', type=int, default=TRAINING['batch'])
    parser.add_argument('--seeds', type=int, nargs='+', default=CONFIG['experiments']['blur_ratios']['seeds'])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive epochs/batch and unique seeds required')
    print('Checking original inputs and GPU...', flush=True)
    cfg, initial, evidence = preflight()
    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)
    print(f'Plan: {len(args.seeds)*3} fresh models, {args.epochs} epochs/model; {CONFIG['experiments']['blur_ratios']['imgsz']} train/test.', flush=True)
    if args.check_only:
        print('CHECK PASSED. No data generation, training or evaluation started.', flush=True)
        return
    out = OUTPUT / 'experiment3_blur_ratios' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    print('NEW EXPERIMENT:', out, flush=True)
    run(out, cfg, initial, evidence, args)


if __name__ == '__main__':
    main()
