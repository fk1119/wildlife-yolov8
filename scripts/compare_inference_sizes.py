"""Evaluate one checkpoint at two inference sizes without retraining."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.yolo')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['MPLBACKEND'] = 'Agg'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        parser.error(f'Checkpoint not found: {weights}')
    from ultralytics import YOLO

    output = ROOT / 'wildlife8' / 'inference_comparisons' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    common = dict(data=str(ROOT / 'wildlife8/dataset.yaml'), split='test',
                  batch=1, device=0, workers=0, conf=0.001, iou=0.7,
                  half=False, augment=False, rect=True, plots=True)
    record = dict(weights=str(weights), weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                  settings=common, sizes=[416, 640], note='Exploratory evaluation on a previously inspected test set; no retraining.')
    (output / 'settings.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    rows = []
    for size in (416, 640):
        metrics = YOLO(str(weights)).val(**common, imgsz=size, project=str(output), name=f'size{size}')
        result = dict(imgsz=size, metrics=metrics.results_dict,
                      per_class=metrics.summary(), speed_ms=metrics.speed)
        (output / f'metrics_{size}.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.item()), encoding='utf-8')
        rows.append(dict(imgsz=size, precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                         mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map)))
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('\nimgsz  Precision  Recall  mAP50  mAP50-95')
    for row in rows:
        print(f"{row['imgsz']:5d}  {row['precision']:.4f}     {row['recall']:.4f}  {row['mAP50']:.4f}  {row['mAP50_95']:.4f}")
    print(f"mAP50-95 change (640 - 416): {(rows[1]['mAP50_95'] - rows[0]['mAP50_95']) * 100:+.2f} percentage points")
    print('DONE. Comparison saved to:', output)


if __name__ == '__main__':
    main()
