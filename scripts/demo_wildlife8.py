"""Run eight-class detection and export per-image counts (not unique animal counts).

Behavior is identical to the original script; this version is restructured for
readability, adds a --device option (CPU fallback), checks the weights file
before loading, and gives clearer errors when inputs or outputs are missing.
"""
import argparse
import csv
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'wildlife8'
DEFAULT_WEIGHTS = DATA_DIR / 'runs/train/weights/best.pt'
DEFAULT_SOURCE = DATA_DIR / 'demo_images'
OUTPUT_DIR = DATA_DIR / 'runs'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

# Must be set before ultralytics is imported.
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.yolo')
os.environ['YOLO_AUTOINSTALL'] = 'false'

from ultralytics import YOLO  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help='Image file or directory (default: wildlife8/demo_images)')
    parser.add_argument('--weights', default=str(DEFAULT_WEIGHTS), help='Path to best.pt')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='NMS overlap threshold; smaller values suppress more overlapping boxes')
    parser.add_argument('--imgsz', type=int, choices=[416, 640], default=416, help='Inference size')
    parser.add_argument('--device', default='0', help="Device: '0' for GPU, 'cpu' otherwise")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error('conf and iou must be between 0 and 1')
    return args


def collect_images(source):
    """Return the sorted list of image paths under a directory, or a single file."""
    if source.is_dir():
        files = sorted(f for f in source.glob('*') if f.suffix.lower() in IMAGE_EXTS)
    else:
        files = [source] if source.suffix.lower() in IMAGE_EXTS else []
    if not files:
        raise FileNotFoundError(f'No input images found at: {source}')
    return files


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f'Weights not found: {weights} (train first, or pass --weights)')

    model = YOLO(str(weights))
    files = collect_images(Path(args.source) if args.source else DEFAULT_SOURCE)

    rows = []
    save_dir = None
    for result in model.predict(
        source=[str(f) for f in files],
        stream=True,
        imgsz=args.imgsz,
        batch=1,
        rect=True,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        project=str(OUTPUT_DIR),
        name='demo',
    ):
        counts = Counter(int(c) for c in result.boxes.cls.cpu().tolist())
        row = {'image': Path(result.path).name}
        row.update({name: counts[i] for i, name in model.names.items()})
        row['total'] = len(result.boxes)
        rows.append(row)
        save_dir = Path(result.save_dir)

    if save_dir is None or not rows:
        raise RuntimeError('Detection produced no results; nothing was saved')

    csv_path = save_dir / 'counts.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f'Processed {len(rows)} image(s)')
    print('Saved detection images and counts.csv:', save_dir)


if __name__ == '__main__':
    main()
