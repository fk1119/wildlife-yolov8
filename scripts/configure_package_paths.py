"""Configure the primary dataset for fresh experiments at this checkout location.

Historical experiment YAML/JSON files are evidence and are not rewritten.
After moving a checkout, start a new run rather than resume an old run.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / 'wildlife8/manifest.json').read_text(encoding='utf-8'))
    data = root / 'wildlife8/data'
    for split, count in [('train', 1753), ('val', 383), ('test', 360)]:
        images = [p for p in (data / 'images' / split).iterdir()
                  if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
        if len(images) != count:
            raise ValueError(f'{split}: expected {count} images, found {len(images)}')
        for image in images:
            label = data / 'labels' / split / (image.stem + '.txt')
            if not label.is_file():
                raise FileNotFoundError(label)
    config = dict(path=str(data), train='images/train', val='images/val',
                  test='images/test', names=dict(enumerate(manifest['classes'])))
    cfg = root / 'wildlife8/dataset.yaml'
    payload = json.dumps(config, ensure_ascii=False, indent=2)
    if not cfg.exists() or cfg.read_text(encoding='utf-8') != payload:
        backup = cfg.with_name('dataset.yaml.original')
        if cfg.exists() and not backup.exists():
            backup.write_bytes(cfg.read_bytes())
        cfg.write_text(payload, encoding='utf-8')
    print(f'PASS: dataset configured for {root}; use new experiment directories.')
    return config


if __name__ == '__main__':
    from project_config import dataset_config
    print('Configured:', dataset_config())
