"""Single configuration loader. Relative resource paths are relative to config.json."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path or os.environ.get('WILDLIFE_CONFIG', ROOT / 'config.json')).resolve()
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    required = {'paths', 'runtime', 'training', 'experiments'}
    if set(value) != required:
        raise ValueError(f'Configuration sections must be {sorted(required)}')
    for key in ('epochs', 'batch', 'seed'):
        number = value['training'][key]
        if type(number) is not int or number < (0 if key == 'seed' else 1):
            raise ValueError(f'Invalid training.{key}')
    for key in ('device', 'workers'):
        if type(value['runtime'][key]) is not int or value['runtime'][key] < 0:
            raise ValueError(f'runtime.{key} must be a nonnegative integer')
    for key in ('inference_sizes', 'training_sizes'):
        sizes = value['experiments'][key]['sizes']
        if len(sizes) != 2 or len(set(sizes)) != 2 or any(type(n) is not int or n < 32 or n % 32 for n in sizes):
            raise ValueError(f'{key}.sizes requires two different positive multiples of 32')
    for key in ('mosaic', 'blur_ratios'):
        n = value['experiments'][key]['imgsz']
        if type(n) is not int or n < 32 or n % 32:
            raise ValueError(f'Invalid {key}.imgsz')
    seeds = value['experiments']['blur_ratios']['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(n) is not int or n < 0 for n in seeds):
        raise ValueError('blur_ratios.seeds must be unique nonnegative integers')
    # This study reuses the reference baseline; varying its protocol would invalidate pairing.
    if value['experiments']['robustness'] != dict(epochs=30, batch=8, imgsz=416, seeds=[42,43]):
        raise ValueError('Robustness uses a fixed reference protocol: 30 epochs, batch8, 416, seeds42/43')
    paths = {}
    for key in ('data', 'initial_weights', 'output', 'baseline_run'):
        raw = value['paths'][key]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f'Empty paths.{key}')
        candidate = Path(raw).expanduser()
        paths[key] = (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
    return value, paths, path


CONFIG, PATHS, CONFIG_FILE = load_config()
DATA = PATHS['data']
INITIAL = PATHS['initial_weights']
OUTPUT = PATHS['output']
BASELINE = PATHS['baseline_run']
DEVICE = CONFIG['runtime']['device']
WORKERS = CONFIG['runtime']['workers']
TRAINING = CONFIG['training']


def dataset_config():
    """Create only the active dataset mapping; do not modify historical run records."""
    import yaml
    names = json.loads((ROOT / 'wildlife8/manifest.json').read_text(encoding='utf-8'))['classes']
    value = dict(path=str(DATA), train='images/train', val='images/val', test='images/test',
                 names=dict(enumerate(names)))
    target = ROOT / 'wildlife8/dataset.yaml'
    if not target.exists() or yaml.safe_load(target.read_text(encoding='utf-8')) != value:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(value, allow_unicode=True), encoding='utf-8')
    return target


def snapshot(directory):
    directory = Path(directory)
    value = dict(configuration=CONFIG, resolved_paths={k:str(v) for k,v in PATHS.items()})
    (directory / 'effective_config.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
