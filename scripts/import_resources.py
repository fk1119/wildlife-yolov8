"""Import allowlisted data and checkpoints from a local ZIP without training."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def safe_target(root, name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or '..' in relative.parts or '\\' in name
            or ':' in name or not relative.parts):
        raise ValueError(f'Unsafe resource path: {name}')
    target = (root / Path(*relative.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Path leaves project: {name}')
    return target


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def import_archive(archive, root=ROOT):
    records = json.loads((root / 'resources_manifest.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        name_set = set(names)
        # Both the resource archive and earlier full backup are supported.
        marker = 'assets/yolov8n.pt'
        prefixes = {n[:-len(marker)] for n in names if n == marker or n.endswith('/' + marker)}
        if len(prefixes) != 1:
            raise ValueError('ZIP must contain exactly one project assets/yolov8n.pt')
        prefix = prefixes.pop()
        plan = []
        for record in records:
            target = safe_target(root, record['path'])
            member = prefix + record['path']
            if member not in name_set or names.count(member) != 1:
                raise ValueError(f'Missing or duplicate ZIP entry: {member}')
            if z.getinfo(member).file_size != record['size']:
                raise ValueError(f'Unexpected resource size: {member}')
            if target.exists():
                if not target.is_file() or digest(target) != record['sha256']:
                    raise ValueError(f'Existing different file; refusing to overwrite: {target}')
            else:
                plan.append((record, target, member))
        for i, (record, target, member) in enumerate(plan, 1):
            target.parent.mkdir(parents=True, exist_ok=True)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.part', delete=False) as output:
                    temp_path = Path(output.name)
                    h = hashlib.sha256()
                    with z.open(member) as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b''):
                            h.update(chunk)
                            output.write(chunk)
                if h.hexdigest() != record['sha256']:
                    raise ValueError(f'Hash mismatch: {member}')
                if target.exists():
                    raise ValueError(f'Target appeared during import: {target}')
                # rename fails if another process created target on Windows.
                temp_path.rename(target)
                temp_path = None
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()
            if i % 500 == 0:
                print(f'Imported {i}/{len(plan)}', flush=True)
    data = json.loads((root / 'wildlife8/manifest.json').read_text(encoding='utf-8'))
    config = dict(path=str((root / 'wildlife8/data').resolve()), train='images/train',
                  val='images/val', test='images/test', names=dict(enumerate(data['classes'])))
    # JSON is a YAML-compatible mapping; use standard library only for import.
    cfg = root / 'wildlife8/dataset.yaml'
    backup = cfg.with_name('dataset.yaml.original')
    if cfg.exists() and not backup.exists():
        backup.write_bytes(cfg.read_bytes())
    cfg.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (root / '.yolo').mkdir(exist_ok=True)
    print(f'PASS: {len(records)} resources verified, {len(plan)} imported; dataset path configured.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    import_archive(args.archive.resolve())


if __name__ == '__main__':
    main()
