"""Check authored links, source syntax, tracked-size risks and honest experiment status."""
import ast
import fnmatch
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]

def main():
    errors=[]
    for p in (ROOT/'scripts').glob('*.py'):
        try: ast.parse(p.read_text(encoding='utf-8-sig'))
        except SyntaxError as exc: errors.append(f'{p.name}: {exc}')
    authored=[ROOT/'README.md',ROOT/'CODE_ORIGIN.md',ROOT/'DATA_ASSET.md',*list((ROOT/'docs').glob('*.md'))]
    for p in authored:
        for raw in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            raw=raw.strip('<>')
            if '://' in raw or raw.startswith('#'): continue
            target=(p.parent/unquote(raw.split('#')[0])).resolve()
            if not target.exists(): errors.append(f'Broken link in {p.name}: {raw}')
    entries=json.loads((ROOT/'experiments.json').read_text(encoding='utf-8'))
    for entry in entries:
        evidence=ROOT/entry['evidence']
        if not evidence.is_file():errors.append(f'Missing evidence: {evidence}')
        if entry['status']=='incomplete':
            state=json.loads(evidence.read_text(encoding='utf-8'))
            if state.get('status')=='complete':errors.append('Experiment completed: update all status documents before submission.')
    candidates=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT).as_posix()
        if rel.startswith(('.git/','.venv/','wildlife8/data/','.yolo/')):continue
        if p.suffix in {'.pt','.zip','.log','.cache','.pyc','.original'}:continue
        if '__pycache__' in p.parts or p.name=='dataset.yaml':continue
        candidates.append(p)
        if p.stat().st_size>=25*1024*1024:errors.append(f'File exceeds browser upload size: {rel}')
    if errors:raise SystemExit('\n'.join(errors))
    print(f'PASS: source syntax, authored links and experiment evidence; {len(candidates)} candidate files, {sum(p.stat().st_size for p in candidates)/1024**2:.1f} MiB.')
    print('Experiment index and evidence paths verified.')

if __name__=='__main__': main()
