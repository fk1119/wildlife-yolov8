"""Run each training arm in a fresh interpreter, with live output and an exact log.

Ultralytics 8.4.147 lazily imports Events after seeding. Events consumes one
Python random draw only on the first training call in a process. Separate
workers prevent arm order from changing augmentation RNG (matching study 4).
"""
import json
import os
from pathlib import Path
import sys

from live_process import run_logged


def train_isolated(initial, job_dir, **training):
    job_dir = Path(job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=False)
    job = job_dir / 'job.json'
    job.write_text(json.dumps(dict(initial=str(Path(initial).resolve()), training=training),
                              ensure_ascii=False, indent=2), encoding='utf-8')
    run_logged([sys.executable, '-u', '-X', 'utf8', str(Path(__file__).resolve()),
                '--job', str(job)], job_dir / 'console.log')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding='utf-8'))
    # Keep imports before YOLO.train consistent in every fresh worker.
    from ultralytics import YOLO
    (args.job.parent / 'worker.json').write_text(json.dumps(dict(
        pid=os.getpid(), interpreter=sys.executable, isolation='fresh-process'), indent=2), encoding='utf-8')
    print(f'TRAIN WORKER pid={os.getpid()}: {job["training"]["name"]}', flush=True)
    model = YOLO(job['initial'])
    model.train(**job['training'])


if __name__ == '__main__':
    main()
