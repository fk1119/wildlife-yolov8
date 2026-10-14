"""Unified entry: python run.py mosaic [--check-only]."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    aliases = {'experiment1_inference_sizes': 'inference_sizes', 'experiment2_training_sizes': 'training_sizes', 'experiment3_mosaic': 'mosaic', 'experiment4_robustness': 'robustness', 'experiment5_blur_ratios': 'blur_ratios'}
    parser.add_argument('experiment', choices=list(aliases) + list(aliases.values()))
    parser.add_argument('--config', type=Path, default=ROOT/'config.json')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume-run', type=Path, help='Mosaic only: same-machine interrupted run')
    args = parser.parse_args()
    args.experiment = aliases.get(args.experiment, args.experiment)
    if args.resume_run and args.experiment != 'mosaic':
        parser.error('--resume-run is only supported for mosaic')
    os.environ['WILDLIFE_CONFIG'] = str(args.config.resolve())
    sys.path.insert(0, str(ROOT/'scripts'))
    from project_config import CONFIG_FILE, OUTPUT, BASELINE
    scripts = dict(inference_sizes='experiment1_inference_sizes.py', training_sizes='experiment2_training_sizes.py',
                   mosaic='experiment3_mosaic.py', blur_ratios='experiment5_blur_ratios.py', robustness='experiment4_robustness.py')
    command = [sys.executable, '-u', '-X', 'utf8', str(ROOT/'scripts'/scripts[args.experiment])]
    if args.check_only:
        if args.experiment == 'robustness':
            from size_experiment_common import preflight
            preflight()
            for file in ('weights/best.pt', 'args.yaml'):
                if not (BASELINE/file).is_file():
                    raise FileNotFoundError(BASELINE/file)
            print('CHECK PASSED. No training started.')
            return
        command.append('--check-only')
    if args.resume_run:
        command += ['--resume-run', str(args.resume_run.resolve())]
    if args.experiment == 'robustness':
        out = OUTPUT / 'experiment4_robustness' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        command += ['--output', str(out)]
    print(f'Configuration: {CONFIG_FILE}', flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if args.experiment == 'robustness':
        subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/summarize_experiment4_robustness.py'),
                        '--output', str(out)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
