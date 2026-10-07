import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import isolated_training as worker


class WorkerTests(unittest.TestCase):
    def test_each_arm_launches_new_interpreter_with_exact_job(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(worker, 'run_logged') as launch:
            root = Path(tmp)
            for name in ('first', 'second'):
                worker.train_isolated(root / 'initial.pt', root / name, name=name,
                                      seed=42, data=str(root / '中文 路径/data.yaml'))
            self.assertEqual(launch.call_count, 2)
            for call in launch.call_args_list:
                command, log = call.args
                self.assertEqual(command[:4], [sys.executable, '-u', '-X', 'utf8'])
                self.assertEqual(Path(command[4]), Path(worker.__file__).resolve())
                job = Path(command[6])
                payload = json.loads(job.read_text(encoding='utf-8'))
                self.assertEqual(payload['training']['seed'], 42)
                self.assertEqual(payload['training']['data'], str(root / '中文 路径/data.yaml'))
                self.assertEqual(log, job.parent / 'console.log')
            with self.assertRaises(FileExistsError):
                worker.train_isolated(root/'initial.pt', root/'first', name='first')

    def test_child_failure_propagates(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(worker, 'run_logged',
                side_effect=subprocess.CalledProcessError(1, ['python'])):
            with self.assertRaises(subprocess.CalledProcessError):
                worker.train_isolated(Path(tmp)/'initial.pt', Path(tmp)/'job', name='bad')


if __name__ == '__main__':
    unittest.main()
