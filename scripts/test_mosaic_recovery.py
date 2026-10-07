"""Regression tests; no GPU, real training, or project result changes."""
import argparse
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import run_mosaic_experiment as runner


class RecoveryTests(unittest.TestCase):
    def test_nested_numpy_json(self):
        value = {'per_class': [{'Instances': np.int64(12), 'AP': np.float32(.5)}],
                 'array': np.array([1, 2])}
        self.assertEqual(json.loads(json.dumps(value, default=runner.json_default)),
                         {'per_class': [{'Instances': 12, 'AP': .5}], 'array': [1, 2]})
        with self.assertRaises(TypeError):
            json.dumps({'bad': object()}, default=runner.json_default)

    def run_case(self, completed=True):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            initial = root / 'initial.pt'
            initial.write_bytes(b'initial')
            cfg = root / 'dataset.yaml'
            cfg.write_text('fixture')
            group = root / 'on'
            weights = group / 'train/weights/best.pt'
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b'preserved model')
            args = argparse.Namespace(epochs=30, imgsz=416, batch=8, seed=42, resume_run=root)
            record = dict(condition='mosaic_on', initial_sha256=runner.sha256(initial),
                dataset_manifest_sha256=runner.MANIFEST_SHA, dataset=str(cfg),
                configuration=dict(epochs=30, imgsz=416, batch=8, mosaic=1, seed=42,
                    optimizer='AdamW', lr0=.001, weight_decay=.0005, workers=0, amp=False, close_mosaic=0))
            (group/'run_record.json').write_text(json.dumps(record))
            (group/'train/results.csv').write_text('epoch\n' + '\n'.join(map(str, range(1, 31 if completed else 30))))
            calls = []
            class FakeYOLO:
                def __init__(self, path):
                    calls.append(('load', path))
                def train(self, **kwargs):
                    raise AssertionError('Completed model must never be retrained')
                def val(self, **kwargs):
                    calls.append(('val', kwargs))
                    return types.SimpleNamespace(results_dict={'AP': np.float64(.75)},
                        summary=lambda: [{'Instances': np.int64(10)}], speed={'inference': np.float32(2)})
            fake_torch = types.SimpleNamespace(__version__='fixture',
                cuda=types.SimpleNamespace(is_available=lambda:False, empty_cache=lambda:None))
            with patch.dict('sys.modules', {'torch':fake_torch, 'ultralytics':types.SimpleNamespace(YOLO=FakeYOLO)}), patch.object(runner, 'train_isolated', side_effect=AssertionError('Must reuse existing model')):
                if not completed:
                    with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                        runner.train_and_test(initial, cfg, group, 1, args)
                    self.assertEqual(calls, [])
                    return
                runner.train_and_test(initial, cfg, group, 1, args)
                result = json.loads((group/'test_results.json').read_text())
                self.assertEqual(result['per_class'][0]['Instances'], 10)
                self.assertEqual(weights.read_bytes(), b'preserved model')
                self.assertEqual(len(calls), 2)
                runner.train_and_test(initial, cfg, group, 1, args)
                self.assertEqual(len(calls), 2, 'Saved matching evaluation should be reused')
                args.batch = 4
                with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                    runner.train_and_test(initial, cfg, group, 1, args)

    def test_recover_evaluation_without_training(self):
        self.run_case()

    def test_reject_partial_training(self):
        self.run_case(completed=False)

    def test_missing_group_trains_then_saves_nested_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            initial = root/'initial.pt'
            initial.write_bytes(b'initial')
            group = root/'off'
            args = argparse.Namespace(epochs=30, imgsz=416, batch=8, seed=42, resume_run=root)
            calls = []
            class FakeYOLO:
                def __init__(self, path):
                    pass
                def train(self, **kwargs):
                    calls.append(kwargs)
                    weights = Path(kwargs['project'])/'train/weights/best.pt'
                    weights.parent.mkdir(parents=True)
                    weights.write_bytes(b'new model')
                def val(self, **kwargs):
                    return types.SimpleNamespace(results_dict={'AP': np.float64(.7)},
                        summary=lambda: [{'Instances': np.int64(5)}], speed={})
            fake_torch = types.SimpleNamespace(__version__='fixture',
                cuda=types.SimpleNamespace(is_available=lambda:False, empty_cache=lambda:None))
            def fake_train(initial, job_dir, **kw):
                FakeYOLO(str(initial)).train(**kw)
            with patch.dict('sys.modules', {'torch':fake_torch, 'ultralytics':types.SimpleNamespace(YOLO=FakeYOLO)}), patch.object(runner, 'train_isolated', fake_train):
                runner.train_and_test(initial, root/'data.yaml', group, 0, args)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['mosaic'], 0)
            self.assertEqual(json.loads((group/'test_results.json').read_text())['per_class'][0]['Instances'], 5)


if __name__ == '__main__':
    unittest.main()
