"""No GPU/training required: verify checkpoint identity and experiment wiring."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from size_experiment_common import METRICS, run_experiment


class WorkflowTest(unittest.TestCase):
    def exercise(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / 'initial.pt'
            initial.write_bytes(b'initial')
            cfg = root / 'dataset.yaml'
            cfg.write_text('names: [bear]', encoding='utf-8')
            events = []

            class FakeYOLO:
                def __init__(self, path):
                    self.path = path

                def train(self, **kw):
                    events.append(('train', self.path, kw))
                    best = Path(kw['project']) / kw['name'] / 'weights/best.pt'
                    best.parent.mkdir(parents=True)
                    best.write_bytes(str(kw['imgsz']).encode())

                def val(self, **kw):
                    events.append(('test', self.path, kw))
                    return SimpleNamespace(results_dict={k: .7 for k in METRICS.values()},
                                           summary=lambda: [{'Class': 'bear', 'mAP50-95': .7}],
                                           speed={'inference': 10.})

            args = SimpleNamespace(epochs=30, batch=8, seed=42)
            out = root / 'new'
            def fake_train(initial, job_dir, **kw):
                FakeYOLO(str(initial)).train(**kw)
            with contextlib.redirect_stdout(io.StringIO()):
                rows = run_experiment(mode, out, cfg, initial, args, {}, FakeYOLO, lambda: None, fake_train)
            training = [e for e in events if e[0] == 'train']
            testing = [e for e in events if e[0] == 'test']
            self.assertEqual([e[2]['imgsz'] for e in training], [416] if mode == 'inference' else [416, 640])
            self.assertTrue(all(e[1] == str(initial) for e in training))
            self.assertEqual([e[2]['imgsz'] for e in testing], [416, 640])
            self.assertTrue(all(e[2]['split'] == 'test' for e in testing))
            self.assertEqual(testing[0][1] == testing[1][1], mode == 'inference')
            self.assertEqual(len(rows), 2)
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'complete')
            for name in ('comparison.csv', 'per_class.csv', 'comparison.json', '实验对照.md'):
                self.assertTrue((out / name).is_file())
            with self.assertRaises(FileExistsError):
                run_experiment(mode, out, cfg, initial, args, {}, FakeYOLO, lambda: None)

    def test_same_checkpoint_two_test_sizes(self):
        self.exercise('inference')

    def test_fresh_models_with_matching_test_sizes(self):
        self.exercise('training')


if __name__ == '__main__':
    unittest.main()
