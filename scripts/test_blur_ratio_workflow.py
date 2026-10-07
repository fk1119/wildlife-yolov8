"""Small synthetic images and fake YOLO: no real training or inference."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import yaml
import experiment3_blur_ratios as experiment


class BlurStudyTest(unittest.TestCase):
    def test_nested_assignment(self):
        groups = experiment.assignments(1753)
        self.assertEqual([len(groups[r]) for r in experiment.RATIOS], [0, 438, 876])
        self.assertTrue(groups[25].items() <= groups[50].items())
        self.assertEqual(groups, experiment.assignments(1753))

    def test_complete_workflow_with_fake_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val', 'test'):
                folder = root / 'original/images' / split
                labels = root / 'original/labels' / split
                folder.mkdir(parents=True)
                labels.mkdir(parents=True)
                for i in range(4):
                    pixels = np.random.default_rng(i).integers(0, 256, (24, 32, 3), dtype=np.uint8)
                    ok, encoded = cv2.imencode('.png', pixels)
                    self.assertTrue(ok)
                    encoded.tofile(folder / f'{i}.png')
                    (labels / f'{i}.txt').write_text('0 0.5 0.5 0.25 0.25\n')
            cfg = root / 'dataset.yaml'
            cfg.write_text(yaml.safe_dump(dict(path=str(root / 'original'), train='images/train',
                val='images/val', test='images/test', names={0: 'bear'})), encoding='utf-8')
            initial = root / 'initial.pt'
            initial.write_bytes(b'initial')
            events = []

            class FakeYOLO:
                def __init__(self, weights):
                    self.weights = weights

                def train(self, **kw):
                    events.append(('train', self.weights, kw))
                    best = Path(kw['project']) / kw['name'] / 'weights/best.pt'
                    best.parent.mkdir(parents=True)
                    best.write_bytes(kw['name'].encode())

                def val(self, **kw):
                    events.append(('val', self.weights, kw))
                    return SimpleNamespace(results_dict={k: .75 for k in experiment.METRICS.values()},
                        summary=lambda: [{'Class': 'bear', 'mAP50-95': .75}], speed={'inference': 1})

            out = root / 'experiment'
            console = io.StringIO()
            jobs = []
            def fake_training(initial, job_dir, **kw):
                jobs.append(job_dir)
                FakeYOLO(str(initial)).train(**kw)
            with patch('ultralytics.YOLO', FakeYOLO), patch.object(experiment, 'train_isolated', fake_training), patch.object(experiment, 'release_gpu'), contextlib.redirect_stdout(console):
                experiment.run(out, cfg, initial, {}, SimpleNamespace(epochs=30, batch=8, seeds=[42, 43]))
            self.assertEqual(len(set(jobs)), 6)
            training = [e for e in events if e[0] == 'train']
            testing = [e for e in events if e[0] == 'val' and e[2]['split'] == 'test']
            validation = [e for e in events if e[0] == 'val' and e[2]['split'] == 'val']
            self.assertEqual(len(training), 6)
            self.assertTrue(all(e[2]['verbose'] for e in events))
            self.assertIn('PREPARE train_blur0/train: 0/4', console.getvalue())
            self.assertIn('PREPARE train_blur0/train: 4/4', console.getvalue())
            self.assertIn('TRAIN 6/6:', console.getvalue())
            self.assertIn('EVALUATE 84/84:', console.getvalue())
            self.assertTrue(all(e[1] == str(initial) for e in training))
            self.assertEqual(len(testing), 60)
            self.assertEqual(len(validation), 24)
            self.assertTrue(all(e[0] == 'train' for e in events[:6]))
            self.assertEqual(json.loads((out / 'validation_selection.json').read_text())['selected_ratio'], 0)
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'complete')
            self.assertTrue((out / 'comparison.csv').is_file())
            for ratio in experiment.RATIOS:
                records = json.loads((out / f'data/train_blur{ratio}/manifest.json').read_text())
                self.assertEqual(sum(r['condition'] != 'clean' for r in records), 4 * ratio // 100)
                for r in records:
                    result = experiment.read_image(out / f'data/train_blur{ratio}/images/train' / r['file'])
                    original = experiment.read_image(Path(r['source']))
                    self.assertEqual(result.shape, original.shape)
                    if r['condition'] == 'clean':
                        np.testing.assert_array_equal(original, result)


if __name__ == '__main__':
    unittest.main()
