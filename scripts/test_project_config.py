import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import project_config as settings
import size_experiment_common as sizes


class ConfigTests(unittest.TestCase):
    def test_relative_paths_follow_config_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'组员 配置'
            root.mkdir()
            cfg = root/'config.json'
            cfg.write_text(json.dumps(settings.CONFIG))
            _, paths, _ = settings.load_config(cfg)
            self.assertEqual(paths['data'], root/'wildlife8/data')
            self.assertEqual(paths['output'], root/'wildlife8')

    def test_reject_invalid_values_and_changed_reference_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)/'config.json'
            for section, key, value in [('training','batch',0), ('runtime','device',-1)]:
                data = copy.deepcopy(settings.CONFIG)
                data[section][key] = value
                cfg.write_text(json.dumps(data))
                with self.assertRaises(ValueError): settings.load_config(cfg)
            data = copy.deepcopy(settings.CONFIG)
            data['experiments']['robustness']['epochs'] = 2
            cfg.write_text(json.dumps(data))
            with self.assertRaises(ValueError): settings.load_config(cfg)

    def test_changed_sizes_device_workers_reach_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root/'initial.pt'; initial.write_bytes(b'initial')
            cfg = root/'data.yaml'; cfg.write_text('names: [bear]')
            config = copy.deepcopy(settings.CONFIG)
            config['experiments']['inference_sizes']['sizes'] = [320, 512]
            events = []
            class FakeModel:
                def __init__(self, path): pass
                def train(self, **kw):
                    events.append(('train',kw))
                    best = Path(kw['project'])/kw['name']/'weights/best.pt'
                    best.parent.mkdir(parents=True); best.write_bytes(b'best')
                def val(self, **kw):
                    events.append(('test',kw))
                    return SimpleNamespace(results_dict={k:.5 for k in sizes.METRICS.values()},
                        summary=lambda:[{'Class':'bear'}],speed={'inference':1.})
            with patch.object(sizes,'CONFIG',config), patch.object(sizes,'DEVICE',2), patch.object(sizes,'WORKERS',3), contextlib.redirect_stdout(io.StringIO()):
                sizes.run_experiment('inference',root/'out',cfg,initial,
                    SimpleNamespace(epochs=2,batch=4,seed=7),{},FakeModel,lambda:None)
            self.assertEqual([kw['imgsz'] for _,kw in events],[320,320,512])
            self.assertTrue(all(kw['device']==2 and kw['workers']==3 for _,kw in events))
            self.assertEqual(events[0][1]['epochs'],2)
            self.assertEqual(events[0][1]['batch'],4)
            self.assertEqual(events[0][1]['seed'],7)


if __name__ == '__main__': unittest.main()
