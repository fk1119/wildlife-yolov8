"""Relocation checks with synthetic files; does not launch training."""
import json
from pathlib import Path
import tempfile
import unittest
from configure_package_paths import configure


class PathsTests(unittest.TestCase):
    def test_nondefault_location_and_second_move(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'组员 A 项目'
            for split, count in [('train',1753), ('val',383), ('test',360)]:
                images = root/'wildlife8/data/images'/split
                labels = root/'wildlife8/data/labels'/split
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                for i in range(count):
                    (images/f'{i}.jpg').touch()
                    (labels/f'{i}.txt').touch()
            (root/'wildlife8/manifest.json').write_text(json.dumps({'classes':list(range(8))}))
            cfg = root/'wildlife8/dataset.yaml'
            cfg.write_text('old configuration')
            historical = root/'wildlife8/old_run/dataset.yaml'
            historical.parent.mkdir()
            historical.write_text('historical evidence')
            configure(root)
            self.assertEqual(Path(json.loads(cfg.read_text())['path']), root/'wildlife8/data')
            first = cfg.read_bytes()
            configure(root)
            self.assertEqual(first, cfg.read_bytes())
            moved = Path(temp)/'Member B project'
            root.rename(moved)
            configure(moved)
            self.assertEqual(Path(json.loads((moved/'wildlife8/dataset.yaml').read_text(encoding='utf-8'))['path']), moved/'wildlife8/data')
            self.assertEqual((moved/'wildlife8/old_run/dataset.yaml').read_text(), 'historical evidence')
            self.assertEqual((moved/'wildlife8/dataset.yaml.original').read_text(), 'old configuration')


if __name__ == '__main__':
    unittest.main()
