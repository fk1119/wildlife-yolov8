import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from import_resources import import_archive, safe_target


class ResourceImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='wildlife_import_test_')
        self.root = Path(self.temp.name)/'project'
        (self.root/'wildlife8').mkdir(parents=True)
        (self.root/'wildlife8/manifest.json').write_text(json.dumps({'classes':['lion']}))
        self.payloads={'assets/yolov8n.pt':b'fixture only, not model bytes',
                       'wildlife8/data/labels/train/lion.txt':b'0 .5 .5 .3 .3'}
        records=[dict(path=p,size=len(b),sha256=hashlib.sha256(b).hexdigest()) for p,b in self.payloads.items()]
        (self.root/'resources_manifest.json').write_text(json.dumps(records))
        self.archive=Path(self.temp.name)/'resources.zip'

    def tearDown(self):
        self.temp.cleanup()

    def make_zip(self,prefix='',tamper=False):
        with zipfile.ZipFile(self.archive,'w') as z:
            for name,data in self.payloads.items():
                if tamper and name.endswith('.pt'):data=b'x'*len(data)
                z.writestr(prefix+name,data)

    def test_import_and_repeated_import(self):
        self.make_zip()
        import_archive(self.archive,self.root)
        import_archive(self.archive,self.root)
        for name,data in self.payloads.items():self.assertEqual((self.root/name).read_bytes(),data)
        cfg=json.loads((self.root/'wildlife8/dataset.yaml').read_text(encoding='utf-8'))
        self.assertEqual(Path(cfg['path']),self.root/'wildlife8/data')

    def test_full_backup_top_folder(self):
        self.make_zip(prefix='full_backup/')
        import_archive(self.archive,self.root)
        self.assertTrue((self.root/'assets/yolov8n.pt').exists())

    def test_tampered_archive_rejected(self):
        self.make_zip(tamper=True)
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):import_archive(self.archive,self.root)
        self.assertFalse((self.root/'assets/yolov8n.pt').exists())

    def test_different_existing_file_is_preserved(self):
        self.make_zip()
        (self.root/'assets').mkdir()
        target=self.root/'assets/yolov8n.pt';target.write_bytes(b'keep my model')
        with self.assertRaisesRegex(ValueError,'refusing to overwrite'):import_archive(self.archive,self.root)
        self.assertEqual(target.read_bytes(),b'keep my model')

    def test_traversal_rejected(self):
        for name in ['../outside','/absolute','C:/outside','..\\outside']:
            with self.subTest(name=name),self.assertRaises(ValueError):safe_target(self.root,name)

if __name__=='__main__':unittest.main()
