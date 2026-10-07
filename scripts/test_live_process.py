import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from live_process import run_logged


class LiveOutputTests(unittest.TestCase):
    def test_output_arrives_before_process_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            signal = root/'received'
            class Sink(io.StringIO):
                def write(self, value):
                    result = super().write(value)
                    if '进度' in self.getvalue(): signal.touch()
                    return result
            script = root/'child.py'
            script.write_text("import sys,time\nfrom pathlib import Path\n"
                "sys.stdout.write('进度 1/2\\r');sys.stdout.flush()\n"
                "for _ in range(100):\n"
                " if Path('received').exists():break\n"
                " time.sleep(.02)\n"
                "else: raise RuntimeError('Output was buffered until exit')\n"
                "sys.stderr.write('完成\\n')\n",encoding='utf-8')
            sink = Sink()
            run_logged([sys.executable,'-u','-X','utf8',str(script)],root/'log',cwd=root,stream=sink)
            expected = '进度 1/2\r完成' + os.linesep
            self.assertEqual(sink.getvalue(),expected)
            self.assertEqual((root/'log').read_bytes(),expected.encode('utf-8'))

    def test_error_keeps_output_and_exit_code(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/'log'
            sink = io.StringIO()
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                run_logged([sys.executable,'-u','-c',"import sys;print('failure',file=sys.stderr);sys.exit(7)"],log,stream=sink)
            self.assertEqual(caught.exception.returncode,7)
            self.assertIn('failure',sink.getvalue())
            self.assertIn(b'failure',log.read_bytes())


if __name__ == '__main__': unittest.main()
