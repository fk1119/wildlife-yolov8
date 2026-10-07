"""Stream a child's combined output to the terminal and an exact byte log."""
import codecs
from pathlib import Path
import subprocess
import sys


def run_logged(command, log_path, cwd=None, stream=None):
    stream = sys.stdout if stream is None else stream
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    with Path(log_path).open('wb') as log:
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=0) as process:
            try:
                while True:
                    chunk = process.stdout.read(8192)
                    if not chunk:
                        break
                    log.write(chunk)
                    log.flush()
                    stream.write(decoder.decode(chunk))
                    stream.flush()
                stream.write(decoder.decode(b'', final=True))
                stream.flush()
                returncode = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
