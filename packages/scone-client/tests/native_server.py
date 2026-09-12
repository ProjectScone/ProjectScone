"""Real Python API contracts, with an independently selected server interpreter."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import pytest
import requests

from scone import Scone


@contextmanager
def native_server(state, fixture):
    configured = os.environ.get('SCONE_TEST_NATIVE_PYTHON')
    if not configured:
        pytest.skip('set SCONE_TEST_NATIVE_PYTHON to run the real Python native server')
    interpreter = shutil.which(configured)
    if interpreter is None:
        pytest.fail('SCONE_TEST_NATIVE_PYTHON must name an executable Python interpreter')
    root = Path(__file__).resolve().parents[2]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    base = 'http://127.0.0.1:' + str(port)
    env = {**os.environ, 'PYTHONPATH': str(root / 'memory' / 'src')}
    with (state / 'server.log').open('a+') as log:
        process = subprocess.Popen([interpreter, '-u', str(Path(__file__).parent / 'fixtures' / fixture),
                                    str(state), str(port)], env=env, stdin=subprocess.PIPE, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0)
                    pytest.fail('native server exited: ' + log.read())
                try:
                    if requests.get(base + '/healthz', timeout=0.2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.025)
            else:
                pytest.fail('native server did not start')
            with Scone(base, 'agent-fixture') as client:
                yield client
        finally:
            if process.poll() is None:
                try:
                    process.communicate(b'stop\n', timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)
                    pytest.fail('native server did not shut down')
            assert process.returncode == 0

