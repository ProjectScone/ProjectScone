"""The CLI must close owned storage before honoring a deployment SIGTERM."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest

from ..paths import PACKAGE_ROOT


@pytest.mark.parametrize('ignore', [False, True])
def test_termination_respects_an_existing_host_handler(ignore):
    from scone_memory.api._signals import termination_unwinds
    observed = []
    handler = signal.SIG_IGN if ignore else lambda signum, frame: observed.append(signum)
    previous = signal.signal(signal.SIGTERM, handler)
    try:
        with termination_unwinds():
            signal.raise_signal(signal.SIGTERM)
        assert observed == ([] if ignore else [signal.SIGTERM])
        assert signal.getsignal(signal.SIGTERM) == handler
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.mark.parametrize('host_kind', ['memory', 'composed', 'conversations'])
def test_sigterm_closes_owned_engine_before_exit(tmp_path, host_kind):
    marker = tmp_path / 'closed'
    launcher = tmp_path / 'launch.py'
    code = '''
import asyncio
from pathlib import Path
import sys
from scone_memory.api import __main__ as host

original_build = host.build_engine
async def build(settings):
    engine = await original_build(settings)
    original_close = engine.close
    async def close():
        await original_close()
        Path(sys.argv[1]).write_text("closed")
    engine.close = close
    return engine
host.build_engine = build
host.main()
'''
    if host_kind == 'conversations':
        code = '''
from pathlib import Path
import sys
from scone_memory.api import conversation_server as host
from scone_memory.runtime.config import Settings
original_close = host.close_engine
async def close(engine):
    await original_close(engine)
    Path(sys.argv[1]).write_text("closed")
host.close_engine = close
sys.exit(host.main(Settings.from_env(), journal=sys.argv[1] + ".journal"))
'''
    launcher.write_text(code)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = {key: value for key, value in os.environ.items() if not key.startswith('SCONE_')}
    env.update(SCONE_SQLITE_PATH=str(tmp_path / 'memory.db'), SCONE_EMBEDDER='hash',
               SCONE_API_KEY='termination-fixture', SCONE_HOST='127.0.0.1', SCONE_PORT=str(port),
               PYTHONPATH=str(PACKAGE_ROOT / 'src'))
    if host_kind == 'composed':
        env['SCONE_CONVERSATIONS_JOURNAL'] = str(tmp_path / 'journal.db')
    with (tmp_path / 'server.log').open('w') as log:
        process = subprocess.Popen([sys.executable, str(launcher), str(marker)], env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=1) as client:
                while True:
                    assert process.poll() is None, (tmp_path / 'server.log').read_text()
                    try:
                        if client.get('/healthz').status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    assert time.monotonic() < deadline, 'server startup timed out'
                    time.sleep(.025)
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=10)
            assert marker.exists(), 'SIGTERM bypassed the engine close boundary'
            assert process.returncode in (0, 143)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
