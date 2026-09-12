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

from scone import Scone, TaskPlan, HumanInput, ModelTask


@contextmanager
def server(state):
    configured = os.environ.get('SCONE_TEST_NATIVE_PYTHON')
    if not configured:
        pytest.skip('set SCONE_TEST_NATIVE_PYTHON to run the real Python agent server')
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
        process = subprocess.Popen([interpreter, '-u', str(Path(__file__).parent / 'fixtures' / 'agent_server.py'),
                                    str(state), str(port)], env=env, stdin=subprocess.PIPE, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0)
                    pytest.fail('agent server exited: ' + log.read())
                try:
                    if requests.get(base + '/healthz', timeout=0.2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.025)
            else:
                pytest.fail('agent server did not start')
            with Scone(base, 'agent-fixture') as client:
                yield client.agents(expected_space='alpha')
        finally:
            if process.poll() is None:
                try:
                    process.communicate(b'stop\n', timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)
                    pytest.fail('agent server did not shut down')
            assert process.returncode == 0


def paused(agents, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = agents.status(run_id)
        if status.status == 'awaiting_input' and not status.active_local:
            return status
        time.sleep(0.01)
    pytest.fail('run did not pause for input')


@pytest.mark.integration
def test_restart_reply_and_explicit_model_continuation(tmp_path):
    with server(tmp_path) as agents:
        assert {model.model_id for model in agents.catalog()[0].models} == {'fast', 'careful'}
        plan = TaskPlan('review', (HumanInput('choose', 'Choose'),
            ModelTask('answer', 'worker', 'careful', 'Answer', ('choose',))))
        saved = agents.save_plan(plan, expected_revision=0)
        assert agents.plan('review') == saved and agents.plans().items == (saved,)
        agents.start('one', plan=saved, question='Plan')
        paused(agents, 'one').match(agents.request('one'))
        pending, = agents.inputs('one')
        answered = agents.respond(pending, response='é' * 2000)
        assert answered.revision == 2
        assert not (tmp_path / 'calls.jsonl').exists()
    with server(tmp_path) as agents:
        assert agents.inputs('one') == (answered,)
        assert not (tmp_path / 'calls.jsonl').exists()
        agents.continue_run('one', continuation_id='approval', responses=(answered,))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = agents.status('one')
            if status.status == 'completed' and not status.active_local:
                break
            time.sleep(0.01)
        assert status.status == 'completed'
        activated, = agents.inputs('one')
        assert activated.revision == 3 and activated.activation_id == 'approval'
        # Explicit retries retain their original identity and never execute again.
        assert agents.respond(pending, response=answered.response) == activated
        agents.continue_run('one', continuation_id='approval', responses=(answered,))
        calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
        assert len(calls) == 1 and calls[0]['model'] == 'careful'
        assert answered.response in json.dumps(calls[0]['messages'], ensure_ascii=False)
        saved = agents.plan('review')
        agents.start('cancelled', plan=saved, question='Stop')
        paused(agents, 'cancelled')
        assert agents.cancel('cancelled').status == 'cancelled'
        assert {run.status for run in agents.runs().items} == {'completed', 'cancelled'}
