"""Real agent API contracts across a process restart."""
from contextlib import contextmanager
import json
import time

import pytest

from scone import TaskPlan, HumanInput, ModelTask
from native_server import native_server


@contextmanager
def server(state):
    with native_server(state, 'agent_server.py') as client:
        yield client.agents(expected_space='alpha')


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
