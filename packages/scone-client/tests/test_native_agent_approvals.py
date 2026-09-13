"""Real SDK/native protocol persists exact proposals through two restarts."""

import json
import time

import pytest

from scone import HandoffAgent, HandoffPlan, HumanInput, ModelTask, TaskPlan
from native_server import native_server


def settled(agents, expected):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = agents.status('one')
        if status.status == expected and not status.active_local:
            status.match(agents.request('one'))
            return status
        time.sleep(0.01)
    pytest.fail('run did not reach ' + expected + ': ' + repr(status))


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.integration
@pytest.mark.parametrize('kind', ['task', 'interactive', 'handoff'])
@pytest.mark.parametrize('decision', ['approve', 'deny'])
def test_native_approval_restart_and_exact_continuation(tmp_path, kind, decision):
    task = ModelTask('send', 'worker', 'careful', 'Send Hello')
    plan = (
        HandoffPlan('review', 'worker', (HandoffAgent('worker', 'careful'),), max_handoffs=0)
        if kind == 'handoff'
        else TaskPlan(
            'review', (task, HumanInput('review', 'Review', ('send',))) if kind == 'interactive' else (task,)
        )
    )
    with native_server(tmp_path, 'agent_approval_server.py') as client:
        agents = client.agents(expected_space='alpha')
        assert client.capabilities().supports('agents.approvals')
        saved = agents.save_plan(plan, expected_revision=0)
        agents.start('one', plan=saved, question='Send a note')
        settled(agents, 'paused')
        (pending,) = agents.approvals('one')
        assert pending.call.model_id == 'careful' and pending.call.arguments() == {'message': 'Hello'}
        assert len(lines(tmp_path / 'models.jsonl')) == 1 and not lines(tmp_path / 'effects.jsonl')
    with native_server(tmp_path, 'agent_approval_server.py') as client:
        agents = client.agents(expected_space='alpha')
        assert agents.approvals('one') == (pending,)
        decided = agents.decide_tool(pending, decision=decision)
        assert decided.revision == 2 and decided.decision_digest
        assert len(lines(tmp_path / 'models.jsonl')) == 1 and not lines(tmp_path / 'effects.jsonl')
    with native_server(tmp_path, 'agent_approval_server.py') as client:
        agents = client.agents(expected_space='alpha')
        assert agents.approvals('one') == (decided,)
        admitted = agents.continue_tools('one', continuation_id='exact', decisions=(decided,))
        settled(agents, 'awaiting_input' if kind == 'interactive' else 'completed')
        (consumed,) = agents.approvals('one')
        assert consumed.revision == 4 and consumed.decision_digest == decided.decision_digest
        repeated = agents.continue_tools('one', continuation_id='exact', decisions=(decided,))
        assert repeated.activation == admitted.activation
        assert agents.decide_tool(pending, decision=decision) == consumed
        models = lines(tmp_path / 'models.jsonl')
        assert len(models) == 2 and {row['model'] for row in models} == {'careful'}
        assert len(lines(tmp_path / 'effects.jsonl')) == (1 if decision == 'approve' else 0)
