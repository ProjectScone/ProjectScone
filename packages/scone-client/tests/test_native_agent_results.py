"""Client result reads preserve scoped evidence, real hops and passive restarts."""
import json
import time

import pytest

from scone import TaskPlan, ModelTask, HandoffPlan, HandoffAgent, SconeError
from native_server import native_server


def completed(agents, run_id):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        status = agents.status(run_id)
        if status.status == 'completed' and not status.active_local:
            return
        time.sleep(0.01)
    pytest.fail('agent run did not complete')


@pytest.mark.integration
def test_retained_source_results_are_read_only_and_refuse_forgotten_evidence(tmp_path):
    with native_server(tmp_path, 'result_server.py') as client:
        source = client.add('Ada studies stars.')
        agents = client.agents(expected_space='alpha')
        plan = agents.save_plan(TaskPlan('research', (ModelTask('answer', 'researcher', 'careful', 'Explain Ada.'),)),
                                expected_revision=0)
        agents.start('one', plan=plan, question='Ada stars')
        completed(agents, 'one')
        result = agents.result('one')
        output = result.results['answer']
        assert output.source_status == 'retained' and output.evidence_ids
        assert output.evidence_packets[0].payload['status'] == 'prepared'
        count = len((tmp_path / 'calls.jsonl').read_text().splitlines())
    with native_server(tmp_path, 'result_server.py') as client:
        agents = client.agents(expected_space='alpha')
        assert agents.result('one').results == result.results
        assert len((tmp_path / 'calls.jsonl').read_text().splitlines()) == count
        client._request('DELETE', '/v1/episodes/' + str(source.episode_id))
        with pytest.raises(SconeError):
            agents.result('one')
        assert len((tmp_path / 'calls.jsonl').read_text().splitlines()) == count


@pytest.mark.integration
def test_actual_handoffs_and_limit_keep_final_answer_semantics(tmp_path):
    with native_server(tmp_path, 'result_server.py') as client:
        agents = client.agents(expected_space='alpha')
        policies = (HandoffAgent('relay', 'relay', ('finisher',)), HandoffAgent('finisher', 'finish'))
        for limit in (0, 1):
            saved = agents.save_plan(HandoffPlan('handoff-' + str(limit), 'relay', policies, limit), expected_revision=0)
            run_id = 'run-' + str(limit)
            agents.start(run_id, plan=saved, question='Delegate this')
            completed(agents, run_id)
            result = agents.result(run_id)
            assert len(result.hops) == limit + 1
            if limit == 0:
                assert result.status == 'handoff_limit' and result.final is None
            else:
                assert result.status == 'completed' and result.final.text == 'Completed answer.'
                assert [hop.output.model_id for hop in result.hops] == ['relay', 'finish']
        calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
        assert [call['model'] for call in calls] == ['relay', 'relay', 'finish']
