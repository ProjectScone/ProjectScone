import json

import pytest

from scone_memory.agents.evidence_loop import ToolCall, ToolStep
from scone_memory.testing.jev_agent_answers import run_trial, validate_trials
from scone_memory.testing.jev_public_qa import SPACE
from scone_memory.testing.public_qa import Question


async def test_trial_runs_native_search_and_read_and_retains_transcript(engine):
    await engine.remember(SPACE, 'Morgan founded Cedar.', kind='file')
    await engine.remember('other', 'PRIVATE_MARKER', kind='file')

    class Reader:
        calls = 0

        async def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 1:
                packet = json.loads(messages[-1]['content'])
                return ToolStep(calls=(ToolCall(id='read', name='read_memory',
                    arguments={'chunk_id': packet['items'][0]['chunk_id']}),))
            return ToolStep(content='Morgan')

    question = Question(id='test', dataset='squad', question='Who founded Cedar?')
    result = await run_trial(engine, Reader(), question)
    assert result['completed'] and result['answer_text'] == 'Morgan'
    assert result['model_calls'] == 2 and result['tool_calls'] == 2
    assert [r['name'] for r in result['tool_outcomes']] == ['search_memory', 'read_memory']
    assert len(result['turns']) == 2
    assert 'Morgan founded Cedar.' in json.dumps(result['evidence_packets'])
    assert 'PRIVATE_MARKER' not in json.dumps(result)


async def test_trial_records_provider_failure_without_error_body(engine):
    await engine.remember(SPACE, 'Cedar exists.', kind='file')

    class Failing:
        async def complete(self, messages, tools):
            raise RuntimeError('secret provider error body')

    result = await run_trial(engine, Failing(), Question(id='test', dataset='squad', question='Cedar?'))
    assert result['completed'] is False and result['error_type'] == 'RuntimeError'
    assert result['answer_text'] == ''
    assert 'secret provider error body' not in json.dumps(result)
    assert result['turns'][0]['error_type'] == 'RuntimeError'


@pytest.mark.parametrize('ids', [[], ['a', 'a'], ['a', 'extra']])
def test_schedule_rejects_missing_duplicate_or_foreign_questions(ids):
    with pytest.raises(ValueError):
        validate_trials([{'id': key} for key in ids], {'a', 'b'})
