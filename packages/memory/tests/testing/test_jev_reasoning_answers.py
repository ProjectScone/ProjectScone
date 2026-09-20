import json
from pathlib import Path
import runpy

import pytest

from scone_memory.testing.jev_answers import PreparedAnswer
from scone_memory.testing.jev_public_qa import digest, save
from scone_memory.testing.public_qa import Question, benchmark_messages
from scone_memory.testing.public_qa_run import request_digest


async def test_reasoning_pilot_preserves_identical_evidence_and_balanced_schedule(tmp_path, monkeypatch):
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'benchmarks/jev_reasoning_answers.py'))
    source, output = tmp_path/'source', tmp_path/'output'
    source.mkdir()
    prepared = []
    for dataset in ('squad', 'hotpotqa'):
        for i in range(100):
            q = Question(id=f'{dataset}:{i}', dataset=dataset, question=f'Question {i}?')
            request = benchmark_messages(q)
            prepared.append(PreparedAnswer(question=q, arm='hybrid_jev', request=request,
                request_sha256=request_digest(request), receipt={}, sources=(), prepare_ms=0))
    (source/'prepared.jsonl').write_text(''.join(p.model_dump_json()+'\n' for p in prepared))
    save(source/'completion.json', {'terminal': True, 'code_and_inputs_unchanged': True,
                                   'prepared_sha256': digest(source/'prepared.jsonl')})
    assert len(namespace['selected_requests'](source, per_dataset=100)) == 200
    with pytest.raises(ValueError):
        namespace['selected_requests'](source, per_dataset=0)
    calls = []

    async def capture(provider, messages, *, timeout):
        calls.append((request_digest(messages), provider._chat.think, provider._max_output_tokens))
        return {'status': 'completed', 'completed': True, 'answer_text': 'sample', 'total_ms': 1.}

    monkeypatch.setenv('SCONE_JEV_API_KEY_ENV', 'PILOT_TEST_KEY')
    monkeypatch.setenv('PILOT_TEST_KEY', 'not-a-real-key')
    monkeypatch.setitem(namespace['run'].__globals__, 'capture_public_reply', capture)
    await namespace['run'](source, output)
    rows = [json.loads(line) for line in (output/'answers.jsonl').read_text().splitlines()]
    assert len(rows) == 80 and len({r['id'] for r in rows}) == 40
    assert sum(r['dataset'] == 'squad' for r in rows) == 40
    for i in range(0, 80, 2):
        assert calls[i][0] == calls[i+1][0]
        assert {calls[i][1], calls[i+1][1]} == {True, False}
        assert calls[i][2] == calls[i+1][2] == 2048
    assert json.loads((output/'completion.json').read_text())['code_and_inputs_unchanged']
    (source/'prepared.jsonl').write_text('changed evidence')
    with pytest.raises(ValueError, match='evidence changed'):
        namespace['selected_requests'](source)
