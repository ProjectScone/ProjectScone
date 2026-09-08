"""Repeated navigation must preserve freshness, isolation, and bounded work."""
import json

import pytest

from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolCall, ToolLoopLimits, ToolStep
from test_chunk_window import document
from test_evidence_tool_loop import Script, binding, search


def read(call_id, chunk_id, **args):
    return ToolStep(calls=(ToolCall(id=call_id, name='read_memory', arguments={'chunk_id':chunk_id, **args}),))


async def setup(engine, monkeypatch):
    await document(engine, metadata={'team':'blue'})
    tools = binding(engine)
    calls = []
    original = tools.prepare

    async def counted(name, arguments):
        calls.append((name, dict(arguments)))
        return await original(name, arguments)

    monkeypatch.setattr(tools, 'prepare', counted)
    model = Script(search())

    async def discovered():
        packet = json.loads(model.requests[-1][0][-1]['content'])
        chunk_id = packet['items'][0]['chunk_id']
        model.steps.extend([read('again', chunk_id, before=1, after=1), ToolStep(content='Grounded answer.')])
        return read('first', chunk_id)

    model.steps.append(discovered)
    return tools, calls, model


async def test_identical_read_reuses_validated_evidence_and_counts_attempt(engine, monkeypatch):
    tools, calls, model = await setup(engine, monkeypatch)
    result = await EvidenceToolLoop(model, tools, limits=ToolLoopLimits(max_tool_calls=3)).run(
        [{'role':'user','content':'Juniper?'}])
    assert [name for name, _ in calls] == ['search_memory', 'read_memory']
    assert result.tool_calls == 3
    assert model.requests[-1][1] == []
    assert len(result.evidence_packets) == 2
    assert result.tool_outcomes[-1].reused is True
    assert result.tool_outcomes[-1].output_bytes < result.tool_outcomes[-2].output_bytes
    packet = json.loads(model.requests[-1][0][-1]['content'])
    assert packet['reuse']['tool_result_number'] == 2
    assert packet['reuse']['new_evidence_count'] == 0
    assert 'items' not in packet
    assert await result.validate()


@pytest.mark.parametrize('changes', [{'after':2}, {'before':0}, {'after':True}, {'unexpected':1}])
async def test_different_or_invalid_read_arguments_never_hit_cache(engine, monkeypatch, changes):
    tools, calls, model = await setup(engine, monkeypatch)
    async def different():
        first = model.steps.pop(0)
        args = dict(first.calls[0].arguments)
        args.update(changes)
        return ToolStep(calls=(ToolCall(id='changed',name='read_memory',arguments=args),))
    original_discovered = model.steps[-1]
    async def discover():
        first = await original_discovered()
        model.steps.insert(0, different)
        return first
    model.steps[-1] = discover
    result = await EvidenceToolLoop(model, tools).run([{'role':'user','content':'Juniper?'}])
    assert [name for name, _ in calls] == ['search_memory','read_memory','read_memory']
    assert result.tool_outcomes[-1].reused is False


async def test_deleted_source_prevents_reuse_before_next_model_request(engine, monkeypatch):
    tools, calls, model = await setup(engine, monkeypatch)
    original_discovered = model.steps[-1]
    async def discover():
        first = await original_discovered()
        duplicate = model.steps.pop(0)
        async def remove():
            packet = json.loads(model.requests[-1][0][-1]['content'])
            await engine.forget('alpha', packet['items'][0]['episode_id'])
            return duplicate
        model.steps.insert(0, remove)
        return first
    model.steps[-1] = discover
    with pytest.raises(RuntimeError, match='evidence changed before reuse'):
        await EvidenceToolLoop(model, tools).run([{'role':'user','content':'Juniper?'}])
    assert len(model.requests) == 3
    assert len(calls) == 2


async def test_read_cache_is_scoped_to_one_run(engine, monkeypatch):
    tools, calls, model = await setup(engine, monkeypatch)
    steps = list(model.steps)
    loop = EvidenceToolLoop(model, tools)
    for _ in range(2):
        model.steps = list(steps)
        result = await loop.run([{'role':'user','content':'Juniper?'}])
        assert result.tool_outcomes[1].reused is False
        assert result.tool_outcomes[2].reused is True
    assert [name for name, _ in calls] == ['search_memory','read_memory'] * 2


@pytest.mark.parametrize("status", ["unavailable", "empty"])
async def test_unsuccessful_reads_are_not_cached(engine, monkeypatch, status):
    from scone_memory.agents.tool_evidence import PreparedToolEvidence

    tools, calls, model = await setup(engine, monkeypatch)
    original = tools.prepare
    async def valid():
        return True
    async def unavailable(name, arguments):
        if name == 'read_memory':
            calls.append((name, dict(arguments)))
            return PreparedToolEvidence(json.dumps({'ok':status=='empty','status':status,'error':'store_error' if status=='unavailable' else None}), (), valid)
        return await original(name, arguments)
    monkeypatch.setattr(tools, 'prepare', unavailable)
    result = await EvidenceToolLoop(model, tools).run([{'role':'user','content':'Juniper?'}])
    assert [name for name, _ in calls] == ['search_memory','read_memory','read_memory']
    assert [row.reused for row in result.tool_outcomes] == [False, False, False]


@pytest.mark.parametrize('external', [False, True])
async def test_reuse_validation_obeys_deadline_and_external_cancellation(external):
    import asyncio
    from scone_memory.agents.tool_evidence import PreparedToolEvidence

    entered = asyncio.Event()
    async def blocking():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return True
    async def valid():
        return True
    class Tools:
        def openai(self):
            return []
        async def prepare(self, name, args):
            packet = json.dumps({'ok':True,'status':'prepared','items':[{'chunk_id':1}]})
            return PreparedToolEvidence(packet, ('chunk:1',), blocking if name=='read_memory' else valid)
    model = Script(search(), read('first',1), read('again',1), ToolStep(content='Must not publish'))
    task = asyncio.create_task(EvidenceToolLoop(model, Tools(),
        limits=ToolLoopLimits(timeout_s=10.0 if external else 0.03)).run([{'role':'user','content':'Juniper?'}]))
    try:
        await asyncio.wait_for(entered.wait(),1)
        if external:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if external else TimeoutError):
            await task
        assert len(model.requests) == 3
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_reuse_receipt_respects_remaining_output_budget():
    from scone_memory.agents.tool_evidence import PreparedToolEvidence

    async def valid():
        return True
    class Tools:
        def openai(self):
            return []
        async def prepare(self, name, arguments):
            payload = json.dumps({'ok':True,'status':'prepared','items':[{'chunk_id':1}]})
            payload = payload.ljust(120 if name=='search_memory' else 260)
            return PreparedToolEvidence(payload, ('chunk:1',), valid)
    model = Script(search(), read('first',1), read('again',1), ToolStep(content='Answer from earlier evidence.'))
    result = await EvidenceToolLoop(model, Tools(), limits=ToolLoopLimits(max_tool_bytes=512)).run(
        [{'role':'user','content':'Juniper?'}])
    assert result.tool_calls == 3 and len(model.requests) == 4
    assert result.tool_outcomes[-1].error == 'tool_output_budget'
    assert result.tool_outcomes[-1].reused is False
    assert len(result.evidence_packets) == 2


@pytest.mark.parametrize('whole_source', [True, False])
async def test_changed_anchor_reuses_only_an_equivalent_whole_source(engine, monkeypatch, whole_source):
    engine.chunk_target = 100
    added = await engine.remember('alpha', 'Juniper ' + 'sample ' * 70, metadata={'team':'blue'})
    chunks = await engine.documents.chunks_of('alpha', added.episode_id)
    assert len(chunks) == 4
    tools = binding(engine)
    calls = []
    original = tools.prepare
    async def counted(name, arguments):
        calls.append(name)
        return await original(name, arguments)
    monkeypatch.setattr(tools,'prepare',counted)
    model = Script(search())
    async def full_read():
        packet = json.loads(model.requests[-1][0][-1]['content'])
        first_id = packet['items'][0]['chunk_id']
        second_id = chunks[-1].chunk_id if first_id != chunks[-1].chunk_id else chunks[0].chunk_id
        model.steps.extend([read('different-anchor',second_id,before=4 if whole_source else 0,
                                 after=4 if whole_source else 0), ToolStep(content='Answer.')])
        return read('full-source',first_id,before=4,after=4)
    model.steps.append(full_read)
    result = await EvidenceToolLoop(model,tools).run([{'role':'user','content':'Juniper?'}])
    assert result.tool_outcomes[-1].reused is whole_source
    assert calls.count('read_memory') == (1 if whole_source else 2)
    assert await result.validate()


async def test_read_rejected_by_output_budget_is_not_cached():
    from scone_memory.agents.tool_evidence import PreparedToolEvidence

    read_attempts = []
    async def valid():
        return True
    class Tools:
        def openai(self):
            return []
        async def prepare(self, name, arguments):
            size = 200
            if name == 'read_memory':
                read_attempts.append(arguments)
                size = 400 if len(read_attempts)==1 else 100
            payload = json.dumps({'ok':True,'status':'prepared','items':[{'chunk_id':1}]}).ljust(size)
            return PreparedToolEvidence(payload, ('chunk:1',), valid)
    model = Script(search(), read('large',1), read('retry',1), ToolStep(content='Answer.'))
    result = await EvidenceToolLoop(model, Tools(), limits=ToolLoopLimits(max_tool_bytes=512)).run(
        [{'role':'user','content':'Juniper?'}])
    assert len(read_attempts) == 2
    assert result.tool_outcomes[1].error == 'tool_output_budget'
    assert result.tool_outcomes[2].status == 'prepared'
    assert result.tool_outcomes[2].reused is False
    assert len(result.evidence_packets) == 2
