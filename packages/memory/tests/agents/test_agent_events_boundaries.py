import asyncio
import pytest
from tests.agents.test_agent_events import agent
from tests.agents.test_task_workflow import memory
from tests.agents.test_evidence_tool_loop import binding
from tests.agents.test_agent_turn_recovery import storage
from tests.agents.test_custom_tools import call, tool
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.progress import AgentEventStream
from scone_memory.agents.turn_journal import TurnJournalPaused


async def test_replayed_custom_result_does_not_report_fresh_execution(memory):
    effects = []
    requests = []

    class Model:
        async def complete(self, messages, tools):
            requests.append(True)
            return ToolStep(content='Done') if any(row['role'] == 'tool' for row in messages) else call()

    selected = agent(Model, [tool(lambda args, ctx: effects.append(args['count']))])
    points, _ = storage()
    with pytest.raises(TurnJournalPaused):
        await selected.run('Q', tools=binding(memory), checkpoints=points, max_new_operations=2)
    assert effects == [3]
    events = AgentEventStream()
    result = await selected.run('Q', tools=binding(memory), checkpoints=points, events=events)
    rows = [row async for row in events]
    assert any(row.kind == 'operation_reused' and row.operation_kind == 'custom' for row in rows)
    outcome = next(row for row in rows if row.kind == 'tool_result')
    assert outcome.reused is True, repr(outcome)
    assert outcome.journal_reused is True and outcome.presentation_reused is False
    assert result.output.tool_outcomes[0].reused is False
    assert effects == [3] and len(requests) == 2


def test_started_iterator_cannot_cross_event_loops():
    first = asyncio.new_event_loop()
    second = asyncio.new_event_loop()
    stream = AgentEventStream()
    reader = stream.__aiter__()

    async def begin():
        producer = stream._begin('a', 'm', 'b' * 64)
        producer.finish('turn_completed')
        assert (await anext(reader)).kind == 'turn_started'

    try:
        first.run_until_complete(begin())
        with pytest.raises(RuntimeError, match='another event loop'):
            second.run_until_complete(anext(reader))

        async def remaining():
            return [row async for row in stream]

        assert [row.kind for row in first.run_until_complete(remaining())] == ['turn_completed']
    finally:
        first.run_until_complete(reader.aclose())
        first.close()
        second.close()


async def test_cancel_terminal_waits_for_cleanup_and_reader_detach_keeps_execution(memory):
    entered = asyncio.Event()
    closing = asyncio.Event()
    release = asyncio.Event()
    closed = []

    class Model:
        async def complete(self, messages, tools):
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closing.set()
            await release.wait()
            closed.append(True)

    stream = AgentEventStream()
    reader = stream.__aiter__()
    running = asyncio.create_task(agent(Model).run('Q', tools=binding(memory), events=stream))
    try:
        await entered.wait()
        running.cancel()
        await closing.wait()
        assert not running.done()
        got = []
        while stream._buffer:
            got.append(await anext(reader))
        assert all(not row.kind.startswith('turn_') or row.kind == 'turn_started' for row in got)
        await reader.aclose()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        tail = [row async for row in stream]
        assert closed == [True] and tail[-1].kind == 'turn_cancelled'
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await reader.aclose()


async def test_cleanup_revocation_cannot_emit_turn_completed(memory):
    source = await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})

    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Juniper uses Polaris.')

        async def aclose(self):
            await memory.forget('alpha', source.episode_id)

    stream = AgentEventStream()
    with pytest.raises(RuntimeError, match='evidence'):
        await agent(Model, initial_search=True).run('Juniper', tools=binding(memory), events=stream)
    rows = [row async for row in stream]
    assert rows[-1].kind == 'turn_failed' and not any(row.kind == 'turn_completed' for row in rows)


async def test_replayed_memory_result_never_reruns_search(memory):
    await memory.remember('alpha', 'Juniper uses Polaris.', metadata={'team': 'blue'})

    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done')

    selected = agent(Model, initial_search=True)
    points, _ = storage()
    scoped = binding(memory)
    with pytest.raises(TurnJournalPaused):
        await selected.run('Juniper', tools=scoped, checkpoints=points, max_new_operations=1)

    async def forbidden(*args, **kwargs):
        raise AssertionError('replayed search executed')

    scoped.prepare = forbidden
    stream = AgentEventStream()
    await selected.run('Juniper', tools=scoped, checkpoints=points, events=stream)
    rows = [row async for row in stream]
    assert any(row.kind == 'operation_reused' and row.operation_kind == 'memory' for row in rows)
    outcome = next(row for row in rows if row.kind == 'tool_result')
    assert outcome.reused is True and outcome.journal_reused is True
    assert outcome.presentation_reused is False


async def test_events_leave_native_result_and_selected_binding_unchanged(memory):
    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done') if any(row['role'] == 'tool' for row in messages) else call()

    selected = agent(Model, [tool(lambda args, ctx: args['count'] * 2)])
    plain = await selected.run('Q', tools=binding(memory))
    stream = AgentEventStream(max_events=1)
    observed = await selected.run('Q', tools=binding(memory), events=stream)
    assert observed == plain
    assert stream.dropped_events > 0


async def test_multiple_lag_gaps_partition_every_sequence():
    from scone_memory.agents.progress import AgentProgressGap

    stream = AgentEventStream(max_events=2)
    emitter = stream._begin('a', 'm', 'b' * 64)
    reader = stream.__aiter__()
    represented = []

    def collect(row):
        if isinstance(row, AgentProgressGap):
            represented.extend(range(row.first_sequence, row.last_sequence + 1))
        else:
            represented.append(row.sequence)

    try:
        for _ in range(5):
            emitter.emit('tool_proposed')
        collect(await anext(reader))
        for _ in range(4):
            emitter.emit('tool_result')
        collect(await anext(reader))
        collect(await anext(reader))
        emitter.finish('turn_completed')
        async for row in reader:
            collect(row)
        assert represented == list(range(1, 12))
    finally:
        await reader.aclose()
