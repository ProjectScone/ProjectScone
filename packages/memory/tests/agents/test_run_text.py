"""A run's answer, readable as it is written, from the service that runs it.

The loop delivers a step's public text to a sink; the run service owns a
process-local window per running step, keyed by space, run and step, so a
reader can follow the answer being written and a later reader gets the
receipt instead. What the window holds is provisional: it is never stored,
it does not survive a restart, and the verified result is what `result`
returns. Withdrawn text -- streamed before a turn that then called tools --
is marked in the window, so a reader clears it rather than keeping it.
"""

import asyncio

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from scone_memory.agents.evidence_loop import ToolCall, ToolStep
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_service import AgentRunService
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan
from scone_memory.retrieval.recall_scope import RecallScope

pytestmark = pytest.mark.asyncio

KEY = b'k' * 32


class Scripted:
    """A streaming model: each turn writes its pieces to the sink it is
    handed, waits at `hold` when asked, and returns its step."""

    def __init__(self, turns, entered: asyncio.Event, release: asyncio.Event, hold_turn: int = 0):
        self.turns, self.entered, self.release, self.hold_turn = list(turns), entered, release, hold_turn
        self.handed: list[bool] = []
        self.turn = 0

    async def complete(self, messages, tools, *, on_public_text=None):
        self.handed.append(on_public_text is not None)
        pieces, step = self.turns.pop(0)
        for index, piece in enumerate(pieces):
            if on_public_text is not None:
                await on_public_text(piece)
            if self.turn == self.hold_turn and index == 0:
                self.entered.set()
                await self.release.wait()
        if not pieces and self.turn == self.hold_turn:
            self.entered.set()
            await self.release.wait()
        self.turn += 1
        if isinstance(step, Exception):
            raise step
        return step


async def service_for(tmp_path, factory, *, public_text: bool):
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    catalog = AgentCatalog(models=[AgentModel('local', 'Local', '1', factory)], agents=[AgentDefinition(
        agent_id='research', instructions='Use evidence.', models=('local',), default_model='local', initial_search=False)])
    plans = AgentPlanStore(tmp_path / 'plans.db', key=KEY)
    plan = AgentTaskPlan(workflow_id='report', tasks=(AgentTask(task_id='find', agent_id='research', prompt='Find evidence.'),))
    plans.save('alpha', plan, catalog=catalog, expected_revision=0)
    service = AgentRunService(tmp_path / 'runs', key=KEY, catalog=catalog, plans=plans, memory=memory,
                              scope_for=lambda space: RecallScope.validated(), max_active=1, public_text=public_text)
    return service, plans, catalog, memory


async def test_a_running_step_offers_its_public_text_as_it_is_written(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    model = Scripted([(['Answer ', 'from model.'], ToolStep(content='Answer from model.'))], entered, release)
    service, plans, _, memory = await service_for(tmp_path, lambda: model, public_text=True)
    try:
        await service.start('alpha', 'one', workflow_id='report', plan_revision=1, question='Question')
        await entered.wait()
        window = service.text_window('alpha', 'one', 'find')
        assert window is not None and not window.closed
        assert window.next_after(0) == (None, (1, 'Answer '))
        release.set()
        await service.wait('alpha', 'one')
        assert (await service.result('alpha', 'one')).results['find']['text'] == 'Answer from model.'
        assert window.last_sequence == 2 and window.closed and not window.failed
        assert model.handed == [True]
    finally:
        release.set(); await service.aclose(); plans.close(); await memory.close()


async def test_without_public_text_no_window_exists_and_the_model_is_not_handed_a_sink(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    model = Scripted([(['x'], ToolStep(content='Answer'))], entered, release)
    service, plans, _, memory = await service_for(tmp_path, lambda: model, public_text=False)
    try:
        await service.start('alpha', 'one', workflow_id='report', plan_revision=1, question='Question')
        await entered.wait()
        assert service.text_window('alpha', 'one', 'find') is None
        release.set(); await service.wait('alpha', 'one')
        assert model.handed == [False]
    finally:
        release.set(); await service.aclose(); plans.close(); await memory.close()


async def test_a_window_is_scoped_to_its_space_run_and_step(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    model = Scripted([(['x'], ToolStep(content='Answer'))], entered, release)
    service, plans, _, memory = await service_for(tmp_path, lambda: model, public_text=True)
    try:
        await service.start('alpha', 'one', workflow_id='report', plan_revision=1, question='Question')
        await entered.wait()
        assert service.text_window('bravo', 'one', 'find') is None
        assert service.text_window('alpha', 'two', 'find') is None
        assert service.text_window('alpha', 'one', 'other') is None
        assert service.text_window('alpha', 'one', 'find') is not None
    finally:
        release.set(); await service.aclose(); plans.close(); await memory.close()


async def test_provisional_text_does_not_survive_the_service(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    release.set()
    model = Scripted([(['x'], ToolStep(content='Answer'))], entered, release)
    service, plans, catalog, memory = await service_for(tmp_path, lambda: model, public_text=True)
    try:
        await service.start('alpha', 'one', workflow_id='report', plan_revision=1, question='Question')
        await service.wait('alpha', 'one')
        await service.aclose()
        reopened = AgentRunService(tmp_path / 'runs', key=KEY, catalog=catalog, plans=plans, memory=memory,
                                   scope_for=lambda space: RecallScope.validated(), public_text=True)
        try:
            assert (await reopened.status('alpha', 'one')).status == 'completed'
            assert reopened.text_window('alpha', 'one', 'find') is None, 'provisional text is not durable, by design'
        finally:
            await reopened.aclose()
    finally:
        plans.close(); await memory.close()


async def test_text_written_before_a_tool_turn_is_marked_withdrawn_in_the_window(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    search = ToolStep(calls=(ToolCall(id='s1', name='search_memory', arguments={'query': 'decision'}),))
    model = Scripted([(['Looking that up.'], search), (['We decided.'], ToolStep(content='We decided.'))], entered, release, hold_turn=1)
    service, plans, _, memory = await service_for(tmp_path, lambda: model, public_text=True)
    try:
        await service.start('alpha', 'one', workflow_id='report', plan_revision=1, question='Question')
        await entered.wait()
        window = service.text_window('alpha', 'one', 'find')
        assert window is not None
        assert window.next_after(0) == (None, (1, 'Looking that up.'))
        assert window.next_after(1) == (None, (2, None)), 'a withdrawal is a chunk with no text'
        assert window.next_after(2) == (None, (3, 'We decided.'))
        release.set(); await service.wait('alpha', 'one')
        assert (await service.result('alpha', 'one')).results['find']['text'] == 'We decided.'
    finally:
        release.set(); await service.aclose(); plans.close(); await memory.close()


async def test_a_failed_step_marks_its_window_failed_and_closed(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    model = Scripted([(['Half an '], RuntimeError('model gone'))], entered, release)
    service, plans, _, memory = await service_for(tmp_path, lambda: model, public_text=True)
    try:
        await service.start('alpha', 'one', workflow_id='report', plan_revision=1, question='Question')
        await entered.wait()
        window = service.text_window('alpha', 'one', 'find')
        assert window is not None
        release.set(); await service.wait('alpha', 'one')
        assert (await service.status('alpha', 'one')).status == 'failed'
        assert window.closed and window.failed
        assert window.next_after(0) == (None, None), 'a failed window offers no text'
    finally:
        release.set(); await service.aclose(); plans.close(); await memory.close()


# --- the registry itself ------------------------------------------------------

async def test_a_step_run_again_gets_a_new_window_and_the_old_one_is_closed():
    from scone_memory.agents.run_text import AgentRunText

    registry = AgentRunText()
    first = registry.open('alpha', 'one', 'find')
    await first.append('provisional')
    second = registry.open('alpha', 'one', 'find')
    assert first.window.closed and not first.window.failed, 'no reader keeps following the old window'
    assert registry.window('alpha', 'one', 'find') is second.window and not second.window.closed


async def test_closing_the_registry_fails_every_open_window_and_leaves_closed_ones_alone():
    from scone_memory.agents.run_text import AgentRunText

    registry = AgentRunText()
    done = registry.open('alpha', 'one', 'find')
    registry.close('alpha', 'one', 'find', failed=False)
    open_window = registry.open('alpha', 'two', 'find')
    registry.close_all()
    assert done.window.closed and not done.window.failed
    assert open_window.window.closed and open_window.window.failed
