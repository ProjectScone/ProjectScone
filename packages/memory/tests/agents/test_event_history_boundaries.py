from dataclasses import replace
import warnings
import pytest
from tests.agents.test_event_history import run_request, events
from scone_memory.agents.event_history import AgentEventHistoryStore
from scone_memory.agents.workflow import WorkflowError


@pytest.mark.parametrize('field', ['kind', 'reused'])
async def test_malformed_event_refusal_never_warns_private_values(tmp_path, run_request, field):
    marker = 'PRIVATE_EVENT_CREDENTIAL'
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        row = replace((await events())[0], **{field: marker})
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            with pytest.raises(WorkflowError, match='invalid_history_event'):
                store.append(run_request, step_id='find', selection_id='find', event=row)
        assert not any(marker in str(item.message) for item in captured), [
            str(item.message) for item in captured
        ]
    finally:
        store.close()


async def test_malformed_request_refusal_never_warns_private_values(tmp_path, run_request):
    marker = 'PRIVATE_REQUEST_CREDENTIAL'
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        bad = run_request.model_copy(update={'max_parallel': marker})
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            with pytest.raises(WorkflowError, match='invalid_history_request'):
                store.read(bad)
        assert not any(marker in str(item.message) for item in captured), [
            str(item.message) for item in captured
        ]
    finally:
        store.close()


@pytest.mark.parametrize('mode', ['json', 'python'])
async def test_checked_request_is_used_for_selection(tmp_path, run_request, mode):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        copied = run_request.model_copy(update={'plan': run_request.plan.model_dump(mode=mode)})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                store.append(copied, step_id='find', selection_id='find', event=(await events())[0])
        except WorkflowError:
            pass
        except AttributeError as error:
            pytest.fail('validated detached request discarded before selection: ' + str(error))
    finally:
        store.close()


async def test_cyclic_native_field_refuses_without_recursive_detachment(tmp_path, run_request):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    value = []
    value.append(value)
    try:
        row = replace((await events())[0], kind=value)
        with pytest.raises(WorkflowError, match='invalid_history_event'):
            store.append(run_request, step_id='find', selection_id='find', event=row)
    finally:
        store.close()


async def test_reuse_metadata_cannot_contradict_itself(tmp_path, run_request):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        row = replace(
            (await events())[0],
            kind='tool_result',
            tool_index=1,
            tool_name='search_memory',
            origin='host',
            status='prepared',
            output_bytes=1,
            reused=False,
            journal_reused=True,
            presentation_reused=False,
        )
        with pytest.raises(WorkflowError, match='invalid_history_event'):
            store.append(run_request, step_id='find', selection_id='find', event=row)
    finally:
        store.close()


async def test_handoff_first_hop_cannot_be_attributed_to_nonroot_agent(tmp_path):
    from scone_memory.agents.catalog import AgentCatalog, AgentModel, AgentDefinition
    from scone_memory.agents.handoff_workflow import AgentHandoffPlan, HandoffAgent
    from scone_memory.agents.plan_store import AgentPlanStore
    from scone_memory.agents.run_store import AgentRunStore
    from scone_memory.retrieval.recall_scope import RecallScope

    def forbidden():
        raise AssertionError('no model')

    catalog = AgentCatalog(
        models=[AgentModel('local', 'Local', '1', forbidden)],
        agents=[
            AgentDefinition(agent_id=name, instructions='Answer', models=('local',), default_model='local')
            for name in ('research', 'write')
        ],
    )
    plans = AgentPlanStore(tmp_path / 'plans.db', key=b'k' * 32)
    runs = AgentRunStore(tmp_path / 'runs.db', key=b'k' * 32)
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        plan = AgentHandoffPlan(
            workflow_id='h',
            root_agent='research',
            agents=(
                HandoffAgent(agent_id='research', model_id='local', can_handoff_to=('write',)),
                HandoffAgent(agent_id='write', model_id='local', can_handoff_to=('research',)),
            ),
        )
        saved = plans.save('alpha', plan, catalog=catalog, expected_revision=0)
        request = runs.register('alpha', 'h-run', plan=saved, question='Q', scope=RecallScope.validated())
        row = replace((await events())[0], agent_id='write', binding=catalog.bind('write').fingerprint)
        with pytest.raises(WorkflowError, match='history_selection_mismatch'):
            store.append(request, step_id='hop-01', selection_id='write', event=row)
    finally:
        store.close()
        runs.close()
        plans.close()


async def test_manifest_write_failure_rolls_back_event_and_retention(tmp_path, run_request, monkeypatch):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32, max_events=1)
    try:
        row = (await events())[0]
        store.append(run_request, step_id='find', selection_id='find', event=row)
        before = store.read(run_request)
        original = store._storage._seal

        def fail(token, payload):
            if token.startswith('state:'):
                raise OSError('manifest unavailable')
            return original(token, payload)

        monkeypatch.setattr(store._storage, '_seal', fail)
        with pytest.raises(WorkflowError, match='history_store_unavailable'):
            store.append(run_request, step_id='find', selection_id='find', event=row)
        assert store.read(run_request) == before
    finally:
        store.close()


@pytest.mark.parametrize('field', ['step_id', 'selection_id'])
async def test_invalid_envelope_identity_refuses_before_recursive_encoding(tmp_path, run_request, field):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    value = 0
    for _ in range(1200):
        value = [value]
    try:
        options = {'step_id': 'find', 'selection_id': 'find', 'event': (await events())[0], field: value}
        with pytest.raises(WorkflowError, match='invalid_history_event'):
            store.append(run_request, **options)
    finally:
        store.close()


@pytest.mark.parametrize('nested', [False, True])
async def test_bad_space_in_unchecked_request_is_private(tmp_path, run_request, nested):
    marker = 'PRIVATE SPACE CREDENTIAL'
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        bad = (
            run_request.model_copy(update={'plan': run_request.plan.model_copy(update={'space': marker})})
            if nested
            else run_request.model_copy(update={'space': marker})
        )
        with pytest.raises(WorkflowError, match='invalid_history_request'):
            store.read(bad)
    finally:
        store.close()


async def test_real_bound_agent_events_roundtrip_all_observed_fields(tmp_path, run_request):
    from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
    from scone_memory.agents.catalog import AgentCatalog, AgentModel, AgentDefinition
    from scone_memory.agents.evidence_loop import ToolStep
    from scone_memory.agents.progress import AgentEventStream
    from scone_memory.integrations.scoped_tools import ScopedMemoryTools

    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done')

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        await memory.remember('alpha', 'Juniper uses Polaris.')
        catalog = AgentCatalog(
            models=[AgentModel('local', 'Local', '1', Model)],
            agents=[
                AgentDefinition(
                    agent_id='research',
                    instructions='Use authorized evidence.',
                    models=('local',),
                    default_model='local',
                )
            ],
        )
        stream = AgentEventStream()
        await catalog.bind('research').run(
            run_request.question,
            tools=ScopedMemoryTools(memory, 'alpha', scope=run_request.recall_scope()),
            events=stream,
        )
        rows = [row async for row in stream]
        assert {
            'tool_proposed',
            'tool_result',
            'operation_started',
            'operation_completed',
            'turn_completed',
        } <= {row.kind for row in rows}
        for row in rows:
            store.append(run_request, step_id='find', selection_id='find', event=row)
        assert [entry.event for entry in store.read(run_request).items] == rows
    finally:
        store.close()
        await memory.close()
