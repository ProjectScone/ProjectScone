"""Encrypted replay of observed metadata never executes a saved run."""

from contextlib import closing
from dataclasses import replace
import sqlite3

import pytest

from scone_memory.agents.event_history import AgentEventHistoryStore
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.progress import AgentEventStream, AgentProgressGap
from scone_memory.agents.run_store import AgentRunStore
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope
from tests.agents.test_run_store import catalog, plan


@pytest.fixture
def run_request(tmp_path):
    plans = AgentPlanStore(tmp_path / 'plans.db', key=b'k' * 32)
    runs = AgentRunStore(tmp_path / 'runs.db', key=b'k' * 32)
    try:
        saved = plans.save('alpha', plan(), catalog=catalog(), expected_revision=0)
        yield runs.register(
            'alpha', 'run_request-1', plan=saved, question='PRIVATE question', scope=RecallScope.validated()
        )
    finally:
        runs.close()
        plans.close()


async def events(capacity=128):
    stream = AgentEventStream(max_events=capacity)
    emitter = stream._begin('research', 'local', catalog().bind('research').fingerprint)
    emitter.finish('turn_completed')
    return [row async for row in stream]


async def test_restart_pagination_privacy_and_no_fabricated_history(tmp_path, run_request):
    path = tmp_path / 'history.db'
    store = AgentEventHistoryStore(path, key=b'k' * 32)
    assert store.read(run_request).available is False
    rows = await events()
    for row in rows:
        store.append(run_request, step_id='find', selection_id='find', event=row)
    first = store.read(run_request, limit=1)
    assert first.available and first.retained_from == 1
    assert first.items[0].event == rows[0] and first.items[0].position == 1
    assert first.items[0].step_id == first.items[0].selection_id == 'find'
    store.close()
    assert b'research' not in path.read_bytes() and b'PRIVATE' not in path.read_bytes()
    store = AgentEventHistoryStore(path, key=b'k' * 32)
    try:
        second = store.read(run_request, after=first.next_after)
        assert [row.event for row in second.items] == rows[1:]
        tail = store.read(run_request, after=second.next_after)
        assert tail.items == () and tail.next_after == second.next_after
    finally:
        store.close()


async def test_retention_floor_and_purge_invalidate_old_generation(tmp_path, run_request):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32, max_events=2)
    try:
        rows = await events()
        store.append(run_request, step_id='find', selection_id='find', event=rows[0])
        cursor = store.read(run_request).next_after
        for _ in range(3):
            store.append(run_request, step_id='find', selection_id='find', event=rows[1])
        page = store.read(run_request, after=cursor)
        assert page.retained_from == 3 and [row.position for row in page.items] == [3, 4]
        assert page.omitted == (2, 2)
        store.purge(run_request)
        assert not store.read(run_request).available
        with pytest.raises(WorkflowError, match='invalid_history_cursor'):
            store.read(run_request, after=cursor)
        store.append(run_request, step_id='find', selection_id='find', event=rows[0])
        with pytest.raises(WorkflowError, match='invalid_history_cursor'):
            store.read(run_request, after=cursor)
    finally:
        store.close()


async def test_native_loss_gap_is_preserved_without_invented_events(tmp_path, run_request):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        rows = await events(1)
        assert isinstance(rows[0], AgentProgressGap)
        for row in rows:
            store.append(run_request, step_id='find', selection_id='find', event=row)
        assert [row.event for row in store.read(run_request).items] == rows
    finally:
        store.close()


async def test_cursor_and_request_are_bound_to_exact_run(tmp_path, run_request):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        store.append(run_request, step_id='find', selection_id='find', event=(await events())[0])
        cursor = store.read(run_request).next_after
        other = run_request.model_copy(update={'run_id': 'run_request-2'})
        assert not store.read(other).available
        for changed in [other, run_request.model_copy(update={'question': 'Changed'})]:
            with pytest.raises(WorkflowError):
                store.read(changed, after=cursor)
        with pytest.raises(WorkflowError, match='history_run_conflict'):
            store.append(
                run_request.model_copy(update={'question': 'Changed'}),
                step_id='find',
                selection_id='find',
                event=(await events())[0],
            )
        with pytest.raises(WorkflowError, match='invalid_history_cursor'):
            store.read(run_request, after=cursor[:-1] + ('0' if cursor[-1] != '0' else '1'))
    finally:
        store.close()


@pytest.mark.parametrize('mutation', ['delete', 'swap', 'corrupt'])
async def test_missing_or_modified_rows_are_not_silent_gaps(tmp_path, run_request, mutation):
    path = tmp_path / 'history.db'
    store = AgentEventHistoryStore(path, key=b'k' * 32)
    for row in await events():
        store.append(run_request, step_id='find', selection_id='find', event=row)
    store.close()
    with closing(sqlite3.connect(path)) as db, db:
        rows = db.execute(
            "SELECT token,payload FROM agent_history WHERE token LIKE 'event:%' ORDER BY token"
        ).fetchall()
        assert len(rows) == 2
        if mutation == 'delete':
            db.execute('DELETE FROM agent_history WHERE token=?', (rows[0][0],))
        else:
            db.execute(
                'UPDATE agent_history SET payload=? WHERE token=?',
                (rows[1][1] if mutation == 'swap' else b'broken', rows[0][0]),
            )
    store = AgentEventHistoryStore(path, key=b'k' * 32)
    try:
        with pytest.raises(WorkflowError, match='history_key_or_integrity'):
            store.read(run_request)
    finally:
        store.close()


@pytest.mark.parametrize(
    'change',
    [
        {'sequence': True},
        {'sequence': 0},
        {'elapsed_s': float('nan')},
        {'tool_name': 'PRIVATE text'},
        {'error': 'PRIVATE error'},
        {'occurred_at': 'yesterday'},
    ],
)
async def test_invalid_native_metadata_is_rejected_before_storage(tmp_path, run_request, change):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        row = replace((await events())[0], **change)
        with pytest.raises(WorkflowError, match='invalid_history_event'):
            store.append(run_request, step_id='find', selection_id='find', event=row)
        assert not store.read(run_request).available
    finally:
        store.close()


async def test_only_declared_selection_and_model_binding_can_be_recorded(tmp_path, run_request):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        row = (await events())[0]
        for kwargs in [
            dict(selection_id='other', event=row),
            dict(selection_id='find', event=replace(row, binding='b' * 64)),
            dict(selection_id='find', event=replace(row, model_id='other')),
        ]:
            with pytest.raises(WorkflowError, match='history_selection_mismatch'):
                store.append(run_request, step_id='find', **kwargs)
    finally:
        store.close()


async def test_concurrent_writers_allocate_one_contiguous_history(tmp_path, run_request):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / 'history.db'
    AgentEventHistoryStore(path, key=b'k' * 32).close()
    ready = Barrier(2)
    row = (await events())[0]

    def append():
        store = AgentEventHistoryStore(path, key=b'k' * 32)
        try:
            ready.wait(timeout=10)
            return store.append(run_request, step_id='find', selection_id='find', event=row)
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        one, two = [future.result(timeout=15) for future in [pool.submit(append), pool.submit(append)]]
    assert sorted([one.position, two.position]) == [1, 2]
    store = AgentEventHistoryStore(path, key=b'k' * 32)
    try:
        assert [row.position for row in store.read(run_request).items] == [1, 2]
    finally:
        store.close()


async def test_store_capacity_limits_and_reopen_with_smaller_retention(tmp_path, run_request):
    path = tmp_path / 'history.db'
    store = AgentEventHistoryStore(path, key=b'k' * 32, max_histories=1)
    row = (await events())[0]
    try:
        store.append(run_request, step_id='find', selection_id='find', event=row)
        with pytest.raises(WorkflowError, match='history_store_limit'):
            store.append(
                run_request.model_copy(update={'run_id': 'second'}),
                step_id='find',
                selection_id='find',
                event=row,
            )
        store.append(run_request, step_id='find', selection_id='find', event=row)
    finally:
        store.close()
    store = AgentEventHistoryStore(path, key=b'k' * 32, max_events=1)
    try:
        store.append(run_request, step_id='find', selection_id='find', event=row)
        page = store.read(run_request)
        assert page.retained_from == 3 and page.omitted == (1, 2)
        assert [row.position for row in page.items] == [3]
    finally:
        store.close()
    with pytest.raises(WorkflowError, match='history_key_or_integrity'):
        AgentEventHistoryStore(path, key=b'x' * 32)


async def test_ciphertext_from_purged_generation_cannot_be_replayed(tmp_path, run_request):
    path = tmp_path / 'history.db'
    store = AgentEventHistoryStore(path, key=b'k' * 32)
    row = (await events())[0]
    try:
        store.append(run_request, step_id='find', selection_id='find', event=row)
        with closing(sqlite3.connect(path)) as db:
            original = db.execute(
                "SELECT token,payload FROM agent_history WHERE token LIKE 'event:%'"
            ).fetchone()
        store.purge(run_request)
        store.append(run_request, step_id='find', selection_id='find', event=row)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('UPDATE agent_history SET payload=? WHERE token=?', (original[1], original[0]))
        with pytest.raises(WorkflowError, match='history_key_or_integrity'):
            store.read(run_request)
    finally:
        store.close()


async def test_missing_manifest_and_future_or_malformed_cursors_refuse(tmp_path, run_request):
    path = tmp_path / 'history.db'
    store = AgentEventHistoryStore(path, key=b'k' * 32)
    try:
        store.append(run_request, step_id='find', selection_id='find', event=(await events())[0])
        for cursor in ['', 'x' * 1000, True, 1]:
            with pytest.raises(WorkflowError, match='invalid_history_cursor'):
                store.read(run_request, after=cursor)
        token, digest, _ = store._identity(run_request)
        with store._storage._access() as db:
            state = store._manifest(db, token, digest)
        with pytest.raises(WorkflowError, match='invalid_history_cursor'):
            store.read(run_request, after=store._cursor(token, state, 50))
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("DELETE FROM agent_history WHERE token LIKE 'state:%'")
        with pytest.raises(WorkflowError, match='history_key_or_integrity'):
            store.read(run_request)
    finally:
        store.close()


async def test_space_isolation_and_malformed_request_copies(tmp_path, run_request):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        store.append(run_request, step_id='find', selection_id='find', event=(await events())[0])
        cursor = store.read(run_request).next_after
        bravo = run_request.model_copy(
            update={'space': 'bravo', 'plan': run_request.plan.model_copy(update={'space': 'bravo'})}
        )
        assert not store.read(bravo).available
        with pytest.raises(WorkflowError, match='invalid_history_cursor'):
            store.read(bravo, after=cursor)
        with pytest.raises(WorkflowError, match='invalid_history_request'):
            store.read(run_request.model_copy(update={'max_parallel': True}))
    finally:
        store.close()


@pytest.mark.parametrize(
    'change',
    [
        {'kind': 'PRIVATE kind'},
        {'operation_kind': 'PRIVATE operation'},
        {'reused': 3},
        {'origin': 'PRIVATE origin'},
    ],
)
async def test_malformed_dataclass_fields_fail_without_serializer_warnings(tmp_path, run_request, change):
    store = AgentEventHistoryStore(tmp_path / 'history.db', key=b'k' * 32)
    try:
        with pytest.raises(WorkflowError, match='invalid_history_event'):
            store.append(
                run_request, step_id='find', selection_id='find', event=replace((await events())[0], **change)
            )
    finally:
        store.close()


async def test_maximum_trap_report_survives_encrypted_store(tmp_path, run_request):
    from scone_memory.agents.traps import ObservationGraph

    detector = ObservationGraph(2)
    for index in range(15):
        assert detector.observe(str(index).encode()) is None
    graph = detector.observe(b'14')
    event = replace((await events())[0], kind='trap_detected', trap_graph=graph)
    path = tmp_path / 'trap-history.db'
    store = AgentEventHistoryStore(path, key=b'k' * 32)
    try:
        store.append(run_request, step_id='find', selection_id='find', event=event)
    finally:
        store.close()
    assert b'trap_detected' not in path.read_bytes()
    with closing(AgentEventHistoryStore(path, key=b'k' * 32)) as reopened:
        assert reopened.read(run_request).items[0].event == event


@pytest.mark.parametrize('change', ['missing', 'wrong_kind', 'metadata', 'counts', 'barrier'])
async def test_store_rejects_invalid_trap_reports(tmp_path, run_request, change):
    from scone_memory.agents.traps import ObservationGraph

    detector = ObservationGraph(2)
    detector.observe(b'a')
    graph = detector.observe(b'a')
    event = replace((await events())[0], kind='trap_detected', trap_graph=graph)
    if change == 'missing':
        event = replace(event, trap_graph=None)
    elif change == 'wrong_kind':
        event = replace(event, kind='turn_failed')
    elif change == 'metadata':
        event = replace(event, tool_name='search_memory')
    elif change == 'counts':
        event = replace(event, trap_graph=replace(graph, edges=()))
    else:
        event = replace(event, trap_graph=replace(graph, nodes=(replace(graph.nodes[0], comparable=False),)))
    with closing(AgentEventHistoryStore(tmp_path / 'trap-history.db', key=b'k' * 32)) as store:
        with pytest.raises(WorkflowError, match='invalid_history_event'):
            store.append(run_request, step_id='find', selection_id='find', event=event)
        assert not store.read(run_request).available
