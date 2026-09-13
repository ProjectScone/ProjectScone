import concurrent.futures
from contextlib import contextmanager
import sqlite3
import threading
import warnings
import pytest
from .test_approval_store import approvals
from scone_memory.agents.approval_store import AgentApprovalStore
from scone_memory.agents.run_store import AgentRunStore
from scone_memory.agents.workflow import WorkflowError


def activated(store, call):
    record = store.request('alpha', 'one', call)
    store.decide('alpha', 'one', record.request_id, decision='approve', actor='owner', expected_revision=1)
    store.activate('alpha', 'one', 'continue', decisions={record.request_id: 2})
    return record


def test_two_connections_only_one_claim(approvals, tmp_path):
    store, runs, _, call = approvals
    record = activated(store, call)
    barrier = threading.Barrier(2)

    def claim():
        local = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
        try:
            barrier.wait(timeout=3)
            try:
                return (
                    AgentApprovalStore(local)
                    .claim('alpha', 'one', record.request_id, activation_id='continue', call=call)
                    .revision
                )
            except WorkflowError as error:
                return error.code
        finally:
            local.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    assert sorted(map(str, results)) == ['4', 'approval_already_claimed']


def test_claim_cancel_race_serializes(approvals, tmp_path):
    store, runs, _, call = approvals
    record = activated(store, call)
    barrier = threading.Barrier(2)

    def operation(cancel):
        local = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
        try:
            barrier.wait(timeout=3)
            if cancel:
                local.request_cancel('alpha', 'one')
                return 'cancelled'
            try:
                return (
                    AgentApprovalStore(local)
                    .claim('alpha', 'one', record.request_id, activation_id='continue', call=call)
                    .revision
                )
            except WorkflowError as error:
                return error.code
        finally:
            local.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(operation, [False, True]))
    assert results in [[4, 'cancelled'], ['run_cancelled', 'cancelled']]
    assert store.get('alpha', 'one', record.request_id).revision == (4 if results[0] == 4 else 3)


def test_activation_rolls_back_all_records(approvals):
    store, _, _, call = approvals
    records = [
        store.request('alpha', 'one', call),
        store.request('alpha', 'one', call.model_copy(update={'operation_digest': 'c' * 64})),
    ]
    for record in records:
        store.decide(
            'alpha', 'one', record.request_id, decision='approve', actor='owner', expected_revision=1
        )
    original = store._put
    count = 0

    def fail(db, record):
        nonlocal count
        count += 1
        if count == 2:
            raise sqlite3.OperationalError('private SQL detail')
        original(db, record)

    store._put = fail
    try:
        with pytest.raises(WorkflowError, match='run_store_unavailable'):
            store.activate('alpha', 'one', 'continue', decisions={record.request_id: 2 for record in records})
    finally:
        store._put = original
    assert [record.revision for record in store.list('alpha', 'one')] == [2, 2]
    store.activate('alpha', 'one', 'continue', decisions={record.request_id: 2 for record in records})
    assert [record.revision for record in store.list('alpha', 'one')] == [3, 3]


def test_claim_lost_commit_ack_is_not_reclaimable(approvals):
    store, runs, _, call = approvals
    record = activated(store, call)
    original = runs._storage._access

    @contextmanager
    def fail(*, write=False):
        with original(write=write) as db:
            yield db
        if write:
            raise WorkflowError('run_store_unavailable')

    runs._storage._access = fail
    try:
        with pytest.raises(WorkflowError, match='run_store_unavailable'):
            store.claim('alpha', 'one', record.request_id, activation_id='continue', call=call)
    finally:
        runs._storage._access = original
    assert store.get('alpha', 'one', record.request_id).revision == 4
    with pytest.raises(WorkflowError, match='approval_already_claimed'):
        store.claim('alpha', 'one', record.request_id, activation_id='continue', call=call)
    assert (
        store.activate('alpha', 'one', 'continue', decisions={record.request_id: 2}).activation_id
        == 'continue'
    )


def test_invalid_copied_call_does_not_disclose_values_in_warnings(approvals):
    store, _, _, call = approvals
    secret = 'PRIVATE-invalid-call-content'
    corrupt = call.model_copy(update={'tool_name': {'secret': secret}})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(WorkflowError, match='invalid_approval_call'):
            store.request('alpha', 'one', corrupt)
    assert not any(secret in str(item.message) for item in caught)
    assert store.list('alpha', 'one') == ()


@pytest.mark.parametrize('activation', [False, True])
def test_authenticated_malformed_space_has_no_private_error(approvals, activation):
    import json

    store, runs, _, call = approvals
    record = activated(store, call)
    name = 'continue' if activation else record.request_id
    token = store._token('alpha', 'one', name, activation=activation)
    with runs._storage._access(write=True) as db:
        (payload,) = db.execute('SELECT payload FROM agent_runs WHERE token=?', (token,)).fetchone()
        value = json.loads(runs._storage._unseal(token, payload))
        value['space'] = 'PRIVATE_INVALID_SPACE'
        db.execute(
            'UPDATE agent_runs SET payload=? WHERE token=?',
            (runs._storage._seal(token, json.dumps(value).encode()), token),
        )
    with pytest.raises(WorkflowError, match='approval_key_or_integrity') as caught:
        if activation:
            store.claim('alpha', 'one', record.request_id, activation_id='continue', call=call)
        else:
            store.get('alpha', 'one', record.request_id)
    assert 'PRIVATE_INVALID_SPACE' not in str(caught.value)


def test_request_collision_with_changed_arguments_refuses(approvals):
    store, _, _, call = approvals
    record = store.request('alpha', 'one', call)
    changed = call.model_copy(update={'arguments_json': '{"message":"Changed"}'})
    with pytest.raises(WorkflowError, match='approval_request_conflict'):
        store.request('alpha', 'one', changed)
    assert store.get('alpha', 'one', record.request_id) == record


def test_request_capacity_is_bounded_and_existing_request_is_idempotent(approvals):
    store, _, _, call = approvals
    first = None
    for index in range(512):
        record = store.request('alpha', 'one', call.model_copy(update={'operation_digest': f'{index:064x}'}))
        first = first or record
    assert len(store.list('alpha', 'one')) == 512
    assert store.request('alpha', 'one', first.call) == first
    with pytest.raises(WorkflowError, match='approval_store_limit'):
        store.request('alpha', 'one', call.model_copy(update={'operation_digest': f'{512:064x}'}))
