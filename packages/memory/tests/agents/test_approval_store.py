"""Approval, activation and execution admission are separate durable decisions."""

import pytest

from scone_memory.agents.approval_models import ApprovalCall
from scone_memory.agents.approval_store import AgentApprovalStore
from scone_memory.agents.plan_store import AgentPlanStore
from scone_memory.agents.run_store import AgentRunStore, _request_bytes
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope
from .test_plan_store import catalog, plan


@pytest.fixture
def approvals(tmp_path):
    plans = AgentPlanStore(tmp_path / 'plans', key=b'k' * 32)
    saved = plans.save('alpha', plan(), catalog=catalog(), expected_revision=0)
    runs = AgentRunStore(tmp_path / 'runs', key=b'k' * 32, max_runs=2)
    request = runs.register(
        'alpha', 'one', plan=saved, question='Private original question', scope=RecallScope.validated()
    )
    call = ApprovalCall(
        step_id='find',
        selection_id='find',
        agent_id='a',
        model_id='local',
        binding=saved.bindings['find'],
        tool_name='deliver',
        tool_revision='1',
        tool_digest='a' * 64,
        arguments_json='{"message":"Private exact proposal"}',
        operation_digest='b' * 64,
    )
    yield AgentApprovalStore(runs), runs, request, call
    runs.close()
    plans.close()


def test_decision_activation_and_single_claim_survive_reopen(approvals, tmp_path):
    store, runs, original, call = approvals
    pending = store.request('alpha', 'one', call)
    assert pending.revision == 1 and pending.decision is None
    assert store.request('alpha', 'one', call) == pending
    decided = store.decide(
        'alpha', 'one', pending.request_id, decision='approve', actor='owner', expected_revision=1
    )
    assert decided.revision == 2 and decided.activation_id is None
    with pytest.raises(WorkflowError, match='approval_not_activated'):
        store.claim('alpha', 'one', pending.request_id, activation_id='continue', call=call)
    activation = store.activate('alpha', 'one', 'continue', decisions={pending.request_id: 2})
    assert activation.decisions == {pending.request_id: 2}
    assert _request_bytes(runs.get('alpha', 'one')) == _request_bytes(original)
    reopened_runs = AgentRunStore(tmp_path / 'runs', key=b'k' * 32)
    try:
        reopened = AgentApprovalStore(reopened_runs)
        claimed = reopened.claim('alpha', 'one', pending.request_id, activation_id='continue', call=call)
        assert claimed.revision == 4 and claimed.consumed_at is not None
        with pytest.raises(WorkflowError, match='approval_already_claimed'):
            store.claim('alpha', 'one', pending.request_id, activation_id='continue', call=call)
    finally:
        reopened_runs.close()
    assert b'Private exact proposal' not in (tmp_path / 'runs').read_bytes()


def test_decisions_are_single_assignment_and_activation_ids_are_immutable(approvals):
    store, _, _, call = approvals
    record = store.request('alpha', 'one', call)
    store.decide('alpha', 'one', record.request_id, decision='deny', actor='owner', expected_revision=1)
    assert (
        store.decide(
            'alpha', 'one', record.request_id, decision='deny', actor='owner', expected_revision=1
        ).revision
        == 2
    )
    for decision, actor in [('approve', 'owner'), ('deny', 'other')]:
        with pytest.raises(WorkflowError, match='approval_decision_conflict'):
            store.decide(
                'alpha', 'one', record.request_id, decision=decision, actor=actor, expected_revision=1
            )
    activation = store.activate('alpha', 'one', 'continue', decisions={record.request_id: 2})
    assert store.activate('alpha', 'one', 'continue', decisions={record.request_id: 2}) == activation
    with pytest.raises(WorkflowError, match='approval_activation_conflict'):
        store.activate('alpha', 'one', 'another', decisions={record.request_id: 2})


def test_cancellation_blocks_decision_activation_and_claim_but_not_inspection(approvals):
    store, runs, _, call = approvals
    record = store.request('alpha', 'one', call)
    store.decide('alpha', 'one', record.request_id, decision='approve', actor='owner', expected_revision=1)
    store.activate('alpha', 'one', 'continue', decisions={record.request_id: 2})
    runs.request_cancel('alpha', 'one')
    assert store.get('alpha', 'one', record.request_id).revision == 3
    for action in [
        lambda: store.request('alpha', 'one', call),
        lambda: store.decide(
            'alpha', 'one', record.request_id, decision='approve', actor='owner', expected_revision=1
        ),
        lambda: store.activate('alpha', 'one', 'continue', decisions={record.request_id: 2}),
        lambda: store.claim('alpha', 'one', record.request_id, activation_id='continue', call=call),
    ]:
        with pytest.raises(WorkflowError, match='run_cancelled'):
            action()


@pytest.mark.parametrize('change', ['selection_id', 'model_id', 'binding', 'agent_id'])
def test_unknown_or_changed_selection_cannot_request_approval(approvals, change):
    store, _, _, call = approvals
    changed = ApprovalCall.model_validate(
        {**call.model_dump(), change: 'f' * 64 if change == 'binding' else 'other'}
    )
    with pytest.raises(WorkflowError):
        store.request('alpha', 'one', changed)
    assert store.list('alpha', 'one') == ()


def test_auxiliary_records_do_not_consume_run_capacity_or_escape_scope(approvals):
    store, runs, request, call = approvals
    record = store.request('alpha', 'one', call)
    assert store.get('bravo', 'one', record.request_id) is None
    store.decide('alpha', 'one', record.request_id, decision='approve', actor='owner', expected_revision=1)
    store.activate('alpha', 'one', 'continue', decisions={record.request_id: 2})
    runs.register('alpha', 'two', plan=request.plan, question='Another', scope=RecallScope.validated())
    assert len(runs.list('alpha').items) == 2 and len(store.list('alpha', 'one')) == 1
    assert store.list('alpha', 'two') == ()
