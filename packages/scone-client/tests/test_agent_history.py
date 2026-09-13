"""Typed history validates identity, shape and cursor continuity without writes."""

from copy import deepcopy

import pytest
from scone import SconeError
from scone.agent_history import HistoryPage
from scone.agent_models import RunRequest
from test_agent_models import REQUEST
from test_agents import fixture, CAPS


def cursor(position, generation='b' * 32):
    return generation + '.' + format(position, '016x') + '.' + 'c' * 64


def event():
    return dict(
        sequence=1,
        invocation_id='d' * 32,
        agent_id='worker',
        model_id='careful',
        binding='a' * 64,
        kind='turn_started',
        occurred_at='2026-09-13T00:00:00Z',
        elapsed_s=0.0,
        operation_id=None,
        operation_kind=None,
        duration_s=None,
        tool_index=None,
        tool_name=None,
        status=None,
        error=None,
        output_bytes=None,
        origin=None,
        reused=None,
        journal_reused=None,
        presentation_reused=None,
    )


def page():
    return dict(
        space='alpha',
        run_id='one',
        available=True,
        items=[
            dict(
                position=1,
                step_id='answer',
                selection_id='answer',
                event=event(),
                collection_id=None,
                activation_id=None,
            )
        ],
        next_after=cursor(1),
        retained_from=1,
        omitted=None,
    )


def parse(value, **kwargs):
    return HistoryPage.from_json(
        value, request=RunRequest.from_json(REQUEST, expected_space='alpha', run_id='one'), **kwargs
    )


def test_typed_selected_model_history_and_tail_cursor():
    result = parse(page())
    assert result.items[0].event.model_id == 'careful'
    tail = {**page(), 'items': []}
    assert parse(tail, after=cursor(1)).items == ()
    with pytest.raises(SconeError):
        parse(tail)


@pytest.mark.parametrize(
    'change',
    [
        {'space': 'wrong'},
        {'run_id': 'wrong'},
        {'available': 1},
        {'next_after': cursor(2)},
        {'retained_from': 2},
        {'omitted': [1, 1]},
        {'extra': 'PRIVATE'},
    ],
)
def test_malformed_history_page_is_refused(change):
    with pytest.raises(SconeError) as error:
        parse({**page(), **change})
    assert 'PRIVATE' not in str(error.value)


@pytest.mark.parametrize(
    'change',
    [
        {'model_id': 'fast'},
        {'binding': 'f' * 64},
        {'kind': 'unknown'},
        {'elapsed_s': float('nan')},
        {'sequence': True},
        {'operation_id': 1},
        {'error': 'PRIVATE'},
        {'text': 'PRIVATE'},
    ],
)
def test_invalid_or_substituted_events_are_refused(change):
    value = page()
    value['items'][0]['event'].update(change)
    with pytest.raises(SconeError) as error:
        parse(value)
    assert 'PRIVATE' not in str(error.value)


def test_history_client_requires_capability_and_only_reads(fixture):
    server, agents = fixture
    server.route(
        'GET', '/v1/capabilities', 200, {**CAPS, 'features': {**CAPS['features'], 'agents.history': True}}
    )
    server.route('GET', '/v1/agent-runs/one/request', 200, REQUEST)
    server.route('GET', '/v1/agent-runs/one/history', 200, page())
    assert agents.history('one').items[0].event.model_id == 'careful'
    assert all(request.method == 'GET' for request in server.requests)


def collection(kind='collection_finished'):
    return dict(
        kind=kind,
        collection_id='e' * 32,
        occurred_at='2026-09-13T00:00:00Z',
        invocation_id='d' * 32,
        last_sequence=8,
        observed_events=5,
        lost_events=3,
        terminal_kind='turn_completed',
        error=None,
    )


def test_collection_coverage_and_retention_gap_remain_distinct():
    value = page()
    value['items'][0].update(position=4, collection_id='e' * 32, event=collection())
    value.update(next_after=cursor(4), retained_from=4, omitted=[1, 3])
    result = parse(value)
    assert result.omitted == (1, 3) and result.items[0].event.lost_events == 3
    assert result.items[0].event.observed_events == 5


@pytest.mark.parametrize(
    'change',
    [
        {'observed_events': 4},
        {'last_sequence': True},
        {'error': 'PRIVATE'},
        {'terminal_kind': None},
        {'invocation_id': None},
        {'collection_id': 'f' * 32},
    ],
)
def test_malformed_collection_is_refused(change):
    value = page()
    value['items'][0].update(collection_id='e' * 32, event={**collection(), **change})
    with pytest.raises(SconeError):
        parse(value)


def test_native_gap_and_legacy_envelope_are_preserved():
    value = page()
    value['items'][0]['event'] = dict(invocation_id='d' * 32, first_sequence=2, last_sequence=5)
    value['items'][0].pop('collection_id')
    value['items'][0].pop('activation_id')
    parsed = parse(value).items[0]
    assert (
        parsed.collection_id is None and parsed.event.first_sequence == 2 and parsed.event.last_sequence == 5
    )


@pytest.mark.parametrize(
    'change',
    [{'next_after': cursor(1, 'f' * 32)}, {'items': []}, {'items': [page()['items'][0], page()['items'][0]]}],
)
def test_page_replay_cannot_reset_generation_or_duplicate_positions(change):
    with pytest.raises(SconeError):
        parse({**page(), **change}, after=cursor(0))


def test_handoff_selection_must_be_reachable_at_exact_hop():
    from scone import HandoffAgent, HandoffPlan

    request = deepcopy(REQUEST)
    request['plan']['plan'] = HandoffPlan(
        'handoff',
        'worker',
        (HandoffAgent('worker', 'careful', ('reviewer',)), HandoffAgent('reviewer', 'careful', ())),
        2,
    ).to_json()
    request['plan']['bindings'] = {'worker': 'a' * 64, 'reviewer': 'b' * 64}
    bound = RunRequest.from_json(request, expected_space='alpha', run_id='one')
    value = page()
    value['items'][0].update(step_id='hop-02', selection_id='reviewer')
    value['items'][0]['event'].update(agent_id='reviewer', binding='b' * 64)
    assert HistoryPage.from_json(value, request=bound).items[0].selection_id == 'reviewer'
    value['items'][0]['step_id'] = 'hop-01'
    with pytest.raises(SconeError):
        HistoryPage.from_json(value, request=bound)


@pytest.mark.parametrize(
    'arguments', [{'limit': True}, {'limit': 0}, {'after': 'PRIVATE'}, {'after': cursor(2**53)}]
)
def test_history_query_is_validated_before_any_request(fixture, arguments):
    server, agents = fixture
    with pytest.raises(SconeError):
        agents.history('one', **arguments)
    assert server.requests == []
