"""Explicit usage negotiation preserves missing observations and model boundaries."""
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest
from scone import SconeError
from scone.agent_results import ModelOutput, parse_result
from scone.agent_models import RunRequest
from test_agents import fixture, CAPS
from test_agent_models import REQUEST, SAVED
from test_agent_results import OUTPUT, RESULT, HUMAN, serve

REPORT = {'prompt_tokens': 12, 'completion_tokens': 5, 'total_tokens': 17}


def output(value, *, model_calls=1, include_usage=True):
    return ModelOutput.from_json({**OUTPUT, 'model_calls': model_calls, 'usage': value},
        task_id='answer', agent_id='worker', model_id='careful', binding='a'*64,
        depends_on=('choose',), include_usage=include_usage)


def test_usage_preserves_partial_categories_and_does_not_infer_totals():
    parsed = output({'calls': [REPORT, {'prompt_tokens': 20, 'completion_tokens': None, 'total_tokens': None}]}, model_calls=2)
    assert parsed.usage.prompt_tokens == 32
    assert parsed.usage.completion_tokens is parsed.usage.total_tokens is None
    assert parsed.usage.calls[0].total_tokens == 17
    assert output(None).usage is None
    assert output({'calls': [{'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': None}]}).usage.total_tokens is None


def test_usage_is_detached_immutable_and_per_call_bounded():
    raw = {'calls': [dict(REPORT)]}
    parsed = output(raw)
    raw['calls'][0]['prompt_tokens'] = 999
    assert parsed.usage.prompt_tokens == 12
    with pytest.raises(FrozenInstanceError):
        parsed.usage.calls[0].prompt_tokens = 1
    report = {'prompt_tokens': 10**9, 'completion_tokens': 0, 'total_tokens': 10**9}
    assert output({'calls': [report]*17}, model_calls=17).usage.total_tokens == 17*10**9


@pytest.mark.parametrize('bad', [True, -1, 1.5, '12', 10**9+1, float('inf'), float('nan'), 10**10000])
def test_report_counts_refuse_non_native_values(bad):
    with pytest.raises(SconeError):
        output({'calls': [{**REPORT, 'prompt_tokens': bad}]})


@pytest.mark.parametrize('bad', [
    {}, {'calls': []}, {'calls': [REPORT, REPORT]}, {'calls': [REPORT]*18},
    {'calls': [REPORT], 'total_tokens': 17}, {'calls': [{**REPORT, 'private': 'secret'}]},
    {'calls': [{'prompt_tokens': 12}]}, {'calls': [None]}, {'calls': [True]},
    {'calls': [{**REPORT, 'total_tokens': 18}]}, {'calls': [{**REPORT, 'total_tokens': 16}]},
    {'calls': [{'prompt_tokens': None, 'completion_tokens': 5, 'total_tokens': 4}]},
])
def test_report_shape_cardinality_and_consistency_are_strict(bad):
    with pytest.raises(SconeError):
        output(bad)


def test_default_parser_refuses_unnegotiated_usage_and_opt_in_requires_key():
    with pytest.raises(SconeError):
        output(None, include_usage=False)
    with pytest.raises(SconeError):
        ModelOutput.from_json(OUTPUT, task_id='answer', agent_id='worker', model_id='careful',
                              binding='a'*64, depends_on=('choose',), include_usage=True)


def test_result_negotiates_usage_and_keeps_result_as_final_remote_read(fixture):
    server, agents = fixture
    serve(server, {**RESULT, 'results': {'choose': HUMAN, 'answer': {**OUTPUT, 'usage': {'calls': [REPORT]}}}})
    server.route('GET', '/v1/capabilities', 200, {**CAPS, 'features': {**CAPS['features'], 'agents.usage': True}})
    parsed = agents.result('one', include_usage=True)
    assert parsed.results['answer'].usage.total_tokens == 17
    assert not hasattr(parsed.results['choose'], 'usage')
    assert server.requests[-1].path == '/v1/agent-runs/one/result'
    assert server.requests[-1].query == {'include_usage': ['true']}
    assert all(request.method == 'GET' for request in server.requests)


@pytest.mark.parametrize('bad', [None, 0, 1, 'true'])
def test_invalid_usage_selection_refused_before_io(fixture, bad):
    server, agents = fixture
    with pytest.raises(SconeError):
        agents.result('one', include_usage=bad)
    assert not server.requests


def test_unsupported_usage_refuses_before_result_request(fixture):
    server, agents = fixture
    serve(server)
    with pytest.raises(SconeError, match='agents.usage'):
        agents.result('one', include_usage=True)
    assert not any(request.path.endswith('/result') for request in server.requests)


def test_default_wire_is_unchanged_and_human_usage_is_always_refused(fixture):
    server, agents = fixture
    serve(server)
    assert agents.result('one').results['answer'].usage is None
    assert server.requests[-1].query == {}
    server.route('GET', '/v1/capabilities', 200, {**CAPS, 'features': {**CAPS['features'], 'agents.usage': True}})
    serve(server, {**RESULT, 'results': {'choose': {**HUMAN, 'usage': None}, 'answer': {**OUTPUT, 'usage': None}}})
    with pytest.raises(SconeError):
        agents.result('one', include_usage=True)


def test_handoff_usage_binds_final_copy_and_limit_reports_actual_hops():
    plan = {'workflow_id': 'handoff', 'root_agent': 'worker', 'agents': [
        {'agent_id': 'worker', 'model_id': 'careful', 'can_handoff_to': ['worker']}], 'max_handoffs': 0}
    request = RunRequest.from_json({**REQUEST, 'plan': {**SAVED, 'plan': plan, 'bindings': {'worker': 'a'*64}}},
                                  expected_space='alpha', run_id='one')
    model = {**OUTPUT, 'task_id': 'hop-01', 'depends_on': [], 'usage': {'calls': [REPORT]}}
    row = {'space': 'alpha', 'run_id': 'one', 'status': 'completed', 'final': model,
           'hops': [{'output': model, 'handoff_to': None}], 'reused_hops': ['hop-01']}
    result = parse_result(row, request, include_usage=True)
    assert len(result.hops) == 1 and result.hops[0].output.usage.total_tokens == 17
    assert result.final == result.hops[0].output
    bad = deepcopy(row)
    bad['final'] = {**bad['final'], 'usage': None}
    with pytest.raises(SconeError):
        parse_result(bad, request, include_usage=True)
    limited = {**row, 'status': 'handoff_limit', 'final': None, 'hops': [{'output': model, 'handoff_to': 'worker'}]}
    assert parse_result(limited, request, include_usage=True).final is None
