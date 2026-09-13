"""Final handoff contracts retain authored identity without changing old plans."""
from dataclasses import FrozenInstanceError
import json

import pytest
from scone import HandoffAgent, HandoffPlan, TaskAnswerRequirements, SconeError
from scone.agent_models import parse_plan
from test_agents import fixture, CAPS
from test_agent_models import SAVED, REQUEST
from test_agent_results import OUTPUT


def handoff(requirements=None):
    return HandoffPlan('handoff', 'worker', (HandoffAgent('worker', 'careful'),), 0,
                       answer_requirements=requirements)


def allow(server, **features):
    server.route('GET', '/v1/capabilities', 200, {**CAPS, 'features': {
        **CAPS['features'], 'agents.handoffs': True, **features}})


def saved(plan):
    return {**SAVED, 'plan': plan.to_json(), 'bindings': {'worker': 'a'*64}}


def test_legacy_handoff_wire_stays_byte_identical_and_null_omits():
    legacy = HandoffPlan('handoff', 'worker', (HandoffAgent('worker', 'careful'),), 0)
    expected = {'workflow_id':'handoff', 'root_agent':'worker',
                'agents':[{'agent_id':'worker','model_id':'careful','can_handoff_to':[]}], 'max_handoffs':0}
    assert json.dumps(legacy.to_json(), separators=(',', ':')) == json.dumps(expected, separators=(',', ':'))
    assert handoff().to_json() == expected
    assert parse_plan({**expected, 'answer_requirements':None}) == legacy


def test_handoff_contract_roundtrip_is_frozen_and_preserves_authored_refs():
    schema = {'$defs':{'word':{'type':'string'}}, 'properties':{'x':{'$ref':'#/$defs/word'}}}
    requirements = TaskAnswerRequirements(format='json_object', output_schema=schema)
    plan = handoff(requirements)
    original = json.loads(json.dumps(schema))
    schema['$defs']['word']['type'] = 'number'
    assert plan.to_json()['answer_requirements'] == {
        'instructions':'', 'max_bytes':64000, 'max_lines':None, 'format':'json_object', 'output_schema':original}
    assert parse_plan(plan.to_json()) == plan
    with pytest.raises(FrozenInstanceError): plan.answer_requirements = None
    with pytest.raises(TypeError): plan.answer_requirements.output_schema['$defs']['word']['type'] = 'number'
    copied = plan.to_json(); copied['answer_requirements']['output_schema']['properties']['x']['$ref'] = '#/other'
    assert plan.to_json()['answer_requirements']['output_schema'] == original


@pytest.mark.parametrize('value', [False, {}, 'text', 1], ids=['bool','dict','string','integer'])
def test_constructor_refuses_untyped_contract(value):
    with pytest.raises(SconeError): handoff(value)


@pytest.mark.parametrize('value', [False, [], {'max_bytes':True}, {'unknown':True},
                                 {'format':'text','output_schema':{}}])
def test_wire_refuses_malformed_contract(value):
    wire = HandoffPlan('handoff','worker',(HandoffAgent('worker','careful'),)).to_json()
    with pytest.raises(SconeError): parse_plan({**wire, 'answer_requirements':value})


@pytest.mark.parametrize('schema', [None, {}], ids=['text','schema'])
def test_save_requires_handoff_specific_capability_and_schema_before_put(fixture, schema):
    server, agents = fixture
    requirements = TaskAnswerRequirements() if schema is None else TaskAnswerRequirements(format='json_object', output_schema=schema)
    plan = handoff(requirements)
    allow(server, **{'agents.output_requirements':True, 'agents.output_schema':True})
    with pytest.raises(SconeError, match='agents.handoffs.output_requirements'):
        agents.save_plan(plan, expected_revision=0)
    assert not any(request.method == 'PUT' for request in server.requests)
    allow(server, **{'agents.handoffs.output_requirements':True})
    if schema is not None:
        with pytest.raises(SconeError, match='agents.output_schema'): agents.save_plan(plan, expected_revision=0)
        assert not any(request.method == 'PUT' for request in server.requests)
        allow(server, **{'agents.handoffs.output_requirements':True, 'agents.output_schema':True})
    server.route('PUT', '/v1/agent-plans/handoff', 200, saved(plan))
    assert agents.save_plan(plan, expected_revision=0).plan == plan
    assert server.requests[-1].json == {'plan':plan.to_json(),'expected_revision':0}


def test_legacy_save_needs_no_new_capability(fixture):
    server, agents = fixture
    allow(server)
    plan = HandoffPlan('handoff','worker',(HandoffAgent('worker','careful'),),0)
    server.route('PUT','/v1/agent-plans/handoff',200,saved(plan))
    assert agents.save_plan(plan, expected_revision=0).plan == plan


@pytest.mark.parametrize('before,after', [(True,1),(False,0),(1,1.0)])
def test_save_refuses_type_changed_schema_acknowledgement(fixture, before, after):
    server, agents = fixture
    plan = handoff(TaskAnswerRequirements(format='json_object',output_schema={'const':before}))
    wire = saved(plan); wire['plan']['answer_requirements']['output_schema']['const'] = after
    allow(server, **{'agents.handoffs.output_requirements':True, 'agents.output_schema':True})
    server.route('PUT','/v1/agent-plans/handoff',200,wire)
    with pytest.raises(SconeError, match='acknowledgement'): agents.save_plan(plan, expected_revision=0)
    assert sum(request.method=='PUT' for request in server.requests) == 1


def test_reordered_schema_keys_acknowledge_and_final_result_stays_string(fixture):
    server, agents = fixture
    plan = handoff(TaskAnswerRequirements(format='json_object', output_schema={'type':'object','title':'Answer'}))
    wire = saved(plan); wire['plan']['answer_requirements']['output_schema'] = {'title':'Answer','type':'object'}
    allow(server, **{'agents.handoffs.output_requirements':True, 'agents.output_schema':True})
    server.route('PUT','/v1/agent-plans/handoff',200,wire)
    assert agents.save_plan(plan,expected_revision=0).plan == plan
    server.route('GET','/v1/agent-runs/one/request',200,{**REQUEST,'plan':wire})
    output = {**OUTPUT,'task_id':'hop-01','depends_on':[], 'text':'{"answer":"kept"}\n'}
    server.route('GET','/v1/agent-runs/one/result',200,{'space':'alpha','run_id':'one','status':'completed',
        'final':output,'hops':[{'output':output,'handoff_to':None}],'reused_hops':['hop-01']})
    result = agents.result('one')
    assert result.final.text == output['text']
    assert result.final == result.hops[-1].output
    assert server.requests[-1].path == '/v1/agent-runs/one/result'
