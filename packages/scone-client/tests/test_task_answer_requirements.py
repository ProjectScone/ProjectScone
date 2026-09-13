"""Authored task contracts remain bounded, immutable and explicitly negotiated."""
import json
from dataclasses import FrozenInstanceError, replace

import pytest
from scone import SconeError
from scone.agent_models import ModelTask, TaskPlan, parse_plan
from test_agents import fixture, CAPS
from test_agent_models import SAVED


def contract(**kwargs):
    from scone import TaskAnswerRequirements
    return TaskAnswerRequirements(**kwargs)


def task(requirements):
    return ModelTask('answer', 'worker', 'careful', 'Answer', answer_requirements=requirements)


def test_defaults_and_legacy_task_wire_are_distinct():
    defaults = {'instructions': '', 'max_bytes': 64000, 'max_lines': None, 'format': 'text'}
    assert contract().to_json() == defaults
    assert 'answer_requirements' not in ModelTask('answer', 'worker', 'careful', 'Answer').to_json()
    authored = task(contract()).to_json()
    assert authored['answer_requirements'] == defaults
    assert parse_plan({'workflow_id': 'w', 'tasks': [authored]}).to_json()['tasks'] == [authored]


def test_schema_preserves_refs_and_is_deeply_detached_and_immutable():
    schema = {'$defs': {'value': {'type': 'string', 'enum': ['a', 'b']}}, 'type': 'object',
              'properties': {'value': {'$ref': '#/$defs/value'}}, 'required': ['value']}
    original = json.loads(json.dumps(schema))
    requirements = contract(format='json_object', output_schema=schema)
    schema['$defs']['value']['enum'].append('c')
    assert requirements.to_json()['output_schema'] == original
    with pytest.raises(TypeError): requirements.output_schema['type'] = 'array'
    with pytest.raises(TypeError): requirements.output_schema['$defs']['value']['enum'][0] = 'c'
    with pytest.raises(FrozenInstanceError): requirements.max_bytes = 1
    exported = requirements.to_json()
    exported['output_schema']['required'].append('other')
    assert requirements.to_json()['output_schema'] == original
    assert replace(requirements, max_lines=2).to_json()['output_schema'] == original


@pytest.mark.parametrize('kwargs', [
    {'instructions': None}, {'instructions': '\ud800'}, {'instructions': 'é'*4001},
    {'max_bytes': True}, {'max_bytes': 0}, {'max_bytes': 128001}, {'max_bytes': 1.0},
    {'max_lines': True}, {'max_lines': 0}, {'max_lines': 1001}, {'max_lines': '2'},
    {'format': 'JSON'}, {'format': True}, {'output_schema': {}},
], ids=['instructions-none','surrogate','instruction-bytes','bool-bytes','zero-bytes','large-bytes','float-bytes',
        'bool-lines','zero-lines','large-lines','string-lines','unknown-format','bool-format','schema-text'])
def test_invalid_basic_fields_refused(kwargs):
    with pytest.raises(SconeError): contract(**kwargs)


def test_instruction_utf8_boundary_and_empty_content_are_preserved():
    assert contract(instructions='é'*4000).instructions == 'é'*4000
    assert contract(instructions='  \n').instructions == '  \n'


@pytest.mark.parametrize('schema', [True, [], {'const': float('nan')}, {'const': float('inf')},
    {1: 'bad'}, {'const': {1}}, {'const': ('a',)}, {'const': '\ud800'}],
    ids=['bool-root','array-root','nan','infinity','numeric-key','set','tuple','surrogate'])
def test_schema_non_json_refused(schema):
    with pytest.raises(SconeError): contract(format='json_object', output_schema=schema)


def test_schema_bounds_are_real_and_cycles_refuse():
    assert contract(format='json_object', output_schema={'examples': [None]*4094})
    with pytest.raises(SconeError): contract(format='json_object', output_schema={'examples': [None]*4095})
    nested = None
    for _ in range(31): nested = [nested]
    assert contract(format='json_object', output_schema={'x': nested})
    with pytest.raises(SconeError): contract(format='json_object', output_schema={'x': [nested]})
    cyclic = {}; cyclic['x'] = cyclic
    with pytest.raises(SconeError): contract(format='json_object', output_schema=cyclic)
    overhead = len(json.dumps({'description': ''}, separators=(',', ':')).encode())
    exact = 'x'*(32768-overhead)
    assert contract(format='json_object', output_schema={'description': exact})
    with pytest.raises(SconeError): contract(format='json_object', output_schema={'description': exact+'x'})


def test_parser_refuses_unknown_contract_fields_and_input_task_contracts():
    base = task(contract()).to_json()
    with pytest.raises(SconeError): parse_plan({'workflow_id':'w','tasks':[{**base,'answer_requirements':{'unknown':True}}]})
    with pytest.raises(SconeError): parse_plan({'kind':'interactive','workflow_id':'w','tasks':[
        {'kind':'input','task_id':'h','prompt':'Choose','depends_on':[],'max_response_bytes':10,'answer_requirements':{}}]})
    assert parse_plan({'workflow_id':'w','tasks':[{**base,'answer_requirements':None}]}).tasks[0].answer_requirements is None


def test_schema_identity_distinguishes_boolean_from_number_but_not_object_order():
    boolean = contract(format='json_object', output_schema={'const': True})
    numeric = contract(format='json_object', output_schema={'const': 1})
    assert boolean != numeric
    assert contract(format='json_object', output_schema={'type':'object','title':'A'}) == contract(
        format='json_object', output_schema={'title':'A','type':'object'})


@pytest.mark.parametrize('schema', [None, {'type':'object'}], ids=['basic','schema'])
def test_save_requires_each_capability_before_writing(fixture, schema):
    server, agents = fixture
    requirements = contract() if schema is None else contract(format='json_object', output_schema=schema)
    plan = TaskPlan('w',(task(requirements),))
    with pytest.raises(SconeError, match='agents.output_requirements'): agents.save_plan(plan,expected_revision=0)
    assert not any(row.method=='PUT' for row in server.requests)
    features = {**CAPS['features'],'agents.output_requirements':True}
    server.route('GET','/v1/capabilities',200,{**CAPS,'features':features})
    if schema is not None:
        with pytest.raises(SconeError, match='agents.output_schema'): agents.save_plan(plan,expected_revision=0)
        assert not any(row.method=='PUT' for row in server.requests)
        features['agents.output_schema']=True
        server.route('GET','/v1/capabilities',200,{**CAPS,'features':features})
    saved = {**SAVED,'plan':plan.to_json(),'bindings':{'answer':'a'*64}}
    server.route('PUT','/v1/agent-plans/w',200,saved)
    assert agents.save_plan(plan,expected_revision=0).plan == plan
    assert server.requests[-1].json == {'plan':plan.to_json(),'expected_revision':0}


def test_save_refuses_changed_schema_boolean_to_number(fixture):
    server, agents = fixture
    plan = TaskPlan('w',(task(contract(format='json_object',output_schema={'const':True})),))
    changed = plan.to_json(); changed['tasks'][0]['answer_requirements']['output_schema']['const']=1
    server.route('GET','/v1/capabilities',200,{**CAPS,'features':{**CAPS['features'],'agents.output_requirements':True,'agents.output_schema':True}})
    server.route('PUT','/v1/agent-plans/w',200,{**SAVED,'plan':changed,'bindings':{'answer':'a'*64}})
    with pytest.raises(SconeError,match='acknowledgement'): agents.save_plan(plan,expected_revision=0)

@pytest.mark.parametrize('field', ['max_bytes', 'max_lines'])
def test_huge_integer_limits_fail_without_stringifying_the_value(field):
    with pytest.raises(SconeError): contract(**{field: 10**10000})


def test_schema_unicode_and_escaped_bytes_match_compact_native_encoding():
    for content in ('é'*100, '\0'*100, '\\"'*100):
        schema = {'description': content}
        parsed = contract(format='json_object', output_schema=schema)
        assert parsed.to_json()['output_schema'] == schema
    with pytest.raises(SconeError): contract(format='json_object', output_schema={'description':'\0'*5462})


def test_sdk_preserves_authored_schema_without_claiming_to_compile_it():
    schema = {'$defs': {'entry': {'type':'string'}}, '$ref':'#/$defs/entry', 'x-annotation':{'enabled':True}}
    assert contract(format='json_object', output_schema=schema).to_json()['output_schema'] == schema
