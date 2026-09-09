"""An application schema must constrain both field shape and actual values."""
import copy

import pytest

from scone_memory.realtime.answer_requirements import AnswerRequirements, validated_requirements


SCHEMA = {'type':'object', 'properties':{'answer':{'type':'string', 'minLength':1}},
          'required':['answer'], 'additionalProperties':False}


@pytest.mark.parametrize('text,accepted', [
    ('{"answer":"oboe"}', True), ('{"instrument":"oboe"}', False),
    ('{"answer":12}', False), ('{"answer":""}', False),
    ('{"answer":"oboe","extra":true}', False), ('{}', False),
    ('{"answer":"oboe","answer":"cello"}', False),
])
def test_schema_enforces_requested_keys_types_and_constraints(text, accepted):
    requirements = AnswerRequirements(format='json_object', output_schema=SCHEMA)
    assert requirements.accepts(text) is accepted


def test_local_references_are_self_contained_for_nested_action_schemas():
    schema = {'$defs':{'answer':{'type':'string', 'enum':['oboe','cello']}},
        'properties':{'answer':{'$ref':'#/$defs/answer'}}, 'required':['answer'], 'additionalProperties':False}
    requirements = AnswerRequirements(format='json_object', output_schema=schema)
    assert requirements.accepts('{"answer":"oboe"}')
    assert not requirements.accepts('{"answer":"flute"}')
    assert '$ref' not in requirements.prompt() and '$defs' not in requirements.prompt()
    assert schema['properties']['answer'] == {'$ref':'#/$defs/answer'}


def test_reference_siblings_preserve_unevaluated_property_semantics():
    schema = {'$defs':{'base':{'type':'object', 'properties':{'answer':{'type':'string'}}}},
              '$ref':'#/$defs/base', 'required':['answer'], 'unevaluatedProperties':False}
    requirements = AnswerRequirements(format='json_object', output_schema=schema)
    assert requirements.accepts('{"answer":"oboe"}')
    assert not requirements.accepts('{"answer":"oboe","extra":1}')


def test_reference_spellings_inside_instance_data_are_not_schema_references():
    schema = {'properties':{'$ref':{'const':'https://example.invalid/value'},
        'payload':{'const':{'$ref':'file:///value'}}}, 'required':['$ref','payload'], 'additionalProperties':False}
    requirements = AnswerRequirements(format='json_object', output_schema=schema)
    assert requirements.accepts('{"$ref":"https://example.invalid/value","payload":{"$ref":"file:///value"}}')


@pytest.mark.parametrize('schema', [
    {'$ref':'https://example.invalid/schema'}, {'$ref':'file:///etc/passwd'},
    {'$defs':{'cycle':{'$ref':'#/$defs/cycle'}}, '$ref':'#/$defs/cycle'},
    {'$ref':'#/$defs/missing'}, {'$dynamicRef':'#node'},
    {'$id':'https://example.invalid/schema'}, {'$schema':'https://example.invalid/dialect'},
    {'type':'array'}, {'properties':{'answer':{'type':'bogus'}}},
    {'properties':{'answer':{'minLength':-1}}}, {'requred':['answer']},
])
def test_invalid_or_non_self_contained_schemas_fail_before_inference(schema):
    with pytest.raises(ValueError):
        AnswerRequirements(format='json_object', output_schema=schema)


@pytest.mark.parametrize('constraint,text,accepted', [
    ({'const':1}, '1.00000000000000000000000001', False),
    ({'minimum':0.3}, '0.29999999999999999999999999', False),
    ({'multipleOf':0.1}, '0.3', True),
    ({'multipleOf':3}, '1e400', False),
    ({'multipleOf':0.1}, '1e400', True),
    ({'type':'integer'}, '1.0', True),
    ({'type':'integer'}, 'true', False),
])
def test_numeric_validation_does_not_round_to_binary_float(constraint, text, accepted):
    schema = {'properties':{'value':constraint}, 'required':['value']}
    requirements = AnswerRequirements(format='json_object', output_schema=schema)
    assert requirements.accepts('{"value":'+text+'}') is accepted


def test_nested_arrays_unions_and_conditions_are_validated():
    schema = {'properties':{'items':{'type':'array', 'minItems':1, 'maxItems':2,
        'items':{'anyOf':[{'type':'string'}, {'type':'integer'}]}}, 'kind':{'enum':['single','pair']}},
        'required':['items','kind'], 'additionalProperties':False,
        'if':{'properties':{'kind':{'const':'pair'}}},
        'then':{'properties':{'items':{'minItems':2}}}}
    requirements = AnswerRequirements(format='json_object', output_schema=schema)
    assert requirements.accepts('{"kind":"pair","items":["oboe",2]}')
    assert not requirements.accepts('{"kind":"pair","items":["oboe"]}')
    assert not requirements.accepts('{"kind":"single","items":[true]}')


@pytest.mark.parametrize('values,accepted', [('[1,2]', True), ('[1,"two"]', False),
    ('[1,2,3,4]', False), ('[1,2,"three"]', True), ('[true,2]', False)])
def test_contains_bounds_count_only_matching_elements(values, accepted):
    requirements = AnswerRequirements(format='json_object', output_schema={'properties':{
        'values':{'type':'array', 'contains':{'type':'integer'}, 'minContains':2, 'maxContains':3}}})
    assert requirements.accepts('{"values":'+values+'}') is accepted


def test_input_and_ownership_snapshots_do_not_share_nested_schema_values():
    original = copy.deepcopy(SCHEMA)
    requirements = AnswerRequirements(format='json_object', output_schema=original)
    original['properties']['answer']['type'] = 'integer'
    snapshot = validated_requirements(requirements)
    requirements.output_schema['properties']['answer']['type'] = 'boolean'
    assert snapshot.accepts('{"answer":"oboe"}')
    assert not snapshot.accepts('{"answer":true}')


def test_schema_requires_object_format_and_cannot_bypass_ownership_validation():
    with pytest.raises(ValueError):
        AnswerRequirements(output_schema=SCHEMA)
    with pytest.raises(ValueError):
        validated_requirements(AnswerRequirements.model_construct(format='json_object',
            output_schema={'$ref':'https://example.invalid/schema'}))


def test_schema_size_and_nesting_are_bounded():
    with pytest.raises(ValueError):
        AnswerRequirements(format='json_object', output_schema={'description':'x'*33000})
    nested = {'type':'string'}
    for _ in range(40):
        nested = {'type':'array', 'items':nested}
    with pytest.raises(ValueError):
        AnswerRequirements(format='json_object', output_schema={'properties':{'value':nested}})


def test_absent_schema_keeps_existing_serialized_requirements():
    assert AnswerRequirements().model_dump(mode='json') == {
        'instructions':'', 'max_bytes':64000, 'max_lines':None, 'format':'text'}


def test_escaped_local_pointers_and_boolean_subschemas():
    schema = {'$defs':{'a/b~c':{'type':'string'}},
        'properties':{'answer':{'$ref':'#/$defs/a~1b~0c'}, 'forbidden':False}, 'required':['answer']}
    requirements = AnswerRequirements(format='json_object', output_schema=schema)
    assert requirements.accepts('{"answer":"oboe"}')
    assert not requirements.accepts('{"answer":"oboe","forbidden":null}')


def test_schema_numeric_limits_do_not_change_syntax_only_mode():
    requirements = AnswerRequirements(format='json_object', output_schema={'properties':{'value':{'type':'number'}}})
    for token in ['1e4097', '1e-4097', '1'*4097]:
        text = '{"value":'+token+'}'
        assert not requirements.accepts(text)
        assert AnswerRequirements(format='json_object').accepts(text)


def test_format_is_an_annotation_not_an_implicit_optional_dependency():
    requirements = AnswerRequirements(format='json_object',
        output_schema={'properties':{'answer':{'type':'string', 'format':'email'}}})
    assert requirements.accepts('{"answer":"oboe"}')


def test_repeated_reference_expansion_is_bounded():
    definitions = {'leaf':{'type':'string'}}
    previous = 'leaf'
    for index in range(8):
        key = f'level{index}'
        definitions[key] = {'anyOf':[{'$ref':f'#/$defs/{previous}'}]*2}
        previous = key
    with pytest.raises(ValueError):
        AnswerRequirements(format='json_object',
            output_schema={'$defs':definitions, 'properties':{'answer':{'$ref':f'#/$defs/{previous}'}}})


def test_missing_optional_dependency_does_not_disable_ordinary_requirements(monkeypatch):
    import builtins
    from scone_memory.realtime.output_schema import _compiled

    real_import = builtins.__import__
    def without_jsonschema(name, *args, **kwargs):
        if name == 'jsonschema' or name.startswith('jsonschema.'):
            raise ImportError('not installed')
        return real_import(name, *args, **kwargs)
    _compiled.cache_clear()
    monkeypatch.setattr(builtins, '__import__', without_jsonschema)
    assert AnswerRequirements(format='json_object').accepts('{"answer":"oboe"}')
    with pytest.raises(ImportError, match=r'scone-memory\[structured-output\]'):
        AnswerRequirements(format='json_object', output_schema=SCHEMA)
