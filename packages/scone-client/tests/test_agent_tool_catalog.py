"""Shared configured tool metadata stays strict, immutable, and additive."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import pytest
from scone import Scone, SconeError
from scone.agent_models import AgentChoice, parse_catalog
from stub_server import StubScone

TOOL = {'name': 'lookup_stock', 'description': 'Read local inventory.', 'revision': 'inventory:1'}
MODEL = {'model_id': 'local', 'label': 'Local model', 'revision': 'v1'}
AGENT = {'agent_id': 'helper', 'default_model': 'local', 'models': [MODEL]}


def catalog():
    return {'agents': [{**deepcopy(AGENT), 'tools': ['lookup_stock']},
                       {**deepcopy(AGENT), 'agent_id': 'other', 'tools': ['lookup_stock']}],
            'tools': [deepcopy(TOOL)]}


def test_shared_tools_are_frozen_detached_and_exported():
    from scone import ToolChoice
    value = catalog()
    agents = parse_catalog(value)
    tool = agents[0].tools[0]
    assert isinstance(tool, ToolChoice)
    assert (tool.name, tool.description, tool.revision) == tuple(TOOL.values())
    assert agents[1].tools == (tool,)
    value['tools'][0]['description'] = 'Changed'
    value['agents'][0]['tools'].clear()
    assert agents[0].tools == (tool,) and tool.description == TOOL['description']
    with pytest.raises(FrozenInstanceError):
        tool.description = 'Changed'
    with pytest.raises(FrozenInstanceError):
        agents[0].tools = ()


def test_legacy_unknown_and_explicit_empty_are_distinct():
    legacy = parse_catalog({'agents': [AGENT]})[0]
    assert legacy.tools is None
    assert AgentChoice(legacy.agent_id, legacy.default_model, legacy.models).tools is None
    known = parse_catalog({'agents': [{**AGENT, 'tools': []}], 'tools': []})[0]
    assert known.tools == ()


@pytest.mark.parametrize('change', ['missing_table', 'missing_selection', 'one_missing_selection',
    'table_null', 'table_false', 'selection_null', 'selection_false', 'selection_string',
    'duplicate_table', 'duplicate_selection', 'unknown_selection', 'extra_summary', 'missing_revision'])
def test_partial_or_contradictory_tool_metadata_refuses(change):
    value = catalog()
    if change == 'missing_table': del value['tools']
    elif change == 'missing_selection':
        for agent in value['agents']: del agent['tools']
    elif change == 'one_missing_selection': del value['agents'][1]['tools']
    elif change == 'table_null': value['tools'] = None
    elif change == 'table_false': value['tools'] = False
    elif change == 'selection_null': value['agents'][0]['tools'] = None
    elif change == 'selection_false': value['agents'][0]['tools'] = False
    elif change == 'selection_string': value['agents'][0]['tools'] = 'lookup_stock'
    elif change == 'duplicate_table': value['tools'].append(deepcopy(TOOL))
    elif change == 'duplicate_selection': value['agents'][0]['tools'].append('lookup_stock')
    elif change == 'unknown_selection': value['agents'][0]['tools'] = ['other_tool']
    elif change == 'extra_summary': value['tools'][0]['parameters'] = {'type': 'object'}
    elif change == 'missing_revision': del value['tools'][0]['revision']
    with pytest.raises(SconeError): parse_catalog(value)


@pytest.mark.parametrize('name', ['search_memory', 'trace_memory', 'read_memory', 'compute_memory',
    'answer', 'unknown_tool', 'custom_tool', 'a.b', 'a:b', '', 'a' * 65, 'é', None, False, 1])
def test_invalid_or_reserved_tool_names_refuse(name):
    value = catalog()
    value['tools'][0]['name'] = name
    for agent in value['agents']: agent['tools'] = [name]
    with pytest.raises(SconeError): parse_catalog(value)


@pytest.mark.parametrize('field,value', [
    ('description', ''), ('description', ' \n'), ('description', '\ud800'),
    ('description', '😀' * 1001), ('description', False),
    ('revision', ''), ('revision', 'v 1'), ('revision', 'a' * 129), ('revision', False),
])
def test_metadata_limits_refuse_without_coercion(field, value):
    response = catalog()
    response['tools'][0][field] = value
    with pytest.raises(SconeError): parse_catalog(response)


def test_native_name_revision_and_utf8_boundaries():
    value = catalog()
    value['tools'][0] = {'name': '_' * 64, 'description': '😀' * 1000, 'revision': '.'}
    for agent in value['agents']: agent['tools'] = ['_' * 64]
    tool = parse_catalog(value)[0].tools[0]
    assert tool.description == '😀' * 1000 and tool.revision == '.'


def test_shared_metadata_budget_counts_table_once_and_limits_escaped_bytes():
    tools = [{'name': 't' + str(i), 'description': 'x' * 3900, 'revision': '1'} for i in range(32)]
    value = {'tools': tools, 'agents': [{**AGENT, 'agent_id': 'a' + str(i),
             'tools': [tool['name'] for tool in tools]} for i in range(32)]}
    assert len(json.dumps(tools, ensure_ascii=False, separators=(',', ':')).encode()) < 128000
    assert len(parse_catalog(value)[0].tools) == 32
    for tool in tools: tool['description'] = 'x' * 4000
    assert len(json.dumps(tools, ensure_ascii=False, separators=(',', ':')).encode()) > 128000
    with pytest.raises(SconeError): parse_catalog(value)
    for tool in tools: tool['description'] = 'x' + '\0' * 1000
    with pytest.raises(SconeError): parse_catalog(value)


def test_more_than_32_tools_refuses():
    value = {'agents': [{**AGENT, 'tools': ['t' + str(i) for i in range(33)]}],
             'tools': [{'name': 't' + str(i), 'description': 'Text', 'revision': '1'} for i in range(33)]}
    with pytest.raises(SconeError): parse_catalog(value)


@pytest.mark.parametrize('new_shape', [False, True])
def test_catalog_transport_needs_no_extra_capability_and_never_executes(new_shape):
    with StubScone() as server, Scone(server.base_url, 'key') as client:
        server.route('GET', '/v1/capabilities', 200, {'schema_version': 1, 'implementation': 'python',
                     'features': {'agents.catalog': True}})
        server.route('GET', '/v1/agents/catalog', 200, catalog() if new_shape else {'agents': [AGENT]})
        agent = client.agents(expected_space='alpha').catalog()[0]
        assert (agent.tools is not None) == new_shape
        assert [r.path for r in server.requests] == ['/v1/capabilities', '/v1/agents/catalog']
        assert all(r.method == 'GET' for r in server.requests)


def test_tool_choice_direct_construction_checks_metadata_and_agent_tuple():
    from scone import ToolChoice
    for name, description, revision in [('answer', 'Text', '1'), ('ok', '\ud800', '1'), ('ok', 'Text', False)]:
        with pytest.raises(SconeError): ToolChoice(name, description, revision)
    tool = ToolChoice('safe', 'Local tool', '1')
    for selected in ([tool], (tool, tool), ('safe',), (False,)):
        with pytest.raises(SconeError): AgentChoice('a', 'm', (), selected)


def test_exact_compact_table_byte_limit_and_selection_order():
    tools = [{'name': 't' + str(i), 'description': 'x' * 3900, 'revision': '1'} for i in range(32)]
    remaining = 128000 - len(json.dumps(tools, ensure_ascii=False, separators=(',', ':')).encode())
    for tool in tools:
        extra = min(remaining, 100)
        tool['description'] += 'x' * extra
        remaining -= extra
    assert remaining == 0
    value = {'agents': [{**AGENT, 'tools': [tool['name'] for tool in reversed(tools)]}], 'tools': tools}
    assert len(json.dumps(tools, ensure_ascii=False, separators=(',', ':')).encode()) == 128000
    assert [tool.name for tool in parse_catalog(value)[0].tools] == value['agents'][0]['tools']
    tools[-1]['description'] += 'x'
    with pytest.raises(SconeError): parse_catalog(value)
