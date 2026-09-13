"""Public agent choices expose bounded tool summaries without executable state."""
import json

from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
from .test_custom_tools import tool


def forbidden():
    raise AssertionError('catalog reads must not create a model')


def test_catalog_deduplicates_selected_tools_and_detaches_public_metadata():
    registrations = [tool(lambda args, ctx: None), tool(lambda args, ctx: None, name='private_unused')]
    catalog = AgentCatalog(models=[AgentModel('local', 'Local', '1', forbidden)],
        tools=registrations, agents=[AgentDefinition(agent_id=name, instructions='PRIVATE instructions',
            models=('local',), default_model='local', tools=('double_count',) if name != 'empty' else ())
            for name in ('first', 'second', 'empty')])
    summaries = catalog.tool_descriptions()
    assert summaries == ({'name': 'double_count', 'description': 'Double a supplied count.', 'revision': '1'},)
    assert [row['tools'] for row in catalog.describe()] == [['double_count'], ['double_count'], []]
    encoded = json.dumps({'agents': catalog.describe(), 'tools': summaries})
    for secret in ('PRIVATE', 'handler', 'parameters', 'private_unused', 'count must'):
        assert secret not in encoded
    summaries[0]['description'] = 'Changed public copy'
    catalog.describe()[0]['tools'].clear()
    assert catalog.tool_descriptions()[0]['description'] == 'Double a supplied count.'
    assert catalog.describe()[0]['tools'] == ['double_count']


def test_catalog_without_application_tools_reports_known_empty_lists():
    catalog = AgentCatalog(models=[AgentModel('local', 'Local', '1', forbidden)],
        agents=[AgentDefinition(agent_id='worker', instructions='Work', models=('local',), default_model='local')])
    assert catalog.tool_descriptions() == ()
    assert catalog.describe()[0]['tools'] == []
