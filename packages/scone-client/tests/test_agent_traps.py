"""Standalone clients validate trap diagnostics without importing the runtime."""
from copy import deepcopy

import pytest
from scone import SconeError
from scone.agent_events import ProgressEvent
from test_agent_history import event, page, parse


def graph():
    return dict(nodes=[dict(node_id=1, first_round=1, visits=2, comparable=True)],
                edges=[dict(source=1, target=1, count=1)], path=[1, 1], pattern=[1], repetitions=2)


def test_typed_trap_history_and_legacy_events():
    value = page()
    value['items'][0]['event'].update(kind='trap_detected', trap_graph=graph())
    result = parse(value).items[0].event
    assert result.trap_graph.path == (1, 1)
    assert result.trap_graph.nodes[0].visits == 2
    assert result.model_id == 'careful'
    assert ProgressEvent.from_json(event()).trap_graph is None


@pytest.mark.parametrize('change', ['missing', 'wrong_kind', 'metadata', 'unknown_field',
    'visits', 'edges', 'pattern', 'boolean', 'barrier', 'identity', 'oversized'])
def test_untrusted_trap_report_is_refused(change):
    report = deepcopy(graph())
    value = dict(event(), kind='trap_detected', trap_graph=report)
    if change == 'missing':
        value.pop('trap_graph')
    elif change == 'wrong_kind':
        value['kind'] = 'turn_failed'
    elif change == 'metadata':
        value['tool_name'] = 'search_memory'
    elif change == 'unknown_field':
        report['nodes'][0]['query'] = 'PRIVATE'
    elif change == 'visits':
        report['nodes'][0]['visits'] = 3
    elif change == 'edges':
        report['edges'] *= 2
    elif change == 'pattern':
        report['repetitions'] = 3
    elif change == 'boolean':
        report['path'] = [True, True]
    elif change == 'barrier':
        report['nodes'][0]['comparable'] = False
    elif change == 'identity':
        report['nodes'][0]['node_id'] = 2
    else:
        report['path'] = [1] * 17
    with pytest.raises(SconeError) as caught:
        ProgressEvent.from_json(value)
    assert 'PRIVATE' not in str(caught.value)
