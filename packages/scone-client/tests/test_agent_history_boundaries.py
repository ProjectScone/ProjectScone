import pytest
from scone import SconeError
from scone.agent_events import ProgressEvent
from test_agent_history import page, parse, cursor, event


@pytest.mark.parametrize(
    'name', ['search_memory', 'trace_memory', 'read_memory', 'compute_memory', 'unknown_tool', 'custom_tool']
)
def test_native_builtin_progress_names_are_accepted(name):
    value = event()
    value.update(kind='tool_proposed', tool_index=1, tool_name=name, origin='model')
    assert ProgressEvent.from_json(value).tool_name == name


from copy import deepcopy
from scone.agent_history import HistoryPage
from scone.agent_models import RunRequest
from test_agent_models import REQUEST


def test_one_collection_cannot_change_task_selection_inside_page():
    raw_request = deepcopy(REQUEST)
    raw_request['plan']['plan']['tasks'].append(
        {
            'task_id': 'other',
            'agent_id': 'worker',
            'model_id': 'careful',
            'prompt': 'Other',
            'depends_on': ['choose'],
        }
    )
    raw_request['plan']['bindings']['other'] = 'a' * 64
    request = RunRequest.from_json(raw_request, expected_space='alpha', run_id='one')
    value = page()
    first = value['items'][0]
    first['collection_id'] = 'e' * 32
    second = deepcopy(first)
    second.update(position=2, step_id='other', selection_id='other')
    second['event'].update(sequence=2, kind='turn_completed')
    value.update(items=[first, second], next_after=cursor(2))
    with pytest.raises(SconeError):
        HistoryPage.from_json(value, request=request)
