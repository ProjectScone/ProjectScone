"""Independent native-wire compatibility and malformed-input regressions."""
import pytest
from scone import SconeError
from scone.agent_models import ModelTask, SavedPlan
from test_agents import fixture
from test_agent_models import SAVED


@pytest.mark.parametrize('label', ['é' * 256, '🙂' * 256])
def test_catalog_accepts_native_unicode_label_character_bound(fixture, label):
    server, agents = fixture
    server.route('GET', '/v1/agents/catalog', 200, {'agents': [{
        'agent_id': 'worker', 'default_model': 'local', 'models': [{
            'model_id': 'local', 'revision': '1', 'label': label}]}]})
    assert agents.catalog()[0].models[0].label == label


def test_invalid_unicode_question_raises_client_error_before_requests(fixture):
    server, agents = fixture
    with pytest.raises(SconeError):
        agents.start('one', plan=SavedPlan.from_json(SAVED, expected_space='alpha'), question='\ud800')
    assert server.requests == []


def test_invalid_unicode_task_prompt_raises_client_error():
    with pytest.raises(SconeError):
        ModelTask('a', 'worker', 'local', '\ud800')


def test_catalog_does_not_expand_native_label_character_limit(fixture):
    server, agents = fixture
    server.route('GET', '/v1/agents/catalog', 200, {'agents': [{
        'agent_id': 'worker', 'default_model': 'local', 'models': [{
            'model_id': 'local', 'revision': '1', 'label': 'é' * 257}]}]})
    with pytest.raises(SconeError):
        agents.catalog()
