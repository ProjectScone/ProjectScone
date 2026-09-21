"""Static catalog adapters admit exactly the currently reviewed recipe version."""
from dataclasses import replace
import json
import shutil

import pytest

from scone_memory.agents.tool_recipe_store import ToolRecipeStore
from tests.agents.test_tool_recipes import capability, context, recipe


@pytest.fixture
def setup(tmp_path):
    effects = []
    primitive = capability(effects)
    store = ToolRecipeStore(tmp_path / 'recipes', key=b'k' * 32)
    try:
        yield store, primitive, effects
    finally:
        store.close()


def approve(store, primitive):
    store.propose('alpha', 'one', recipe(primitive), proposed_by='author', tools=[primitive])
    store.decide('alpha', 'one', decision='approve', actor='human', reason='Reviewed', expected_revision=1)


async def test_adapters_registered_before_proposal_use_exact_approved_version(setup):
    store, primitive, effects = setup
    inspect, invoke = store.execution_tools(space='alpha', tools=[primitive])
    assert invoke.requires_approval and not inspect.requires_approval
    approve(store, primitive)
    description = json.loads(await inspect.invoke({'proposal_id': 'one'}, context()))['result']
    assert description['proposal_id'] == 'one'
    assert description['tool']['name'] == 'quadruple'
    assert description['tool']['parameters']['properties']['count']['type'] == 'integer'
    assert not effects
    args = {'proposal_id': 'one', 'tool_revision': description['tool']['revision'], 'arguments_json': '{"count":3}'}
    result = json.loads(await invoke.invoke(args, context()))
    assert result['result'] == 12
    assert result['verified_accuracy'] is False and result['source_status'] == 'unverified'
    assert effects == [(3, 'alpha'), (6, 'alpha')]


@pytest.mark.parametrize('failure', ['pending', 'denied', 'revoked', 'revision', 'scope', 'dependency', 'closed'])
async def test_adapter_refuses_unapproved_or_changed_authority_without_effects(setup, failure):
    store, primitive, effects = setup
    store.propose('alpha', 'one', recipe(primitive), proposed_by='author', tools=[primitive])
    if failure != 'pending':
        store.decide('alpha', 'one', decision='deny' if failure == 'denied' else 'approve',
                     actor='human', reason='Reviewed', expected_revision=1)
    revision = '0' * 64 if failure in ('pending', 'denied') else store.bind('alpha', 'one', tools=[primitive]).revision
    inspect, invoke = store.execution_tools(space='alpha', tools=[primitive])
    if failure == 'revoked':
        store.revoke('alpha', 'one', actor='human', reason='Retired', expected_revision=2)
    elif failure == 'revision':
        revision = '0' * 64
    elif failure == 'dependency':
        inspect, invoke = store.execution_tools(space='alpha', tools=[replace(primitive, revision='2')])
    elif failure == 'closed':
        store.close()
    with pytest.raises(RuntimeError):
        await invoke.invoke({'proposal_id': 'one', 'tool_revision': revision, 'arguments_json': '{"count":3}'},
                            context('bravo' if failure == 'scope' else 'alpha'))
    assert not effects


@pytest.mark.parametrize('raw', ['{"count":3,"count":4}', '{"count":true}', '[]', '{"count":NaN}', '{"count":3,"other":1}'])
async def test_inner_arguments_are_unambiguous_and_validated_before_effects(setup, raw):
    store, primitive, effects = setup
    approve(store, primitive)
    _, invoke = store.execution_tools(space='alpha', tools=[primitive])
    revision = store.bind('alpha', 'one', tools=[primitive]).revision
    with pytest.raises(RuntimeError):
        await invoke.invoke({'proposal_id': 'one', 'tool_revision': revision, 'arguments_json': raw}, context())
    assert not effects


async def test_adapter_identity_survives_reopen_but_not_authority_copy(setup, tmp_path):
    store, primitive, effects = setup
    approve(store, primitive)
    _, original = store.execution_tools(space='alpha', tools=[primitive])
    revision = store.bind('alpha', 'one', tools=[primitive]).revision
    store.close()
    reopened = ToolRecipeStore(tmp_path / 'recipes', key=b'k' * 32)
    shutil.copyfile(tmp_path / 'recipes', tmp_path / 'copy')
    (tmp_path / 'copy').chmod(0o600)
    copied = ToolRecipeStore(tmp_path / 'copy', key=b'k' * 32)
    try:
        _, resumed = reopened.execution_tools(space='alpha', tools=[primitive])
        _, foreign = copied.execution_tools(space='alpha', tools=[primitive])
        assert resumed.revision == original.revision
        assert foreign.revision != original.revision
        with pytest.raises(RuntimeError):
            await foreign.invoke({'proposal_id': 'one', 'tool_revision': revision, 'arguments_json': '{"count":3}'}, context())
        assert not effects
        packet = json.loads(await resumed.invoke({'proposal_id': 'one', 'tool_revision': revision, 'arguments_json': '{"count":3}'}, context()))
        assert packet['result'] == 12
    finally:
        reopened.close()
        copied.close()
