"""Agent authors a recipe; only a reviewed, pinned version can become a tool."""
import json
import time
from dataclasses import replace

import pytest

from scone_memory.agents.custom_tools import AgentTool, ToolContext
from scone_memory.retrieval.recall_scope import RecallScope


def capability(effects, *, revision='1'):
    async def double(arguments, context):
        effects.append((arguments['count'], context.space))
        return {'value': arguments['count'] * 2}
    return AgentTool('double', 'Double a count.', revision,
        {'type': 'object', 'properties': {'count': {'type': 'integer'}},
         'required': ['count'], 'additionalProperties': False}, double)


def recipe(tool):
    from scone_memory.agents.recipe_models import ToolRecipe
    from scone_memory.agents.tool_recipes import capability_digest
    return ToolRecipe.model_validate_json(json.dumps({
        'name': 'quadruple', 'description': 'Double a count twice.',
        'requirements': 'Return four times the user count using reviewed arithmetic tools.',
        'inputs': [{'name': 'count', 'kind': 'integer'}],
        'steps': [
            {'name': 'first', 'tool': 'double', 'digest': capability_digest(tool),
             'arguments': {'count': {'kind': 'input', 'name': 'count'}}},
            {'name': 'second', 'tool': 'double', 'digest': capability_digest(tool),
             'arguments': {'count': {'kind': 'step', 'name': 'first', 'path': ['value']}}},
        ], 'result': {'kind': 'step', 'name': 'second', 'path': ['value']},
    }))


def context(space='alpha'):
    return ToolContext(space, RecallScope.validated(), None, time.monotonic() + 10)


async def test_proposal_review_reopen_and_revoke(tmp_path):
    from scone_memory.agents.tool_recipe_store import ToolRecipeStore
    effects = []
    tool = capability(effects)
    path = tmp_path / 'recipes.db'
    store = ToolRecipeStore(path, key=b'k' * 32)
    proposer = store.proposal_tool(space='alpha', proposed_by='author-agent', tools=[tool])
    payload = json.loads(await proposer.invoke({'proposal_id': 'version-1',
        'recipe_json': recipe(tool).model_dump_json()}, context()))
    assert payload['result']['status'] == 'pending'
    assert not effects
    with pytest.raises(ValueError, match='approved'):
        store.bind('alpha', 'version-1', tools=[tool])
    proposal = store.get('alpha', 'version-1')
    assert proposal.proposed_by == 'author-agent' and proposal.requirements == recipe(tool).requirements
    assert proposal.dependencies()[0]['name'] == 'double'
    store.decide('alpha', 'version-1', decision='approve', actor='human-owner',
                 reason='Reviewed both arithmetic steps and the input contract.', expected_revision=1)
    store.close()
    assert b'Reviewed both' not in path.read_bytes() and b'quadruple' not in path.read_bytes()
    reopened = ToolRecipeStore(path, key=b'k' * 32)
    try:
        approved = reopened.bind('alpha', 'version-1', tools=[tool])
        assert approved.requires_approval
        result = json.loads(await approved.invoke({'count': 3}, context()))
        assert result['result'] == 12
        assert result['source_status'] == 'unverified' and result['verified_accuracy'] is False
        assert effects == [(3, 'alpha'), (6, 'alpha')]
        reopened.revoke('alpha', 'version-1', actor='human-owner', reason='Retired.', expected_revision=2)
        with pytest.raises(RuntimeError):
            await approved.invoke({'count': 3}, context())
        assert len(effects) == 2
    finally:
        reopened.close()


async def test_pending_denied_changed_and_cross_space_tools_never_execute(tmp_path):
    from scone_memory.agents.tool_recipe_store import ToolRecipeStore
    effects = []
    tool = capability(effects)
    store = ToolRecipeStore(tmp_path / 'recipes', key=b'k' * 32)
    try:
        store.propose('alpha', 'one', recipe(tool), proposed_by='agent', tools=[tool])
        store.decide('alpha', 'one', decision='deny', actor='owner', reason='Not needed.', expected_revision=1)
        with pytest.raises(ValueError):
            store.bind('alpha', 'one', tools=[tool])
        store.propose('alpha', 'two', recipe(tool), proposed_by='agent', tools=[tool])
        store.decide('alpha', 'two', decision='approve', actor='owner', reason='Reviewed.', expected_revision=1)
        with pytest.raises(ValueError, match='dependency'):
            store.bind('alpha', 'two', tools=[replace(tool, revision='2')])
        with pytest.raises(ValueError):
            store.bind('bravo', 'two', tools=[tool])
        approved = store.bind('alpha', 'two', tools=[tool])
        with pytest.raises(RuntimeError):
            await approved.invoke({'count': 3}, context('bravo'))
        assert effects == []
    finally:
        store.close()


async def test_agent_can_discover_recipe_contract_and_pinned_capabilities(tmp_path):
    from scone_memory.agents.tool_recipe_store import ToolRecipeStore
    from scone_memory.agents.tool_recipes import capability_digest
    tool = capability([])
    store = ToolRecipeStore(tmp_path / 'recipes', key=b'k' * 32)
    try:
        discover = store.capabilities_tool(space='alpha', tools=[tool])
        packet = json.loads(await discover.invoke({}, context()))
        description = packet['result']
        assert description['tools'][0]['digest'] == capability_digest(tool)
        assert description['tools'][0]['tool']['parameters']['properties']['count']['type'] == 'integer'
        assert 'steps' in description['recipe_schema']['properties']
        with pytest.raises(RuntimeError):
            await discover.invoke({}, context('bravo'))
    finally:
        store.close()


@pytest.mark.parametrize('mutation', ['unknown', 'forward', 'cycle', 'duplicate', 'boolean_index', 'guarded', 'direct'])
def test_unsafe_or_unresolved_recipes_are_refused_before_proposal(tmp_path, mutation):
    from scone_memory.agents.recipe_models import ToolRecipe
    from scone_memory.agents.tool_recipe_store import ToolRecipeStore
    tool = capability([])
    if mutation == 'guarded':
        tool = replace(tool, requires_approval=True)
    if mutation == 'direct':
        tool = replace(tool, return_direct=True)
    raw = json.loads(recipe(tool).model_dump_json())
    if mutation == 'unknown':
        raw['steps'][0]['tool'] = 'missing'
    if mutation == 'forward':
        raw['steps'][0]['arguments']['count'] = {'kind': 'step', 'name': 'second'}
    if mutation == 'cycle':
        raw['steps'][0]['tool'] = 'quadruple'
    if mutation == 'duplicate':
        raw['steps'][1]['name'] = 'first'
    if mutation == 'boolean_index':
        raw['steps'][1]['arguments']['count']['path'] = [True]
    store = ToolRecipeStore(tmp_path / 'recipes', key=b'k' * 32)
    try:
        with pytest.raises(ValueError):
            store.propose('alpha', 'bad', ToolRecipe.model_validate_json(json.dumps(raw)), proposed_by='agent', tools=[tool])
        assert store.get('alpha', 'bad') is None
    finally:
        store.close()


async def test_revocation_between_steps_prevents_later_effects(tmp_path):
    from scone_memory.agents.tool_recipe_store import ToolRecipeStore
    effects = []
    store = ToolRecipeStore(tmp_path / 'recipes', key=b'k' * 32)
    async def revoking(arguments, ctx):
        effects.append(arguments['count'])
        store.revoke('alpha', 'one', actor='owner', reason='Stop now.', expected_revision=2)
        return {'value': arguments['count'] * 2}
    tool = replace(capability(effects), handler=revoking)
    try:
        store.propose('alpha', 'one', recipe(tool), proposed_by='agent', tools=[tool])
        store.decide('alpha', 'one', decision='approve', actor='owner', reason='Reviewed.', expected_revision=1)
        bound = store.bind('alpha', 'one', tools=[tool])
        with pytest.raises(RuntimeError):
            await bound.invoke({'count': 3}, context())
        assert effects == [3]
    finally:
        store.close()


def test_version_conflicts_and_review_compare_and_swap(tmp_path):
    from scone_memory.agents.tool_recipe_store import ToolRecipeStore
    tool = capability([])
    path = tmp_path / 'recipes'
    store = ToolRecipeStore(path, key=b'k' * 32)
    peer = ToolRecipeStore(path, key=b'k' * 32)
    try:
        original = store.propose('alpha', 'one', recipe(tool), proposed_by='agent', tools=[tool])
        assert peer.propose('alpha', 'one', recipe(tool), proposed_by='agent', tools=[tool]) == original
        with pytest.raises(ValueError, match='conflict'):
            store.propose('alpha', 'one', recipe(tool).model_copy(update={'description': 'Different'}), proposed_by='agent', tools=[tool])
        with pytest.raises(ValueError, match='independent'):
            store.decide('alpha', 'one', decision='approve', actor='agent', reason='Self approval', expected_revision=1)
        peer.decide('alpha', 'one', decision='deny', actor='owner', reason='Denied.', expected_revision=1)
        with pytest.raises(ValueError, match='revision'):
            store.decide('alpha', 'one', decision='approve', actor='owner', reason='Stale review.', expected_revision=1)
        assert store.get('alpha', 'one').status == 'denied'
    finally:
        store.close()
        peer.close()


@pytest.mark.parametrize('name', ['answer', 'search_memory', 'unknown_tool', 'recipe_capabilities', 'propose_tool_recipe'])
def test_reserved_recipe_names_fail_before_human_review(name):
    from scone_memory.agents.recipe_models import ToolRecipe
    raw = json.loads(recipe(capability([])).model_dump_json())
    raw['name'] = name
    with pytest.raises(ValueError, match='reserved'):
        ToolRecipe.model_validate_json(json.dumps(raw))


async def test_revoke_before_queued_sync_dependency_dispatch(tmp_path):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from scone_memory.agents.tool_recipe_store import ToolRecipeStore

    effects = []
    def double(arguments, tool_context):
        effects.append(arguments['count'])
        return {'value': arguments['count'] * 2}
    tool = replace(capability([]), handler=double)
    store = ToolRecipeStore(tmp_path / 'recipes', key=b'k' * 32)
    gate, started = threading.Event(), threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    previous_executor = loop._default_executor
    loop.set_default_executor(pool)
    pending = None
    def occupy_worker():
        started.set()
        gate.wait(timeout=5)
    try:
        store.propose('alpha', 'one', recipe(tool), proposed_by='agent', tools=[tool])
        store.decide('alpha', 'one', decision='approve', actor='human', reason='Reviewed', expected_revision=1)
        approved = store.bind('alpha', 'one', tools=[tool])
        pool.submit(occupy_worker)
        assert started.wait(timeout=1)
        pending = asyncio.create_task(approved.invoke({'count': 3}, context()))
        # Wait until actual dependency worker is queued, without dispatching it.
        for _ in range(100):
            if pool._work_queue.qsize():
                break
            await asyncio.sleep(.001)
        assert pool._work_queue.qsize() == 1
        assert effects == []
        store.revoke('alpha', 'one', actor='human', reason='Revoke before worker dispatch', expected_revision=2)
        gate.set()
        with pytest.raises(RuntimeError):
            await pending
        assert effects == []
    finally:
        gate.set()
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        loop._default_executor = previous_executor
        pool.shutdown(wait=True)
        store.close()




def test_authoring_revision_binds_store_space_and_author(tmp_path):
    from scone_memory.agents.tool_recipe_store import ToolRecipeStore
    tool = capability([])
    first = ToolRecipeStore(tmp_path / 'first', key=b'k' * 32)
    second = ToolRecipeStore(tmp_path / 'second', key=b'k' * 32)
    try:
        revisions = {store.proposal_tool(space=space, proposed_by=actor, tools=[tool]).revision
                     for store, space, actor in [(first, 'alpha', 'one'), (first, 'bravo', 'one'),
                        (first, 'alpha', 'two'), (second, 'alpha', 'one')]}
        assert len(revisions) == 4
    finally:
        first.close()
        second.close()


async def test_deadline_during_dispatch_authorization_blocks_sync_handler(tmp_path):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from scone_memory.agents.tool_recipe_store import ToolRecipeStore
    import sqlite3
    import time
    effects = []
    def double(arguments, tool_context):
        effects.append(arguments['count'])
        return {'value': arguments['count'] * 2}
    tool = replace(capability([]), handler=double)
    path = tmp_path / 'recipes'
    store = ToolRecipeStore(path, key=b'k' * 32)
    gate, started = threading.Event(), threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    previous_executor = loop._default_executor
    loop.set_default_executor(pool)
    pending = locked = None
    def occupy_worker():
        started.set()
        gate.wait(timeout=5)
    try:
        store.propose('alpha', 'one', recipe(tool), proposed_by='agent', tools=[tool])
        store.decide('alpha', 'one', decision='approve', actor='human', reason='Reviewed', expected_revision=1)
        approved = store.bind('alpha', 'one', tools=[tool])
        pool.submit(occupy_worker)
        assert started.wait(timeout=1)
        pending = asyncio.create_task(approved.invoke({'count': 3}, replace(context(), deadline=time.monotonic() + .2)))
        for _ in range(100):
            if pool._work_queue.qsize():
                break
            await asyncio.sleep(.001)
        assert pool._work_queue.qsize() == 1
        # Guard reaches SQLite only after event-loop admission completed.
        locked = sqlite3.connect(path, isolation_level=None)
        locked.execute('BEGIN EXCLUSIVE')
        gate.set()
        with pytest.raises(RuntimeError):
            await pending
        assert effects == []
        locked.execute('ROLLBACK')
        locked.close()
        locked = None
        pool.shutdown(wait=True)
        assert effects == []
    finally:
        gate.set()
        if locked is not None:
            locked.close()
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        loop._default_executor = previous_executor
        pool.shutdown(wait=True)
        store.close()


def test_human_review_backlog_pages_only_requested_space(tmp_path):
    from scone_memory.agents.tool_recipe_store import ToolRecipeStore
    tool = capability([])
    store = ToolRecipeStore(tmp_path / 'recipes', key=b'k' * 32)
    try:
        for space in ('alpha', 'bravo'):
            for name in ('one', 'two', 'three'):
                store.propose(space, name, recipe(tool), proposed_by='agent', tools=[tool])
        seen, cursor = [], None
        for _ in range(3):
            page = store.list('alpha', limit=1, after=cursor)
            assert len(page.items) == 1 and page.items[0].space == 'alpha'
            seen.append(page.items[0].proposal_id)
            cursor = page.next_after
            if cursor is not None:
                with pytest.raises(ValueError, match='cursor'):
                    store.list('bravo', after=cursor)
        assert set(seen) == {'one', 'two', 'three'} and cursor is None
    finally:
        store.close()


def test_bound_recipe_revision_identifies_live_review_authority(tmp_path):
    from scone_memory.agents.tool_recipe_store import ToolRecipeStore
    tool = capability([])
    path = tmp_path / 'recipes'
    store = ToolRecipeStore(path, key=b'k' * 32)
    clone = None
    try:
        store.propose('alpha', 'one', recipe(tool), proposed_by='agent', tools=[tool])
        store.decide('alpha', 'one', decision='approve', actor='owner', reason='Reviewed.', expected_revision=1)
        copied = tmp_path / 'copy'
        copied.write_bytes(path.read_bytes())
        copied.chmod(0o600)
        clone = ToolRecipeStore(copied, key=b'k' * 32)
        assert store.bind('alpha', 'one', tools=[tool]).revision != clone.bind('alpha', 'one', tools=[tool]).revision
    finally:
        if clone is not None:
            clone.close()
        store.close()
