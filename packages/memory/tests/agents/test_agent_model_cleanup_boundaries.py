"""Independent cleanup privacy and cancellation regressions."""

import asyncio
import traceback
import pytest
from .test_agent_model_cleanup import registry
from .test_task_workflow import memory
from .test_evidence_tool_loop import binding
from scone_memory.agents.evidence_loop import ToolStep


@pytest.mark.parametrize(
    'kind',
    [
        'property',
        'property_cancel',
        'sync_raise',
        'sync_value',
        'async_raise',
        'async_cancel',
        'async_self_cancel',
    ],
)
async def test_cleanup_boundary_refuses_without_private_disclosure(memory, kind):
    marker = 'PRIVATE_CLEANUP_CREDENTIAL'

    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done')

    if kind in ('property', 'property_cancel'):

        def getter(self):
            raise (asyncio.CancelledError(marker) if kind == 'property_cancel' else RuntimeError(marker))

        Model.aclose = property(getter)
    elif kind == 'sync_raise':

        def close(self):
            raise RuntimeError(marker)

        Model.aclose = close
    elif kind == 'sync_value':
        Model.aclose = lambda self: marker
    elif kind == 'async_self_cancel':

        async def close(self):
            asyncio.current_task().cancel(marker)

        Model.aclose = close
    elif kind == 'async_cancel':

        async def close(self):
            raise asyncio.CancelledError(marker)

        Model.aclose = close
    else:

        async def close(self):
            raise RuntimeError(marker)

        Model.aclose = close
    caught = None
    try:
        await registry(Model).bind('worker').run('Q', tools=binding(memory))
    except BaseException as error:
        caught = error
    assert caught is not None, 'invalid cleanup was silently ignored'
    rendered = ''.join(traceback.format_exception(caught))
    assert marker not in rendered, rendered


async def test_swallowed_model_cancel_still_closes_and_refuses(memory):
    entered = asyncio.Event()
    closed = []

    class Model:
        async def complete(self, messages, tools):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return ToolStep(content='Done')

        async def aclose(self):
            closed.append(True)

    task = asyncio.create_task(registry(Model).bind('worker').run('Q', tools=binding(memory)))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [True]


async def test_cancelled_cleanup_failure_does_not_log_or_leak(memory):
    entered = asyncio.Event()
    release = asyncio.Event()
    errors = []
    loop = asyncio.get_running_loop()
    prior = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, ctx: errors.append(ctx))

    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done')

        async def aclose(self):
            entered.set()
            await release.wait()
            raise RuntimeError('PRIVATE_LATE_CLEANUP')

    task = asyncio.create_task(registry(Model).bind('worker').run('Q', tools=binding(memory)))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert 'PRIVATE_LATE_CLEANUP' not in ''.join(traceback.format_exception(caught.value))
        await asyncio.sleep(0)
        assert errors == []
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        loop.set_exception_handler(prior)


async def test_cleanup_elapsed_cannot_publish_late_result(memory):
    import time
    from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel
    from scone_memory.agents.evidence_loop import ToolLoopLimits

    closed = []

    class Model:
        async def complete(self, messages, tools):
            return ToolStep(content='Done')

        async def aclose(self):
            time.sleep(0.03)
            closed.append(True)

    selected = AgentCatalog(
        models=[AgentModel('m', 'Model', '1', Model)],
        agents=[
            AgentDefinition(
                agent_id='a',
                instructions='Answer',
                models=('m',),
                default_model='m',
                initial_search=False,
                limits=ToolLoopLimits(timeout_s=0.01),
            )
        ],
    ).bind('a')
    with pytest.raises(TimeoutError):
        await selected.run('Q', tools=binding(memory))
    assert closed == [True]


async def test_pause_primary_preserved_with_sanitized_cleanup_note(memory):
    from .test_agent_turn_recovery import storage
    from .test_custom_tools import call, tool
    from scone_memory.agents.turn_journal import TurnJournalPaused

    class Model:
        async def complete(self, messages, tools):
            return call()

        async def aclose(self):
            raise RuntimeError('PRIVATE_PAUSE_CLEANUP')

    points, _ = storage()
    selected = registry(Model, registered=[tool(lambda args, ctx: None)]).bind('worker')
    with pytest.raises(TurnJournalPaused) as caught:
        await selected.run('Q', tools=binding(memory), checkpoints=points, max_new_operations=1)
    assert 'agent_model_cleanup_failed' in caught.value.__notes__
    assert 'PRIVATE_PAUSE_CLEANUP' not in ''.join(traceback.format_exception(caught.value))


async def test_declared_cleanup_attribute_error_cannot_be_treated_as_absence(memory):
    calls = []

    class Model:
        async def complete(self, messages, tools):
            calls.append(True)
            return ToolStep(content='Done')

        @property
        def aclose(self):
            raise AttributeError('PRIVATE_MISSING_CLOSE_CLIENT')

    with pytest.raises(RuntimeError, match='agent_model_cleanup_failed'):
        await registry(Model).bind('worker').run('Q', tools=binding(memory))
    assert calls == []
