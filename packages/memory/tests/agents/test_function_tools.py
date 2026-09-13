"""Ordinary annotated functions reuse the native agent tool contract."""
from __future__ import annotations

import asyncio
import functools
import importlib.util
import json
import sys
import threading
from enum import Enum
from typing import Annotated

import pytest

from scone_memory.agents.custom_tools import ToolContext
from scone_memory.agents.function_tools import function_tool
from scone_memory.agents.evidence_loop import ToolStep
from scone_memory.agents.task_workflow import AgentTask, AgentTaskPlan, AgentWorkflow
from scone_memory.agents.workflow import WorkflowError
from scone_memory.retrieval.recall_scope import RecallScope
from .test_custom_tools import agents, call
from .test_evidence_tool_loop import Script, binding
from .test_task_workflow import memory


@pytest.mark.parametrize('asynchronous', [False, True])
async def test_function_signature_description_defaults_scope_and_execution_thread(memory, asynchronous):
    invoked = []
    def calculate(count: Annotated[int, 'Number of items.'], /, factor: int = 2, *, context: ToolContext) -> object:
        """Multiply a count by the requested factor."""
        invoked.append((count, factor, context.space, threading.get_ident()))
        return {'total': count * factor}
    async def calculate_async(count: Annotated[int, 'Number of items.'], /, factor: int = 2, *, context: ToolContext) -> object:
        return calculate(count, factor, context=context)
    registration = function_tool(calculate_async if asynchronous else calculate, name='calculate',
        description=calculate.__doc__, revision='1', context_parameter='context')
    schema = registration.openai()['function']['parameters']
    assert schema['required'] == ['count'] and schema['additionalProperties'] is False
    assert schema['properties']['count']['description'] == 'Number of items.'
    assert schema['properties']['factor']['default'] == 2
    assert 'context' not in schema['properties']
    model = Script(call({'count': 3}, name='calculate'), ToolStep(content='Six.'))
    result = await agents(model, [registration], names=('calculate',)).bind('worker').run('Double three', tools=binding(memory))
    assert result.output.text == 'Six.' and invoked[0][:3] == (3, 2, 'alpha')
    assert (invoked[0][3] == threading.get_ident()) is asynchronous
    assert json.loads(model.requests[1][0][-1]['content'])['result'] == {'total': 6}


@pytest.mark.parametrize('arguments', [{'count': True}, {'count': '3'}, {'count': 3, 'context': {}}, {'count': 3, 'extra': 1}, {}])
async def test_invalid_arguments_and_context_injection_never_reach_function(memory, arguments):
    invoked = []
    def calculate(count: int, context: ToolContext) -> object:
        invoked.append(1)
        return count
    registration = function_tool(calculate, description='Calculate.', revision='1', context_parameter='context')
    model = Script(call(arguments, name='calculate'), ToolStep(content='Refused.'))
    result = await agents(model, [registration], names=('calculate',)).bind('worker').run('Question', tools=binding(memory))
    assert not invoked and result.output.tool_outcomes[0].error == 'invalid_arguments'


async def test_default_values_are_snapshotted_and_fresh_for_every_invocation(memory):
    values = ['original']
    def collect(items: list[str] = values) -> object:
        """Collect the configured labels."""
        items.append('called')
        return items
    registered = function_tool(collect, revision='1')
    first = agents(Script(call({}, name='collect'), ToolStep(content='Done.')), [registered], names=('collect',)).bind('worker')
    values.append('MUTATED')
    collect.__defaults__ = (['REPLACED'],)
    for _ in range(2):
        context = ToolContext('alpha', RecallScope.validated(), None, asyncio.get_running_loop().time()+5)
        assert json.loads(await registered.invoke({}, context))['result'] == ['original', 'called']
    assert first.tools[0].info()['parameters']['properties']['items']['default'] == ['original']
    changed = agents(Script(), [function_tool(collect, revision='1')], names=('collect',)).bind('worker')
    assert first.fingerprint != changed.fingerprint


async def test_bound_methods_and_partial_functions_preserve_parameter_binding():
    class Counter:
        def total(self, count: int, *, offset: int = 0) -> object:
            return count + offset
    registered = function_tool(functools.partial(Counter().total, offset=2), name='total',
                               description='Add the configured offset.', revision='1')
    context = ToolContext('alpha', RecallScope.validated(), None, asyncio.get_running_loop().time()+5)
    assert json.loads(await registered.invoke({'count': 3}, context))['result'] == 5


def test_defaults_that_lose_their_python_type_during_union_roundtrip_refuse():
    class Color(Enum):
        RED = 'red'
    def enum_default(color: Color | str = Color.RED) -> object:
        return color
    def tuple_default(values: tuple[int, ...] | list[int] = (1, 2)) -> object:
        return values
    for function in (enum_default, tuple_default):
        with pytest.raises(ValueError, match='default'):
            function_tool(function, revision='1', description='Use the configured value.',
                          annotation_namespace={'Color': Color})


def test_string_annotation_expressions_are_not_executed():
    invoked = []
    def unsafe(value: 'mark()') -> object:
        return value
    with pytest.raises(ValueError):
        function_tool(unsafe, description='Unsafe annotation.', revision='1',
                      annotation_namespace={'mark': lambda: invoked.append(1) or int})
    assert not invoked


@pytest.mark.skipif(sys.version_info < (3, 14), reason='Deferred annotations require Python 3.14')
def test_deferred_annotation_expression_is_not_evaluated_during_inspection():
    invoked = []
    namespace = {'mark': lambda: invoked.append(1) or int}
    source = 'def deferred(value: mark()):\n    return value\n'
    exec(compile(source, '<deferred-function-test>', 'exec', dont_inherit=True), namespace)
    with pytest.raises(ValueError):
        function_tool(namespace['deferred'], description='Inspect safely.', revision='1')
    assert not invoked


@pytest.mark.skipif(sys.version_info < (3, 14), reason='Deferred annotations require Python 3.14')
def test_readable_deferred_annotation_expression_is_rejected_without_execution(tmp_path):
    source = tmp_path / 'deferred.py'
    source.write_text('invoked = []\ndef mark():\n    invoked.append(1)\n    return int\ndef deferred(value: mark()):\n    return value\n')
    spec = importlib.util.spec_from_file_location('deferred_fixture', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError):
        function_tool(module.deferred, description='Inspect safely.', revision='1')
    assert module.invoked == []


@pytest.mark.skipif(sys.version_info < (3, 14), reason='Deferred annotations require Python 3.14')
def test_changed_source_cannot_redefine_an_already_loaded_function_contract(tmp_path):
    source = tmp_path / 'loaded.py'
    source.write_text('def actual(value: int):\n    return value + 1\n')
    spec = importlib.util.spec_from_file_location('loaded_fixture', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = function_tool(module.actual, description='Increment.', revision='1')
    assert first.openai()['function']['parameters']['properties']['value']['type'] == 'integer'
    source.write_text('def actual(value: str):\n    return value + 1\n')
    with pytest.raises(ValueError, match='source'):
        function_tool(module.actual, description='Increment.', revision='1')


@pytest.mark.parametrize('failure', ['variadic', 'unannotated', 'bad_default', 'context_name', 'context_type', 'generator', 'metadata'])
def test_incompatible_function_contracts_fail_during_registration(failure):
    def variadic(*values: int) -> object:
        return values
    def unannotated(value):
        return value
    def bad_default(value: int = 'bad') -> object:
        return value
    def context_type(context: int) -> object:
        return context
    def generator(value: int):
        yield value
    def ordinary(value: int) -> object:
        return value
    chosen = {'variadic': variadic, 'unannotated': unannotated, 'bad_default': bad_default,
              'context_type': context_type, 'generator': generator}.get(failure, ordinary)
    options = {'context_parameter': 'missing' if failure == 'context_name' else 'context'} if failure.startswith('context_') else {}
    with pytest.raises(ValueError):
        function_tool(chosen, revision='1', description='' if failure == 'metadata' else 'Function.', **options)


async def test_inferred_function_workflow_reopen_reuses_result_and_default_changes_refuse(tmp_path, memory):
    invoked = []
    def build(factor):
        def calculate(count: int, multiplier: int = factor) -> object:
            invoked.append(1)
            return count * multiplier
        registry = agents(Script(call({'count': 3}, name='calculate'), ToolStep(content='Six.')),
            [function_tool(calculate, description='Multiply.', revision='1')], names=('calculate',))
        return AgentWorkflow(tmp_path/'journal', key=b'k'*32, catalog=registry,
            plan=AgentTaskPlan(workflow_id='multiply', tasks=(AgentTask(task_id='one', agent_id='worker', prompt='Double three.'),)),
            memory=memory, space='alpha', scope=RecallScope.validated())
    first = build(2)
    try:
        await first.run('r', 'Question')
    finally:
        first.close()
    reopened = build(2)
    try:
        assert (await reopened.run('r', 'Question')).reused_steps == ('one',)
    finally:
        reopened.close()
    changed = build(3)
    try:
        with pytest.raises(WorkflowError, match='binding_mismatch'):
            await changed.run('r', 'Question')
        assert invoked == [1]
    finally:
        changed.close()
