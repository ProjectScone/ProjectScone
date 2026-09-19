"""Interpret reviewed recipes using only explicitly provided host capabilities."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
import inspect
import hashlib
import json

from .custom_tools import AgentTool, ToolContext, _check_context, _check_dispatch, _encoded, snapshot_tools
from .recipe_models import RecipeLiteral, RecipeReference, RecipeValue, ToolRecipe


def capability_digest(tool: AgentTool) -> str:
    encoded = json.dumps(tool.info(), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest()


def recipe_dependencies(recipe: ToolRecipe, tools: Sequence[AgentTool]) -> tuple[AgentTool, ...]:
    selected = {tool.name: tool for tool in snapshot_tools(tools)}
    used: dict[str, AgentTool] = {}
    for step in recipe.steps:
        tool = selected.get(step.tool)
        if tool is None or capability_digest(tool) != step.digest:
            raise ValueError('recipe dependency changed or unavailable')
        if tool.requires_approval or tool.return_direct:
            raise ValueError('recipe dependency requires separate call admission')
        used[tool.name] = tool
    return tuple(used.values())


def _resolve(value: RecipeValue, arguments: dict[str, object], results: dict[str, object]) -> object:
    if isinstance(value, RecipeLiteral):
        return json.loads(value.value_json)
    assert isinstance(value, RecipeReference)
    current = (arguments if value.kind == 'input' else results)[value.name]
    for part in value.path:
        if type(part) is str and isinstance(current, dict) and part in current:
            current = current[part]
        elif type(part) is int and isinstance(current, list) and part < len(current):
            current = current[part]
        else:
            raise ValueError('recipe result path unavailable')
    return json.loads(_encoded(current, 16000))


def reviewed_recipe_tool(recipe: ToolRecipe, *, revision: str, tools: Sequence[AgentTool],
                         admit: Callable[[ToolContext], None],
                         dispatch_admit: Callable[[ToolContext], None]) -> AgentTool:
    """Host compiler; callers supply current version authorization, not model text."""
    recipe = ToolRecipe.model_validate_json(recipe.model_dump_json())
    def guarded(tool: AgentTool) -> AgentTool:
        original = tool.handler
        if inspect.iscoroutinefunction(original):
            async def asynchronous(arguments: dict[str, object], context: ToolContext) -> object:
                dispatch_admit(context)
                _check_context(context)
                return await original(arguments, context)
            return replace(tool, handler=asynchronous)
        def synchronous(arguments: dict[str, object], context: ToolContext) -> object:
            # This runs inside the worker, after any queue delay.
            dispatch_admit(context)
            _check_dispatch(context)
            return original(arguments, context)
        return replace(tool, handler=synchronous)

    dependencies = {tool.name: guarded(tool) for tool in recipe_dependencies(recipe, tools)}

    async def execute(arguments: dict[str, object], context: ToolContext) -> object:
        results: dict[str, object] = {}
        remaining = 64000
        for step in recipe.steps:
            _check_context(context)
            admit(context)
            tool = dependencies[step.tool]
            values = {name: _resolve(value, arguments, results) for name, value in step.arguments.items()}
            # Revalidate immediately before dispatch; resolving inputs grants no authority.
            _check_context(context)
            admit(context)
            payload = await tool.invoke(values, context, remaining_output_bytes=remaining)
            remaining -= len(payload.encode())
            packet = json.loads(payload)
            if packet.get('ok') is not True or 'result' not in packet:
                raise ValueError('recipe dependency refused its arguments')
            results[step.name] = packet['result']
        _check_context(context)
        admit(context)
        return _resolve(recipe.result, arguments, results)

    return AgentTool(recipe.name, recipe.description, revision, recipe.parameters(), execute,
                     requires_approval=True)
