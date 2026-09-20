"""Static agent tools for discovering and invoking a reviewed recipe version."""
from collections.abc import Callable
import json

from .custom_tools import AgentTool, ToolContext, _check_context


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate recipe argument')
        result[key] = value
    return result


def recipe_execution_tools(*, space: str, revision: str,
                           bind: Callable[[str], AgentTool]) -> tuple[AgentTool, AgentTool]:
    """Host adapter; bind must enforce current approval and dependency identity."""
    def current(arguments: dict[str, object], context: ToolContext) -> AgentTool:
        if context.space != space:
            raise ValueError('recipe execution scope changed')
        proposal_id = arguments['proposal_id']
        assert isinstance(proposal_id, str)
        tool = bind(proposal_id)
        _check_context(context)
        return tool

    async def inspect(arguments: dict[str, object], context: ToolContext) -> object:
        tool = current(arguments, context)
        return {'proposal_id': arguments['proposal_id'], 'tool': tool.info()}

    async def invoke(arguments: dict[str, object], context: ToolContext) -> object:
        tool = current(arguments, context)
        if arguments['tool_revision'] != tool.revision:
            raise ValueError('reviewed recipe version changed')
        raw = arguments['arguments_json']
        assert isinstance(raw, str)
        values = json.loads(raw, object_pairs_hook=_unique)
        if not isinstance(values, dict) or tool.prepare_arguments(values) is None:
            raise ValueError('invalid recipe arguments')
        # The outer adapter is the journaled, human-approved call. The inner
        # compiler still checks current version authority before each dispatch.
        result = json.loads(await tool.invoke(values, context))
        if result.get('ok') is not True or 'result' not in result:
            raise ValueError('recipe invocation refused')
        return result['result']

    identifier = {'type': 'string', 'minLength': 1, 'maxLength': 128}
    return (
        AgentTool('inspect_tool_recipe',
            'Inspect a currently approved recipe by proposal ID to obtain its exact tool revision and input schema.',
            revision, {'type': 'object', 'properties': {'proposal_id': identifier},
                       'required': ['proposal_id'], 'additionalProperties': False}, inspect),
        AgentTool('invoke_tool_recipe',
            'Request execution of an approved recipe using its proposal ID, exact inspected tool revision and JSON inputs; human call approval is required.',
            revision, {'type': 'object', 'properties': {
                'proposal_id': identifier,
                'tool_revision': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                'arguments_json': {'type': 'string', 'maxLength': 14000}},
                'required': ['proposal_id', 'tool_revision', 'arguments_json'], 'additionalProperties': False},
            invoke, requires_approval=True),
    )
