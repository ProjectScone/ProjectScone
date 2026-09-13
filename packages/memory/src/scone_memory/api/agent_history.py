"""Read-only delivery of authorized native agent observations."""

from collections.abc import Awaitable, Callable
import re

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from ..agents.run_service import AgentRunService
from ..agents.workflow import WorkflowError
from .agent_runs import _failure, _response


def _selection(request: Request) -> tuple[int, str | None]:
    query = request.query_params
    if any(name not in ('limit', 'after') or len(query.getlist(name)) != 1 for name in query):
        raise WorkflowError('invalid_history_query')
    raw = query.get('limit', '50')
    if re.fullmatch(r'[1-9][0-9]{0,2}', raw) is None or int(raw) > 100:
        raise WorkflowError('invalid_history_query')
    after = query.get('after')
    if after is not None and (not after or len(after) > 4096):
        raise WorkflowError('invalid_history_cursor')
    return int(raw), after


def mount_agent_history_routes(
    app: FastAPI,
    service: AgentRunService,
    space_for: Callable[..., Awaitable[str]],
    assert_current_space: Callable[[Request, str], None],
) -> None:
    @app.get('/v1/agent-runs/{run_id}/history')
    async def history(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            limit, after = _selection(request)
            page = await service.history_for_delivery(
                space,
                run_id,
                after=after,
                limit=limit,
                admission_guard=lambda: assert_current_space(request, space),
            )
            payload = {
                'space': space,
                'run_id': run_id,
                'available': page.available,
                'items': [entry.model_dump(mode='json') for entry in page.items],
                'next_after': page.next_after,
                'retained_from': page.retained_from,
                'omitted': page.omitted,
            }
            assert_current_space(request, space)
            return _response(payload)
        except (WorkflowError, ValueError) as error:
            return _failure(error)
