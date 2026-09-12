"""Scoped HTTP configuration for host-registered agents and saved task plans."""
from collections.abc import Awaitable, Callable

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..agents.catalog import AgentCatalog
from ..agents.plan_store import AgentPlan, AgentPlanStore, PlanConfigurationChanged, SavedAgentPlan
from ..agents.workflow import WorkflowError


class _SavePlan(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    expected_revision: int = Field(ge=0, le=2**63 - 2)
    plan: AgentPlan


async def _body(request: Request) -> bytes:
    limit = 128000
    declared = request.headers.get('content-length', '')
    if declared.isdigit() and int(declared) > limit:
        raise WorkflowError('plan_request_limit')
    content = bytearray()
    async for part in request.stream():
        if len(content) + len(part) > limit:
            raise WorkflowError('plan_request_limit')
        content.extend(part)
    return bytes(content)


def _failure(error: WorkflowError | ValueError) -> JSONResponse:
    code = error.code if isinstance(error, WorkflowError) else 'invalid_agent_plan'
    status = 503
    if code == 'plan_scope_changed':
        status = 403
    elif code == 'plan_revision_conflict':
        status = 409
    elif code == 'plan_request_limit':
        status = 413
    elif code.startswith('invalid_') or code in {'plan_payload_limit', 'plan_store_limit'}:
        status = 422
    return JSONResponse({'error': code, 'code': code}, status_code=status,
                        headers={'Cache-Control': 'no-store'})


def mount_agent_plan_routes(app: FastAPI, catalog: AgentCatalog, store: AgentPlanStore,
                            space_for: Callable[..., Awaitable[str]]) -> None:
    """The host dependency must authenticate, enforce roles and reject deleted spaces."""
    def present(saved: SavedAgentPlan) -> dict[str, object]:
        current = True
        try:
            saved.checked_plan(catalog)
        except PlanConfigurationChanged:
            current = False
        return {**saved.model_dump(mode='json'), 'configuration_current': current}

    def response(value: object) -> JSONResponse:
        return JSONResponse(value, headers={'Cache-Control': 'no-store'})

    @app.get('/v1/agents/catalog')
    async def choices(space: str = Depends(space_for)) -> JSONResponse:
        return response({'agents': catalog.describe()})

    @app.get('/v1/agent-plans')
    async def plans(limit: int = 50, after: str | None = None, space: str = Depends(space_for)) -> JSONResponse:
        try:
            page = store.list(space, limit=limit, after=after)
            return response({'items': [present(item) for item in page.items], 'next_after': page.next_after})
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.get('/v1/agent-plans/{workflow_id}')
    async def get_plan(workflow_id: str, space: str = Depends(space_for)) -> JSONResponse:
        try:
            saved = store.get(space, workflow_id)
            if saved is None:
                return JSONResponse({'error': 'agent_plan_not_found'}, status_code=404,
                                    headers={'Cache-Control': 'no-store'})
            return response(present(saved))
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.put('/v1/agent-plans/{workflow_id}')
    async def save_plan(workflow_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            raw = await _body(request)
            current_space = await space_for(request)
            if current_space != space:
                raise WorkflowError('plan_scope_changed')
            body = _SavePlan.model_validate_json(raw)
            if body.plan.workflow_id != workflow_id:
                raise ValueError('plan identity mismatch')
            return response(present(store.save(space, body.plan, catalog=catalog,
                                               expected_revision=body.expected_revision)))
        except (WorkflowError, ValueError) as error:
            return _failure(error)
