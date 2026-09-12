"""Authenticated background run admission and non-executing result reads."""
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Self

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..agents.catalog import Identifier
from ..agents.run_service import AgentRunService
from ..agents.workflow import WorkflowError


class _CancelRun(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)


class _StartRun(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    run_id: Identifier
    workflow_id: Identifier
    plan_revision: int = Field(ge=1, le=2**63 - 1)
    question: str = Field(min_length=1, max_length=4000)

    @model_validator(mode='after')
    def valid(self) -> Self:
        if not self.question.strip() or len(self.question.encode()) > 4000:
            raise ValueError('invalid run question')
        return self


async def _body(request: Request) -> bytes:
    declared = request.headers.get('content-length', '')
    if declared.isdigit() and int(declared) > 8192:
        raise WorkflowError('run_request_limit')
    content = bytearray()
    async for part in request.stream():
        if len(content) + len(part) > 8192:
            raise WorkflowError('run_request_limit')
        content.extend(part)
    return bytes(content)


def _response(value: object, status: int = 200) -> JSONResponse:
    return JSONResponse(value, status_code=status, headers={'Cache-Control': 'no-store'})


def _failure(error: WorkflowError | ValueError) -> JSONResponse:
    code = error.code if isinstance(error, WorkflowError) else 'invalid_agent_run'
    status = 503
    if code in {'run_busy', 'busy'}:
        status = 429
    elif code in {'run_request_conflict', 'plan_revision_conflict', 'plan_configuration_changed',
                  'run_scope_changed', 'outcome_unknown', 'run_cancelled', 'sources_invalid',
                  'not_completed', 'run_not_owned', 'binding_mismatch'}:
        status = 409
    elif code in {'agent_plan_not_found', 'space_deleted'}:
        status = 404
    elif code == 'run_request_limit':
        status = 413
    elif code.startswith('invalid_') or code == 'run_store_limit':
        status = 422
    response = _response({'error': code, 'code': code}, status)
    if status == 429:
        response.headers['Retry-After'] = '1'
    return response


def mount_agent_run_routes(app: FastAPI, service: AgentRunService,
                           space_for: Callable[..., Awaitable[str]],
                           assert_current_space: Callable[[Request, str], None]) -> None:
    """Host dependencies enforce authentication, roles and unchanged key scope."""
    @app.post('/v1/agent-runs')
    async def start(request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            body = _StartRun.model_validate_json(await _body(request))
            current = await space_for(request)
            if current != space:
                assert_current_space(request, space)
            result = await service.start(space, body.run_id, workflow_id=body.workflow_id,
                plan_revision=body.plan_revision, question=body.question,
                admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response(asdict(result), 202)
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.get('/v1/agent-runs')
    async def runs(request: Request, limit: int = 20, after: str | None = None,
                   space: str = Depends(space_for)) -> JSONResponse:
        try:
            page = await service.list(space, limit=limit, after=after)
            assert_current_space(request, space)
            return _response({'items': [asdict(item) for item in page.items], 'next_after': page.next_after})
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.get('/v1/agent-runs/{run_id}')
    async def status(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            progress = await service.status(space, run_id)
            assert_current_space(request, space)
            return _response(asdict(progress)) if progress else _response({'error': 'agent_run_not_found'}, 404)
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.get('/v1/agent-runs/{run_id}/request')
    async def invocation(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            saved = await service.request(space, run_id)
            assert_current_space(request, space)
            return _response(saved.model_dump(mode='json')) if saved else _response({'error': 'agent_run_not_found'}, 404)
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.get('/v1/agent-runs/{run_id}/result')
    async def result(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            answer = await service.result(space, run_id)
            assert_current_space(request, space)
            return _response({'space': space, **asdict(answer)}) if answer else _response({'error': 'agent_result_not_found'}, 404)
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.post('/v1/agent-runs/{run_id}/cancel')
    async def cancel(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            raw = await _body(request)
            _CancelRun.model_validate_json(raw if raw.strip() else b'{}')
            progress = await service.cancel(space, run_id,
                admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response(asdict(progress)) if progress else _response({'error': 'agent_run_not_found'}, 404)
        except (WorkflowError, ValueError) as error:
            return _failure(error)
