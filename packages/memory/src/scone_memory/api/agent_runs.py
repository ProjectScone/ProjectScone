"""Authenticated background run admission and non-executing result reads."""
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Self

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..agents.catalog import Identifier
from ..agents.handoff_workflow import HandoffResult
from ..agents.input_store import AgentInputRecord
from ..agents.run_service import AgentRunService
from ..agents.workflow import WorkflowError


class _CancelRun(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)


class _InputResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    response: str = Field(min_length=1, max_length=4000)
    expected_revision: int = Field(ge=1, le=3)


class _ContinueRun(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    continuation_id: Identifier
    responses: dict[Identifier, int] = Field(min_length=1, max_length=32)


class _StartRun(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    run_id: Identifier
    workflow_id: Identifier
    plan_revision: int = Field(ge=1, le=2**63 - 1)
    question: str = Field(min_length=1, max_length=4000)
    max_parallel: int = Field(default=1, ge=1, le=8)

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
                  'not_completed', 'run_not_owned', 'binding_mismatch', 'input_request_conflict',
                  'input_revision_conflict', 'input_response_conflict', 'input_activation_conflict',
                  'input_not_answered', 'input_not_ready', 'interactive_plan_required'}:
        status = 409
    elif code in {'agent_plan_not_found', 'space_deleted', 'run_not_found', 'input_not_found', 'input_task_not_found'}:
        status = 404
    elif code == 'run_request_limit':
        status = 413
    elif code.startswith('invalid_') or code == 'run_store_limit':
        status = 422
    response = _response({'error': code, 'code': code}, status)
    if status == 429:
        response.headers['Retry-After'] = '1'
    return response


def _input(record: AgentInputRecord) -> dict[str, object]:
    return record.model_dump(mode='json', exclude={'invocation_digest'})


def mount_agent_run_routes(app: FastAPI, service: AgentRunService,
                           space_for: Callable[..., Awaitable[str]],
                           assert_current_space: Callable[[Request, str], None]) -> None:
    """Host dependencies enforce authentication, roles and unchanged key scope."""
    @app.get('/v1/agents/run-policy')
    async def policy(request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            result = await service.policy(space)
            assert_current_space(request, space)
            return _response(result)
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.post('/v1/agent-runs')
    async def start(request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            body = _StartRun.model_validate_json(await _body(request))
            current = await space_for(request)
            if current != space:
                assert_current_space(request, space)
            result = await service.start(space, body.run_id, workflow_id=body.workflow_id,
                plan_revision=body.plan_revision, question=body.question, max_parallel=body.max_parallel,
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
            if answer is None:
                return _response({'error': 'agent_result_not_found'}, 404)
            payload = answer.model_dump(mode='json') if isinstance(answer, HandoffResult) else asdict(answer)
            return _response({'space': space, **payload})
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

    @app.get('/v1/agent-runs/{run_id}/inputs')
    async def inputs(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            records = await service.inputs(space, run_id,
                admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response({'space': space, 'run_id': run_id, 'items': [_input(record) for record in records]})
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.post('/v1/agent-runs/{run_id}/inputs/{task_id}/response')
    async def respond(run_id: str, task_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            body = _InputResponse.model_validate_json(await _body(request))
            assert_current_space(request, space)
            record = await service.respond(space, run_id, task_id, response=body.response,
                expected_revision=body.expected_revision, admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response(_input(record))
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.post('/v1/agent-runs/{run_id}/continue')
    async def continue_run(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            body = _ContinueRun.model_validate_json(await _body(request))
            assert_current_space(request, space)
            result = await service.continue_run(space, run_id, continuation_id=body.continuation_id,
                responses=body.responses, admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response(asdict(result), 202)
        except (WorkflowError, ValueError) as error:
            return _failure(error)
