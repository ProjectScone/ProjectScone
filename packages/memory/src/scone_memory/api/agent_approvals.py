"""Authenticated exact-call review and explicitly selected tool continuation."""
from collections.abc import Awaitable, Callable
from dataclasses import asdict
import json
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .agent_runs import _body, _failure, _response
from ..agents.approval_models import Decision, Digest, ToolApprovalRecord
from ..agents.approval_store import decision_digest
from ..agents.catalog import Identifier
from ..agents.run_service import AgentRunService
from ..agents.workflow import WorkflowError


class _Decision(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    decision: Decision
    expected_revision: int = Field(ge=1, le=1)


class _Continuation(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    continuation_id: Identifier
    decisions: dict[Digest, Annotated[int, Field(ge=2, le=2)]] = Field(min_length=1, max_length=32)


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WorkflowError('invalid_approval_request')
        result[key] = value
    return result


async def _control_body(request: Request) -> object:
    raw = await _body(request)
    try:
        return json.loads(raw, object_pairs_hook=_unique)
    except (ValueError, RecursionError):
        raise WorkflowError('invalid_approval_request') from None


def _record(record: ToolApprovalRecord) -> dict[str, object]:
    return {**record.model_dump(mode='json', exclude={'invocation_digest'}),
            'decision_digest': decision_digest(record) if record.revision >= 2 else None}


def mount_agent_approval_routes(app: FastAPI, service: AgentRunService,
    space_for: Callable[..., Awaitable[str]], assert_current_space: Callable[[Request, str], None],
    actor_for: Callable[[Request], str]) -> None:
    """Host supplies current role/scope authorization and the decision actor."""
    @app.get('/v1/agent-runs/{run_id}/approvals')
    async def approvals(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            records = await service.approvals(space, run_id,
                admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response({'space': space, 'run_id': run_id, 'items': [
                _record(record) for record in records]})
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.post('/v1/agent-runs/{run_id}/approvals/{request_id}/decision')
    async def decide(run_id: str, request_id: str, request: Request,
                     space: str = Depends(space_for)) -> JSONResponse:
        try:
            body = _Decision.model_validate(await _control_body(request))
            assert_current_space(request, space)
            actor = actor_for(request)
            record = await service.decide_tool(space, run_id, request_id,
                decision=body.decision, actor=actor, expected_revision=body.expected_revision,
                admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response(_record(record))
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.post('/v1/agent-runs/{run_id}/approval-continuations')
    async def continue_tools(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            body = _Continuation.model_validate(await _control_body(request))
            assert_current_space(request, space)
            result = await service.continue_tools(space, run_id, continuation_id=body.continuation_id,
                decisions=body.decisions, admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response({'status': asdict(result.status),
                'activation': result.activation.model_dump(mode='json', exclude={'invocation_digest'})}, 202)
        except (WorkflowError, ValueError) as error:
            return _failure(error)
