"""Current-authority HTTP review of immutable tool proposals; no execution."""
from collections.abc import Awaitable, Callable
import json
import re
from typing import Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..agents.tool_recipe_store import ToolRecipeConflict, ToolRecipeProposal, ToolRecipeStore, validate_proposal_id
from ..agents.workflow import WorkflowError


class _Reason(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    reason: str = Field(min_length=1, max_length=2000)

    @field_validator('reason')
    @classmethod
    def meaningful(cls, value: str) -> str:
        if not value.strip() or len(value.encode()) > 2000:
            raise ValueError('review reason required')
        return value


class _Decision(_Reason):
    decision: Literal['approve', 'deny']
    expected_revision: int = Field(ge=1, le=1)


class _Revocation(_Reason):
    expected_revision: int = Field(ge=2, le=2)


def _response(value: object, status: int = 200) -> JSONResponse:
    return JSONResponse(value, status_code=status, headers={'Cache-Control': 'no-store'})


def _failure(code: str, status: int) -> JSONResponse:
    return _response({'error': code, 'code': code}, status)


def _present(record: ToolRecipeProposal) -> dict[str, object]:
    return {**record.model_dump(mode='json', exclude={'dependencies_json'}),
            'dependencies': record.dependencies(), 'revision': record.revision, 'status': record.status}


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate request field')
        result[key] = value
    return result


async def _body(request: Request) -> object:
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > 8192:
            raise OverflowError('request limit')
        chunks.extend(chunk)
    return json.loads(bytes(chunks), object_pairs_hook=_unique)


def mount_tool_recipe_routes(app: FastAPI, store: ToolRecipeStore,
    space_for: Callable[..., Awaitable[str]], assert_current_space: Callable[[Request, str], None],
    actor_for: Callable[[Request], str]) -> None:
    async def current(request: Request, space: str) -> None:
        await space_for(request)  # Recheck deleted space as well as current key/role.
        assert_current_space(request, space)

    @app.get('/v1/tool-recipes')
    async def proposals(request: Request, space: str = Depends(space_for)) -> JSONResponse:
        query = request.query_params
        if any(key not in ('limit', 'after') or len(query.getlist(key)) != 1 for key in query):
            return _failure('invalid_tool_recipe_query', 422)
        raw = query.get('limit', '50')
        if re.fullmatch(r'[1-9][0-9]{0,2}', raw) is None or int(raw) > 100:
            return _failure('invalid_tool_recipe_query', 422)
        try:
            page = store.list(space, limit=int(raw), after=query.get('after'))
        except ValueError:
            return _failure('invalid_tool_recipe_query', 422)
        except WorkflowError:
            return _failure('tool_recipe_unavailable', 503)
        await current(request, space)
        return _response({'space': space, 'items': [_present(item) for item in page.items], 'next_after': page.next_after})

    @app.get('/v1/tool-recipes/{proposal_id}')
    async def proposal(proposal_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            validate_proposal_id(proposal_id)
        except (ValueError, WorkflowError):
            return _failure('invalid_tool_recipe_identifier', 422)
        try:
            record = store.get(space, proposal_id)
        except (ValueError, WorkflowError):
            return _failure('tool_recipe_unavailable', 503)
        await current(request, space)
        return _failure('tool_recipe_not_found', 404) if record is None else _response(_present(record))

    async def review(proposal_id: str, request: Request, space: str, *, revoke: bool) -> JSONResponse:
        try:
            validate_proposal_id(proposal_id)
        except (ValueError, WorkflowError):
            return _failure('invalid_tool_recipe_identifier', 422)
        if request.query_params:
            return _failure('invalid_tool_recipe_review', 422)
        try:
            raw = await _body(request)
            body = _Revocation.model_validate(raw) if revoke else _Decision.model_validate(raw)
        except OverflowError:
            return _failure('tool_recipe_request_limit', 413)
        except (ValueError, RecursionError):
            return _failure('invalid_tool_recipe_review', 422)
        await current(request, space)
        actor = actor_for(request)
        try:
            if store.get(space, proposal_id) is None:
                return _failure('tool_recipe_not_found', 404)
            if isinstance(body, _Decision):
                record = store.decide(space, proposal_id, decision=body.decision, actor=actor,
                                      reason=body.reason, expected_revision=body.expected_revision,
                                      admission_guard=lambda: assert_current_space(request, space))
            else:
                record = store.revoke(space, proposal_id, actor=actor, reason=body.reason,
                                      expected_revision=body.expected_revision,
                                      admission_guard=lambda: assert_current_space(request, space))
        except ToolRecipeConflict:
            return _failure('tool_recipe_revision_conflict', 409)
        except (ValueError, WorkflowError):
            return _failure('tool_recipe_unavailable', 503)
        await current(request, space)
        return _response(_present(record))

    @app.post('/v1/tool-recipes/{proposal_id}/decision')
    async def decide(proposal_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        return await review(proposal_id, request, space, revoke=False)

    @app.post('/v1/tool-recipes/{proposal_id}/revoke')
    async def revoke(proposal_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        return await review(proposal_id, request, space, revoke=True)
