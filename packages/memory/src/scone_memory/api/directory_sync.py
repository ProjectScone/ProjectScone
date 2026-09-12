"""Authenticated controls for preconfigured local directory collections."""
from collections.abc import Awaitable, Callable
from dataclasses import asdict
import json
from typing import TypeVar

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..agents.catalog import Identifier
from ..agents.workflow import WorkflowError
from ..ingestion.directory_service import DirectorySyncService
from .responses import LedgerJSONResponse


class StartSync(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    run_id: Identifier
    collection_id: Identifier
    delete_missing: bool = False
    expected_configuration: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')


class ControlSync(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    expected_revision: int = Field(ge=0, le=2**31 - 1)


Payload = TypeVar('Payload', bound=BaseModel)


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in pairs:
        if key in values:
            raise ValueError('duplicate request key')
        values[key] = value
    return values


async def _decode(request: Request, model: type[Payload]) -> Payload:
    raw = bytearray()
    async for part in request.stream():
        if len(raw) + len(part) > 8192:
            raise WorkflowError('sync_request_limit')
        raw.extend(part)
    try:
        value: object = json.loads(raw, object_pairs_hook=_unique)
    except RecursionError:
        raise ValueError('nested request limit') from None
    return model.model_validate(value)


def _response(value: object, status: int = 200) -> JSONResponse:
    return LedgerJSONResponse(jsonable_encoder(value), status_code=status, headers={'Cache-Control': 'no-store'})


def _failure(error: WorkflowError | ValueError | OSError) -> JSONResponse:
    code = error.code if isinstance(error, WorkflowError) else (
        'sync_unavailable' if isinstance(error, OSError) else 'invalid_sync_request')
    statuses = {
        'sync_busy': 429, 'sync_owned_elsewhere': 409, 'sync_request_conflict': 409,
        'sync_configuration_changed': 409, 'sync_result_terminal': 409,
        'sync_result_unavailable': 409, 'sync_resume_required': 409,
        'sync_attempt_limit': 409, 'sync_cancelled': 409,
        'sync_not_found': 404, 'sync_collection_not_found': 404, 'space_deleted': 404,
        'sync_delete_forbidden': 403, 'sync_request_limit': 413, 'sync_store_limit': 429,
    }
    status = 422 if code.startswith('invalid_') else statuses.get(code, 503)
    response = _response({'error': code, 'code': code}, status)
    if status == 429:
        response.headers['Retry-After'] = '1'
    return response


def mount_directory_sync_routes(app: FastAPI, service: DirectorySyncService,
                                space_for: Callable[..., Awaitable[str]],
                                assert_current_space: Callable[[Request, str], None]) -> None:
    @app.get('/v1/sync-collections')
    async def collections(request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            values = await service.catalog(space)
            assert_current_space(request, space)
            return _response({'items': [asdict(value) for value in values]})
        except (WorkflowError, ValueError, OSError) as error:
            return _failure(error)

    @app.post('/v1/sync-runs')
    async def start(request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            body = await _decode(request, StartSync)
            assert_current_space(request, space)
            value = await service.start(space, body.run_id, collection_id=body.collection_id,
                delete_missing=body.delete_missing, expected_configuration=body.expected_configuration, admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response(asdict(value), 202)
        except (WorkflowError, ValueError, OSError) as error:
            return _failure(error)

    @app.get('/v1/sync-runs')
    async def history(request: Request, limit: int = 20, after: str | None = None,
                      space: str = Depends(space_for)) -> JSONResponse:
        try:
            value = await service.list(space, limit=limit, after=after)
            assert_current_space(request, space)
            return _response(asdict(value))
        except (WorkflowError, ValueError, OSError) as error:
            return _failure(error)

    @app.get('/v1/sync-runs/{run_id}')
    async def status(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            value = await service.status(space, run_id)
            assert_current_space(request, space)
            if value is None:
                raise WorkflowError('sync_not_found')
            return _response(asdict(value))
        except (WorkflowError, ValueError, OSError) as error:
            return _failure(error)

    @app.get('/v1/sync-runs/{run_id}/request')
    async def invocation(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            value = await service.request(space, run_id)
            assert_current_space(request, space)
            if value is None:
                raise WorkflowError('sync_not_found')
            return _response(value.model_dump(mode='json'))
        except (WorkflowError, ValueError, OSError) as error:
            return _failure(error)

    @app.get('/v1/sync-runs/{run_id}/result')
    async def result(run_id: str, request: Request, limit: int = 20, after: int | None = None,
                     space: str = Depends(space_for)) -> JSONResponse:
        try:
            value = await service.result(space, run_id, limit=limit, after=after)
            assert_current_space(request, space)
            return _response({'space': space, 'run_id': run_id, **asdict(value)})
        except (WorkflowError, ValueError, OSError) as error:
            return _failure(error)

    @app.post('/v1/sync-runs/{run_id}/resume')
    async def resume(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            body = await _decode(request, ControlSync)
            assert_current_space(request, space)
            value = await service.resume(space, run_id, expected_revision=body.expected_revision,
                admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response(asdict(value), 202)
        except (WorkflowError, ValueError, OSError) as error:
            return _failure(error)

    @app.post('/v1/sync-runs/{run_id}/cancel')
    async def cancel(run_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            body = await _decode(request, ControlSync)
            assert_current_space(request, space)
            value = await service.cancel(space, run_id, expected_revision=body.expected_revision,
                admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response(asdict(value), 202)
        except (WorkflowError, ValueError, OSError) as error:
            return _failure(error)
