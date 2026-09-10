"""Host-authorized configuration routes and bounded self-hosted model discovery."""

import asyncio
from collections.abc import Callable
import json

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel, ConfigDict, Field

from ..runtime.model_connections import (
    ModelConnection, ModelConnectionConflict, ModelConnectionError, ModelConnectionStore, api_key,
)


class _Replace(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    expected_revision: int = Field(ge=0, le=2**63 - 3)
    connection: ModelConnection | None


class _Probe(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    connection: ModelConnection


async def probe_model_connection(connection: ModelConnection, *,
                                 transport: httpx.AsyncBaseTransport | None = None) -> dict[str, object]:
    """Read GET /models only; this does not test inference or download a model."""
    connection = ModelConnection.model_validate(connection)
    token = api_key(connection)
    headers = {'Accept': 'application/json', 'Accept-Encoding': 'identity'}
    if token is not None:
        headers['Authorization'] = 'Bearer ' + token
    timeout = min(connection.timeout_s, 10)
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(timeout=timeout, transport=transport,
                                         trust_env=False, follow_redirects=False) as client:
                async with client.stream('GET', connection.base_url + 'models', headers=headers) as response:
                    if response.status_code != 200:
                        raise ModelConnectionError('Self-hosted model discovery request failed')
                    if response.headers.get('content-encoding', 'identity').lower() != 'identity':
                        raise ModelConnectionError('Self-hosted model discovery returned an unsupported response')
                    body = bytearray()
                    async for part in response.aiter_bytes():
                        if len(body) + len(part) > 256000:
                            raise ModelConnectionError('Self-hosted model discovery response exceeds its size limit')
                        body.extend(part)
        payload = json.loads(body)
    except (httpx.HTTPError, TimeoutError, ValueError):
        raise ModelConnectionError('Self-hosted model discovery is unavailable or returned an invalid response') from None
    data = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(data, list) or len(data) > 2048:
        raise ModelConnectionError('Self-hosted model discovery returned an invalid model list')
    models: list[str] = []
    for item in data:
        model = item.get('id') if isinstance(item, dict) else None
        if not isinstance(model, str) or not model.strip() or len(model) > 160 or any(ord(char) < 32 for char in model):
            raise ModelConnectionError('Self-hosted model discovery returned an invalid model list')
        if model not in models:
            models.append(model)
    return {'models': models, 'model_available': connection.model in models}


async def _body(request: Request) -> object:
    content = bytearray()
    async for part in request.stream():
        if len(content) + len(part) > 16000:
            raise HTTPException(400, 'Model connection request exceeds its size limit')
        content.extend(part)
    try:
        return json.loads(content)
    except ValueError:
        raise HTTPException(400, 'Invalid model connection request') from None


def mount_model_connection_routes(app: FastAPI, store: ModelConnectionStore,
                                  authorize: Callable[..., object], *,
                                  on_change: Callable[[], None] | None = None) -> None:
    """Every endpoint uses the host's explicit administration dependency."""
    if not callable(authorize):
        raise ValueError('A host administration dependency is required')
    if on_change is not None and not callable(on_change):
        raise ValueError('on_change must be callable')
    dependencies = [Depends(authorize)]

    @app.get('/v1/model-connections', dependencies=dependencies)
    async def snapshot():
        try:
            return store.snapshot()
        except ModelConnectionError:
            return JSONResponse({'error': 'Self-hosted model settings are unavailable'}, status_code=503)

    @app.put('/v1/model-connections/{role}', dependencies=dependencies)
    async def replace(role: str, request: Request):
        try:
            body = _Replace.model_validate(await _body(request))
            saved = store.replace(role, body.connection, expected_revision=body.expected_revision)
        except ModelConnectionConflict:
            return JSONResponse({'error': 'Model settings changed; reload the saved revision before saving'}, status_code=409)
        except ValueError:
            return JSONResponse({'error': 'Invalid model connection; use a self-hosted endpoint and valid model settings (speech requires voice)'}, status_code=400)
        except ModelConnectionError:
            return JSONResponse({'error': 'Self-hosted model settings could not be saved'}, status_code=503)
        except HTTPException as error:
            return JSONResponse({'error': error.detail}, status_code=error.status_code)
        if on_change is not None:
            try:
                on_change()
            except Exception:
                return JSONResponse({'error': 'Model settings were saved, but runtime refresh failed',
                                     'saved': saved}, status_code=503)
        return saved

    @app.post('/v1/model-connections/probe', dependencies=dependencies)
    async def probe(request: Request):
        try:
            body = _Probe.model_validate(await _body(request))
            return await probe_model_connection(body.connection)
        except ValueError:
            return JSONResponse({'error': 'Invalid model connection; use a self-hosted endpoint and valid model settings'}, status_code=400)
        except ModelConnectionError:
            return JSONResponse({'error': 'Self-hosted model discovery is unavailable; check the endpoint and server token configuration'}, status_code=502)
        except HTTPException as error:
            return JSONResponse({'error': error.detail}, status_code=error.status_code)
