"""Bounded document admission and explicit controls over retained imports."""
from collections.abc import Awaitable, Callable
from dataclasses import asdict

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..agents.catalog import Identifier
from ..agents.workflow import WorkflowError
from ..ingestion.document_ocr import PdfOcrSelection
from ..ingestion.import_service import DocumentImportService


class StartDocumentImport(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    import_id: Identifier
    attachment_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    filename: str = Field(min_length=1, max_length=1024)
    pdf_ocr: PdfOcrSelection | None = None


class ControlDocumentImport(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', hide_input_in_errors=True)
    expected_revision: int = Field(ge=0, le=2**31 - 1)


async def _body(request: Request) -> bytes:
    content = bytearray()
    async for part in request.stream():
        if len(content) + len(part) > 8192:
            raise WorkflowError('import_request_limit')
        content.extend(part)
    return bytes(content)


def _response(value: object, status: int = 200) -> JSONResponse:
    return JSONResponse(jsonable_encoder(value), status_code=status, headers={'Cache-Control': 'no-store'})


def _failure(error: WorkflowError | ValueError) -> JSONResponse:
    code = error.code if isinstance(error, WorkflowError) else 'invalid_document_import'
    status = 503
    if code in {'import_busy', 'busy'}:
        status = 429
    elif code in {'import_request_conflict', 'import_parser_changed', 'import_not_owned',
                  'sources_invalid', 'not_completed', 'binding_mismatch', 'retries_exhausted',
                  'import_cancelled', 'outcome_unknown'}:
        status = 409
    elif code in {'import_not_found', 'space_deleted'}:
        status = 404
    elif code == 'import_request_limit':
        status = 413
    elif code.startswith('invalid_') or code == 'import_store_limit':
        status = 422
    response = _response({'error': code, 'code': code}, status)
    if status == 429:
        response.headers['Retry-After'] = '1'
    return response


def mount_document_job_routes(app: FastAPI, service: DocumentImportService,
                              space_for: Callable[..., Awaitable[str]],
                              assert_current_space: Callable[[Request, str], None]) -> None:
    @app.post('/v1/document-jobs')
    async def start(request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            body = StartDocumentImport.model_validate_json(await _body(request))
            assert_current_space(request, space)
            result = await service.start(space, body.import_id, attachment_id=body.attachment_id,
                filename=body.filename, pdf_ocr=body.pdf_ocr,
                admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response(asdict(result), 202)
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.get('/v1/document-jobs')
    async def history(request: Request, limit: int = 20, after: str | None = None,
                      space: str = Depends(space_for)) -> JSONResponse:
        try:
            result = await service.list(space, limit=limit, after=after)
            assert_current_space(request, space)
            return _response(asdict(result))
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.get('/v1/document-jobs/{import_id}')
    async def status(import_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            result = await service.status(space, import_id)
            assert_current_space(request, space)
            return _response(asdict(result)) if result else _response({'error': 'import_not_found'}, 404)
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.get('/v1/document-jobs/{import_id}/request')
    async def invocation(import_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            result = await service.request(space, import_id)
            assert_current_space(request, space)
            return _response(result.model_dump(mode='json')) if result else _response({'error': 'import_not_found'}, 404)
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.get('/v1/document-jobs/{import_id}/result')
    async def result(import_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            saved = await service.request(space, import_id)
            result = await service.result(space, import_id) if saved else None
            assert_current_space(request, space)
            if result is None or saved is None:
                return _response({'error': 'import_not_found'}, 404)
            return _response({'space': space, 'import_id': import_id, **asdict(result),
                **({'pdf_ocr': saved.spec.pdf_ocr.model_dump()} if saved.spec.pdf_ocr else {})})
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.post('/v1/document-jobs/{import_id}/resume')
    async def resume(import_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            body = ControlDocumentImport.model_validate_json(await _body(request))
            assert_current_space(request, space)
            result = await service.resume(space, import_id, expected_revision=body.expected_revision,
                admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response(asdict(result), 202)
        except (WorkflowError, ValueError) as error:
            return _failure(error)

    @app.post('/v1/document-jobs/{import_id}/cancel')
    async def cancel(import_id: str, request: Request, space: str = Depends(space_for)) -> JSONResponse:
        try:
            body = ControlDocumentImport.model_validate_json(await _body(request))
            assert_current_space(request, space)
            result = await service.cancel(space, import_id, expected_revision=body.expected_revision,
                admission_guard=lambda: assert_current_space(request, space))
            assert_current_space(request, space)
            return _response(asdict(result)) if result else _response({'error': 'import_not_found'}, 404)
        except (WorkflowError, ValueError) as error:
            return _failure(error)
