"""Local background document execution over durable requests and stage journals."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
import fcntl
import hashlib
import hmac
import os
from pathlib import Path
import stat

from ..agents.workflow import WorkflowError, WorkflowStatus, _integer, _name, _private_file
from ..core.models import Added
from ..core.validation import check_space
from ..memory.engine import MemoryEngine
from .document_ocr import PdfOcrSelection
from .file_workflow import DocumentIngestionWorkflow
from .files import DocumentIngested, document_provenance, extraction_filename
from .formats.registry import DocumentParser
from .formats.types import DocumentLimits
from .import_store import DocumentImportRequest, DocumentImportSpec, DocumentImportStore, ImportConflict


@dataclass(frozen=True)
class ImportParserBinding:
    revision: str
    parser: DocumentParser

    def __post_init__(self) -> None:
        _name(self.revision)
        if not callable(getattr(self.parser, 'parse', None)):
            raise ValueError('document parser required')


@dataclass(frozen=True)
class DocumentImportStatus:
    import_id: str
    space: str
    filename: str
    attachment_id: str
    created_at: str
    attempt: int
    max_attempts: int
    revision: int
    status: str
    active_local: bool
    completed_steps: tuple[str, ...]
    inflight: str | None
    outcome_unknown: bool
    error_class: str | None


@dataclass(frozen=True)
class DocumentImportStatusPage:
    items: tuple[DocumentImportStatus, ...]
    next_after: str | None


class DocumentImportService:
    """Owns bounded local tasks; restart and receipt reads never replay them.

    The host owns the memory engine and authentication. Admission outlives its
    caller, while aclose cancels and joins owned work. Locks coordinate duplicate
    execution, not a distributed queue. Results require current source validation.
    """
    def __init__(self, directory: str | Path, *, key: bytes, memory: MemoryEngine,
                 parser_for: Callable[[PdfOcrSelection | None], ImportParserBinding],
                 limits: DocumentLimits = DocumentLimits(), max_active: int = 2,
                 max_imports: int = 4096, deadline_s: float = 120.0, max_attempts: int = 3) -> None:
        _integer(max_active, 1, 16)
        _integer(max_attempts, 1, 4)
        if type(key) is not bytes or len(key) != 32:
            raise WorkflowError('key_must_be_32_bytes')
        if type(deadline_s) not in (int, float) or not 0 < deadline_s <= 300:
            raise WorkflowError('invalid_budget')
        if not callable(parser_for):
            raise ValueError('host parser factory required')
        target = Path(directory).absolute()
        target.mkdir(mode=0o700, exist_ok=True)
        info = target.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise WorkflowError('private_directory_required')
        self._attempts = max_attempts
        self._limits = DocumentLimits.model_validate(limits.model_dump())
        self._directory, self._key, self._memory = target, key, memory
        self._parser_for, self._maximum, self._deadline = parser_for, max_active, deadline_s
        self._imports = DocumentImportStore(target / 'requests.sqlite', key=key, max_imports=max_imports)
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._workflows: dict[tuple[str, str], DocumentIngestionWorkflow] = {}
        self._owners: dict[tuple[str, str], int] = {}
        self._failures: dict[tuple[str, str], str] = {}
        self._closing = self._closed = False

    def require_host(self, memory: MemoryEngine) -> None:
        if memory is not self._memory:
            raise ValueError('document imports must use the host memory engine')

    def _available(self) -> None:
        if self._closed or self._closing:
            raise WorkflowError('import_service_closed')

    async def _space(self, space: str) -> None:
        self._available()
        check_space(space)
        if await self._memory.space_deleted(space) is not None:
            raise WorkflowError('space_deleted')
        self._available()

    def _path(self, request: DocumentImportRequest) -> Path:
        digest = hmac.new(self._key, b'document-execution:' + request.space.encode()
            + b'\0' + request.import_id.encode(), hashlib.sha256).hexdigest()
        return self._directory / (digest + '.sqlite')

    def _claim(self, request: DocumentImportRequest) -> int:
        descriptor = _private_file(Path(str(self._path(request)) + '.owner'))
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return descriptor
        except BlockingIOError:
            os.close(descriptor)
            raise WorkflowError('import_busy') from None
        except BaseException:
            os.close(descriptor)
            raise

    def _open(self, request: DocumentImportRequest) -> DocumentIngestionWorkflow:
        binding = self._parser_for(request.spec.pdf_ocr)
        if binding.revision != request.spec.parser_revision or request.spec.limits != self._limits:
            raise WorkflowError('import_parser_changed')
        return DocumentIngestionWorkflow(self._memory, self._path(request), key=self._key,
            parser=binding.parser, parser_revision=binding.revision, limits=request.spec.limits,
            deadline=request.spec.deadline_s, max_retries=request.spec.max_attempts - 1,
            automatic_retries=False)

    def _progress(self, request: DocumentImportRequest) -> WorkflowStatus | None:
        workflow = self._workflows.get((request.space, request.import_id))
        if workflow is not None:
            return workflow.status(request.import_id, space=request.space,
                attachment_id=request.spec.attachment_id, filename=request.spec.filename)
        if not self._path(request).exists():
            return None
        workflow = self._open(request)
        try:
            return workflow.status(request.import_id, space=request.space,
                attachment_id=request.spec.attachment_id, filename=request.spec.filename)
        finally:
            workflow.close()

    def _status(self, request: DocumentImportRequest) -> DocumentImportStatus:
        identity = (request.space, request.import_id)
        active = identity in self._tasks
        try:
            progress = self._progress(request)
        except WorkflowError as error:
            return DocumentImportStatus(request.import_id, request.space, request.spec.filename,
                request.spec.attachment_id, request.created_at.isoformat(), request.attempt, request.spec.max_attempts,
                request.revision, 'unavailable', active, (), None, bool(request.attempt), error.code)
        unknown = not active and bool(request.attempt) and not (progress and progress.status == 'completed')
        status = progress.status if progress else ('interrupted' if request.attempt else 'registered')
        if request.cancel_requested_at is not None and not active and status != 'completed':
            status = 'cancelled'
        elif status == 'running' and not active:
            status = 'interrupted'
        return DocumentImportStatus(request.import_id, request.space, request.spec.filename,
            request.spec.attachment_id, request.created_at.isoformat(), request.attempt, request.spec.max_attempts,
            request.revision, status, active, progress.completed_steps if progress else (),
            progress.inflight if progress else None, unknown,
            progress.error_class if progress else self._failures.get(identity))

    async def request(self, space: str, import_id: str) -> DocumentImportRequest | None:
        await self._space(space)
        return self._imports.get(space, import_id)

    async def status(self, space: str, import_id: str) -> DocumentImportStatus | None:
        request = await self.request(space, import_id)
        return self._status(request) if request else None

    async def list(self, space: str, *, limit: int = 20, after: str | None = None) -> DocumentImportStatusPage:
        await self._space(space)
        page = self._imports.list(space, limit=limit, after=after)
        return DocumentImportStatusPage(tuple(self._status(request) for request in page.items), page.next_after)

    async def start(self, space: str, import_id: str, *, attachment_id: str, filename: str,
                    pdf_ocr: PdfOcrSelection | None = None,
                    admission_guard: Callable[[], None] | None = None) -> DocumentImportStatus:
        await self._space(space)
        binding = self._parser_for(pdf_ocr)
        spec = DocumentImportSpec(attachment_id=attachment_id, filename=filename,
            pdf_ocr=pdf_ocr, parser_revision=binding.revision, limits=self._limits,
            deadline_s=float(self._deadline), max_attempts=self._attempts)
        prior = self._imports.get(space, import_id)
        if prior is not None:
            if prior.spec != spec:
                raise ImportConflict()
            return self._status(prior)
        original, _ = await self._memory.attachment(space, attachment_id)
        extraction_filename(original, filename)
        self._available()
        if len(self._tasks) >= self._maximum:
            raise WorkflowError('import_busy')
        if admission_guard is not None:
            admission_guard()
        request = self._imports.register(space, import_id, spec)
        # Registration may have lost a race to another process. Never reinterpret
        # that process's admitted request as a fresh execution.
        if request.attempt or request.cancel_requested_at is not None:
            return self._status(request)
        return self._admit(request, resume=False)

    def _admit(self, request: DocumentImportRequest, *, resume: bool) -> DocumentImportStatus:
        self._available()
        identity = (request.space, request.import_id)
        if identity in self._tasks or len(self._tasks) >= self._maximum:
            raise WorkflowError('import_busy')
        if request.attempt >= request.spec.max_attempts:
            raise WorkflowError('retries_exhausted')
        descriptor = self._claim(request)
        workflow = None
        try:
            workflow = self._open(request)
            current = self._imports.start_attempt(request.space, request.import_id,
                expected_revision=request.revision, resume=resume)
            self._workflows[identity] = workflow
            admission = self._status(current)
            self._owners[identity] = descriptor
            self._failures.pop(identity, None)
            task = asyncio.create_task(self._execute(current, workflow))
            self._tasks[identity] = task
            task.add_done_callback(lambda finished: self._finished(identity))
            return replace(admission, active_local=True, status='running', outcome_unknown=False)
        except BaseException:
            self._workflows.pop(identity, None)
            self._owners.pop(identity, None)
            if workflow is not None:
                workflow.close()
            os.close(descriptor)
            raise

    async def resume(self, space: str, import_id: str, *, expected_revision: int,
                     admission_guard: Callable[[], None] | None = None) -> DocumentImportStatus:
        _integer(expected_revision, 0, 2**31 - 1)
        request = await self.request(space, import_id)
        if request is None:
            raise WorkflowError('import_not_found')
        if request.revision != expected_revision:
            raise ImportConflict()
        if admission_guard is not None:
            admission_guard()
        current = self._status(request)
        if current.status == 'completed':
            await self.result(space, import_id)
            return current
        if current.status in {'sources_invalid', 'unavailable'}:
            raise WorkflowError(current.error_class or 'sources_invalid')
        return self._admit(request, resume=True)

    async def _execute(self, request: DocumentImportRequest, workflow: DocumentIngestionWorkflow) -> None:
        try:
            await workflow.run(request.import_id, space=request.space,
                attachment_id=request.spec.attachment_id, filename=request.spec.filename)
        except asyncio.CancelledError:
            pass
        except WorkflowError as error:
            self._failures[(request.space, request.import_id)] = error.code
        except Exception:
            self._failures[(request.space, request.import_id)] = 'import_execution_failed'

    def _finished(self, identity: tuple[str, str]) -> None:
        try:
            self._workflows[identity].close()
        except Exception:
            self._failures[identity] = 'import_close_failed'
        finally:
            self._workflows.pop(identity, None)
            self._tasks.pop(identity, None)
            descriptor = self._owners.pop(identity, None)
            if descriptor is not None:
                os.close(descriptor)

    async def wait(self, space: str, import_id: str) -> DocumentImportStatus | None:
        await self._space(space)
        task = self._tasks.get((space, import_id))
        if task is not None:
            await asyncio.shield(task)
        return await self.status(space, import_id)

    async def cancel(self, space: str, import_id: str, *, expected_revision: int,
                     admission_guard: Callable[[], None] | None = None) -> DocumentImportStatus | None:
        _integer(expected_revision, 0, 2**31 - 1)
        request = await self.request(space, import_id)
        if request is None:
            return None
        if request.revision != expected_revision:
            raise ImportConflict()
        if admission_guard is not None:
            admission_guard()
        task = self._tasks.get((space, import_id))
        descriptor = None
        if task is None:
            try:
                descriptor = self._claim(request)
            except WorkflowError as error:
                raise WorkflowError('import_not_owned') from error
        try:
            current = self._status(request)
            if current.status == 'completed':
                return current
            try:
                updated = self._imports.request_cancel(space, import_id, expected_revision=expected_revision)
            except ImportConflict:
                raise
            except Exception:
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                raise
            if updated is None:
                raise WorkflowError('import_not_found')
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            return self._status(updated)
        finally:
            if descriptor is not None:
                os.close(descriptor)

    async def result(self, space: str, import_id: str) -> DocumentIngested | None:
        request = await self.request(space, import_id)
        if request is None:
            return None
        if (space, import_id) in self._tasks:
            raise WorkflowError('not_completed')
        workflow = self._open(request)
        try:
            result = await workflow.read_result(import_id, space=space,
                attachment_id=request.spec.attachment_id, filename=request.spec.filename)
            if result is None:
                raise WorkflowError('not_completed')
            added = Added.model_validate(result.results['index'])
            evidence = await document_provenance(self._memory, space, added.episode_id)
            return DocumentIngested(added, evidence.original, evidence.manifest,
                evidence.format, len(evidence.segments), evidence.filename)
        finally:
            workflow.close()

    def close_idle(self) -> None:
        if self._tasks or self._owners:
            raise WorkflowError('import_busy')
        if not self._closed:
            self._closing = True
            self._imports.close()
            self._closed = True

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closing = True
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.close_idle()
