"""Encrypted page checkpoints for retained PDFs, followed by recoverable indexing."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING, cast

from ..agents.workflow import JSONValue, StepContext, WorkflowError, WorkflowRunner, WorkflowStatus, WorkflowStep
from ..core.errors import InvalidInput, NotFound
from ..core.models import Added, Attachment
from ..core.validation import check_space
from ..ocr.types import OcrEngine, OcrResult
from .documents import PdfIngested, ingest_pdf, pdf_provenance
from .pdf import ParsedPdf, PdfLimits
from .pdf_ocr import OcrPdfOptions, OcrPdfParser, assemble_ocr_pdf

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine


@dataclass(frozen=True)
class PdfOcrIngested(PdfIngested):
    reused_pages: tuple[int, ...]
    reused_index: bool


@dataclass(frozen=True)
class _PreparedParser:
    original: str
    parsed: ParsedPdf

    async def parse(self, data: bytes, limits: PdfLimits) -> ParsedPdf:
        if hashlib.sha256(data).hexdigest() != self.original:
            raise InvalidInput('prepared PDF does not match its original')
        return self.parsed


def _dependency_revision(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return 'unavailable'


def _run_key(run_id: str, stage: str) -> str:
    if not isinstance(run_id, str) or not 1 <= len(run_id) <= 128:
        raise WorkflowError('invalid_identifier')
    try:
        identifier = hashlib.sha256(run_id.encode('utf-8')).hexdigest()
    except UnicodeError:
        raise WorkflowError('invalid_identifier') from None
    return f'pdf-ocr:{identifier}:{stage}'


class PdfOcrWorkflow:
    """Caller-owned page and index execution, with no whole-document deadline.

    Each page gets ``limits.timeout_seconds`` and its own encrypted receipt.
    The caller supplies a stable key and recognizer revision, and retains the
    source before starting. A retry revalidates the source before reusing work.
    This is not a background queue; cancel the active coroutine to interrupt it.
    """
    def __init__(self, memory: MemoryEngine, path: str | Path, *, key: bytes,
                 engine: OcrEngine, recognizer_revision: str,
                 options: OcrPdfOptions = OcrPdfOptions(), limits: PdfLimits = PdfLimits(),
                 index_timeout_seconds: float = 120.0, max_retries: int = 1,
                 max_checkpoint_bytes: int = 1_000_000):
        self._memory = memory
        self._parser = OcrPdfParser(engine, options=options)
        self._limits = limits
        self._prepared: _PreparedParser | None = None
        self._running = False
        self._closed = False
        self._scope: dict[str, JSONValue] = {
            'format': 'pdf-ocr-workflow-v1', 'recognizer_revision': recognizer_revision,
            'text_strategy': 'layout-text-fallback-v1',
            'options': cast(JSONValue, json.loads(options.model_dump_json())),
            'limits': cast(JSONValue, json.loads(limits.model_dump_json())),
            'pypdf': _dependency_revision('pypdf'), 'renderer': _dependency_revision('pypdfium2'),
        }
        self._pages = WorkflowRunner(path, key=key,
            steps=(WorkflowStep('ocr', recognizer_revision, self._recognize, idempotent=True, retryable=True),),
            source_verifier=self._verify_page, deadline=limits.timeout_seconds,
            max_retries=max_retries, max_payload_bytes=max_checkpoint_bytes)
        try:
            self._index = WorkflowRunner(path, key=key,
                steps=(WorkflowStep('index', 'pdf-v1', self._store, idempotent=True, retryable=True),),
                source_verifier=self._verify_index, deadline=index_timeout_seconds,
                max_retries=max_retries)
            try:
                self._binding = WorkflowRunner(path, key=key,
                    steps=(WorkflowStep('source', 'pdf-source-v1', self._bind_source, idempotent=True),),
                    source_verifier=self._verify_binding, deadline=limits.timeout_seconds, max_retries=0)
            except BaseException:
                self._index.close()
                raise
        except BaseException:
            self._pages.close()
            raise

    def close(self) -> None:
        if self._running:
            raise WorkflowError('busy')
        self._pages.close()
        self._index.close()
        self._binding.close()
        self._closed = True

    def page_status(self, run_id: str, *, space: str, attachment_id: str, page: int) -> WorkflowStatus | None:
        if type(page) is not int or not 1 <= page <= self._limits.max_pages:
            raise InvalidInput('PDF checkpoint page is outside its page limit')
        return self._pages.status(_run_key(run_id, f'page:{page}'), space=space,
            scope=self._scope, inputs={'original': attachment_id, 'page': page})

    async def _original(self, space: str, identifier: str) -> tuple[Attachment, bytes]:
        if await self._memory.space_deleted(space):
            raise WorkflowError('sources_invalid')
        try:
            original, raw = await self._memory.attachment(space, identifier)
        except (OSError, sqlite3.OperationalError):
            raise
        except Exception:
            raise WorkflowError('sources_invalid') from None
        if (original.media_type != 'application/pdf' or hashlib.sha256(raw).hexdigest() != identifier
                or not raw.startswith(b'%PDF-')):
            raise WorkflowError('sources_invalid')
        return original, raw

    async def _verify_page(self, context: StepContext) -> bool:
        inputs = cast(dict[str, JSONValue], context.inputs)
        await self._original(context.space, cast(str, inputs['original']))
        if 'ocr' in context.completed:
            self._result(context.completed['ocr'])
        return True

    async def _verify_binding(self, context: StepContext) -> bool:
        await self._original(context.space, cast(str, context.inputs))
        return True

    async def _bind_source(self, context: StepContext) -> JSONValue:
        return None

    def _result(self, value: JSONValue) -> OcrResult:
        result = OcrResult.model_validate_json(cast(str, value))
        if (result.width * result.height > self._parser.options.max_pixels
                or len(result.regions) > self._parser.options.max_regions):
            raise InvalidInput('OCR checkpoint exceeds its recognition limits')
        return result

    async def _recognize(self, context: StepContext) -> JSONValue:
        inputs = cast(dict[str, JSONValue], context.inputs)
        _, raw = await self._original(context.space, cast(str, inputs['original']))
        result = await self._parser.recognize_page(raw, cast(int, inputs['page']),
                                                  timeout_seconds=self._limits.timeout_seconds)
        return result.model_dump_json()

    async def _verify_index(self, context: StepContext) -> bool:
        inputs = cast(dict[str, JSONValue], context.inputs)
        identifier = cast(str, inputs['original'])
        await self._original(context.space, identifier)
        if 'index' in context.completed:
            receipt = cast(dict[str, JSONValue], context.completed['index'])
            added = Added.model_validate(receipt['added'])
            evidence = await pdf_provenance(self._memory, context.space, added.episode_id)
            if evidence.original.attachment_id != identifier or evidence.manifest.attachment_id != receipt['manifest']:
                return False
        return True

    async def _store(self, context: StepContext) -> JSONValue:
        inputs = cast(dict[str, JSONValue], context.inputs)
        original, raw = await self._original(context.space, cast(str, inputs['original']))
        prepared = self._prepared
        if prepared is None or hashlib.sha256(prepared.parsed.model_dump_json().encode()).hexdigest() != inputs['parsed']:
            raise InvalidInput('prepared PDF does not match its indexing checkpoint')
        result = await ingest_pdf(self._memory, context.space, raw, filename=original.filename,
                                 limits=self._limits, parser=prepared, embedding_checkpoint=context.checkpoints)
        return {'added': cast(JSONValue, result.added.model_dump(mode='json')), 'manifest': result.manifest.attachment_id}

    async def run(self, run_id: str, *, space: str, attachment_id: str) -> PdfOcrIngested:
        check_space(space)
        index_key = _run_key(run_id, 'index')
        if self._closed:
            raise WorkflowError('closed')
        if self._running:
            raise WorkflowError('busy')
        self._running = True
        try:
            await self._binding.run(_run_key(run_id, 'source'), space=space,
                                    scope=self._scope, inputs=attachment_id)
            original, raw = await self._original(space, attachment_id)
            parsed = await self._parser.inspect(raw, self._limits)
            recognized: dict[int, OcrResult] = {}
            reused: list[int] = []
            text_bytes = len(parsed.text.encode())
            for page in parsed.pages:
                if not page.empty and self._parser.options.mode != 'all_pages':
                    continue
                receipt = await self._pages.run(_run_key(run_id, f'page:{page.number}'), space=space,
                    scope=self._scope, inputs={'original': attachment_id, 'page': page.number})
                result = self._result(receipt.results['ocr'])
                text_bytes += sum(len(r.text.encode()) for r in result.regions) + max(0, len(result.regions) - 1) - (page.end - page.start)
                if text_bytes > self._limits.max_text_bytes:
                    raise InvalidInput('PDF OCR text exceeds its byte limit')
                recognized[page.number] = result
                if receipt.reused_steps:
                    reused.append(page.number)
            complete = assemble_ocr_pdf(parsed, recognized, self._limits)
            self._prepared = _PreparedParser(attachment_id, complete)
            receipt = await self._index.run(index_key, space=space, scope=self._scope,
                inputs={'original': attachment_id, 'parsed': hashlib.sha256(complete.model_dump_json().encode()).hexdigest()})
            saved = cast(dict[str, JSONValue], receipt.results['index'])
            added = Added.model_validate(saved['added'])
            manifest, _ = await self._memory.attachment(space, cast(str, saved['manifest']))
            return PdfOcrIngested(added, original, manifest, tuple(p.number for p in complete.pages if p.empty),
                                  tuple(reused), bool(receipt.reused_steps))
        except (FileNotFoundError, NotFound):
            raise WorkflowError('sources_invalid') from None
        except (OSError, sqlite3.OperationalError):
            raise WorkflowError('verification_unavailable') from None
        finally:
            self._prepared = None
            self._running = False
