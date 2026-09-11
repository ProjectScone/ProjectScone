"""Durable extract/index stages over retained sources and the shared workflow journal."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ..agents.workflow import JSONValue, StepContext, WorkflowResult, WorkflowRunner, WorkflowStatus, WorkflowStep
from ..core.errors import InvalidInput
from ..core.validation import check_space
from .files import DocumentManifest, digest, document_provenance, encode_manifest, prepare_document, store_document
from .formats.registry import BuiltinDocumentParser, DocumentParser
from .formats.types import DocumentLimits, validate_document

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine


class DocumentIngestionWorkflow:
    """Caller-owned encrypted journal; one active document per journal.

    Sources must already be retained. Extraction outputs live in the configured
    blob store; the encrypted journal keeps only identities and stage receipts.
    A stable parser_revision must cover the parser, options and model revision.
    Cancellation interrupts the active call; retrying the run explicitly resumes.
    This runner is not a background queue or a distributed worker lease.
    """
    def __init__(self, memory: MemoryEngine, path: str | Path, *, key: bytes,
                 parser_revision: str, parser: DocumentParser | None = None,
                 limits: DocumentLimits = DocumentLimits(), deadline: float = 120,
                 max_retries: int = 1):
        self._memory = memory
        self._parser = parser or BuiltinDocumentParser()
        self._limits = limits
        self._runner = WorkflowRunner(path, key=key, source_verifier=self._verify,
            deadline=deadline, max_retries=max_retries, steps=(
                WorkflowStep('extract', parser_revision, self._extract, idempotent=True, retryable=True),
                WorkflowStep('index', 'document-v1', self._index, idempotent=True, retryable=True)))

    def close(self) -> None:
        self._runner.close()

    def _scope(self) -> dict[str, JSONValue]:
        return {'limits': cast(JSONValue, json.loads(self._limits.model_dump_json()))}

    def status(self, run_id: str, *, space: str, attachment_id: str) -> WorkflowStatus | None:
        return self._runner.status(run_id, space=space, scope=self._scope(), inputs=attachment_id)

    async def run(self, run_id: str, *, space: str, attachment_id: str) -> WorkflowResult:
        check_space(space)
        return await self._runner.run(run_id, space=space, scope=self._scope(), inputs=attachment_id)

    async def _verify(self, context: StepContext) -> bool:
        if not isinstance(context.inputs, str) or await self._memory.space_deleted(context.space):
            return False
        original, raw = await self._memory.attachment(context.space, context.inputs)
        if digest(raw) != context.inputs or not original.filename:
            return False
        if 'extract' in context.completed:
            manifest = await self._manifest(context)
            if manifest.original_sha256 != original.attachment_id or manifest.filename != original.filename:
                return False
        if 'index' in context.completed:
            result = cast(dict[str, JSONValue], context.completed['index'])
            episode_id = cast(int, result['episode_id'])
            evidence = await document_provenance(self._memory, context.space, episode_id)
            extraction = cast(dict[str, JSONValue], context.completed['extract'])
            if evidence.original.attachment_id != original.attachment_id or evidence.manifest.attachment_id != extraction['manifest_id']:
                return False
        return True

    async def _extract(self, context: StepContext) -> JSONValue:
        original, raw = await self._memory.attachment(context.space, cast(str, context.inputs))
        if not original.filename:
            raise InvalidInput('document attachment must retain a filename')
        manifest = await prepare_document(raw, original.filename, parser=self._parser, limits=self._limits)
        retained = await self._memory.attach(context.space, encode_manifest(manifest),
                                              'application/json', filename='document-provenance.json')
        if retained.media_type != 'application/json':
            raise InvalidInput('document manifest has an incompatible media type')
        return {'manifest_id': retained.attachment_id, 'segments': len(manifest.parsed.segments)}

    async def _manifest(self, context: StepContext) -> DocumentManifest:
        extraction = cast(dict[str, JSONValue], context.completed['extract'])
        identifier = cast(str, extraction['manifest_id'])
        attachment, raw = await self._memory.attachment(context.space, identifier)
        if digest(raw) != identifier or attachment.media_type != 'application/json':
            raise InvalidInput('document extraction checkpoint is invalid')
        manifest = DocumentManifest.model_validate_json(raw)
        validate_document(manifest.parsed, self._limits)
        return manifest

    async def _index(self, context: StepContext) -> JSONValue:
        manifest = await self._manifest(context)
        original, _ = await self._memory.attachment(context.space, manifest.original_sha256)
        result = await store_document(self._memory, context.space, original, manifest)
        return cast(JSONValue, json.loads(result.added.model_dump_json()))
