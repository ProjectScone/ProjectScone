"""Encrypted, immutable document import requests and versioned control intent."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
from pathlib import Path
import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..agents._encrypted_store import EncryptedRecordStore
from ..agents.catalog import Identifier
from ..agents.workflow import WorkflowError, _integer, _name
from ..core.validation import check_space
from .document_ocr import PdfOcrSelection
from .formats.registry import extension
from .formats.types import DocumentLimits


class ImportConflict(WorkflowError):
    def __init__(self) -> None:
        super().__init__('import_request_conflict')


class DocumentImportSpec(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    attachment_id: str = Field(pattern=r'^[a-f0-9]{64}$')
    filename: str = Field(min_length=1, max_length=1024)
    parser_revision: Identifier
    limits: DocumentLimits = Field(default_factory=DocumentLimits)
    pdf_ocr: PdfOcrSelection | None = None
    deadline_s: float = Field(default=120.0, gt=0, le=300, allow_inf_nan=False)
    max_attempts: int = Field(default=3, ge=1, le=4)

    @model_validator(mode='after')
    def valid(self) -> Self:
        if not self.filename.strip() or '\0' in self.filename:
            raise ValueError('invalid extraction filename')
        self.filename.encode('utf-8')
        if self.pdf_ocr is not None and extension(self.filename) != '.pdf':
            raise ValueError('PDF OCR requires a PDF extraction filename')
        extension(self.filename)
        return self


class DocumentImportRequest(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    space: str
    import_id: Identifier
    spec: DocumentImportSpec
    created_at: datetime
    revision: int = Field(default=0, ge=0, le=2**31 - 1)
    attempt: int = Field(default=0, ge=0, le=2**31 - 1)
    last_started_at: datetime | None = None
    cancel_requested_at: datetime | None = None

    @model_validator(mode='after')
    def valid(self) -> Self:
        check_space(self.space)
        if self.created_at.tzinfo is None or self.revision < self.attempt:
            raise ValueError('invalid import control state')
        if (self.attempt == 0) != (self.last_started_at is None):
            raise ValueError('import attempt has no matching start timestamp')
        for stamp in (self.last_started_at, self.cancel_requested_at):
            if stamp is not None and (stamp.tzinfo is None or stamp < self.created_at):
                raise ValueError('invalid import control timestamp')
        return self


@dataclass(frozen=True)
class DocumentImportPage:
    items: tuple[DocumentImportRequest, ...]
    next_after: str | None


class DocumentImportStore:
    """Registration never starts work. Control revisions protect concurrent intent.

    The execution owner must separately hold the job's lock before recording an
    attempt. This registry is not an execution lease or a completion receipt.
    """
    def __init__(self, path: str | Path, *, key: bytes, max_imports: int = 4096) -> None:
        _integer(max_imports, 1, 100000)
        self._key, self._maximum = key, max_imports
        self._storage = EncryptedRecordStore(path, key=key, table='document_imports',
            metadata='document_import_meta', application_id=0x5343494D,
            domain='scone-document-imports-v1', label='import')

    def _prefix(self, space: str) -> str:
        check_space(space)
        return hmac.new(self._key, b'import-space:' + space.encode(), hashlib.sha256).hexdigest() + ':'

    def _token(self, space: str, import_id: str) -> str:
        _name(import_id)
        return self._prefix(space) + hmac.new(self._key, b'import-id:' + import_id.encode(), hashlib.sha256).hexdigest()

    def _decode(self, token: str, payload: object, space: str) -> DocumentImportRequest:
        try:
            request = DocumentImportRequest.model_validate_json(self._storage._unseal(token, payload))
        except ValueError:
            raise WorkflowError('import_key_or_integrity') from None
        if request.space != space or self._token(request.space, request.import_id) != token:
            raise WorkflowError('import_key_or_integrity')
        return request

    def get(self, space: str, import_id: str) -> DocumentImportRequest | None:
        token = self._token(space, import_id)
        with self._storage._access() as db:
            row = db.execute('SELECT payload FROM document_imports WHERE token=?', (token,)).fetchone()
            return self._decode(token, row[0], space) if row else None

    def register(self, space: str, import_id: str, spec: DocumentImportSpec) -> DocumentImportRequest:
        token = self._token(space, import_id)
        saved = DocumentImportRequest(space=space, import_id=import_id,
            spec=DocumentImportSpec.model_validate(spec.model_dump()), created_at=datetime.now(timezone.utc))
        payload = self._storage._seal(token, saved.model_dump_json().encode())
        with self._storage._access(write=True) as db:
            row = db.execute('SELECT payload FROM document_imports WHERE token=?', (token,)).fetchone()
            if row:
                previous = self._decode(token, row[0], space)
                if previous.spec != saved.spec:
                    raise ImportConflict()
                return previous
            if db.execute('SELECT COUNT(*) FROM document_imports').fetchone()[0] >= self._maximum:
                raise WorkflowError('import_store_limit')
            db.execute('INSERT INTO document_imports VALUES (?, ?)', (token, payload))
        return saved

    def start_attempt(self, space: str, import_id: str, *, expected_revision: int,
                      resume: bool = False) -> DocumentImportRequest:
        _integer(expected_revision, 0, 2**31 - 1)
        if type(resume) is not bool:
            raise WorkflowError('invalid_resume_flag')
        token = self._token(space, import_id)
        with self._storage._access(write=True) as db:
            row = db.execute('SELECT payload FROM document_imports WHERE token=?', (token,)).fetchone()
            if row is None:
                raise WorkflowError('import_not_found')
            request = self._decode(token, row[0], space)
            if request.revision != expected_revision:
                raise ImportConflict()
            if request.cancel_requested_at is not None and not resume:
                raise WorkflowError('import_cancelled')
            if request.attempt and not resume:
                raise WorkflowError('import_resume_required')
            updated = DocumentImportRequest.model_validate({**request.model_dump(),
                'revision': request.revision + 1, 'attempt': request.attempt + 1,
                'last_started_at': datetime.now(timezone.utc), 'cancel_requested_at': None})
            db.execute('UPDATE document_imports SET payload=? WHERE token=?',
                (self._storage._seal(token, updated.model_dump_json().encode()), token))
        return updated

    def request_cancel(self, space: str, import_id: str, *,
                       expected_revision: int | None = None) -> DocumentImportRequest | None:
        if expected_revision is not None:
            _integer(expected_revision, 0, 2**31 - 1)
        token = self._token(space, import_id)
        with self._storage._access(write=True) as db:
            row = db.execute('SELECT payload FROM document_imports WHERE token=?', (token,)).fetchone()
            if row is None:
                return None
            request = self._decode(token, row[0], space)
            if expected_revision is not None and request.revision != expected_revision:
                raise ImportConflict()
            if request.cancel_requested_at is not None:
                return request
            updated = DocumentImportRequest.model_validate({**request.model_dump(),
                'revision': request.revision + 1, 'cancel_requested_at': datetime.now(timezone.utc)})
            db.execute('UPDATE document_imports SET payload=? WHERE token=?',
                (self._storage._seal(token, updated.model_dump_json().encode()), token))
        return updated

    def list(self, space: str, *, limit: int = 50, after: str | None = None) -> DocumentImportPage:
        _integer(limit, 1, 100)
        prefix = self._prefix(space)
        if after is not None and (not isinstance(after, str) or not after.startswith(prefix)
                                  or re.fullmatch(r'[0-9a-f]{64}', after[len(prefix):]) is None):
            raise WorkflowError('invalid_import_cursor')
        with self._storage._access() as db:
            rows = db.execute('SELECT token,payload FROM document_imports WHERE token>? AND token<? ORDER BY token LIMIT ?',
                (after or prefix, prefix + '~', limit + 1)).fetchall()
            items = tuple(self._decode(token, payload, space) for token, payload in rows[:limit])
            return DocumentImportPage(items, rows[limit - 1][0] if len(rows) > limit else None)

    def close(self) -> None:
        self._storage.close()
