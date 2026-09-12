"""Explicit local import storage and host-owned parser revisions."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import stat
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..agents.catalog import Identifier
from ..core.errors import InvalidInput
from ..ingestion.document_media import DocumentMedia
from ..ingestion.document_ocr import DocumentOcr, PdfOcrSelection
from ..ingestion.formats.registry import BuiltinDocumentParser
from ..ingestion.formats.types import DocumentLimits
from ..ingestion.import_service import DocumentImportService, ImportParserBinding
from ..memory.engine import MemoryEngine


class DocumentJobsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    schema_version: Literal[1]
    state_dir: str = Field(min_length=1, max_length=4096)
    key_env: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z_][A-Za-z0-9_]*$')
    parser_revision: Identifier
    max_active: int = Field(default=2, ge=1, le=16)
    max_imports: int = Field(default=4096, ge=1, le=100000)
    max_attempts: int = Field(default=3, ge=1, le=4)
    deadline_s: float = Field(default=120.0, gt=0, le=300, allow_inf_nan=False)
    limits: DocumentLimits = Field(default_factory=DocumentLimits)

    @field_validator('schema_version', mode='before')
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError('schema version must be an integer')
        return value

    @field_validator('state_dir')
    @classmethod
    def directory(cls, value: str) -> str:
        if not value.strip() or '\0' in value:
            raise ValueError('state directory required')
        return value

    @classmethod
    def read(cls, path: Path) -> DocumentJobsConfig:
        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError('duplicate configuration key')
                result[name] = value
            return result
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, 'rb') as source:
                info = os.fstat(source.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                    raise ValueError('private configuration required')
                raw = source.read(8193)
            if len(raw) > 8192:
                raise ValueError('configuration byte limit')
            json.loads(raw, object_pairs_hook=unique)
            return cls.model_validate_json(raw)
        except (OSError, ValueError, RecursionError):
            raise ValueError('Document job configuration requires bounded JSON in an owned 0600 regular file') from None


def load_document_imports(path: str, memory: MemoryEngine, *, document_ocr: DocumentOcr | None = None,
                          ocr_identity: str = '', document_media: DocumentMedia | None = None) -> DocumentImportService:
    """Open private state without admitting jobs or calling a parser.

    The operator revision must change when native OCR executables, trained data
    or custom parser behavior change. Installed Python package versions and
    each request's OCR choices are additionally bound into the journal revision.
    """
    config_path = Path(path).expanduser().absolute()
    config = DocumentJobsConfig.read(config_path)
    secret = os.environ.get(config.key_env, '')
    if re.fullmatch(r'[a-fA-F0-9]{64}', secret) is None:
        raise ValueError('Document job encryption key environment variable must contain 64 hex characters')
    target = Path(config.state_dir).expanduser()
    if not target.is_absolute():
        target = config_path.parent / target
    target.mkdir(mode=0o700, exist_ok=True)
    info = target.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Document job state directory must be owned and private')
    dependencies = {}
    for package in ('scone-memory', 'pypdf', 'pypdfium2', 'python-docx', 'openpyxl', 'python-pptx', 'beautifulsoup4'):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = 'uninstalled'

    def parser_for(selection: PdfOcrSelection | None) -> ImportParserBinding:
        fingerprint = {'implementation': 'document-import-v1', 'operator': config.parser_revision,
            'dependencies': dependencies, 'ocr': selection.model_dump() if selection else None,
            'ocr_identity': ocr_identity if selection else None,
            'dpi': document_ocr.dpi if selection and document_ocr else None}
        if document_media is not None and selection is None:
            fingerprint['media_revision'] = document_media.revision
        revision = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        if selection is None:
            return ImportParserBinding(revision, document_media.parser() if document_media else BuiltinDocumentParser())
        if document_ocr is None:
            raise InvalidInput('document OCR is not configured on this server')
        return ImportParserBinding(revision, document_ocr.parser(selection))

    return DocumentImportService(target, key=bytes.fromhex(secret), memory=memory, parser_for=parser_for,
        limits=config.limits, max_active=config.max_active, max_imports=config.max_imports,
        deadline_s=config.deadline_s, max_attempts=config.max_attempts)
