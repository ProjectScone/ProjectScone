"""Private operator configuration for passive local directory-sync hosts."""
from __future__ import annotations

import hashlib
import hmac
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import stat
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..agents.catalog import Identifier
from ..core.validation import check_space
from ..core.errors import InvalidInput
from ..ingestion.directory_service import DirectoryCollection, DirectorySyncService
from ..ingestion.directory_sync import DirectorySync
from ..ingestion.document_media import DocumentMedia, MEDIA_DOCUMENT_EXTENSIONS
from ..ingestion.document_ocr import DocumentOcr, PdfOcrSelection
from ..ingestion.formats.registry import BuiltinDocumentParser, DocumentParser
from ..ingestion.formats.types import DocumentLimits
from ..ingestion.import_service import ImportParserBinding
from ..ingestion.source_scan import DirectoryScanner, ScanLimits, default_extensions
from ..memory.engine import MemoryEngine


class CollectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    collection_id: Identifier
    label: str = Field(min_length=1, max_length=160)
    space: str
    root: str = Field(min_length=1, max_length=4096)
    parser_revision: Identifier
    allow_delete_missing: bool = False
    limits: DocumentLimits = Field(default_factory=DocumentLimits)
    scan_limits: ScanLimits = Field(default_factory=ScanLimits)
    extensions: frozenset[str] | None = Field(default=None, min_length=1, max_length=256)
    pdf_ocr: PdfOcrSelection | None = None

    @field_validator('space')
    @classmethod
    def valid_space(cls, value: str) -> str:
        check_space(value)
        return value


class DirectorySyncConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    schema_version: Literal[1]
    state_dir: str = Field(min_length=1, max_length=4096)
    key_env: str = Field(pattern=r'^[A-Za-z_][A-Za-z0-9_]{0,127}$')
    store_id: str = Field(min_length=1, max_length=512)
    collections: tuple[CollectionConfig, ...] = Field(min_length=1, max_length=256)
    max_active: int = Field(default=2, ge=1, le=16)
    max_runs: int = Field(default=4096, ge=1, le=100000)
    max_attempts: int = Field(default=3, ge=1, le=4)
    deadline_s: float = Field(default=300.0, gt=0, le=3600, allow_inf_nan=False)

    @field_validator('schema_version', mode='before')
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError('integer schema version required')
        return value


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('duplicate configuration key')
        result[name] = value
    return result


def _read(path: Path) -> DirectorySyncConfig:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise ValueError('owned private regular configuration required')
            raw = stream.read(131073)
        if len(raw) > 131072:
            raise ValueError('configuration exceeds byte limit')
        json.loads(raw, object_pairs_hook=_unique)
        return DirectorySyncConfig.model_validate_json(raw)
    except (OSError, ValueError, RecursionError):
        raise ValueError('Directory sync configuration requires valid bounded JSON with unique keys '
                         'in an owned 0600 regular file without symbolic or additional hard links') from None


def _resolve(value: str, parent: Path) -> Path:
    if not value.strip() or '\0' in value:
        raise ValueError('directory sync requires nonempty local paths')
    path = Path(value).expanduser()
    return path if path.is_absolute() else parent / path


def _parser(config: CollectionConfig, document_ocr: DocumentOcr | None,
            ocr_identity: str, document_media: DocumentMedia | None) -> ImportParserBinding:
    selected: dict[str, DocumentParser] = {}
    if document_media is not None:
        selected.update({suffix: document_media for suffix in MEDIA_DOCUMENT_EXTENSIONS})
    if config.pdf_ocr is not None:
        if document_ocr is None:
            raise ValueError('collection PDF OCR requires configured host OCR')
        selected['.pdf'] = document_ocr.parser(config.pdf_ocr)
    dependencies: dict[str, str] = {}
    for package in ('scone-memory', 'pypdf', 'pypdfium2', 'python-docx', 'openpyxl', 'python-pptx', 'beautifulsoup4'):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = 'uninstalled'
    fingerprint = {
        'implementation': 'directory-collection-parser-v1', 'operator': config.parser_revision,
        'dependencies': dependencies, 'pdf_ocr': config.pdf_ocr.model_dump() if config.pdf_ocr else None,
        'ocr_identity': ocr_identity if config.pdf_ocr else None,
        'ocr_dpi': document_ocr.dpi if config.pdf_ocr and document_ocr else None,
        'media_revision': document_media.revision if document_media else None,
    }
    revision = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return ImportParserBinding(revision, BuiltinDocumentParser(parsers=selected))


def load_directory_sync(path: str, memory: MemoryEngine, *, document_ocr: DocumentOcr | None = None,
                        ocr_identity: str = '', document_media: DocumentMedia | None = None) -> DirectorySyncService:
    """Load collections without scans, parser calls, model downloads or admission.

    The operator revision must change with OCR binaries/trained data or custom
    parser behavior; installed package versions, host OCR settings and configured
    media revisions are also bound. State must stay outside every source root.
    """
    try:
        config_path = Path(path).expanduser().absolute()
        config = _read(config_path)
        secret = os.environ.get(config.key_env, '')
        if re.fullmatch('[a-fA-F0-9]{64}', secret) is None:
            raise ValueError('Directory sync key environment variable must contain 64 hex characters')
        key = bytes.fromhex(secret)
        target = _resolve(config.state_dir, config_path.parent)
        canonical_target = target.resolve()
        roots: set[tuple[str, int, int]] = set()
        identities: set[tuple[str, str]] = set()
        prepared: list[tuple[CollectionConfig, DirectoryScanner, ImportParserBinding]] = []
        for collection in config.collections:
            extensions = collection.extensions
            if extensions is None:
                extensions = default_extensions() | (MEDIA_DOCUMENT_EXTENSIONS if document_media else frozenset())
            scanner = DirectoryScanner(_resolve(collection.root, config_path.parent),
                                       limits=collection.scan_limits, extensions=extensions)
            info = scanner.root.stat()
            identity = (collection.space, collection.collection_id)
            root = (collection.space, info.st_dev, info.st_ino)
            if identity in identities or root in roots:
                raise ValueError('duplicate configured collection or source root')
            if canonical_target.is_relative_to(scanner.root):
                raise ValueError('directory sync state must be outside every source root')
            identities.add(identity)
            roots.add(root)
            prepared.append((collection, scanner, _parser(collection, document_ocr, ocr_identity, document_media)))
        target.mkdir(mode=0o700, exist_ok=True)
        info = target.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('directory sync state must be an owned private directory')
        collections: list[DirectoryCollection] = []
        for collection, scanner, binding in prepared:
            token = hmac.new(key, json.dumps([collection.space, collection.collection_id]).encode(), hashlib.sha256).hexdigest()
            sync = DirectorySync(memory, scanner.root, space=collection.space,
                journal=target / (token + '.journal'), key=key, store_id=config.store_id,
                parser_revision=binding.revision, parser=binding.parser, limits=collection.limits,
                scan_limits=scanner.limits, extensions=scanner.extensions)
            collections.append(DirectoryCollection(collection.collection_id, collection.label, sync,
                                                   allow_delete_missing=collection.allow_delete_missing))
        registry_binding = json.dumps(['directory-registry-v1', config.store_id], separators=(',', ':')).encode()
        registry_key = hmac.new(key, registry_binding, hashlib.sha256).digest()
        return DirectorySyncService(target, key=registry_key, memory=memory, collections=tuple(collections),
            max_active=config.max_active, max_runs=config.max_runs,
            deadline_s=config.deadline_s, max_attempts=config.max_attempts)
    except InvalidInput:
        raise ValueError('Directory sync collection configuration is invalid') from None
