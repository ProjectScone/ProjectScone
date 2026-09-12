"""Explicit local document model selection for standard memory and conversation hosts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.errors import InvalidInput
from ..ingestion.document_media import DocumentMedia
from ..ingestion.formats.media import MediaDocumentParser
from ..providers.transcription.document import LocalDocumentTranscriber


class DocumentMediaConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    schema_version: Literal[1]
    base_url: str = Field(min_length=1, max_length=2048)
    model: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
    ffmpeg_executable: str = Field(min_length=1, max_length=4096)
    api_key_env: str | None = Field(default=None, min_length=1, max_length=128, pattern=r'^[A-Za-z_][A-Za-z0-9_]*$')
    timeout_seconds: float = Field(default=120.0, gt=0, le=600, allow_inf_nan=False)
    max_duration_seconds: float = Field(default=60.0, gt=0, le=600, allow_inf_nan=False)
    max_response_bytes: int = Field(default=4_000_000, ge=1024, le=12_000_000)
    max_segments: int = Field(default=10000, ge=1, le=10000)

    @field_validator('schema_version', mode='before')
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError('schema version must be an integer')
        return value


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate media configuration key')
        result[key] = value
    return result


def _read(path: str) -> DocumentMediaConfig:
    try:
        descriptor = os.open(Path(path).expanduser(), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as source:
            info = os.fstat(source.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise ValueError('private regular configuration file required')
            raw = source.read(16385)
        if len(raw) > 16384:
            raise ValueError('media configuration exceeds its byte limit')
    except (OSError, ValueError):
        raise ValueError('Document media configuration file must be readable, owned, regular, '
                         '0600, without symbolic or additional hard links, and at most 16384 bytes') from None
    try:
        json.loads(raw, object_pairs_hook=_unique)
        return DocumentMediaConfig.model_validate_json(raw)
    except (ValueError, RecursionError):
        raise ValueError('Document media configuration contents must be valid JSON with unique '
                         'keys and supported settings') from None


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _decoder_digest(path: str) -> str:
    """Fingerprint a stable regular executable, following installer symlinks."""
    maximum = 512 * 1024 * 1024
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
                raise ValueError('invalid decoder size or file type')
            digest = hashlib.sha256()
            count = 0
            while block := source.read(min(1024 * 1024, maximum - count + 1)):
                count += len(block)
                if count > maximum:
                    raise ValueError('decoder exceeds fingerprint byte limit')
                digest.update(block)
            if (count != before.st_size
                    or _file_identity(before) != _file_identity(os.fstat(source.fileno()))
                    or _file_identity(before) != _file_identity(os.stat(path))):
                raise ValueError('decoder changed while fingerprinting')
        return digest.hexdigest()
    except (OSError, ValueError):
        raise ValueError('Document media decoder must be a readable, stable, nonempty regular '
                         'executable of at most 512 MiB') from None


def load_document_media(path: str) -> DocumentMedia:
    """Load a private configuration without contacting or starting its model.

    Nonsecret settings, decoder bytes and operator model revision bind extraction.
    Decoder contents are hashed at load without execution; installers may use
    symlinks. Restart after decoder changes. Operators must bump model_revision
    for changed weights, dynamic libraries or server behavior at an unchanged
    endpoint. Credential rotation alone does not invalidate retained evidence.
    Explicit host injection can bypass this loader.
    """
    config = _read(path)
    key = os.environ.get(config.api_key_env) if config.api_key_env else None
    if config.api_key_env and key is None:
        raise ValueError('Document media configured credential is missing')
    try:
        provider = LocalDocumentTranscriber(base_url=config.base_url, model=config.model, api_key=key,
            timeout=config.timeout_seconds, max_response_bytes=config.max_response_bytes,
            max_segments=config.max_segments)
        parser = MediaDocumentParser(provider, ffmpeg_executable=config.ffmpeg_executable,
                                     max_duration_seconds=config.max_duration_seconds)
    except (OSError, ValueError, InvalidInput, RecursionError):
        raise ValueError('Document media provider settings, decoder path or credential are invalid') from None
    settings = config.model_dump(exclude={'api_key_env'})
    binding = json.dumps({'implementation': 'local-document-media-v2', 'settings': settings,
                          'decoder_sha256': _decoder_digest(config.ffmpeg_executable)},
                         sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return DocumentMedia(parser, revision='media-' + hashlib.sha256(binding).hexdigest())
