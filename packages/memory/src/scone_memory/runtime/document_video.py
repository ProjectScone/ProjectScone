"""Private, explicit configuration for installed video decoding and OCR."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.errors import InvalidInput
from ..ingestion.document_video import DocumentVideo
from ..ingestion.video_frames import VideoFrameDecoder, VideoFramePolicy
from ..ingestion.video_ocr import VideoDocumentParser
from ..ocr.tesseract import TesseractOcr
from ..ocr.types import OcrResult
from .document_media import _decoder_digest, _unique


class DocumentVideoConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    schema_version: Literal[1]
    model_revision: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
    ffmpeg_executable: str = Field(min_length=1, max_length=4096)
    ffprobe_executable: str = Field(min_length=1, max_length=4096)
    ocr_executable: str = Field(min_length=1, max_length=4096)
    language: str = Field(default='eng', min_length=1, max_length=128)
    page_segmentation: int = Field(default=3, ge=0, le=13)
    policy: VideoFramePolicy = Field(default_factory=VideoFramePolicy)

    @field_validator('schema_version', mode='before')
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError('schema version must be an integer')
        return value

    @field_validator('ffmpeg_executable', 'ffprobe_executable', 'ocr_executable')
    @classmethod
    def installed(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError('installed absolute executable required')
        return value


def _read(path: str) -> DocumentVideoConfig:
    try:
        descriptor = os.open(Path(path).expanduser(), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as source:
            info = os.fstat(source.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise ValueError('private regular configuration required')
            raw = source.read(16385)
        if len(raw) > 16384:
            raise ValueError('configuration byte limit')
        json.loads(raw, object_pairs_hook=_unique)
        return DocumentVideoConfig.model_validate_json(raw)
    except (OSError, ValueError, RecursionError):
        raise ValueError('Document video configuration requires bounded JSON with unique supported settings '
                         'in an owned 0600 regular file without symbolic or additional hard links') from None


class _PinnedOcr:
    def __init__(self, engine: TesseractOcr, executable: str, digest: str) -> None:
        self._engine, self._executable, self._digest = engine, executable, digest

    def _check(self) -> None:
        try:
            if _decoder_digest(self._executable) == self._digest:
                return
        except ValueError:
            pass
        raise InvalidInput('configured video OCR executable changed; reload configuration explicitly')

    async def recognize(self, image: bytes, *, max_pixels: int = 20_000_000, max_regions: int = 10_000,
                        timeout_seconds: float = 30.0) -> OcrResult:
        self._check()
        result = await self._engine.recognize(image, max_pixels=max_pixels, max_regions=max_regions,
                                             timeout_seconds=timeout_seconds)
        self._check()
        return result


def load_document_video(path: str) -> DocumentVideo:
    """Load without running tools, contacting services or downloading models.

    Executable bytes and every setting bind the revision. The operator must bump
    model_revision for changed trained data, shared libraries or OCR behavior.
    Host injection supports any caller-owned OcrEngine through VideoDocumentParser.
    """
    config = _read(path)
    try:
        digests = {name: _decoder_digest(executable) for name, executable in (
            ('ffmpeg', config.ffmpeg_executable), ('ffprobe', config.ffprobe_executable),
            ('ocr', config.ocr_executable))}
        decoder = VideoFrameDecoder(ffmpeg_path=config.ffmpeg_executable, ffprobe_path=config.ffprobe_executable)
        engine = TesseractOcr(executable=config.ocr_executable, language=config.language,
                              page_segmentation=config.page_segmentation)
        binding = json.dumps({'implementation': 'local-document-video-v1',
            'settings': config.model_dump(), 'executables': digests}, sort_keys=True, separators=(',', ':')).encode()
        revision = 'video-' + hashlib.sha256(binding).hexdigest()
        parser = VideoDocumentParser(decoder, _PinnedOcr(engine, config.ocr_executable, digests['ocr']),
                                     model_revision=revision, policy=config.policy)
    except (ValueError, OSError, InvalidInput):
        raise ValueError('Document video decoder or OCR settings are invalid') from None
    return DocumentVideo(parser, revision=revision)
