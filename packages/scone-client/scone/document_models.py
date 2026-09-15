"""Standalone durable-document request, control and provenance models."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Mapping, Optional
from types import MappingProxyType
import re

from ._wire import boolean, digest, identifier, integer, invalid, names, record, text, timestamp

MAX_REVISION = 2**31 - 1
_OCR_LANGUAGE = re.compile(r'[A-Za-z0-9_]{1,32}(\+[A-Za-z0-9_]{1,32}){0,7}')
VIDEO_EXTENSIONS = frozenset({'.mp4', '.m4v', '.mov', '.webm', '.mkv', '.avi', '.mpeg', '.mpg', '.mpegts'})


def filename(value: object) -> str:
    result = text(value, 1024, 'document filename')
    if '\0' in result:
        raise invalid('document filename')
    return result


def seconds(value: object, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise invalid('document duration')
    if not 0 < value <= maximum:
        raise invalid('document duration')
    return float(value)


def instant(value: str) -> datetime:
    return datetime.fromisoformat(timestamp(value).replace('Z', '+00:00'))


@dataclass(frozen=True)
class PdfOcr:
    mode: str
    reading_order: str
    #: One of the languages the host offers (``DocumentFormats.pdf_ocr_languages``); None reads
    #: with the host's own language and is not sent.
    language: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in ('missing_text', 'all_pages') or self.reading_order not in ('provider', 'columns_ltr', 'columns_rtl'):
            raise invalid('PDF OCR selection')
        if self.language is not None and (not isinstance(self.language, str) or _OCR_LANGUAGE.fullmatch(self.language) is None):
            raise invalid('PDF OCR language')

    @classmethod
    def from_json(cls, value: object) -> Optional[PdfOcr]:
        if value is None:
            return None
        row = record(value)
        if not {'mode', 'reading_order'} <= set(row) <= {'mode', 'reading_order', 'language'}:
            raise invalid('PDF OCR selection')
        language = row.get('language')
        if 'language' in row and not isinstance(language, str):
            raise invalid('PDF OCR language')
        return cls(text(row.get('mode'), 32), text(row.get('reading_order'), 32),
                   language if isinstance(language, str) else None)

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {'mode': self.mode, 'reading_order': self.reading_order}
        if self.language is not None:
            result['language'] = self.language
        return result


def check_ocr(name: str, selection: Optional[PdfOcr]) -> None:
    if selection is not None and (not isinstance(selection, PdfOcr) or PurePosixPath(name).suffix.lower() != '.pdf'):
        raise invalid('PDF OCR requires PDF filename')


def check_video_ocr(name: str, selected: bool, pdf_ocr: Optional[PdfOcr]) -> None:
    boolean(selected)
    if selected and (pdf_ocr is not None or PurePosixPath(name).suffix.lower() not in VIDEO_EXTENSIONS):
        raise invalid('video OCR requires video filename and cannot be combined with PDF OCR')


@dataclass(frozen=True)
class ParserLimits:
    max_input_bytes: int
    max_text_bytes: int
    max_segments: int
    max_archive_entries: int
    max_archive_bytes: int
    timeout_seconds: float

    @classmethod
    def from_json(cls, value: object) -> ParserLimits:
        row = record(value)
        return cls(integer(row.get('max_input_bytes'), 1, 25*1024*1024),
                   integer(row.get('max_text_bytes'), 1, 2000000), integer(row.get('max_segments'), 1, 20000),
                   integer(row.get('max_archive_entries'), 1, 10000), integer(row.get('max_archive_bytes'), 1, 100000000),
                   seconds(row.get('timeout_seconds'), 120))


@dataclass(frozen=True)
class DocumentSpec:
    attachment_id: str
    filename: str
    parser_revision: str
    limits: ParserLimits
    pdf_ocr: Optional[PdfOcr]
    deadline_s: float
    max_attempts: int
    video_ocr: bool = False

    @classmethod
    def from_json(cls, value: object) -> DocumentSpec:
        row = record(value)
        name, ocr = filename(row.get('filename')), PdfOcr.from_json(row.get('pdf_ocr'))
        check_ocr(name, ocr)
        video = boolean(row.get('video_ocr', False))
        check_video_ocr(name, video, ocr)
        return cls(digest(row.get('attachment_id')), name, identifier(row.get('parser_revision')),
                   ParserLimits.from_json(row.get('limits')), ocr, seconds(row.get('deadline_s'), 300),
                   integer(row.get('max_attempts'), 1, 4), video)


@dataclass(frozen=True)
class DocumentRequest:
    space: str
    import_id: str
    spec: DocumentSpec
    created_at: str
    revision: int
    attempt: int
    last_started_at: Optional[str]
    cancel_requested_at: Optional[str]

    @classmethod
    def from_json(cls, value: object, *, expected_space: str, import_id: str) -> DocumentRequest:
        row = record(value)
        if row.get('space') != expected_space or row.get('import_id') != identifier(import_id):
            raise invalid('document identity')
        created = timestamp(row.get('created_at'))
        revision, attempt = integer(row.get('revision'), 0, MAX_REVISION), integer(row.get('attempt'), 0, MAX_REVISION)
        started = timestamp(row['last_started_at']) if row.get('last_started_at') is not None else None
        cancelled = timestamp(row['cancel_requested_at']) if row.get('cancel_requested_at') is not None else None
        spec = DocumentSpec.from_json(row.get('spec'))
        if revision < attempt or attempt > spec.max_attempts or (attempt == 0) != (started is None):
            raise invalid('document attempt')
        if any(instant(value) < instant(created) for value in (started, cancelled) if value is not None):
            raise invalid('document control timestamp')
        return cls(expected_space, import_id, spec, created, revision, attempt, started, cancelled)


@dataclass(frozen=True)
class DocumentStatus:
    space: str
    import_id: str
    filename: str
    attachment_id: str
    created_at: str
    attempt: int
    max_attempts: int
    revision: int
    status: str
    active_local: bool
    completed_steps: tuple[str, ...]
    inflight: Optional[str]
    outcome_unknown: bool
    error_class: Optional[str]

    @classmethod
    def from_json(cls, value: object, *, expected_space: str, import_id: Optional[str] = None) -> DocumentStatus:
        row = record(value)
        saved_id = identifier(row.get('import_id'))
        if row.get('space') != expected_space or (import_id is not None and saved_id != import_id):
            raise invalid('document status identity')
        maximum = integer(row.get('max_attempts'), 1, 4)
        attempt, revision = integer(row.get('attempt'), 0, maximum), integer(row.get('revision'), 0, MAX_REVISION)
        completed = names(row.get('completed_steps'), 2)
        inflight = identifier(row['inflight']) if row.get('inflight') is not None else None
        state = text(row.get('status'), 64)
        if state not in ('registered', 'created', 'running', 'interrupted', 'completed', 'failed', 'cancelled',
                         'deadline', 'sources_invalid', 'verification_unavailable', 'unavailable',
                         'outcome_unknown', 'retry_not_allowed'):
            raise invalid('document status')
        stages = ('extract', 'index')
        if (revision < attempt or completed != stages[:len(completed)]
                or (inflight is not None and (len(completed) == 2 or inflight != stages[len(completed)]))
                or (state == 'completed' and (completed != stages or inflight is not None))):
            raise invalid('document stage progress')
        return cls(expected_space, saved_id, filename(row.get('filename')), digest(row.get('attachment_id')),
                   timestamp(row.get('created_at')), attempt, maximum, revision, state,
                   boolean(row.get('active_local')), completed, inflight, boolean(row.get('outcome_unknown')),
                   text(row['error_class'], 256) if row.get('error_class') is not None else None)

    def match(self, request: DocumentRequest, *, control: bool = True) -> None:
        if (self.space != request.space or self.import_id != request.import_id or self.filename != request.spec.filename
                or self.attachment_id != request.spec.attachment_id or self.max_attempts != request.spec.max_attempts
                or instant(self.created_at) != instant(request.created_at)
                or (control and (self.revision != request.revision or self.attempt != request.attempt))):
            raise invalid('document request binding')


@dataclass(frozen=True)
class DocumentAttachment:
    attachment_id: str
    media_type: str
    bytes: int
    filename: Optional[str]

    @classmethod
    def from_json(cls, value: object) -> DocumentAttachment:
        row = record(value)
        return cls(digest(row.get('attachment_id')), text(row.get('media_type'), 256),
                   integer(row.get('bytes'), 1), text(row['filename'], 4096) if row.get('filename') is not None else None)


@dataclass(frozen=True)
class DocumentStored:
    episode_id: int
    deduplicated: bool
    chunks: int
    outcome: str

    @classmethod
    def from_json(cls, value: object) -> DocumentStored:
        row = record(value)
        duplicate = boolean(row.get('deduplicated'))
        outcome = row.get('outcome')
        # Durable file ingestion uses an immutable dedup key, never keyed replacement.
        if outcome != ('duplicate' if duplicate else 'accepted') or row.get('replaced') is not None or row.get('reason') is not None:
            raise invalid('document storage outcome')
        return cls(integer(row.get('episode_id'), 1), duplicate, integer(row.get('chunks'), 0), str(outcome))


@dataclass(frozen=True)
class DocumentResult:
    space: str
    import_id: str
    filename: str
    format: str
    segments: int
    original: DocumentAttachment
    manifest: DocumentAttachment
    added: DocumentStored
    pdf_ocr: Optional[PdfOcr]
    video_ocr: bool = False

    @classmethod
    def from_json(cls, value: object, *, request: DocumentRequest) -> DocumentResult:
        row = record(value)
        original, manifest = DocumentAttachment.from_json(row.get('original')), DocumentAttachment.from_json(row.get('manifest'))
        ocr = PdfOcr.from_json(row.get('pdf_ocr'))
        video = boolean(row.get('video_ocr', False))
        if (row.get('space') != request.space or row.get('import_id') != request.import_id
                or row.get('filename') != request.spec.filename or original.attachment_id != request.spec.attachment_id
                or original.bytes > request.spec.limits.max_input_bytes or manifest.media_type != 'application/json'
                or manifest.attachment_id == original.attachment_id or ocr != request.spec.pdf_ocr
                or video != request.spec.video_ocr):
            raise invalid('document result binding')
        segments = integer(row.get('segments'), 0 if video else 1, request.spec.limits.max_segments)
        added = DocumentStored.from_json(row.get('added'))
        if segments == 0 and added.chunks != 0:
            raise invalid('visual-only document chunks')
        return cls(request.space, request.import_id, request.spec.filename, text(row.get('format'), 256),
                   segments, original, manifest, added, ocr, video)


@dataclass(frozen=True)
class DocumentFormat:
    available: bool
    parser: str
    requires: Optional[str]


@dataclass(frozen=True)
class DocumentFormats:
    formats: Mapping[str, DocumentFormat]
    max_input_bytes: int
    pdf_ocr_available: bool
    pdf_ocr_modes: tuple[str, ...]
    pdf_reading_orders: tuple[str, ...]
    video_ocr_available: bool = False
    video_ocr_extensions: tuple[str, ...] = ()
    #: Languages a scan may choose in PdfOcr.language; empty when the host offers none.
    pdf_ocr_languages: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, value: object) -> DocumentFormats:
        row = record(value)
        formats = record(row.get('formats'))
        if len(formats) > 1024:
            raise invalid('document formats')
        parsed = {}
        for suffix, raw in formats.items():
            if re.fullmatch(r'\.[a-z0-9_-]{1,32}', suffix) is None:
                raise invalid('document extension')
            entry = record(raw)
            parsed[suffix] = DocumentFormat(boolean(entry.get('available')), text(entry.get('parser'), 512),
                text(entry['requires'], 1024) if entry.get('requires') is not None else None)
        ocr = record(row.get('pdf_ocr'))
        modes, orders = names(ocr.get('modes'), 2), names(ocr.get('reading_orders'), 3)
        if set(modes) - {'missing_text', 'all_pages'} or set(orders) - {'provider', 'columns_ltr', 'columns_rtl'}:
            raise invalid('document OCR choices')
        languages: tuple[str, ...] = ()
        if 'languages' in ocr:
            offered = ocr['languages']
            if (not isinstance(offered, list) or len(offered) > 64
                    or any(not isinstance(one, str) or _OCR_LANGUAGE.fullmatch(one) is None for one in offered)
                    or len(set(offered)) != len(offered)):
                raise invalid('document OCR languages')
            languages = tuple(offered)
        video_available = False
        video_extensions: tuple[str, ...] = ()
        if 'video_ocr' in row:
            video = record(row['video_ocr'])
            video_available = boolean(video.get('available'))
            video_extensions = names(video.get('extensions'), 32)
            if (set(video_extensions) - VIDEO_EXTENSIONS or video.get('extraction') != 'sampled-frame-text'
                    or boolean(video.get('includes_audio')) or record(video.get('selection')) != {'video_ocr': True}
                    or type(record(video['selection']).get('video_ocr')) is not bool):
                raise invalid('document video OCR choices')
        return cls(MappingProxyType(parsed), integer(row.get('max_input_bytes'), 1, 25*1024*1024),
                   boolean(ocr.get('available')), modes, orders, video_available, video_extensions, languages)
