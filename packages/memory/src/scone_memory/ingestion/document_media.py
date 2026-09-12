"""Host-owned audio transcription for the existing document ingestion surfaces."""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from ..core.errors import InvalidInput
from .extraction_checkpoint import ExtractionCheckpoints, checkpoint_dispatch_allowed
from .formats.media import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, MediaDocumentParser
from .formats.media_checkpoint import TRANSCRIPTION_CHECKPOINT_KEY
from .formats.media_windows import WINDOW_BINDING_KEY
from .formats.registry import BuiltinDocumentParser
from .formats.types import DocumentLimits, ParsedDocument

# .ts already means TypeScript in document dispatch. Use .mpegts for transport streams.
MEDIA_DOCUMENT_EXTENSIONS = (AUDIO_EXTENSIONS | VIDEO_EXTENSIONS) - {'.ts'}


@dataclass(frozen=True)
class DocumentMedia:
    """An explicit parser and operator revision; construction performs no inference.

    Change revision when the model, decoder, limits or transcription behavior
    changes. Durable imports bind it so completed work cannot be resumed under a
    different extraction configuration. The host owns provider resources.
    """
    media_parser: MediaDocumentParser = field(repr=False)
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.media_parser, MediaDocumentParser):
            raise ValueError('document media requires a configured MediaDocumentParser')
        if not isinstance(self.revision, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', self.revision) is None:
            raise ValueError('document media requires a bounded operator revision')

    def parser(self) -> BuiltinDocumentParser:
        return BuiltinDocumentParser(parsers={suffix: self for suffix in MEDIA_DOCUMENT_EXTENSIONS})

    def formats(self) -> dict[str, dict[str, object]]:
        available = self.media_parser.decoder_available
        return {suffix: {'available': available, 'parser': 'media-transcription', 'extraction': 'audio-only',
                         'requires': 'configured ffmpeg and timestamped transcriber'}
                for suffix in sorted(MEDIA_DOCUMENT_EXTENSIONS)}

    async def parse(self, data: bytes, filename: str, limits: DocumentLimits) -> ParsedDocument:
        return await self._parse(data, filename, limits)

    async def parse_checkpointed(self, data: bytes, filename: str, limits: DocumentLimits,
                                 checkpoints: ExtractionCheckpoints) -> ParsedDocument:
        return await self._parse(data, filename, limits, checkpoints)

    async def _parse(self, data: bytes, filename: str, limits: DocumentLimits,
                     checkpoints: ExtractionCheckpoints | None = None) -> ParsedDocument:
        if checkpoints is not None and checkpoint_dispatch_allowed(self.media_parser):
            revision = self.revision.encode('ascii')
            saved_revision = checkpoints.get('media-host-binding')
            if saved_revision is None:
                if (checkpoints.get(TRANSCRIPTION_CHECKPOINT_KEY) is not None
                        or checkpoints.get(WINDOW_BINDING_KEY) is not None):
                    raise InvalidInput('media transcription checkpoint has no host revision binding')
                checkpoints.put('media-host-binding', revision)
            elif saved_revision != revision:
                raise InvalidInput('media transcription checkpoints do not match the host revision')
            parsed = await self.media_parser.parse_checkpointed(data, filename, limits, checkpoints)
        else:
            parsed = await self.media_parser.parse(data, filename, limits)
        return ParsedDocument.model_validate({**parsed.model_dump(), 'metadata': {
            **parsed.metadata, 'transcriber_revision': self.revision,
        }})
