"""Host-owned video OCR, selected explicitly for each import."""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from .formats.media import VIDEO_EXTENSIONS
from .video_ocr import VideoDocumentParser

VIDEO_DOCUMENT_EXTENSIONS = VIDEO_EXTENSIONS - {'.ts'}


@dataclass(frozen=True)
class DocumentVideo:
    """Change revision whenever the injected parser's behavior changes."""
    parser: VideoDocumentParser = field(repr=False)
    revision: str

    def __post_init__(self) -> None:
        if not isinstance(self.parser, VideoDocumentParser):
            raise ValueError('document video requires a configured VideoDocumentParser')
        if not isinstance(self.revision, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', self.revision) is None:
            raise ValueError('document video requires a bounded operator revision')


def video_choices(config: DocumentVideo | None) -> dict[str, object]:
    return {'available': config is not None, 'selection': {'video_ocr': True},
            'extensions': sorted(VIDEO_DOCUMENT_EXTENSIONS), 'extraction': 'sampled-frame-text',
            'includes_audio': False}
