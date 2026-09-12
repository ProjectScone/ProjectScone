"""Optional parser recovery contracts; storage and encryption belong to the caller."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .formats.types import DocumentLimits, ParsedDocument
    from .pdf import ParsedPdf, PdfLimits


class ExtractionCheckpoints(Protocol):
    """Opaque bounded receipts scoped to one active extraction attempt.

    The owner must bind receipts to source, space, parser revision and run.
    Parsers additionally validate their own configuration and receipt contents.
    The workflow journal implements this with encrypted intermediate checkpoints.
    """
    def get(self, key: str) -> bytes | None: ...
    def put(self, key: str, value: bytes) -> None: ...


@runtime_checkable
class CheckpointedDocumentParser(Protocol):
    async def parse_checkpointed(self, data: bytes, filename: str, limits: DocumentLimits,
                                checkpoints: ExtractionCheckpoints) -> ParsedDocument: ...


@runtime_checkable
class CheckpointedPdfParser(Protocol):
    async def parse_checkpointed(self, data: bytes, limits: PdfLimits,
                                checkpoints: ExtractionCheckpoints) -> ParsedPdf: ...


def checkpoint_dispatch_allowed(parser: object) -> bool:
    """An inherited recovery path must not bypass a newer public parse override.

    Extensions replacing parse opt into recovery by also replacing
    parse_checkpointed. Unchanged inherited implementations remain eligible.
    """
    instance = getattr(parser, '__dict__', {})
    if isinstance(instance, dict):
        if 'parse_checkpointed' in instance:
            return True
        if 'parse' in instance:
            return False
    owners = type(parser).__mro__
    plain = next((index for index, owner in enumerate(owners) if 'parse' in vars(owner)), len(owners))
    recovery = next((index for index, owner in enumerate(owners) if 'parse_checkpointed' in vars(owner)), len(owners))
    return recovery < len(owners) and recovery <= plain
