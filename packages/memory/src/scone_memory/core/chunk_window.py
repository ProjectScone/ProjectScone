"""Optional bounded document navigation; older custom stores may omit it."""
from typing import Protocol, runtime_checkable

from .models import Chunk


def validate_chunk_window(episode_id: int, start_ordinal: int, limit: int) -> None:
    if (type(episode_id) is not int or not 0 < episode_id < 2**63
            or type(start_ordinal) is not int or not 0 <= start_ordinal < 2**31
            or type(limit) is not int or not 1 <= limit <= 20):
        raise ValueError('invalid chunk window bounds')


@runtime_checkable
class ChunkWindowLookup(Protocol):
    async def page_chunks(self, space: str, episode_id: int, *, start_ordinal: int,
                          limit: int) -> list[Chunk]:
        """At most limit rows, ordinal >= start, ordered by ordinal then chunk ID.

        Apply space and episode before the limit. Never enumerate other sources
        or materialize every chunk of the document to implement this operation.
        """
        ...
