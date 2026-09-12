"""One coherent passage instead of three fragments of it.

Small chunks match precisely and read badly. Three neighbouring
fragments of one paragraph are three citations to the same thought, and
between them they crowd the rest of the answer out of the limit. The
leading frameworks solve this by indexing a hierarchy at ingestion — a
parent node holding children — and merging children back into the parent
at retrieval, which means choosing the hierarchy before anyone has asked
a question, and re-indexing to change it.

Nothing here needs a hierarchy or a re-index, because every chunk already
carries the byte span it came from. Neighbours from the same episode are
merged by reading the span that contains them, so the shape of a merge is
decided by what was actually retrieved rather than by a decision made at
ingestion.

Three rules it keeps:

- **A merged passage says what went into it.** ``from_chunks`` names every
  chunk absorbed, because a citation nobody can check is worse than three
  that can.
- **It keeps the best score of its parts, never their sum.** A sum would
  make a merged passage outrank everything by arithmetic rather than by
  relevance.
- **It is a passage, not a document.** Chunks further apart than
  ``max_merged`` bytes are left as fragments and the reason is said,
  because silently returning the whole document would be worse than not
  merging at all.

No model is called and nothing is re-embedded: the merged text is read
from the episode this framework already stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from ..core.errors import InvalidInput
from ..core.models import RecallItem
from ..memory.engine import check_space

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

#: Bytes a merged passage may span. Past this the fragments are further
#: apart than one readable passage, and merging them would return most of
#: a document to answer a question about a sentence.
MAX_MERGED = 4_000
#: Chunks it takes to be worth merging. One chunk is already a passage.
MIN_LEAVES = 2
#: Episodes whose text is read to build merged passages in one call.
MAX_EPISODES = 50


@dataclass(frozen=True)
class Merged:
    """What was joined up, and exactly what went into each one."""

    items: tuple[RecallItem, ...]
    #: Merged passages produced.
    merged: int = 0
    #: Chunks that went into them. Always at least twice ``merged``.
    absorbed: int = 0
    #: The chunk a merged passage is reported under, to every chunk that
    #: went into it, the reported one included.
    from_chunks: dict[int, tuple[int, ...]] = field(default_factory=dict)
    #: Episodes whose fragments were too far apart to be one passage.
    too_far: int = 0
    why: str = ""

    def record(self) -> dict[str, object]:
        return {"merged": self.merged, "absorbed": self.absorbed, "too_far": self.too_far,
                "why": self.why,
                "from_chunks": {str(k): list(v) for k, v in self.from_chunks.items()},
                "items": [item.model_dump() for item in self.items]}


async def merge_neighbours(
    engine: "MemoryEngine",
    space: str,
    items: Sequence[RecallItem],
    *,
    max_merged: int = MAX_MERGED,
    min_leaves: int = MIN_LEAVES,
    episodes: int = MAX_EPISODES,
) -> Merged:
    """Join neighbouring chunks of one episode into the passage holding them."""
    check_space(space)
    if not 1 <= max_merged <= 1_000_000:
        raise InvalidInput(f"max_merged must be from 1 to 1000000, not {max_merged}")
    if min_leaves < 2:
        raise InvalidInput(f"merging takes at least two chunks, not {min_leaves}")

    together: dict[int, list[RecallItem]] = {}
    for item in items:
        together.setdefault(item.episode_id, []).append(item)

    kept: list[RecallItem] = []
    from_chunks: dict[int, tuple[int, ...]] = {}
    merged = absorbed = far = 0
    read = 0
    for episode_id, group in together.items():
        if len(group) < min_leaves:
            kept.extend(group)
            continue
        first, last = min(i.start for i in group), max(i.end for i in group)
        if last - first > max_merged:
            far += 1
            kept.extend(group)
            continue
        if read >= episodes:
            kept.extend(group)
            continue
        read += 1
        try:
            episode = await engine.episode(space, episode_id)
        except Exception:
            # A passage we cannot read is not a passage. The fragments we
            # already have are a worse answer than the whole, and a better
            # one than nothing.
            kept.extend(group)
            continue
        text = _span(episode.content, first, last)
        if not text.strip():
            kept.extend(group)
            continue
        best = max(group, key=lambda i: i.score)
        kept.append(best.model_copy(update={
            "text": text, "start": first, "end": last,
            "first_line": None, "last_line": None, "declaration": None}))
        from_chunks[best.chunk_id] = tuple(sorted(i.chunk_id for i in group))
        merged += 1
        absorbed += len(group)

    why = (f"{merged} passage(s) joined from {absorbed} chunk(s)" if merged
           else "nothing to join: no episode returned neighbouring chunks")
    if far:
        why += (f"; {far} episode(s) had fragments too far apart to be one passage "
                f"(over {max_merged} bytes) and were left as they were")
    return Merged(items=tuple(kept), merged=merged, absorbed=absorbed,
                  from_chunks=from_chunks, too_far=far, why=why)


def _span(content: str, start: int, end: int) -> str:
    """The text between two byte offsets, cut on a character boundary.

    Offsets are bytes because that is what a chunk records, and a cut in
    the middle of a character would corrupt the quote rather than shorten
    it.
    """
    raw = content.encode()
    if start < 0 or end > len(raw) or end <= start:
        return ""
    return raw[start:end].decode("utf-8", errors="ignore")
