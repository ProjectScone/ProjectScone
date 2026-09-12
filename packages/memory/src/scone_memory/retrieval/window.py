"""A precise hit, answered with the passage around it.

Merging joins neighbours, so it needs two or more hits in one episode and
does nothing for one. The commoner shape is the opposite: a single
sentence matches exactly, and the sentence alone does not answer the
question because the answer is in the sentence after it.

The leading RAG framework calls this a sentence window and builds it at
ingestion — embed single sentences, keep the surrounding window in
metadata, re-index to change the window size. Ours reads the window from
the episode **at retrieval**, because every chunk already carries the
byte span it came from. So the size is the caller's, a different size is
a different argument rather than a different index, and nothing has to be
decided before anyone has asked a question.

Three rules, each with a test:

- **A window is quoted, never assembled.** The text is the episode's own
  bytes between two offsets, so ``raw[start:end] == text.encode()`` holds
  and a caller can check the quote against the source. Both edges sit on
  character boundaries for that to be true of text, not just of lengths.
- **A window cut short says so.** Asking for 500 bytes before a chunk
  that begins 40 bytes into the episode gets 40, and ``clipped`` counts
  it — otherwise a caller cannot tell a full window from the edge of a
  document.
- **A source confirmed gone is dropped, not served.** The same rule
  merging learned: the fragment we happen to be holding is deleted text,
  and handing it back under a clean reason is worse than returning less.

No model is called and nothing is re-embedded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from ..core.errors import Gone, InvalidInput, NotFound, SconeError
from ..core.models import RecallItem
from ..memory.engine import check_space

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

#: Bytes a window may reach in either direction. Past this it is not a
#: window, it is the document.
MAX_WINDOW = 100_000
#: Episodes read in one call, and the most a caller may ask for. A read
#: that fails counts: it is work done, and a budget that only counted
#: successes bounded nothing. Past the budget the rest are returned
#: unwidened and counted in ``not_read`` rather than dropped.
MAX_EPISODES = 100


def _tail(byte: int) -> bool:
    """Is this byte the continuation of a character that starts earlier?"""
    return 0x80 <= byte < 0xC0


@dataclass(frozen=True)
class Widened:
    """What was widened, and what was left as it was."""

    items: tuple[RecallItem, ...] = ()
    #: Items given a wider span.
    widened: int = 0
    #: Items whose window met the start or end of the episode, so it is
    #: shorter than asked for.
    clipped: int = 0
    #: Items whose window moved a byte or two to sit on character
    #: boundaries. Counted apart from ``clipped`` because the two mean
    #: different things to a caller: this one is a partial character
    #: dropped, not a document that ran out.
    aligned: int = 0
    #: Items whose episode is confirmed absent; dropped rather than served
    #: from text this space no longer holds.
    gone: int = 0
    #: Items whose episode could not be read, which is not a finding that
    #: it is absent. Left as they were.
    unread: int = 0
    #: Items left as they were because the episode budget was spent
    #: before their episode was reached. Not a failure to read and not a
    #: finding about the episode: nobody looked at it.
    not_read: int = 0
    #: Chunk id to the span it was widened to, so a caller can see what
    #: was read rather than trust that it was.
    by_chunk: dict[int, tuple[int, int]] = field(default_factory=dict)
    why: str = ""

    def record(self) -> dict[str, object]:
        return {"widened": self.widened, "clipped": self.clipped, "aligned": self.aligned,
                "gone": self.gone, "unread": self.unread, "not_read": self.not_read,
                "why": self.why,
                "by_chunk": {str(k): list(v) for k, v in self.by_chunk.items()},
                "items": [item.model_dump() for item in self.items]}


async def widen(engine: "MemoryEngine", space: str, items: Sequence[RecallItem], *,
                before: int = 500, after: int = 500,
                episodes: int = MAX_EPISODES) -> Widened:
    """Return each item with the episode's text around it."""
    check_space(space)
    # Byte distances, checked as exact integers at the door. A fractional
    # window passed the `< 0` test and then failed much later as a
    # TypeError from a byte slice, which tells a caller nothing about
    # what they asked for. `bool` is excluded deliberately: `True` is an
    # integer to `isinstance` and is not a distance.
    for name, value in (("before", before), ("after", after)):
        if type(value) is not int:
            raise InvalidInput(f"{name} is a whole number of bytes, not {value!r}")
    if before < 0 or after < 0:
        raise InvalidInput("a window cannot reach a negative distance")
    if before == 0 and after == 0:
        raise InvalidInput("a window of nothing is not a window: give before or after")
    if before > MAX_WINDOW or after > MAX_WINDOW:
        raise InvalidInput(f"a window may reach at most {MAX_WINDOW} bytes either way, not "
                           f"{max(before, after)}; past that it is the document, not a window")
    # MAX_EPISODES documented a cap that nothing enforced, so a caller
    # could ask for more reads than the module says it will do.
    if type(episodes) is not int or not 1 <= episodes <= MAX_EPISODES:
        raise InvalidInput(f"episodes is a whole number from 1 to {MAX_EPISODES}, not "
                           f"{episodes!r}")

    kept: list[RecallItem] = []
    by_chunk: dict[int, tuple[int, int]] = {}
    widened = clipped = moved = vanished = unread = unbudgeted = 0
    read: dict[int, Optional[str]] = {}
    # An episode we tried and could not read. Held apart from `read`
    # because "we could not read it" is not "it is not there", and held
    # at all because a read that failed is still a read: leaving it out
    # let one budgeted episode become one failed read per item.
    unavailable: set[int] = set()
    for item in items:
        if item.episode_id in unavailable:
            unread += 1
            kept.append(item)
            continue
        if item.episode_id not in read:
            if len(read) + len(unavailable) >= episodes:
                unbudgeted += 1
                kept.append(item)
                continue
            try:
                read[item.episode_id] = (await engine.episode(space, item.episode_id)).content
            except (Gone, NotFound):
                read[item.episode_id] = None
            except SconeError:
                unavailable.add(item.episode_id)
                unread += 1
                kept.append(item)
                continue
        content = read[item.episode_id]
        if content is None:
            vanished += 1
            continue
        raw = content.encode()
        start, stop = max(0, item.start - before), min(len(raw), item.end + after)
        first, last = start, stop
        if first >= last or item.end > len(raw):
            # The span does not sit in the text we just read, so this is
            # not the episode the chunk came from any more.
            unread += 1
            kept.append(item)
            continue
        if first > item.start - before or last < item.end + after:
            clipped += 1
        # A window is counted in bytes and read as text, so an edge can
        # land inside a character. Both edges move inward to the nearest
        # character start; the chunk's own span is one, so neither edge
        # can cross the hit. Dropping the partial character while leaving
        # the offsets on it -- which is what decoding leniently did --
        # left the item's span naming bytes the item's text did not
        # contain, and a window is quoted, never assembled.
        while first < last and _tail(raw[first]):
            first += 1
        while first < last < len(raw) and _tail(raw[last]):
            last -= 1
        if (first, last) != (start, stop):
            moved += 1
        text = raw[first:last].decode()
        kept.append(item.model_copy(update={
            "text": text, "start": first, "end": last,
            "first_line": None, "last_line": None, "declaration": None}))
        by_chunk[item.chunk_id] = (first, last)
        widened += 1

    why = f"{widened} item(s) widened" if widened else "nothing to widen"
    if clipped:
        why += (f"; {clipped} met the start or end of their episode, so those windows are "
                f"shorter than asked for")
    if moved:
        why += (f"; {moved} moved to a character boundary, so those windows are a byte or two "
                f"shorter than asked for -- a partial character dropped, not a document that "
                f"ran out")
    if vanished:
        why += (f"; {vanished} item(s) are no longer there and were dropped rather than served "
                f"from text this space has deleted")
    if unread:
        why += (f"; {unread} item(s) could not be read and stand as they were -- that is a "
                f"failure to widen, not a finding that there was nothing to widen")
    if unbudgeted:
        why += (f"; {unbudgeted} item(s) stand as they were because the budget of {episodes} "
                f"episode(s) was spent before theirs was reached -- nobody looked at those, "
                f"which is not a finding about them either")
    return Widened(items=tuple(kept), widened=widened, clipped=clipped, aligned=moved,
                   gone=vanished, unread=unread, not_read=unbudgeted, by_chunk=by_chunk, why=why)
