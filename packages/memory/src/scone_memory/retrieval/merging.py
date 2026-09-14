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
- **It is a passage, not a document.** An episode's fragments are joined
  in clusters, each spanning at most ``max_merged`` bytes, taken in the
  order they sit in the episode. A fragment too far from the rest is left
  as it was and the reason is said, because silently returning the whole
  document would be worse than not merging at all, and one far fragment
  no longer stops the ones beside each other from joining.
- **It says how much of itself was retrieved.** The merged span holds the
  text between fragments too, so ``shares`` gives, for every merged
  passage, the part of its bytes that retrieved chunks cover (overlaps
  counted once). ``min_share`` leaves a merge sparser than that as
  fragments, the way the reference merges children into a parent only
  when enough of them were retrieved.

No model is called and nothing is re-embedded: the merged text is read
from the episode this framework already stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from ..core.errors import Gone, InvalidInput, NotFound, SconeError
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
    #: Episodes with a fragment left out of every passage because it sat
    #: too far from the others.
    too_far: int = 0
    #: The chunk a merged passage is reported under, to the share of its
    #: bytes that retrieved chunks cover, to three places.
    shares: dict[int, float] = field(default_factory=dict)
    #: Clusters left as fragments because retrieved chunks covered less of
    #: the merged span than ``min_share``.
    too_sparse: int = 0
    #: Episodes with fragments to join that were not read because the
    #: episode budget was spent first. Nobody looked at them.
    not_read: int = 0
    #: Episodes confirmed absent. Their fragments are dropped: the text
    #: they quote has been deleted, and serving it would hand back content
    #: this space no longer holds.
    gone: int = 0
    #: Episodes the store would not answer for, which is not a finding
    #: that they are absent. Their fragments stand unjoined.
    unread: int = 0
    max_merged: int = MAX_MERGED
    min_share: float = 0.0
    why: str = ""

    def record(self) -> dict[str, object]:
        return {"merged": self.merged, "absorbed": self.absorbed, "too_far": self.too_far,
                "too_sparse": self.too_sparse, "gone": self.gone, "unread": self.unread,
                "not_read": self.not_read, "why": self.why,
                "from_chunks": {str(k): list(v) for k, v in self.from_chunks.items()},
                "shares": {str(k): v for k, v in self.shares.items()},
                "rules": {"max_merged": self.max_merged, "min_share": self.min_share, "measured": False},
                "items": [item.model_dump() for item in self.items]}


async def merge_neighbours(
    engine: "MemoryEngine",
    space: str,
    items: Sequence[RecallItem],
    *,
    max_merged: int = MAX_MERGED,
    min_leaves: int = MIN_LEAVES,
    episodes: int = MAX_EPISODES,
    min_share: float = 0.0,
) -> Merged:
    """Join neighbouring chunks of one episode into the passages holding them."""
    check_space(space)
    if not 1 <= max_merged <= 1_000_000:
        raise InvalidInput(f"max_merged must be from 1 to 1000000, not {max_merged}")
    if min_leaves < 2:
        raise InvalidInput(f"merging takes at least two chunks, not {min_leaves}")
    if isinstance(min_share, bool) or not isinstance(min_share, (int, float)) or not 0 <= min_share <= 1:
        raise InvalidInput(f"min_share is the share of a merged passage retrieved, from 0 to 1, not {min_share!r}")

    together: dict[int, list[RecallItem]] = {}
    for item in items:
        together.setdefault(item.episode_id, []).append(item)

    # Decided per episode, emitted in the caller's order. A ranked list is
    # the caller's data: walking it by episode and appending group by group
    # would rearrange it, which is a change nobody asked for and nothing
    # reports. A merged passage takes the place of its best fragment.
    joined: dict[int, RecallItem] = {}
    dropped: set[int] = set()
    from_chunks: dict[int, tuple[int, ...]] = {}
    shares: dict[int, float] = {}
    merged = absorbed = far = sparse = vanished = unread = unbudgeted = 0
    read = 0
    for episode_id, group in together.items():
        if len(group) < min_leaves:
            continue
        clusters = _clusters(group, max_merged)
        if any(len(cluster) < min_leaves for cluster in clusters):
            far += 1
        clusters = [cluster for cluster in clusters if len(cluster) >= min_leaves]
        if not clusters:
            continue
        if read >= episodes:
            unbudgeted += 1
            continue
        read += 1
        try:
            episode = await engine.episode(space, episode_id)
        except (Gone, NotFound):
            # Confirmed absent, which is a different fact from a store that
            # would not answer. The episode's text is deleted, so fragments
            # quoting it are deleted too: returning them would serve
            # content this space no longer holds.
            vanished += 1
            dropped.update(item.chunk_id for item in group)
            continue
        except SconeError:
            # Not confirmed anything. The fragments are a worse answer than
            # the whole passage and a better one than nothing, so they
            # stand -- and the reason says the merge did not happen rather
            # than implying there was nothing to do.
            unread += 1
            continue
        except Exception:
            unread += 1
            continue
        for cluster in clusters:
            first, last = cluster[0].start, max(i.end for i in cluster)
            share = round(_covered(cluster) / (last - first), 3)
            if share < min_share:
                sparse += 1
                continue
            text = _span(episode.content, first, last)
            if not text.strip():
                unread += 1
                break
            best = max(cluster, key=lambda i: i.score)
            joined[best.chunk_id] = best.model_copy(update={
                "text": text, "start": first, "end": last,
                "first_line": None, "last_line": None, "declaration": None})
            dropped.update(item.chunk_id for item in cluster if item.chunk_id != best.chunk_id)
            from_chunks[best.chunk_id] = tuple(sorted(i.chunk_id for i in cluster))
            shares[best.chunk_id] = share
            merged += 1
            absorbed += len(cluster)

    kept = [joined.get(item.chunk_id, item) for item in items
            if item.chunk_id not in dropped or item.chunk_id in joined]

    why = (f"{merged} passage(s) joined from {absorbed} chunk(s)" if merged
           else "nothing to join: no episode returned neighbouring chunks")
    if far:
        why += (f"; {far} episode(s) had fragments too far apart to be one passage "
                f"(over {max_merged} bytes), and those were left as they were")
    if sparse:
        why += (f"; {sparse} passage(s) would have been under {min_share} retrieved and were left "
                f"as fragments")
    if unbudgeted:
        why += (f"; {unbudgeted} episode(s) past the budget of {episodes} read were not joined -- "
                f"nobody looked at them")
    if vanished:
        why += (f"; {vanished} episode(s) are no longer there and their fragments were dropped "
                f"rather than served from text this space has deleted")
    if unread:
        why += (f"; {unread} episode(s) could not be read, so their fragments stand unjoined -- "
                f"that is a failure to merge, not a finding that there was nothing to merge")
    return Merged(items=tuple(kept), merged=merged, absorbed=absorbed, from_chunks=from_chunks, too_far=far,
                  shares=shares, too_sparse=sparse, gone=vanished, unread=unread, not_read=unbudgeted,
                  max_merged=max_merged, min_share=float(min_share), why=why)


def _clusters(group: Sequence[RecallItem], max_merged: int) -> list[list[RecallItem]]:
    """An episode's fragments in the order they sit, cut into runs that each
    span at most ``max_merged`` bytes from the first start to the last end."""
    clusters: list[list[RecallItem]] = []
    for item in sorted(group, key=lambda i: (i.start, i.end)):
        # Every fragment already in the run ends within the cap of its start,
        # so this one's own end is the only one that can break it.
        if clusters and item.end - clusters[-1][0].start <= max_merged:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def _covered(cluster: Sequence[RecallItem]) -> int:
    """Bytes the fragments cover, overlaps counted once."""
    total, reach = 0, None
    for item in sorted(cluster, key=lambda i: i.start):
        begin = item.start if reach is None else max(item.start, reach)
        if item.end > begin:
            total += item.end - begin
        reach = item.end if reach is None else max(reach, item.end)
    return total


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
