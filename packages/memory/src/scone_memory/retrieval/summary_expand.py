"""A summary hit, answered with the chunks it cites.

A broad question finds a summary: the stored summary tree
(``summary_tree``) puts one beside the chunks, and the ordinary lanes
return it as a passage. A summary is the model's word, though, and a
reader who has to quote the document wants the document's own text. The
leading RAG framework's document summary index retrieves over summaries
and hands back the nodes under the chosen one -- every node of the
document. Ours hands back the chunks the summary *cites*, because every
sentence of a stored summary carries a quote this framework found in
what it was written from, and the account of those citations is kept.

Resolution runs downward. A level-one summary cites chunks. A summary
above cites a node of the level below (and, where a lone remainder was
carried up, a chunk), with a quote at an offset of that node's text; the
quote lands in one or two of that node's sentences, and only *their*
citations are followed, so the root of a long document expands to the
chunks its own sentences rest on, not to the document.

The rules, each with a test:

- **Only what is cited.** A chunk under a summary that no citation it
  rests on names is not served; ``covers`` is the span, not the evidence.
- **Checked the way it was made.** A citation is followed only when the
  text it names holds its quote at its offsets. A chunk that is not one of
  the summarized document's chunks is ``missing``; one that does not hold
  the quote is ``unquoted``; a node that is not stored below is
  ``unresolved``. None of them is served.
- **A document that went, or changed, is refused with a reason.** The
  summarized episode confirmed forgotten serves nothing: forgetting a
  document leaves its summaries, and the chunks they cite are deleted
  text. A document whose content hash no longer matches the one the
  summary was written from is refused too. A document that could not be
  read is refused as ``unread``, which is not a finding that it is gone.
  A refused summary stands as it was.
- **Capped, and the cap says when it cut.** ``max_chunks`` bounds the
  chunks added across the answer; a summary the cap cut is kept, followed
  by what fit, even in ``replace`` mode, since part of what it cites does
  not stand for all of it. Reads of citation accounts are budgeted
  (``MAX_READS``, a failed read counts), and a summary past the budget
  stands as it was and is counted in ``not_read``.
- **Provenance travels.** Every chunk added carries ``via_summary``: the
  summary's episode and chunk, its level and index, the document it
  summarizes, the mode, and where in that chunk's own text the quotes it
  rests on sit. Offsets, not the quotes again: a second copy of a
  passage's text is a surface withholding would not scan.

A cited chunk already in the answer is not repeated. No model is called
and nothing is re-embedded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional, Sequence, cast

from ..core.errors import Gone, InvalidInput, NotFound, SconeError
from ..core.models import Chunk, Episode, RecallItem
from .summary_tree import StoredSummary, _hash, stored_summaries

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

ExpandMode = Literal["replace", "follow"]
EXPAND_MODES: tuple[ExpandMode, ...] = ("replace", "follow")
#: Chunks an expansion adds across one answer, unless the caller says otherwise.
DEFAULT_MAX_CHUNKS = 20
#: The most a caller may ask expansion to add.
MAX_CHUNKS = 200
#: Citation accounts read in one call. A read that fails counts.
MAX_READS = 200

_REASONS = {
    "malformed": "their metadata does not say what they summarize",
    "source_gone": "the document they summarize is forgotten, so none of its chunks are served",
    "unread": "their document could not be read, which is not a finding that it is gone",
    "content_changed": "their document's content hash no longer matches the one they were written from",
    "detail_unreadable": "their account of citations could not be read",
    "nothing_cited": "none of their citations reached a chunk of the document holding its quote",
}


def check_expansion(mode: object, max_chunks: object) -> None:
    """Refuse a mode or a cap that is not one, before anything is searched."""
    if not isinstance(mode, str) or mode not in EXPAND_MODES:
        raise InvalidInput(f"expand_summaries is {' or '.join(EXPAND_MODES)}, not {mode!r}")
    if type(max_chunks) is not int or not 1 <= max_chunks <= MAX_CHUNKS:
        raise InvalidInput(f"expand_max_chunks is a whole number from 1 to {MAX_CHUNKS}, not {max_chunks!r}")


@dataclass(frozen=True)
class Expanded:
    """What was expanded, what was refused and why, and where the cap cut."""

    items: tuple[RecallItem, ...] = ()
    mode: ExpandMode = "follow"
    max_chunks: int = DEFAULT_MAX_CHUNKS
    #: Distinct summaries among the items.
    summaries: int = 0
    #: Summaries whose citations were followed to chunks.
    expanded: int = 0
    chunks_added: int = 0
    #: Cited chunks not added because the answer already held them.
    already_present: int = 0
    #: Cited chunks the cap left out, and the summaries it cut.
    capped: int = 0
    cut: tuple[int, ...] = ()
    #: Citations naming a chunk that is not one of the summarized document's.
    missing: int = 0
    #: Citations naming a chunk of the document that does not hold the quote at its offsets.
    unquoted: int = 0
    #: Citations naming a node that is not stored below, or nothing this reads.
    unresolved: int = 0
    #: Summaries refused, each with its episode and a reason.
    refused: tuple[dict[str, object], ...] = ()
    #: Summaries left as they were because the read budget was spent.
    not_read: int = 0
    reads: int = 0
    #: Summary episode to the chunk ids it added.
    by_summary: dict[int, tuple[int, ...]] = field(default_factory=dict)
    why: str = ""

    def record(self) -> dict[str, object]:
        return {"mode": self.mode, "max_chunks": self.max_chunks, "summaries": self.summaries,
                "expanded": self.expanded, "chunks_added": self.chunks_added,
                "already_present": self.already_present, "capped": self.capped, "cut": list(self.cut),
                "missing": self.missing, "unquoted": self.unquoted, "unresolved": self.unresolved,
                "refused": [dict(one) for one in self.refused], "not_read": self.not_read, "reads": self.reads,
                "by_summary": {str(k): list(v) for k, v in self.by_summary.items()}, "why": self.why}


class _Refused(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Unbudgeted(Exception):
    pass


@dataclass
class _Source:
    episode: Episode
    chunks: dict[int, Chunk]
    content_hash: str
    nodes: Optional[dict[tuple[int, int], StoredSummary]] = None


@dataclass
class _Walk:
    """One call's reads and tallies, shared by every summary it resolves."""

    engine: "MemoryEngine"
    space: str
    reads: int = 0
    missing: int = 0
    unquoted: int = 0
    unresolved: int = 0
    sources: dict[int, Optional[_Source]] = field(default_factory=dict)
    accounts: dict[str, Optional[list[dict[str, object]]]] = field(default_factory=dict)

    async def source(self, episode_id: int) -> _Source:
        if episode_id in self.sources:
            known = self.sources[episode_id]
            if known is None:
                raise _Refused("source_gone")
            return known
        try:
            episode = await self.engine.episode(self.space, episode_id)
            chunks = await self.engine.documents.chunks_of(self.space, episode_id)
        except (Gone, NotFound):
            self.sources[episode_id] = None
            raise _Refused("source_gone")
        except SconeError:
            raise _Refused("unread")
        found = _Source(episode, {chunk.chunk_id: chunk for chunk in chunks}, _hash(episode.content))
        self.sources[episode_id] = found
        return found

    async def account(self, attachment_id: str) -> Optional[list[dict[str, object]]]:
        """A node's sentences with their citations, or None when they cannot be read."""
        if attachment_id in self.accounts:
            return self.accounts[attachment_id]
        if self.reads >= MAX_READS:
            raise _Unbudgeted()
        self.reads += 1
        sentences: Optional[list[dict[str, object]]] = None
        try:
            _, raw = await self.engine.attachment(self.space, attachment_id)
            body = json.loads(raw)
            listed = body.get("sentences") if isinstance(body, dict) else None
            if isinstance(listed, list) and all(isinstance(one, dict) for one in listed):
                sentences = cast(list[dict[str, object]], listed)
        except (SconeError, ValueError):
            sentences = None
        self.accounts[attachment_id] = sentences
        return sentences

    async def nodes(self, summary_of: int, source: _Source) -> dict[tuple[int, int], StoredSummary]:
        if source.nodes is None:
            source.nodes = {(one.level, one.index): one for one in await stored_summaries(self.engine, self.space, summary_of)}
        return source.nodes

    async def cited(self, summary_of: int, source: _Source, level: int, sentences: list[dict[str, object]],
                    spans: Optional[list[tuple[int, int]]], into: dict[int, list[tuple[int, int]]]) -> None:
        """Follow the citations of the sentences ``spans`` land in (all, with
        None) down to chunks, collecting where each chunk holds its quotes in ``into``."""
        below: dict[tuple[int, int], list[tuple[int, int]]] = {}
        offset = 0
        for sentence in sentences:
            text = sentence.get("text")
            length = len(text) if isinstance(text, str) else 0
            begin, offset = offset, offset + length + 1
            if spans is not None and not any(start < begin + length and end > begin for start, end in spans):
                continue
            citations = sentence.get("citations")
            for citation in citations if isinstance(citations, list) else ():
                passage = citation.get("passage") if isinstance(citation, dict) else None
                quote = citation.get("quote") if isinstance(citation, dict) else None
                start = citation.get("start") if isinstance(citation, dict) else None
                end = citation.get("end") if isinstance(citation, dict) else None
                if not (isinstance(passage, str) and isinstance(quote, str) and type(start) is int and type(end) is int):
                    self.unresolved += 1
                    continue
                kind, _, rest = passage.partition(":")
                if kind == "chunk" and rest.isdigit():
                    chunk = source.chunks.get(int(rest))
                    if chunk is None:
                        self.missing += 1
                    elif chunk.text[start:end] != quote:
                        self.unquoted += 1
                    else:
                        into.setdefault(chunk.chunk_id, []).append((start, end))
                    continue
                named = rest.split(":")
                if kind != "node" or len(named) != 2 or not all(part.isdigit() for part in named):
                    self.unresolved += 1
                    continue
                key = (int(named[0]), int(named[1]))
                node = (await self.nodes(summary_of, source)).get(key)
                # A node is followed only below the level citing it, so a
                # forged account cannot send the walk round in a circle,
                # and only when it was written from this same content.
                if (node is None or key[0] >= level or node.content_hash != source.content_hash
                        or node.text[start:end] != quote):
                    self.unresolved += 1
                    continue
                below.setdefault(key, []).append((start, end))
        for key, landed in below.items():
            node = (await self.nodes(summary_of, source))[key]
            lower = await self.account(node.detail)
            if lower is None:
                self.unresolved += len(landed)
                continue
            await self.cited(summary_of, source, key[0], lower, landed, into)


def _placed(meta: dict[str, str]) -> tuple[int, int, int, str, str]:
    try:
        return (int(meta["summary_of"]), int(meta["summary_level"]), int(meta["summary_index"]),
                meta["summary_detail"], meta["summary_content_hash"])
    except (KeyError, ValueError):
        raise _Refused("malformed")


async def expand_summaries(engine: "MemoryEngine", space: str, items: Sequence[RecallItem], *,
                           mode: ExpandMode = "follow", max_chunks: int = DEFAULT_MAX_CHUNKS) -> Expanded:
    """Each summary among ``items`` followed by (``follow``) or replaced by
    (``replace``) the chunks its citations rest on, in document order."""
    from ..memory.engine import check_space

    check_space(space)
    check_expansion(mode, max_chunks)
    walk = _Walk(engine, space)
    present = {item.chunk_id for item in items}
    # A summary's outcome, by episode, so a summary recalled as several
    # chunks is resolved once: "replaced" drops its other chunks with it.
    handled: dict[int, str] = {}
    kept: list[RecallItem] = []
    refused: list[dict[str, object]] = []
    cut: list[int] = []
    by_summary: dict[int, tuple[int, ...]] = {}
    expanded = added = repeated = capped = unbudgeted = 0
    for item in items:
        if "summary_of" not in item.metadata:
            kept.append(item)
            continue
        if item.episode_id in handled:
            if handled[item.episode_id] != "replaced":
                kept.append(item)
            continue
        handled[item.episode_id] = "stood"
        quotes: dict[int, list[tuple[int, int]]] = {}
        try:
            summary_of, level, index, detail, written_from = _placed(item.metadata)
            source = await walk.source(summary_of)
            if written_from != source.content_hash:
                raise _Refused("content_changed")
            sentences = await walk.account(detail)
            if sentences is None:
                raise _Refused("detail_unreadable")
            await walk.cited(summary_of, source, level, sentences, None, quotes)
            if not quotes:
                raise _Refused("nothing_cited")
        except (_Refused, _Unbudgeted) as stopped:
            if isinstance(stopped, _Refused):
                refused.append({"episode_id": item.episode_id, "reason": stopped.reason})
            else:
                unbudgeted += 1
            kept.append(item)  # a summary not resolved stands as it was
            continue
        fresh = [source.chunks[chunk_id] for chunk_id in quotes if chunk_id not in present]
        repeated += len(quotes) - len(fresh)
        fresh.sort(key=lambda chunk: chunk.ordinal)
        room = max_chunks - added
        taken, left = fresh[:room], len(fresh) - min(len(fresh), room)
        if left:
            capped += left
            cut.append(item.episode_id)
        if mode == "follow" or left:
            kept.append(item)
        else:
            handled[item.episode_id] = "replaced"
        episode = source.episode
        for chunk in taken:
            present.add(chunk.chunk_id)
            kept.append(RecallItem(
                chunk_id=chunk.chunk_id, episode_id=chunk.episode_id, text=chunk.text, score=item.score,
                created_at=chunk.created_at, source=episode.source, tags=episode.tags,
                metadata=dict(episode.metadata), start=chunk.start, end=chunk.end,
                via_summary={"episode_id": item.episode_id, "chunk_id": item.chunk_id, "level": level, "index": index,
                             "summary_of": summary_of, "mode": mode,
                             "cited": [list(span) for span in dict.fromkeys(quotes[chunk.chunk_id])]}))
        added += len(taken)
        expanded += 1
        by_summary[item.episode_id] = tuple(chunk.chunk_id for chunk in taken)

    summaries = len(handled)
    why = f"expanded {expanded} of {summaries} summary hit(s) to {added} cited chunk(s)" if summaries \
        else "no summary among the items"
    if repeated:
        why += f"; {repeated} cited chunk(s) were already in the answer and are not repeated"
    if capped:
        why += (f"; the cap of {max_chunks} chunk(s) left {capped} cited chunk(s) out, so {len(cut)} summary hit(s) "
                f"stand beside the part of what they cite that fit")
    for reason, said in _REASONS.items():
        count = sum(1 for one in refused if one["reason"] == reason)
        if count:
            why += f"; {count} summary hit(s) were refused and stand as they were: {said}"
    if walk.missing:
        why += f"; {walk.missing} citation(s) named a chunk not among the document's chunks and were not served"
    if walk.unquoted:
        why += f"; {walk.unquoted} citation(s) named a chunk that does not hold the quote and were not served"
    if walk.unresolved:
        why += f"; {walk.unresolved} citation(s) could not be followed to a stored summary below"
    if unbudgeted:
        why += (f"; {unbudgeted} summary hit(s) stand as they were because the budget of {MAX_READS} read(s) was "
                f"spent before they were resolved")
    return Expanded(items=tuple(kept), mode=mode, max_chunks=max_chunks, summaries=summaries, expanded=expanded,
                    chunks_added=added, already_present=repeated, capped=capped, cut=tuple(cut),
                    missing=walk.missing, unquoted=walk.unquoted, unresolved=walk.unresolved,
                    refused=tuple(refused), not_read=unbudgeted, reads=walk.reads, by_summary=by_summary, why=why)
