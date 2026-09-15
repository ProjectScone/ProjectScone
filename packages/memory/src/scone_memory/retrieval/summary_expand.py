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
  ``unresolved``. None of them is served. A node is found by the key its
  build stored it under, one read, never a walk over the space.
- **The phrases hold.** A cited chunk the recall's ``require`` or
  ``exclude`` would have dropped is dropped here too and counted, and a
  summary that lost any stands beside what was left, even in ``replace``.
- **A document that went, or changed, is refused with a reason.** The
  summarized episode confirmed forgotten serves nothing: forgetting a
  document leaves its summaries, and the chunks they cite are deleted
  text. A document whose content hash no longer matches the one the
  summary was written from is refused too. A document that could not be
  read is refused as ``unread``, and one never stored here as
  ``source_unknown``; neither is a finding that it is gone.
  A refused summary stands as it was.
- **Capped, and the cap says when it cut.** ``max_chunks`` bounds the
  chunks added across the answer; a summary the cap cut is kept, followed
  by what fit, even in ``replace`` mode, since part of what it cites does
  not stand for all of it. A summary the cap left no room for is not
  counted as expanded. Reads of citation accounts and of nodes are budgeted
  (``MAX_READS``, a failed read counts), and a summary past the budget
  stands as it was and is counted in ``not_read``.
- **Provenance travels.** Every chunk added carries ``via_summary``: the
  summary's episode and chunk, its level and index, the document it
  summarizes, the mode, and where in that chunk's own text the quotes it
  rests on sit. Offsets, not the quotes again: a second copy of a
  passage's text is a surface withholding would not scan. Its lines and
  declaration are worked out as a direct hit's are.

A cited chunk already in the answer is not repeated. No model is called
and nothing is re-embedded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional, Sequence, cast

from ..core.errors import Gone, InvalidInput, NotFound, SconeError
from ..core.models import Chunk, Episode, RecallItem
from .phrases import Phrases, checked_phrases
from .recall import placed_in_file
from .summary_tree import StoredSummary, _hash, stored_summary, summary_key

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

ExpandMode = Literal["replace", "follow"]
EXPAND_MODES: tuple[ExpandMode, ...] = ("replace", "follow")
#: Chunks an expansion adds across one answer, unless the caller says otherwise.
DEFAULT_MAX_CHUNKS = 20
#: The most a caller may ask expansion to add.
MAX_CHUNKS = 200
#: Citation accounts and nodes read in one call. A read that fails counts.
MAX_READS = 200

_REASONS = {
    "malformed": "their metadata does not say what they summarize",
    "source_gone": "the document they summarize is forgotten, so none of its chunks are served",
    "source_unknown": "the document they name is not stored here and never was, as in a note carried from another store",
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
    #: Cited chunks the recall's phrases dropped: one lacking a required phrase, one holding an excluded one.
    dropped_required: int = 0
    dropped_excluded: int = 0
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
                "dropped_required": self.dropped_required, "dropped_excluded": self.dropped_excluded,
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
    #: Nodes looked up by (fan_in, level, index), None when no node is stored there.
    nodes: dict[tuple[int, int, int], Optional[StoredSummary]] = field(default_factory=dict)


@dataclass
class _Walk:
    """One call's reads and tallies, shared by every summary it resolves."""

    engine: "MemoryEngine"
    space: str
    reads: int = 0
    missing: int = 0
    unquoted: int = 0
    unresolved: int = 0
    #: A document read, or the reason it was refused.
    sources: dict[int, _Source | str] = field(default_factory=dict)
    accounts: dict[str, Optional[list[dict[str, object]]]] = field(default_factory=dict)

    async def source(self, episode_id: int) -> _Source:
        if episode_id in self.sources:
            known = self.sources[episode_id]
            if isinstance(known, str):
                raise _Refused(known)
            return known
        try:
            episode = await self.engine.episode(self.space, episode_id)
            chunks = await self.engine.documents.chunks_of(self.space, episode_id)
        except NotFound as absent:
            # Gone is the NotFound a tombstone answers with; a bare NotFound is an id that never meant anything here.
            reason = "source_gone" if isinstance(absent, Gone) else "source_unknown"
            self.sources[episode_id] = reason
            raise _Refused(reason)
        except SconeError:
            raise _Refused("unread")
        found = _Source(episode, {chunk.chunk_id: chunk for chunk in chunks}, _hash(episode.content))
        self.sources[episode_id] = found
        return found

    def spend(self) -> None:
        """Count one read, or stop the summary being resolved when the budget is spent."""
        if self.reads >= MAX_READS:
            raise _Unbudgeted()
        self.reads += 1

    async def account(self, attachment_id: str) -> Optional[list[dict[str, object]]]:
        """A node's sentences with their citations, or None when they cannot be read."""
        if attachment_id in self.accounts:
            return self.accounts[attachment_id]
        self.spend()
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

    async def node(self, summary_of: int, source: _Source, fan_in: int, key: tuple[int, int]) -> Optional[StoredSummary]:
        """The node stored at ``key`` of this document's tree, by the key its build kept it under: one
        read, where listing the document's summaries walks every episode in the space."""
        place = (fan_in, *key)
        if place in source.nodes:
            return source.nodes[place]
        self.spend()  # a lookup is a read, whether or not a node is there
        found: Optional[StoredSummary] = None
        try:
            found = stored_summary(await self.engine.episode_by_key(
                self.space, summary_key(summary_of, source.content_hash, fan_in, *key)))
        except SconeError:
            found = None
        source.nodes[place] = found
        return found

    async def cited(self, summary_of: int, source: _Source, fan_in: int, level: int, sentences: list[dict[str, object]],
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
                # A node is followed only below the level citing it, so a
                # forged account cannot send the walk round in a circle,
                # and only when it was written from this same content.
                if key[0] >= level:
                    self.unresolved += 1
                    continue
                node = await self.node(summary_of, source, fan_in, key)
                if node is None or node.content_hash != source.content_hash or node.text[start:end] != quote:
                    self.unresolved += 1
                    continue
                below.setdefault(key, []).append((start, end))
        for key, landed in below.items():
            found = source.nodes[(fan_in, *key)]
            assert found is not None, "only a node found and checked above is followed"
            lower = await self.account(found.detail)
            if lower is None:
                self.unresolved += len(landed)
                continue
            await self.cited(summary_of, source, fan_in, key[0], lower, landed, into)


def _placed(meta: dict[str, str]) -> tuple[int, int, int, int, str, str]:
    try:
        return (int(meta["summary_of"]), int(meta["summary_level"]), int(meta["summary_index"]), int(meta["summary_fan_in"]),
                meta["summary_detail"], meta["summary_content_hash"])
    except (KeyError, ValueError):
        raise _Refused("malformed")


async def expand_summaries(engine: "MemoryEngine", space: str, items: Sequence[RecallItem], *,
                           mode: ExpandMode = "follow", max_chunks: int = DEFAULT_MAX_CHUNKS,
                           require: Sequence[str] = (), exclude: Sequence[str] = ()) -> Expanded:
    """Each summary among ``items`` followed by (``follow``) or replaced by
    (``replace``) the chunks its citations rest on, in document order, and
    only those holding every ``require`` phrase and no ``exclude`` one."""
    from ..memory.engine import check_space

    check_space(space)
    check_expansion(mode, max_chunks)
    required, excluded = checked_phrases(require, exclude)
    phrases = Phrases(required, excluded) if required or excluded else None
    dropped = {"required": 0, "excluded": 0}
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
    # Summaries the cap cut after some of what they cite fit, and those it left no room for at all.
    cut_partly = cut_whole = 0
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
            summary_of, level, index, fan_in, detail, written_from = _placed(item.metadata)
            source = await walk.source(summary_of)
            if written_from != source.content_hash:
                raise _Refused("content_changed")
            sentences = await walk.account(detail)
            if sentences is None:
                raise _Refused("detail_unreadable")
            await walk.cited(summary_of, source, fan_in, level, sentences, None, quotes)
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
        lost = 0
        if phrases is not None:
            # Before the cap, so a chunk that could not be served takes no place.
            passing = []
            for chunk in fresh:
                fits, rule = phrases.passes(chunk.text)
                if fits:
                    passing.append(chunk)
                else:
                    dropped[rule] += 1
                    lost += 1
            fresh = passing
        fresh.sort(key=lambda chunk: chunk.ordinal)
        room = max_chunks - added
        taken, left = fresh[:room], len(fresh) - min(len(fresh), room)
        if left:
            capped += left
            cut.append(item.episode_id)
            if taken:
                cut_partly += 1
            else:
                cut_whole += 1
        # Part of what a summary cites does not stand for all of it, so only a whole one is replaced.
        whole = not left and not lost
        if mode == "follow" or not whole:
            kept.append(item)
        else:
            handled[item.episode_id] = "replaced"
        episode = source.episode
        for chunk in taken:
            present.add(chunk.chunk_id)
            first_line, last_line, declaration = placed_in_file(episode, chunk)
            kept.append(RecallItem(
                chunk_id=chunk.chunk_id, episode_id=chunk.episode_id, text=chunk.text, score=item.score,
                created_at=chunk.created_at, source=episode.source, tags=episode.tags,
                metadata=dict(episode.metadata), start=chunk.start, end=chunk.end,
                first_line=first_line, last_line=last_line, declaration=declaration,
                via_summary={"episode_id": item.episode_id, "chunk_id": item.chunk_id, "level": level, "index": index,
                             "summary_of": summary_of, "mode": mode,
                             "cited": [list(span) for span in dict.fromkeys(quotes[chunk.chunk_id])]}))
        added += len(taken)
        if taken or whole:
            # One that added nothing because the cap or the phrases left nothing was not expanded.
            expanded += 1
            by_summary[item.episode_id] = tuple(chunk.chunk_id for chunk in taken)

    summaries = len(handled)
    why = f"expanded {expanded} of {summaries} summary hit(s) to {added} cited chunk(s)" if summaries \
        else "no summary among the items"
    if repeated:
        why += f"; {repeated} cited chunk(s) were already in the answer and are not repeated"
    if dropped["required"] or dropped["excluded"]:
        why += (f"; {dropped['required'] + dropped['excluded']} cited chunk(s) were dropped by the phrases "
                f"({dropped['required']} lacking a required one, {dropped['excluded']} holding an excluded one) and "
                f"not served, so a summary that lost any stands beside what was left")
    if capped:
        why += f"; the cap of {max_chunks} chunk(s) left {capped} cited chunk(s) out"
        if cut_partly:
            why += f", so {cut_partly} summary hit(s) stand beside the part of what they cite that fit"
        if cut_whole:
            why += f"; {cut_whole} summary hit(s) stand as they were because the cap left no room for what they cite"
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
                    dropped_required=dropped["required"], dropped_excluded=dropped["excluded"],
                    refused=tuple(refused), not_read=unbudgeted, reads=walk.reads, by_summary=by_summary, why=why)
