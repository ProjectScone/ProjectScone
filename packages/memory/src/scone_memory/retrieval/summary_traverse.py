"""Recall by descending stored summary trees, from the top of each document to its chunks.

A broad question over several long documents is badly served by the
passages that score best on their own: the section that answers it is a
run of chunks, and only one of them may say the words the question does.
The leading framework's tree index answers it by descent: start at the
root summaries, keep the children most like the question, and go down
until the leaves. This is that descent over the summary trees
``summary_tree`` stores, with no model at query time.

The walk:

- **Where it starts.** The documents in scope are those named in
  ``episode_ids``, or every document with a stored tree, kept only when
  the scope's kind, source prefix, dates and tags fit the document. Each
  starts at the highest level of summaries written from its present
  content. The trees are found by one walk over the space's episodes, as
  listing a document's summaries is, and episodes the listing left out (a
  store's cap on what it lists, or episodes stored meanwhile) are counted in
  ``unlisted``. With no ``episode_ids``, a tree whose document the listing
  did not return (a document forgotten, or replaced, leaves its summaries
  behind) cannot be fitted to the scope and is not read: it is counted in
  ``unlisted_trees``, so trees left behind cost a call nothing but the count;
  more than ``MAX_DOCUMENTS`` in scope
  is refused rather than cut to some of them, and so is a step with more
  than ``MAX_CANDIDATES`` candidates (a wide unjoined top, or a forged account).
- **How it descends.** Every candidate at a step is scored against the
  question -- by the cosine of the question's embedding with the vectors
  already stored for the node's chunks (its best chunk), and with ``text``
  also by BM25 over the step's candidates, fused by rank -- and the
  ``branching`` best across all documents are kept, as the reference's
  retriever pools the children of every node it kept before choosing. A
  kept node's children are what its account says it was written from: the
  nodes below it and the chunks. The chunks a first-level summary names are
  leaves; a chunk a higher summary names was carried up beside its nodes
  unsummarized, so it is a candidate at the next step with them, and a leaf
  only if that step keeps it. The leaves reached come back branch first --
  in the order of the ranks down their path, so every chunk under the
  summary kept first at a step comes before any under the next, whatever
  depth each tree reaches -- and within a branch in their own ranking, the
  same scoring as a step's; the first ``limit`` are returned, each with the
  path of summaries that led there. Branch first, because a chunk's own
  words are the signal a broad question defeats: ranked by those alone,
  chunks of a lower branch displace the section the descent chose. ``score`` is the
  item's place in that order (``1 / (1 + place)``), ``similarity`` its own
  cosine. ``max_depth`` bounds the steps.
- **What it never returns.** A summary written from other content than its
  document's (the content hash does not match) is not a candidate and is
  counted in ``stale``. A chunk named below a summary that is not one of
  the document's is ``missing``; a node named that is not stored from this
  content, or whose account cannot be read, is ``unresolved``. A document
  named and forgotten is refused (``source_gone``), and so is one that could
  not be read (``unread``) or was never stored (``source_unknown``), and one
  the listing left out is refused as ``unlisted``. The walk awaits reads, and
  a document or a summary can be forgotten during any of them, so the chunks
  about to be returned, and the chunks of every summary on their paths, are
  read again in one read after the last one, with nothing awaited between
  that read and the answer; a document whose chunks moved is read again for
  its reason and serves none of them, and one with a summary on a path gone
  or changed is refused as ``tree_changed``. A document refused there is not
  counted among those descended, nor its chunks among the leaves.
- **The bounds say when they cut.** ``limit`` leaves out leaves reached
  (``cut_by_limit``), ``max_depth`` leaves candidates reached but not scored
  (``cut_by_depth``), and chunks under no summary a descent starts from
  (a group the model wrote nothing about, a remainder left at an unjoined
  top) are counted in ``uncovered``, since no descent reaches them.
- **Its cost is counted.** One embedding call for the question; a step whose
  candidates do not all have stored vectors (an index that cannot hand them
  back, a vector missing, vectors another embedder wrote) is embedded in one
  call, so every score at a step is on one scale. ``embed_calls`` and
  ``embedded_texts`` say what was spent, and ``vectors`` which steps read
  the index and which were embedded.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from ..core.errors import InvalidInput, SconeError
from ..core.models import Chunk, Episode, RecallItem
from ..core.validation import KINDS, MAX_LIMIT, MAX_QUERY, MAX_SOURCE, normalise_tags, normalise_time
from .episode_scope import episode_fits
from .fusion import RRF_K
from .lexical import Bm25
from .recall import placed_in_file
from .summary_expand import _Refused, _Source, _Walk
from .summary_tree import MAX_LEVELS, StoredSummary, stored_summary

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..memory.engine import MemoryEngine

#: Summaries kept at each step unless the caller says otherwise, and the most a caller may ask for.
DEFAULT_BRANCHING = 2
MAX_BRANCHING = 10
#: Steps down a tree; a stored tree has at most this many levels, so the default never cuts one.
MAX_DEPTH = MAX_LEVELS
DEFAULT_LIMIT = 10
#: Documents with trees one call descends; more in scope is refused, not cut.
MAX_DOCUMENTS = 100
#: Candidates one step scores (the leaves' ranking counts as a step). A stored tree keeps a step far below
#: it -- the kept summaries' children are at most ``branching`` times its fan_in -- so a step past it is a
#: wide unjoined top or a forged account, and is refused rather than scored in part.
MAX_CANDIDATES = 1000

_REASONS = {
    "source_gone": "their document is forgotten",
    "source_unknown": "the document named is not stored here and never was",
    "unread": "their document could not be read, which is not a finding that it is gone",
    "content_changed": "every summary they hold was written from other content than their document's",
    "changed_while_read": "their document's chunks changed while the tree was read",
    "unlisted": "their document was not in the space's listing, so their tree was not seen whole",
    "tree_changed": "a summary on the path to their chunks was forgotten or changed while the tree was read",
}


@dataclass(frozen=True)
class Traversal:
    """The chunks a descent reached, how it got there, and what it refused or cut."""

    items: tuple[RecallItem, ...] = ()
    query: str = ""
    limit: int = DEFAULT_LIMIT
    branching: int = DEFAULT_BRANCHING
    max_depth: int = MAX_DEPTH
    text: bool = False
    #: Documents whose trees were descended, less any refused when the chunks were read again.
    documents: tuple[int, ...] = ()
    #: Documents named that have no summary tree.
    untreed: tuple[int, ...] = ()
    #: Documents with trees the scope's kind, source prefix, dates or tags left out.
    out_of_scope: int = 0
    #: Documents refused, each with its episode and a reason.
    refused: tuple[dict[str, object], ...] = ()
    #: Each step: how many candidates, the summaries kept, and the chunks kept (those carried up beside summaries).
    steps: tuple[dict[str, object], ...] = ()
    #: Distinct chunks reached of the documents not refused, and those the limit left out.
    leaves: int = 0
    cut_by_limit: int = 0
    #: Candidates (summary nodes, or chunks carried up beside them) reached but not scored because the depth ran out.
    cut_by_depth: int = 0
    #: Summaries of the documents in scope written from other content.
    stale: int = 0
    #: Chunks named below a summary that are not the document's.
    missing: int = 0
    #: Nodes named below a summary not stored from this content, or accounts that could not be read.
    unresolved: int = 0
    #: Chunks of the descended documents under no summary a descent starts from.
    uncovered: int = 0
    #: Steps (the leaves' ranking among them) scored by stored vectors, and embedded.
    vectors: dict[str, int] = field(default_factory=lambda: {"index": 0, "embedded": 0})
    embed_calls: int = 0
    embedded_texts: int = 0
    #: Episodes walked to find the trees, and episodes the space holds that the listing did not return.
    walked: int = 0
    unlisted: int = 0
    #: With no episodes named, trees whose document the listing did not return; not read, fitted or descended.
    unlisted_trees: int = 0
    why: str = ""

    def record(self) -> dict[str, object]:
        return {"items": [item.model_dump(mode="json") for item in self.items], "query": self.query, "limit": self.limit,
                "branching": self.branching, "max_depth": self.max_depth, "text": self.text,
                "documents": list(self.documents), "untreed": list(self.untreed), "out_of_scope": self.out_of_scope,
                "refused": [dict(one) for one in self.refused], "steps": [dict(step) for step in self.steps],
                "leaves": self.leaves, "cut_by_limit": self.cut_by_limit, "cut_by_depth": self.cut_by_depth,
                "stale": self.stale, "missing": self.missing, "unresolved": self.unresolved, "uncovered": self.uncovered,
                "vectors": dict(self.vectors), "embed_calls": self.embed_calls, "embedded_texts": self.embedded_texts,
                "walked": self.walked, "unlisted": self.unlisted, "unlisted_trees": self.unlisted_trees, "why": self.why}


@dataclass(frozen=True)
class _Candidate:
    """One thing a step scores: a summary node, or a chunk, with the path of summaries above it and their ranks."""

    document: int
    text: str
    path: tuple[dict[str, object], ...]
    node: Optional[StoredSummary] = None
    chunk: Optional[Chunk] = None
    #: The ranks down the path, and for a chunk kept at a step, its own rank there: the order leaves come back in.
    ranks: tuple[int, ...] = ()
    #: The episodes of the summaries on the path, which the last read confirms.
    through: tuple[int, ...] = ()


@dataclass
class _Scored:
    candidate: _Candidate
    similarity: float
    text_rank: Optional[int]
    score: float


def check_traversal(query: object, limit: object, branching: object, max_depth: object, text: object,
                    episode_ids: object, kind: object, source_prefix: object) -> None:
    """Refuse what cannot be one, before anything is read."""
    if not isinstance(query, str) or not query.strip() or len(query.strip()) > MAX_QUERY:
        raise InvalidInput(f"query must be 1..={MAX_QUERY} chars")
    if type(limit) is not int or not 1 <= limit <= MAX_LIMIT:
        raise InvalidInput(f"limit is a whole number from 1 to {MAX_LIMIT}, not {limit!r}")
    if type(branching) is not int or not 1 <= branching <= MAX_BRANCHING:
        raise InvalidInput(f"branching is a whole number from 1 to {MAX_BRANCHING}, not {branching!r}")
    if type(max_depth) is not int or not 1 <= max_depth <= MAX_DEPTH:
        raise InvalidInput(f"max_depth is a whole number from 1 to {MAX_DEPTH}, not {max_depth!r}")
    if type(text) is not bool:
        raise InvalidInput(f"text is a boolean, not {text!r}")
    if episode_ids is not None and (not isinstance(episode_ids, (list, tuple)) or not episode_ids
                                    or any(type(one) is not int for one in episode_ids)):
        raise InvalidInput(f"episode_ids is a non-empty list of episode ids, not {episode_ids!r}")
    if kind is not None and kind not in KINDS:
        raise InvalidInput(f"kind must be one of {KINDS}, got {kind!r}")
    if source_prefix is not None and (not isinstance(source_prefix, str) or len(source_prefix) > MAX_SOURCE):
        raise InvalidInput(f"source_prefix must be at most {MAX_SOURCE} chars")


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    norms = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(a * b for a, b in zip(left, right)) / norms if norms else 0.0


class _Scorer:
    """Scores one step's candidates at a time, counting what it spends."""

    def __init__(self, engine: "MemoryEngine", space: str, query: str, text: bool) -> None:
        self.engine, self.space, self.query, self.text = engine, space, query, text
        self.embed_calls = self.embedded_texts = 0
        self.vectors = {"index": 0, "embedded": 0}
        self.question: list[float] = []
        #: Each summary note's chunks as first read, so the last read can confirm the summaries on a path.
        self.held: dict[int, list[Chunk]] = {}

    async def chunks_of(self, episode_id: int) -> list[Chunk]:
        if episode_id not in self.held:
            self.held[episode_id] = await self.engine.documents.chunks_of(self.space, episode_id)
        return self.held[episode_id]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_calls += 1
        self.embedded_texts += len(texts)
        return await self.engine.embedder.embed(list(texts))

    async def stored(self, candidates: Sequence[_Candidate]) -> Optional[list[list[list[float]]]]:
        """Each candidate's stored chunk vectors, or None when any candidate lacks one."""
        reader = getattr(self.engine.vectors, "vectors_of", None)
        if not callable(reader) or self.engine.vector_block is not None:
            return None
        owned: list[list[int]] = []
        for candidate in candidates:
            if candidate.node is not None:
                owned.append([chunk.chunk_id for chunk in await self.chunks_of(candidate.node.episode_id)])
            else:
                assert candidate.chunk is not None, "a candidate is a node or a chunk"
                owned.append([candidate.chunk.chunk_id])
        wanted = [chunk_id for ids in owned for chunk_id in ids]
        found = await reader(self.space, wanted)
        if not all(ids for ids in owned) or any(chunk_id not in found for chunk_id in wanted):
            return None
        return [[found[chunk_id] for chunk_id in ids] for ids in owned]

    async def rank(self, candidates: Sequence[_Candidate]) -> list[_Scored]:
        """The candidates best first: by cosine, or with ``text`` by cosine and BM25 fused by rank; ties keep their order."""
        if not candidates:
            return []
        if len(candidates) > MAX_CANDIDATES:
            raise InvalidInput(f"a step of the descent has {len(candidates)} candidates and one step scores at most "
                               f"{MAX_CANDIDATES}; name fewer documents or narrow the scope")
        if not self.question:
            from ..core.embedding import embed_queries
            self.embed_calls += 1
            self.embedded_texts += 1
            [self.question] = await embed_queries(self.engine.embedder, [self.query])
        stored = await self.stored(candidates)
        if stored is not None:
            self.vectors["index"] += 1
            similarities = [max(_cosine(self.question, vector) for vector in vectors) for vectors in stored]
        else:
            self.vectors["embedded"] += 1
            similarities = [_cosine(self.question, vector) for vector in await self.embed([c.text for c in candidates])]
        text_ranks: dict[int, int] = {}
        scores = {position: similarities[position] for position in range(len(candidates))}
        if self.text:
            lexical = Bm25()
            for position, candidate in enumerate(candidates):
                lexical.add(position, candidate.text)
            text_ranks = {position: rank for rank, (position, _) in enumerate(lexical.search(self.query, len(candidates)), start=1)}
            # Fused by rank, as the lanes are. Equal cosines share a rank, so
            # the order they happened to be listed in is not a vote; a
            # candidate BM25 does not match has no text rank to add.
            scores = {position: 1 / (RRF_K + 1 + sum(1 for other in similarities if other > similarities[position]))
                      + (1 / (RRF_K + text_ranks[position]) if position in text_ranks else 0.0)
                      for position in range(len(candidates))}
        order = sorted(range(len(candidates)), key=lambda position: -scores[position])
        return [_Scored(candidates[position], similarities[position], text_ranks.get(position), scores[position])
                for position in order]


async def _account(engine: "MemoryEngine", space: str, detail: str) -> Optional[list[str]]:
    """What a stored node says it was written from, or None when that cannot be read."""
    try:
        _, raw = await engine.attachment(space, detail)
        body = json.loads(raw)
    except (SconeError, ValueError):
        return None
    listed = body.get("written_from") if isinstance(body, dict) else None
    if not isinstance(listed, list) or not all(isinstance(one, str) for one in listed):
        return None
    return listed


def _branch(scored: _Scored) -> tuple[int, ...]:
    """The ranks down a chunk's path, top first: compared in order, every chunk under the summary kept first at
    a step comes before any under the next, whether its tree ends at that step or goes further down."""
    return scored.candidate.ranks


def _leaf(scored: _Scored) -> Chunk:
    assert scored.candidate.chunk is not None, "only chunks are leaves"
    return scored.candidate.chunk


def _step(node: StoredSummary, scored: _Scored, rank: int, text: bool) -> dict[str, object]:
    """One summary on a path: where it sits, its place among those kept at its step, and how it scored."""
    return {"episode_id": node.episode_id, "level": node.level, "index": node.index, "rank": rank,
            "similarity": round(scored.similarity, 6), **({"text_rank": scored.text_rank} if text else {})}


async def traverse_summaries(engine: "MemoryEngine", space: str, query: str, *, limit: int = DEFAULT_LIMIT,
                             branching: int = DEFAULT_BRANCHING, max_depth: int = MAX_DEPTH, text: bool = False,
                             episode_ids: Optional[Sequence[int]] = None, kind: Optional[str] = None,
                             source_prefix: Optional[str] = None, tags: Sequence[str] = (),
                             since: Optional[str] = None, until: Optional[str] = None) -> Traversal:
    """The chunks a descent of the stored summary trees in scope reaches, best ``limit`` first."""
    from ..memory.engine import check_space

    check_space(space)
    check_traversal(query, limit, branching, max_depth, text, episode_ids, kind, source_prefix)
    query = query.strip()
    wanted_tags = set(normalise_tags(tags))
    since_at = normalise_time(since) if since else None
    until_at = normalise_time(until) if until else None

    listed = await engine.documents.recent_episodes(space, (await engine.documents.counts(space)).episodes)
    # Counted again after the listing: a store that caps what it lists, or episodes stored while it listed,
    # leave episodes out -- the oldest ones -- and a tree among them was not found.
    unlisted = max(0, (await engine.documents.counts(space)).episodes - len(listed))
    living = {episode.episode_id: episode for episode in listed}
    trees: dict[int, list[StoredSummary]] = {}
    for episode in listed:
        summary_of = episode.metadata.get("summary_of", "")
        row = stored_summary(episode) if summary_of.isdigit() else None
        if row is not None:
            trees.setdefault(int(summary_of), []).append(row)

    refused: list[dict[str, object]] = []
    untreed: list[int] = []
    in_scope: list[Episode] = []
    absent: list[int] = []
    out_of_scope = unlisted_trees = 0
    for document in (list(dict.fromkeys(episode_ids)) if episode_ids is not None else sorted(trees)):
        found = living.get(document)
        if found is None and episode_ids is None:
            # A tree whose document the listing did not return: a document forgotten or replaced leaves its
            # summaries behind, and nothing it holds can be fitted to the scope. Counted, not read.
            unlisted_trees += 1
        elif found is None:
            absent.append(document)
        elif not wanted_tags.issubset(found.tags) or not episode_fits(found, kind, source_prefix, since_at, until_at):
            out_of_scope += 1
        elif document not in trees:
            untreed.append(document)
        else:
            in_scope.append(found)
    if len(in_scope) > MAX_DOCUMENTS:
        raise InvalidInput(f"{len(in_scope)} documents with summary trees are in scope and one descent takes at most "
                           f"{MAX_DOCUMENTS}; name the documents or narrow the scope")

    walk = _Walk(engine, space)
    for document in absent:
        try:
            await walk.source(document)
            # Not in the listing, and there now: its tree was not seen whole.
            refused.append({"episode_id": document, "reason": "unlisted"})
        except _Refused as stopped:
            refused.append({"episode_id": document, "reason": stopped.reason})

    stale = missing = unresolved = uncovered = 0
    sources: dict[int, _Source] = {}
    places: dict[int, dict[tuple[int, int, int], StoredSummary]] = {}
    pool: list[_Candidate] = []
    for episode in in_scope:
        try:
            source = await walk.source(episode.episode_id)
        except _Refused as stopped:
            refused.append({"episode_id": episode.episode_id, "reason": stopped.reason})
            continue
        fresh = [row for row in trees[episode.episode_id] if row.content_hash == source.content_hash]
        stale += len(trees[episode.episode_id]) - len(fresh)
        if not fresh:
            refused.append({"episode_id": episode.episode_id, "reason": "content_changed"})
            continue
        sources[episode.episode_id] = source
        places[episode.episode_id] = {(row.fan_in, row.level, row.index): row for row in fresh}
        top = max(row.level for row in fresh)
        tops = sorted((row for row in fresh if row.level == top), key=lambda row: row.index)
        uncovered += max(0, len(source.chunks) - sum(row.chunks for row in tops))
        pool.extend(_Candidate(episode.episode_id, row.text, (), node=row) for row in tops)

    scorer = _Scorer(engine, space, query, text)
    steps: list[dict[str, object]] = []
    leaves: dict[int, _Candidate] = {}
    cut_by_depth = 0
    while pool:
        if len(steps) == max_depth:
            cut_by_depth = len(pool)
            break
        chosen = (await scorer.rank(pool))[:branching]
        steps.append({"pool": len(pool),
                      "chosen": [one.candidate.node.episode_id for one in chosen if one.candidate.node is not None],
                      "chunks": [one.candidate.chunk.chunk_id for one in chosen if one.candidate.chunk is not None]})
        below: list[_Candidate] = []
        for rank, scored in enumerate(chosen, start=1):
            document = scored.candidate.document
            node = scored.candidate.node
            if node is None:
                carried = _leaf(scored)
                leaves.setdefault(carried.chunk_id, _Candidate(document, carried.text, scored.candidate.path,
                                                               chunk=carried, ranks=(*scored.candidate.ranks, rank),
                                                               through=scored.candidate.through))
                continue
            written_from = await _account(engine, space, node.detail)
            if written_from is None:
                unresolved += 1
                continue
            # Read now (once, if its step read it to score it), so the last read can confirm every summary on a path.
            await scorer.chunks_of(node.episode_id)
            path = (*scored.candidate.path, _step(node, scored, rank, text))
            ranks, through = (*scored.candidate.ranks, rank), (*scored.candidate.through, node.episode_id)
            for named in written_from:
                kind_of, _, rest = named.partition(":")
                if kind_of == "chunk" and rest.isdigit():
                    chunk = sources[document].chunks.get(int(rest))
                    if chunk is None:
                        missing += 1
                    elif node.level > 1:
                        # Carried up beside the nodes below unsummarized: it competes with them at the next step.
                        below.append(_Candidate(document, chunk.text, path, chunk=chunk, ranks=ranks, through=through))
                    else:
                        leaves.setdefault(chunk.chunk_id, _Candidate(document, chunk.text, path, chunk=chunk, ranks=ranks,
                                                                     through=through))
                    continue
                parts = rest.split(":")
                child = (places[document].get((node.fan_in, int(parts[0]), int(parts[1])))
                         if kind_of == "node" and len(parts) == 2 and all(part.isdigit() for part in parts) else None)
                # Only below the node naming it, so a forged account cannot send the descent round in a circle.
                if child is None or child.level >= node.level:
                    unresolved += 1
                    continue
                below.append(_Candidate(document, child.text, path, node=child, ranks=ranks, through=through))
        pool = below

    # The branch first, then the chunk: a broad question is answered by the section the descent chose, and a
    # chunk of a lower branch whose own words fit better must not push that section out of the answer. The
    # sort is stable, so a branch's chunks keep their own ranking.
    ranked_leaves = sorted(((scored, _leaf(scored)) for scored in await scorer.rank(list(leaves.values()))),
                           key=lambda pair: _branch(pair[0]))
    # Every read is done but the last: the chunks about to be returned, and the chunks of every summary on their
    # paths, read again in one read. A document forgotten or changed during the walk serves none of its chunks,
    # and nor does one whose path runs through a summary that went.
    while ranked_leaves:
        taken = ranked_leaves[:limit]
        summaries = {summary for scored, _ in taken for summary in scored.candidate.through}
        wanted = {chunk.chunk_id for _, chunk in taken} | {
            chunk.chunk_id for summary in summaries for chunk in scorer.held[summary]}
        try:
            fresh_chunks = {chunk.chunk_id: chunk for chunk in await engine.documents.get_chunks(space, sorted(wanted))}
        except SconeError:
            # The read that confirms the answer failed, so no document's chunks were confirmed, returned or cut.
            for document in dict.fromkeys(chunk.episode_id for _, chunk in ranked_leaves):
                refused.append({"episode_id": document, "reason": "unread"})
            ranked_leaves = []
            break
        moved = next((chunk.episode_id for _, chunk in taken if fresh_chunks.get(chunk.chunk_id) != chunk), None)
        if moved is not None:
            refused.append({"episode_id": moved, "reason": await walk.recheck(moved)})
        else:
            # A summary with no chunks when first read was already gone, so it confirms nothing.
            moved = next((chunk.episode_id for scored, chunk in taken for summary in scored.candidate.through
                          if not scorer.held[summary]
                          or any(fresh_chunks.get(held.chunk_id) != held for held in scorer.held[summary])), None)
            if moved is None:
                break
            refused.append({"episode_id": moved, "reason": "tree_changed"})
        ranked_leaves = [(scored, chunk) for scored, chunk in ranked_leaves if chunk.episode_id != moved]

    # Nothing is awaited from here on, so what was confirmed is what is returned.
    items: list[RecallItem] = []
    for place, (scored, chunk) in enumerate(ranked_leaves[:limit]):
        held_in = sources[chunk.episode_id].episode
        first_line, last_line, declaration = placed_in_file(held_in, chunk)
        items.append(RecallItem(
            chunk_id=chunk.chunk_id, episode_id=chunk.episode_id, text=chunk.text, score=round(1 / (1 + place), 6),
            similarity=round(scored.similarity, 6), created_at=chunk.created_at, source=held_in.source,
            tags=held_in.tags, metadata=dict(held_in.metadata), start=chunk.start, end=chunk.end,
            first_line=first_line, last_line=last_line, declaration=declaration,
            via_tree={"summary_of": held_in.episode_id, "path": [dict(step) for step in scored.candidate.path],
                      **({"rank": scored.candidate.ranks[-1]} if len(scored.candidate.ranks) > len(scored.candidate.path) else {}),
                      **({"text_rank": scored.text_rank} if text else {})}))
    reached, cut_by_limit = len(ranked_leaves), max(0, len(ranked_leaves) - limit)
    # A document is refused before the descent or instead of it, or when its chunks were read again: only the last
    # were descended, and they are not counted among the documents that were.
    descended = tuple(sorted(set(sources) - {one["episode_id"] for one in refused}))
    why = f"descended {len(steps)} level(s) of {len(descended)} document tree(s) to {reached} chunk(s)"
    if not in_scope and not absent:
        why = "no document with a summary tree is in scope"
    if cut_by_limit:
        why += f"; the limit of {limit} left {cut_by_limit} chunk(s) out"
    if cut_by_depth:
        why += (f"; the depth of {max_depth} left {cut_by_depth} summary node(s) unscored, so the chunks under them "
                f"were not reached")
    if stale:
        why += f"; {stale} summary node(s) were written from other content than their document's and were not used"
    if missing:
        why += f"; {missing} chunk(s) named below a summary are not the document's and were not returned"
    if unresolved:
        why += (f"; {unresolved} node(s) named below a summary are not stored from this content, or their account "
                f"could not be read, and were not descended")
    if uncovered:
        why += f"; {uncovered} chunk(s) are under no summary the descent starts from, so no descent reaches them"
    if unlisted:
        why += (f"; the store listed {len(listed)} episode(s) and holds {len(listed) + unlisted}, so a tree among the "
                f"{unlisted} it left out was not found")
    if unlisted_trees:
        why += (f"; {unlisted_trees} summary tree(s) name a document the space's listing did not return (a document "
                f"forgotten or replaced leaves its tree behind{', and the listing left episodes out' if unlisted else ''}), "
                f"and were not read or descended")
    if untreed:
        why += f"; {len(untreed)} document(s) named have no summary tree"
    if out_of_scope:
        why += f"; {out_of_scope} document(s) with trees are outside the scope"
    for reason, said in _REASONS.items():
        count = sum(1 for one in refused if one["reason"] == reason)
        if count:
            why += f"; {count} document tree(s) were refused: {said}"
    return Traversal(items=tuple(items), query=query, limit=limit, branching=branching, max_depth=max_depth, text=text,
                     documents=descended, untreed=tuple(untreed), out_of_scope=out_of_scope, refused=tuple(refused),
                     steps=tuple(steps), leaves=reached, cut_by_limit=cut_by_limit, cut_by_depth=cut_by_depth,
                     stale=stale, missing=missing, unresolved=unresolved, uncovered=uncovered,
                     vectors=dict(scorer.vectors), embed_calls=scorer.embed_calls,
                     embedded_texts=scorer.embedded_texts, walked=len(listed), unlisted=unlisted,
                     unlisted_trees=unlisted_trees, why=why)
