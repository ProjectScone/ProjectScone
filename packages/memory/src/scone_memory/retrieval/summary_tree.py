"""A document's summary tree: every level written from the one below it, every sentence quoting down.

A long document answers a broad question badly in chunks: "what does
this report conclude?" is spread over forty passages, and the five that
score best are five fragments. The leading frameworks build a tree of
summaries at ingestion (RAPTOR: cluster, summarize, repeat) and index the
summaries beside the leaves, so a broad question finds a summary and a
narrow one finds a chunk. The summaries are the model's word.

This builds that tree our way. Level zero is the document's chunks in
their order. Each level above groups ``fan_in`` adjacent nodes of the
level below and writes one node from them with the evidence-first
synthesizer, so every sentence carries a quote this code found in the
node it cites -- a chunk's text at level one, a lower summary's text
above that -- and a citation resolves downward to the chunk quotes it
rests on. A group the model wrote nothing about leaves no node, and is
counted. Adjacency is the grouping, not a clustering: a document's own
order is a structure nobody has to guess at.

The nodes are stored as notes the framework wrote, each saying which
episode it summarizes, its level, the chunks it covers and the content
hash of the document at the time, so the ordinary lanes retrieve them
beside the chunks, a reader can see what a passage is, and a summary of
a document that has since changed can be told from a current one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import TYPE_CHECKING, Optional, Sequence

from ..core.errors import InvalidInput, NotFound
from ..providers.llm import ChatModel
from .synthesis import Passage, Synthesis, SynthesisLimits, synthesize_passages

if TYPE_CHECKING:
    from ..core.models import Episode
    from ..memory.engine import MemoryEngine

#: Nodes one node is written from.
DEFAULT_FAN_IN = 6
MAX_FAN_IN = 24
#: Levels above the chunks a tree may have; a document whose tree would
#: be taller stops here with its top level unjoined, and says so.
MAX_LEVELS = 5
#: Chunks a tree is built over; a longer document is refused, not cut.
MAX_CHUNKS = 600
QUESTION = "What does this part of the document say? Every sentence must rest on a quote."


@dataclass(frozen=True)
class Node:
    """One summary: its text, where it sits, and what it was written from."""

    level: int
    index: int
    text: str
    #: Ids of the nodes it was written from: ``chunk:<id>`` at level one, ``node:<level>:<index>`` above.
    written_from: tuple[str, ...]
    #: The chunk ids under it, at any depth.
    covers: tuple[int, ...]
    #: The synthesizer's own account of this node: rounds, drops, reasons.
    synthesis: Synthesis
    #: The episode it was stored as, when stored.
    episode_id: Optional[int] = None

    @property
    def id(self) -> str:
        return f"node:{self.level}:{self.index}"


@dataclass(frozen=True)
class SummaryTree:
    episode_id: int
    content_hash: str
    chunks: int
    fan_in: int
    levels: int
    nodes: tuple[Node, ...]
    #: Groups the model wrote nothing about, per level.
    empty_groups: dict[int, int] = field(default_factory=dict)
    #: Whether the top level has more than one node because MAX_LEVELS stopped the tree.
    unjoined: bool = False
    model_calls: int = 0
    stored: bool = False
    reasons: tuple[str, ...] = ()

    @property
    def root(self) -> Optional[Node]:
        top = [node for node in self.nodes if node.level == self.levels]
        return top[0] if len(top) == 1 else None

    def record(self) -> dict[str, object]:
        return {"episode_id": self.episode_id, "content_hash": self.content_hash, "chunks": self.chunks,
                "fan_in": self.fan_in, "levels": self.levels, "unjoined": self.unjoined, "stored": self.stored,
                "model_calls": self.model_calls, "empty_groups": {str(k): v for k, v in sorted(self.empty_groups.items())},
                "reasons": list(self.reasons),
                "nodes": [{"id": node.id, "level": node.level, "index": node.index, "text": node.text,
                           "written_from": list(node.written_from), "covers": list(node.covers),
                           "episode_id": node.episode_id, "sentences": len(node.synthesis.sentences),
                           "notes_dropped": node.synthesis.notes_dropped_unquoted + node.synthesis.notes_dropped_unknown
                           + node.synthesis.notes_dropped_malformed}
                          for node in self.nodes]}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def summary_metadata(tree_of: "Episode", node: Node, model_name: str) -> dict[str, str]:
    """What a stored summary says about itself, in the episode's metadata."""
    return {"summary_of": str(tree_of.episode_id), "summary_level": str(node.level), "summary_index": str(node.index),
            "summary_covers": ",".join(str(c) for c in node.covers), "summary_written_from": ",".join(node.written_from),
            "summary_content_hash": _hash(tree_of.content), "summary_model": model_name,
            "summary_citations": json.dumps([[c.passage_id, c.quote] for s in node.synthesis.sentences for c in s.citations],
                                            ensure_ascii=False)}


async def build_summary_tree(engine: "MemoryEngine", model: ChatModel, space: str, episode_id: int, *,
                             fan_in: int = DEFAULT_FAN_IN, limits: Optional[SynthesisLimits] = None,
                             max_levels: int = MAX_LEVELS, store: bool = False,
                             model_name: str = "") -> SummaryTree:
    """The tree for one episode, level by level; stored as notes when ``store``."""
    if not 2 <= fan_in <= MAX_FAN_IN:
        raise InvalidInput(f"fan_in is between 2 and {MAX_FAN_IN}")
    if not 1 <= max_levels <= MAX_LEVELS:
        raise InvalidInput(f"max_levels is between 1 and {MAX_LEVELS}")
    episode = await engine.episode(space, episode_id)
    chunks = await engine.documents.chunks_of(space, episode_id)
    content_hash = _hash(episode.content)
    if not chunks:
        raise NotFound(f"episode {episode_id} has no chunks to summarize")
    if len(chunks) > MAX_CHUNKS:
        raise InvalidInput(f"a summary tree is built over at most {MAX_CHUNKS} chunks; episode {episode_id} has {len(chunks)}")
    bound = limits or SynthesisLimits(max_passages=fan_in)
    if bound.max_passages < fan_in:
        raise InvalidInput("limits.max_passages must reach fan_in")
    level_below: list[tuple[str, str, tuple[int, ...]]] = [
        (f"chunk:{chunk.chunk_id}", chunk.text, (chunk.chunk_id,)) for chunk in chunks]
    nodes: list[Node] = []
    empty: dict[int, int] = {}
    calls = 0
    level = 0
    reasons: list[str] = []
    while len(level_below) > 1 and level < max_levels:
        level += 1
        written: list[tuple[str, str, tuple[int, ...]]] = []
        wrote = 0
        for start in range(0, len(level_below), fan_in):
            group = level_below[start:start + fan_in]
            if len(group) == 1 and start > 0:
                # A lone remainder is carried up as it is, not summarized
                # from itself: a summary of one node adds a step and no words.
                written.append(group[0])
                continue
            passages = [Passage(identifier, text, episode.source, episode.created_at) for identifier, text, _ in group]
            made = await synthesize_passages(model, QUESTION, passages, limits=bound)
            calls += made.model_calls
            if not made.sentences:
                empty[level] = empty.get(level, 0) + 1
                continue
            text = " ".join(sentence.text for sentence in made.sentences)
            covers = tuple(c for _, _, under in group for c in under)
            node = Node(level, len([n for n in nodes if n.level == level]), text,
                        tuple(identifier for identifier, _, _ in group), covers, made)
            nodes.append(node)
            written.append((node.id, text, covers))
            wrote += 1
        if not wrote:
            reasons.append(f"level {level}: the model wrote nothing about any group; the tree stops below it")
            level -= 1
            break
        level_below = written
    unjoined = len(level_below) > 1 and level >= 1
    if unjoined:
        reasons.append(f"{len(level_below)} nodes remain at level {level}: MAX_LEVELS or an empty level stopped the tree")
    tree = SummaryTree(episode_id, content_hash, len(chunks), fan_in, level, tuple(nodes),
                       empty_groups=empty, unjoined=unjoined, model_calls=calls, reasons=tuple(reasons))
    if not store or not nodes:
        return tree
    stored: list[Node] = []
    for node in nodes:
        where = f"{episode.source}#summary/{node.level}/{node.index}" if episode.source else f"episode:{episode_id}#summary/{node.level}/{node.index}"
        added = await engine.remember(space, node.text, kind="note", source=where,
                                      metadata=summary_metadata(episode, node, model_name or type(model).__name__),
                                      dedup_key=f"summary:{episode_id}:{content_hash}:{node.level}:{node.index}", replace=True)
        stored.append(Node(node.level, node.index, node.text, node.written_from, node.covers, node.synthesis, added.episode_id))
    return SummaryTree(episode_id, content_hash, len(chunks), fan_in, level, tuple(stored),
                       empty_groups=empty, unjoined=unjoined, model_calls=calls, stored=True, reasons=tuple(reasons))


@dataclass(frozen=True)
class StoredSummary:
    episode_id: int
    level: int
    index: int
    text: str
    covers: tuple[int, ...]
    #: Whether the document has changed since this summary was written.
    stale: bool

    def record(self) -> dict[str, object]:
        return {"episode_id": self.episode_id, "level": self.level, "index": self.index, "text": self.text,
                "covers": list(self.covers), "stale": self.stale}


async def stored_summaries(engine: "MemoryEngine", space: str, episode_id: int) -> tuple[StoredSummary, ...]:
    """The summaries stored for an episode, top level first, each saying whether it is stale."""
    episode = await engine.episode(space, episode_id)
    current = _hash(episode.content)
    found = await engine.episodes(space, {"summary_of": str(episode_id)})
    rows = []
    for one in found:
        meta = one.metadata
        try:
            level, index = int(meta.get("summary_level", "")), int(meta.get("summary_index", ""))
            covers = tuple(int(c) for c in meta.get("summary_covers", "").split(",") if c)
        except ValueError:
            continue
        rows.append(StoredSummary(one.episode_id, level, index, one.content, covers,
                                  stale=meta.get("summary_content_hash") != current))
    return tuple(sorted(rows, key=lambda s: (-s.level, s.index)))
