"""The few tools a turn needs, chosen from the many a host has.

A host with eighteen tools puts eighteen schemas in front of the model on
every turn; a host with two hundred cannot, and a model shown two
hundred chooses worse than one shown eight. The leading framework's
object index retrieves tools for a query the way it retrieves nodes.
This does the same with the engine's own embedder, deterministically:
each tool is embedded once as its name and its sentence, a query is
embedded the same way, and the tools closest to it are offered -- with
a word of the query that is a tool's name or in its sentence counting
too, by how few tools share the word, so "search memory for ..." offers
`search_memory` whatever the vectors say and "the" counts for nothing.
Some tools a host wants always in front of the model; ``always`` keeps
them whatever the query.

What is offered is a suggestion to the host, not a gate: the toolbox
runs any tool it holds. The selection says its scores, what it left
out, and the embedder it used, so a host can see why a tool was or was
not in front of the model. Bounded: at most `MAX_TOOLS` are indexed
and a query is read up to `MAX_QUERY` characters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Optional, Sequence

from ..core.errors import InvalidInput
from ..core.ports import Embedder
from .tools import ToolSpec

MAX_TOOLS = 500
MAX_QUERY = 2_000
DEFAULT_LIMIT = 8
_WORD = re.compile(r"[a-z0-9]+")
#: Words that say nothing about which tool: they are in every sentence.
_STOP = frozenset("a an and are as at be by for from in is it its of on or that the this to what which with".split())


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower())) - _STOP


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


@dataclass(frozen=True)
class Offered:
    """One tool the selection offers, and why."""

    name: str
    score: float
    similarity: float
    #: Words of the query that are the tool's name or in its sentence.
    words: tuple[str, ...]


@dataclass(frozen=True)
class Selection:
    """The tools to put in front of the model, best first, with what the
    choice rested on and what it left out."""

    offered: tuple[Offered, ...]
    always: tuple[str, ...]
    left_out: tuple[str, ...]
    embedder: str
    query: str
    #: Said when no tool resembled the query at all: the offer is then
    #: the always-on tools and the rest in name order, not a ranking.
    note: Optional[str] = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.offered)

    def record(self) -> dict[str, object]:
        return {"offered": [{"name": item.name, "score": round(item.score, 4), "similarity": round(item.similarity, 4),
                             "words": list(item.words)} for item in self.offered],
                "always": list(self.always), "left_out": list(self.left_out), "embedder": self.embedder,
                "basis": "cosine similarity of the query to each tool's name and sentence under the engine's "
                         "embedder, plus a term for each query word that is the tool's name or in its sentence, "
                         "weighted by how few tools share the word", "note": self.note}


class ToolIndex:
    """Tools embedded once, chosen per query."""

    def __init__(self, tools: Sequence[ToolSpec], embedder: Embedder, vectors: Sequence[Sequence[float]]) -> None:
        if len(tools) != len(vectors):
            raise InvalidInput("a tool index needs one vector per tool")
        self.tools = tuple(tools)
        self.embedder = embedder
        self._vectors = tuple(tuple(vector) for vector in vectors)
        self._words = tuple(_words(f"{tool.name.replace('_', ' ')} {tool.summary}") | {tool.name} for tool in self.tools)
        # A word shared by every tool's sentence ("entity", "memory") says
        # little; one that only this tool uses says most.
        counts: dict[str, int] = {}
        for words in self._words:
            for word in words:
                counts[word] = counts.get(word, 0) + 1
        total = len(self.tools)
        self._weight = {word: (math.log(total / count) / math.log(total) if total > 1 else 1.0)
                        for word, count in counts.items()}

    @classmethod
    async def build(cls, tools: Iterable[ToolSpec], embedder: Embedder) -> "ToolIndex":
        listed = tuple(tools)
        if not listed:
            raise InvalidInput("a tool index needs at least one tool")
        if len(listed) > MAX_TOOLS:
            raise InvalidInput(f"a tool index holds at most {MAX_TOOLS} tools")
        names = [tool.name for tool in listed]
        if len(set(names)) != len(names):
            raise InvalidInput("a tool index needs distinct tool names")
        vectors = await embedder.embed([f"{tool.name.replace('_', ' ')}: {tool.summary}" for tool in listed])
        return cls(listed, embedder, vectors)

    async def select(self, query: str, *, limit: int = DEFAULT_LIMIT, always: Sequence[str] = ()) -> Selection:
        """The ``limit`` tools closest to the query, the ``always`` ones
        first whatever the query."""
        if not isinstance(query, str) or not query.strip():
            raise InvalidInput("a selection needs a query")
        if len(query) > MAX_QUERY:
            raise InvalidInput(f"a query is read for tools up to {MAX_QUERY} characters")
        if type(limit) is not int or not 1 <= limit <= MAX_TOOLS:
            raise InvalidInput(f"limit must be from 1 to {MAX_TOOLS}")
        known = {tool.name for tool in self.tools}
        kept = tuple(dict.fromkeys(always))
        unknown = [name for name in kept if name not in known]
        if unknown:
            raise InvalidInput(f"not in this index: {', '.join(unknown)}")
        [asked] = await self.embedder.embed([query])
        query_words = _words(query)
        scored: list[Offered] = []
        for tool, vector, words in zip(self.tools, self._vectors, self._words):
            similarity = _cosine(asked, vector)
            hits = tuple(sorted(query_words & words))
            # A word of the query that names the tool or is in its sentence
            # is worth up to a fifth of a perfect vector match, by how few
            # tools share it: enough that "search memory" offers
            # search_memory ahead of whatever the hash of its words
            # resembles, and nothing for a word every tool uses.
            named = tool.name in query_words or tool.name.lower() in query.lower()
            score = similarity + 0.2 * sum(self._weight.get(word, 0.0) for word in hits) + (0.5 if named else 0.0)
            scored.append(Offered(tool.name, score, similarity, hits))
        scored.sort(key=lambda item: (-item.score, item.name))
        offered = [item for item in scored if item.name in kept]
        offered.sort(key=lambda item: kept.index(item.name))
        for item in scored:
            if len(offered) >= max(limit, len(kept)):
                break
            if item.name not in kept:
                offered.append(item)
        chosen = {item.name for item in offered}
        left_out = tuple(item.name for item in scored if item.name not in chosen)
        note = None
        if all(item.score <= 0.0 for item in scored):
            note = ("no tool resembled the query under this embedder; beyond the always-on tools this offer is in "
                    "name order, not a ranking")
        return Selection(tuple(offered), kept, left_out, self.embedder.id, query, note)
