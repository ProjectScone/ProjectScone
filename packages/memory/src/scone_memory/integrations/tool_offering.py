"""The few tools a turn needs, chosen from the many a host has.

A host with eighteen tools puts eighteen schemas in front of the model on
every turn; a host with two hundred cannot, and a model shown two
hundred chooses worse than one shown eight. The leading framework's
object index retrieves tools for a query the way it retrieves nodes.
This does the same with the engine's own embedder, deterministically:
each tool is embedded once as its name and its sentence, a query is
embedded the same way, and the tools closest to it are offered -- with
a word of the query that is in a tool's name or sentence counting too,
by how few tools share the word (up to a fifth of a perfect vector
match per word; a word the lexical lane calls a stopword counts for
nothing), and a tool named whole in the query (`search_memory`) given
half a match on top. Under the hash embedder the vectors are the words
too, so "search memory for ..." offers `search_memory`; under a model
embedder a tool the vectors put clearly closer can still come first.
Some tools a host wants always in front of the model; ``always`` keeps
them first whatever the query, and past ``limit`` if there are more of
them than that.

What is offered is a suggestion to the host, not a gate: the toolbox
runs any tool it holds. The selection says each offered tool's score,
similarity, matched words and whether it was named, the score of each
tool left out, the embedder it used, and -- when no tool scored above
nothing -- that the order beyond the always-on tools is what little
the vectors said and then the name, not a ranking. Bounded: at most
`MAX_TOOLS` are indexed and a query is read up to `MAX_QUERY`
characters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Optional, Sequence

from ..core.errors import InvalidInput
from ..core.ports import Embedder
from ..retrieval.lexical import STOPWORDS, tokenize
from .tools import ToolSpec

MAX_TOOLS = 500
MAX_QUERY = 2_000
DEFAULT_LIMIT = 8
#: What a matched word is worth, at most, against a perfect vector match
#: of 1.0; and what a tool named whole in the query is worth on top.
WORD_WEIGHT = 0.2
NAME_BONUS = 0.5


def _words(text: str) -> set[str]:
    """The words of a text as the lexical lane reads them (folded,
    normalised, unspaced scripts in n-grams), without its stopwords."""
    return set(tokenize(text)) - STOPWORDS


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
    #: Words of the query that are in the tool's name or sentence.
    words: tuple[str, ...]
    #: The query named the tool whole, as a word of its own.
    named: bool = False


@dataclass(frozen=True)
class Selection:
    """The tools to put in front of the model, best first, with what the
    choice rested on and what it left out."""

    offered: tuple[Offered, ...]
    always: tuple[str, ...]
    left_out: tuple[str, ...]
    embedder: str
    query: str
    #: Said when no tool scored above nothing: the offer beyond the
    #: always-on tools is then what little the vectors said and then the
    #: name, not a ranking.
    note: Optional[str] = None
    #: The score of every tool not offered, so the "why not" is on record.
    left_out_scores: tuple[tuple[str, float], ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.offered)

    def record(self) -> dict[str, object]:
        return {"offered": [{"name": item.name, "score": round(item.score, 4), "similarity": round(item.similarity, 4),
                             "words": list(item.words), "named": item.named} for item in self.offered],
                "always": list(self.always), "left_out": list(self.left_out),
                "left_out_scores": {name: round(score, 4) for name, score in self.left_out_scores},
                "embedder": self.embedder,
                "basis": (f"cosine similarity of the query to each tool's name and sentence under the engine's "
                          f"embedder; plus up to {WORD_WEIGHT} for each query word in the tool's name or sentence, "
                          f"by how few tools share the word; plus {NAME_BONUS} when the query names the tool whole"),
                "note": self.note}


class ToolIndex:
    """Tools embedded once, chosen per query."""

    def __init__(self, tools: Sequence[ToolSpec], embedder: Embedder, vectors: Sequence[Sequence[float]]) -> None:
        listed = tuple(tools)
        _checked(listed)
        if len(listed) != len(vectors):
            raise InvalidInput("a tool index needs one vector per tool")
        self.tools = listed
        self.embedder = embedder
        self._vectors = tuple(tuple(vector) for vector in vectors)
        self._words = tuple(_words(f"{tool.name.replace('_', ' ')} {tool.summary}") for tool in self.tools)
        # A tool is named whole when its name stands as a word of its own
        # in the query: `search_memory`, not the `add` in `address`.
        self._named = tuple(re.compile(r"(?<![\w])" + re.escape(tool.name) + r"(?![\w])", re.IGNORECASE)
                            for tool in self.tools)
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
        _checked(listed)
        vectors = await embedder.embed([f"{tool.name.replace('_', ' ')}: {tool.summary}" for tool in listed])
        if len(vectors) != len(listed):
            raise InvalidInput("the embedder returned a vector count that is not the tool count")
        return cls(listed, embedder, vectors)

    async def select(self, query: str, *, limit: int = DEFAULT_LIMIT, always: Sequence[str] = ()) -> Selection:
        """The ``limit`` tools closest to the query, the ``always`` ones
        first whatever the query -- and past ``limit`` when there are more
        of them than that: a host that names them wants them shown."""
        if not isinstance(query, str) or not query.strip():
            raise InvalidInput("a selection needs a query")
        if len(query) > MAX_QUERY:
            raise InvalidInput(f"a query is read for tools up to {MAX_QUERY} characters")
        if type(limit) is not int or not 1 <= limit <= MAX_TOOLS:
            raise InvalidInput(f"limit must be from 1 to {MAX_TOOLS}")
        known = {tool.name for tool in self.tools}
        kept = tuple(dict.fromkeys(str(name) for name in always))
        unknown = [name for name in kept if name not in known]
        if unknown:
            raise InvalidInput(f"not among these tools: {', '.join(unknown)}")
        from ..core.embedding import embed_queries
        embedded = await embed_queries(self.embedder, [query])
        if len(embedded) != 1:
            raise InvalidInput("the embedder returned a vector count that is not one")
        asked = embedded[0]
        query_words = _words(query)
        scored: list[Offered] = []
        for tool, vector, words, whole in zip(self.tools, self._vectors, self._words, self._named):
            similarity = _cosine(asked, vector)
            hits = tuple(sorted(query_words & words))
            named = whole.search(query) is not None
            score = (similarity + WORD_WEIGHT * sum(self._weight.get(word, 0.0) for word in hits)
                     + (NAME_BONUS if named else 0.0))
            scored.append(Offered(tool.name, score, similarity, hits, named))
        scored.sort(key=lambda item: (-item.score, item.name))
        offered = [item for item in scored if item.name in kept]
        offered.sort(key=lambda item: kept.index(item.name))
        for item in scored:
            if len(offered) >= max(limit, len(kept)):
                break
            if item.name not in kept:
                offered.append(item)
        chosen = {item.name for item in offered}
        left = [item for item in scored if item.name not in chosen]
        note = None
        if all(item.score <= 0.0 for item in scored):
            note = ("no tool scored above nothing under this embedder; beyond the always-on tools this offer is "
                    "what little the vectors said and then the name, not a ranking")
        return Selection(tuple(offered), kept, tuple(item.name for item in left), self.embedder.id, query, note,
                         tuple((item.name, item.score) for item in left))


def _checked(listed: Sequence[ToolSpec]) -> None:
    if not listed:
        raise InvalidInput("a tool index needs at least one tool")
    if len(listed) > MAX_TOOLS:
        raise InvalidInput(f"a tool index holds at most {MAX_TOOLS} tools")
    names = [tool.name for tool in listed]
    if len(set(names)) != len(names):
        raise InvalidInput("a tool index needs distinct tool names")
