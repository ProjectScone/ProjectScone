"""What a passage is under, derived with no model, for the lane that finds what a passage lacks.

The lexical lane finds the words a passage has, and only those. A chunk
under the heading "Refunds" in a document titled "Billing rules" need not
say either word, and a query about billing refunds then misses it. What
the chunk is *under* is not a guess: the headings enclosing it, the
document's title and the name its source was stored under are all in
the source already. This derives them, keeps only what the chunk itself
lacks, and hands the words to a second index beside the text lane, so a
passage becomes findable by its context without a word of its stored
text changing.

Only structure is indexed: headings, title, source name. A document's
frequent words were tried too and left out -- spread over every chunk
they make the lane fire on mentions rather than on structure, and the
lane's weight (see ``retrieval.recall.CONTEXT_WEIGHT``) is set high
enough that what it says must be worth it.

Nothing here is a claim about the passage. The words are a place to look,
fused by rank with the other lanes, and a recall says in ``lanes`` which
passages the context lane put where.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from ..core.ports import context_index
from ..retrieval.lexical import tokenize
from .structure import DocumentStructure, parse_structure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.models import Chunk
    from ..core.ports import NewEpisode

#: Words one chunk's context may carry, in priority order: headings, title,
#: source words. The rest are left out and counted.
MAX_TERMS = 32
#: A first line longer than this is a sentence, not a title.
MAX_TITLE_CHARS = 120

_NAME_SPLIT = re.compile(r"[\s_./\\-]+")


@dataclass(frozen=True)
class ChunkContext:
    """The words a chunk is under and does not say, and what was left out."""

    headings: tuple[str, ...]
    title: Optional[str]
    source_words: tuple[str, ...]
    #: The words in index order, each once, for the context index.
    words: tuple[str, ...]
    omitted: int

    def text(self) -> str:
        return " ".join(self.words)

    def record(self) -> dict[str, object]:
        return {"headings": list(self.headings), "title": self.title, "source_words": list(self.source_words),
                "terms": len(self.words), "omitted": self.omitted}


def heading_path(structure: DocumentStructure, start: int) -> tuple[str, ...]:
    """Titles of the headings enclosing byte ``start``, outermost first."""
    enclosing = [section for section in structure.sections
                 if section.level >= 1 and section.title and section.start <= start < section.end]
    return tuple(section.title for section in sorted(enclosing, key=lambda section: (section.level, section.start)))


def document_title(content: str, structure: Optional[DocumentStructure]) -> Optional[str]:
    """The top heading, or a short first line that does not read as a sentence."""
    if structure is not None:
        headed = [section for section in structure.sections if section.level >= 1 and section.title]
        if headed:
            first = min(headed, key=lambda section: (section.start, section.level))
            return first.title
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > MAX_TITLE_CHARS or line[-1] in ".!?;:," or not tokenize(line):
            return None
        return line
    return None


def source_words(source: Optional[str]) -> tuple[str, ...]:
    """The words of the source's own name: the last path segment without its extension."""
    if not source:
        return ()
    name = source.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return tuple(word for word in (token for part in _NAME_SPLIT.split(name) for token in tokenize(part)))


def chunk_context(content: str, *, start: int, chunk_text: str, source: Optional[str],
                  structure: Optional[DocumentStructure] = None) -> ChunkContext:
    """The words a chunk of ``content`` starting at byte ``start`` is under and does not say."""
    parsed = structure if structure is not None else parse_structure(content)
    headings = heading_path(parsed, start)
    title = document_title(content, parsed)
    named = source_words(source)
    has = set(tokenize(chunk_text))
    words: list[str] = []
    seen: set[str] = set(has)
    omitted = 0
    for phrase in (*headings, *([title] if title else []), *named):
        fresh = [token for token in tokenize(phrase) if token not in seen]
        if not fresh:
            continue
        if len(words) + len(fresh) > MAX_TERMS:
            omitted += len(fresh)
            continue
        words.extend(fresh)
        seen.update(fresh)
    return ChunkContext(headings, title, named, tuple(words), omitted)


async def index_episode_context(documents: object, space: str, episode: "NewEpisode", chunks: Sequence["Chunk"]) -> int:
    """Index the context of every chunk of ``episode`` that has any; the count indexed.

    A store without a context index is left alone: the lane is reported
    as absent at recall, not silently empty."""
    keeper = context_index(documents)
    if keeper is None or not chunks:
        return 0
    structure = parse_structure(episode.content)
    indexed = 0
    for chunk in chunks:
        found = chunk_context(episode.content, start=chunk.start, chunk_text=chunk.text, source=episode.source,
                              structure=structure)
        if found.words:
            await keeper.index_context(space, chunk.chunk_id, found.text())
            indexed += 1
    return indexed
