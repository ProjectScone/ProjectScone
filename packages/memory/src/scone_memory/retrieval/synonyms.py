"""Synonyms a caller wrote down, applied to the text lane's query.

The lexical lane finds the words a passage has, and only those. A
passage that says "automobile" is invisible to a query about a "car"
unless somebody wrote down that in this corpus the two are one word.
This is that list, and nothing more: no model proposes a synonym, the
list is the caller's, read from a file or given in code, and every
expansion is on the record, so a reader can see which words were added
to which query and why a passage was found.

Only the text lane sees the added words. The vector lane's query stays
as the caller wrote it: an embedder already knows what it knows about
"car" and "automobile", and padding its input with a list would move the
vector in ways nobody measured. The lanes are fused by rank, so a
passage found only through an added word competes on rank, never on a
score the addition inflated.

Matching is the lane's own: a term matches when its tokens, as the lexical
tokenizer makes them, occur in that order among the query's tokens, so
case, possessives and stopwords are treated exactly as the index treats
them. Longer terms match first, and a phrase matches as a phrase, not as
its words.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..core.errors import InvalidInput
from .lexical import tokenize

#: Groups one list may hold; more is refused, not cut.
MAX_GROUPS = 2_000
#: Terms one group may hold; more is refused, not cut.
MAX_PER_GROUP = 16
#: Characters one term may run to; longer is refused.
MAX_TERM_CHARS = 64
#: Words one query may gain; the rest are left out and ``capped`` says so.
MAX_ADDED = 12


@dataclass(frozen=True)
class Expansion:
    """What one query matched and gained, and what it did not."""

    query: str
    #: The query the text lane searched: the query, then the added terms.
    text_query: str
    matched: tuple[str, ...]
    added: tuple[str, ...]
    #: Terms the groups offered before the bound; ``added`` is the first of them.
    offered: int
    capped: bool

    def record(self) -> dict[str, object]:
        return {"matched": list(self.matched), "added": list(self.added), "offered": self.offered, "capped": self.capped}


class Synonyms:
    """A caller's synonym groups; every term of a group stands for every other."""

    def __init__(self, groups: Iterable[Iterable[str]]) -> None:
        rows = [tuple(str(term).strip() for term in group) for group in groups]
        if len(rows) > MAX_GROUPS:
            raise InvalidInput(f"{len(rows)} synonym groups over the bound of {MAX_GROUPS}; refused rather than cut")
        self._groups: tuple[tuple[str, ...], ...] = tuple(rows)
        self._surface: dict[tuple[str, ...], str] = {}
        self._members: dict[tuple[str, ...], list[int]] = {}
        for index, group in enumerate(rows):
            if len(group) > MAX_PER_GROUP:
                raise InvalidInput(f"group {index + 1} has {len(group)} terms, over the bound of {MAX_PER_GROUP}; refused rather than cut")
            if len(group) < 2:
                raise InvalidInput(f"group {index + 1} needs more than one term to be a synonym group")
            for term in group:
                if not term or len(term) > MAX_TERM_CHARS:
                    raise InvalidInput(f"a synonym is 1 to {MAX_TERM_CHARS} characters, not {term!r}")
                key = tuple(tokenize(term))
                if not key:
                    raise InvalidInput(f"{term!r} is nothing to the lexical tokenizer, so it cannot be a synonym")
                self._surface.setdefault(key, term)
                self._members.setdefault(key, []).append(index)
        self._keys = sorted(self._members, key=len, reverse=True)

    @classmethod
    def from_lines(cls, text: str) -> "Synonyms":
        """One group per line, terms separated by commas; ``#`` starts a comment."""
        groups = []
        for line in text.splitlines():
            body = line.split("#", 1)[0].strip()
            if body:
                groups.append([term.strip() for term in body.split(",") if term.strip()])
        return cls(groups)

    @classmethod
    def from_file(cls, path: str | Path) -> "Synonyms":
        where = Path(path)
        if not where.is_file():
            raise InvalidInput(f"synonyms file not found: {where}")
        return cls.from_lines(where.read_text(encoding="utf-8"))

    def record(self) -> dict[str, object]:
        return {"groups": len(self._groups), "terms": sum(len(group) for group in self._groups)}

    def expand(self, query: str) -> Expansion:
        """The query with the other members of every matched group appended."""
        tokens = tokenize(query)
        matched_keys: list[tuple[str, ...]] = []
        position = 0
        while position < len(tokens):
            for key in self._keys:
                if tuple(tokens[position:position + len(key)]) == key:
                    matched_keys.append(key)
                    position += len(key)
                    break
            else:
                position += 1
        seen = set(matched_keys)
        offered: list[str] = []
        for key in matched_keys:
            for index in self._members[key]:
                for term in self._groups[index]:
                    term_key = tuple(tokenize(term))
                    if term_key not in seen:
                        seen.add(term_key)
                        offered.append(term)
        added = tuple(offered[:MAX_ADDED])
        text_query = f"{query} {' '.join(added)}" if added else query
        return Expansion(query, text_query, tuple(dict.fromkeys(self._surface[key] for key in matched_keys)),
                         added, len(offered), len(offered) > MAX_ADDED)
