"""Fitting the context put in front of a chunk into the embedder's window.

A model embeds at most so many tokens and drops the rest without saying
so. What is put in front of a chunk -- the headings above it, a table's
header row, the source and date -- comes first, so a long enough prefix
cuts off the end of the very chunk it explains. ``fitted`` shortens the
context until the input fits: the outermost headings first, then the
whole heading line, then the table context. The chunk is never cut; one
too long on its own is embedded without context and counted.

An embedder that can count its own tokens (``count_tokens``, as the local
models can with their tokenizer) is asked. Otherwise tokens are estimated,
since the hashing and remote embedders carry no tokenizer. The estimate
errs high: a word counts one token for every four letters of each of its
parts (a camel-case name split at its capitals), a run of digits one for
every two, each mark one, each character of a script written without
spaces one, and two more for the markers a model adds. Measured on this
project's 1,173 documentation chunks against BGE's own tokenizer it never
counted fewer tokens, and counted 1.36 times as many at the median. Erring
high has a cost the tokenizer does not: at a 2,000-character chunk target
the estimate called 268 of those chunks over 512 tokens with their heading
line, where the tokenizer found 39.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Literal, Optional

from ..retrieval.lexical import UNSPACED_CHAR

#: Names the estimate and the order context is shortened in. Part of the
#: vector writer's identity when the budget is on.
BUDGET_VERSION = "estimate-v1"
#: The same order, counted by the embedder's own tokenizer.
TOKENIZER_VERSION = "tokenizer-v1"

Outcome = Literal["fits", "shortened", "dropped", "body_over"]

_PIECE = re.compile(r"[^\W\d_]+|\d+|[^\w\s]|_")
_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[^\W\d_]")
#: The start and end markers an encoder model puts around every input.
_MARKERS = 2


def estimated_tokens(text: str) -> int:
    """An estimate of the tokens an encoder model reads for ``text``, meant to err high."""
    count = _MARKERS
    for piece in _PIECE.findall(text):
        if piece.isdigit():
            count += math.ceil(len(piece) / 2)
        elif piece[0].isalpha() or UNSPACED_CHAR.search(piece):
            count += len(UNSPACED_CHAR.findall(piece))
            count += sum(math.ceil(len(part) / 4) for part in _PART.findall(UNSPACED_CHAR.sub(" ", piece)))
        else:
            count += 1
    return count


def fitted(*, body: str, heading: str, table: Optional[str], wrap: Callable[[str], str],
           budget: int, count: Callable[[str], int] = estimated_tokens) -> tuple[str, Outcome]:
    """The embedding input for ``body``, with as much of its context as fits ``budget``.

    ``heading`` is the heading line (titles joined by " > "), ``table`` the
    chunk with its table context in front, or None, and ``wrap`` adds the
    source and date the engine puts in front of every input."""
    text = table if table is not None else body

    def compose(line: str, rest: str) -> str:
        return wrap(f"{line}\n{rest}" if line else rest)

    whole = compose(heading, text)
    if count(whole) <= budget:
        return whole, "fits"
    titles = heading.split(" > ") if heading else []
    while titles:
        titles.pop(0)
        candidate = compose(" > ".join(titles), text)
        if count(candidate) <= budget:
            return candidate, "shortened" if titles or table is not None else "dropped"
    # Without a table, the bare chunk was the last input tried above.
    bare = compose("", body)
    if count(bare) <= budget:
        return bare, "dropped"
    return bare, "body_over"
