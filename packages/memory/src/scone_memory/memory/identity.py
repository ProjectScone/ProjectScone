"""How the memory decides that two names are the same entity.

``entity_key`` is identity: case folded and spacing collapsed, nothing
looser. ``join_match`` is the one rule every join follows to decide whether
a claim's object names the thing another claim's subject is about: the
keys must be equal, and an object that cannot name one thing (prose, a
quotation, a pronoun) never joins, while a value whose case can change its
meaning ('MB' against 'mb') joins only when spelled exactly the same. The
entity projection's classifier decides the same cases, so the graph a
person sees and the one retrieval walks agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..core.validation import entity_key
from ..entities.classify import join_block_reason

__all__ = ["JoinMatch", "entity_key", "join_match", "lookup_keys"]

_NEVER_JOIN = frozenset({"prose", "quoted_text", "pronoun"})


@dataclass(frozen=True)
class JoinMatch:
    key: str
    #: "literal" when the two texts are identical, "normalised" when they
    #: met only after folding case and spacing.
    match: Literal["literal", "normalised"]


def join_match(object_text: str, subject_text: str) -> JoinMatch | None:
    if not object_text.strip() or not subject_text.strip():
        return None
    key = entity_key(object_text)
    if key != entity_key(subject_text):
        return None
    blocked = join_block_reason(object_text)
    if blocked in _NEVER_JOIN:
        return None
    if object_text == subject_text:
        return JoinMatch(key, "literal")
    if blocked is not None and object_text.strip() != subject_text.strip():
        return None
    return JoinMatch(key, "normalised")


def lookup_keys(object_text: str) -> tuple[str, ...]:
    """The subject spellings to look up for an object: its key and its exact
    text, only the exact text when folding could change its meaning, and
    nothing when it cannot name one thing."""
    if not object_text.strip():
        return ()
    blocked = join_block_reason(object_text)
    if blocked in _NEVER_JOIN:
        return ()
    if blocked is not None:
        return tuple(dict.fromkeys((object_text, object_text.strip())))
    return tuple(dict.fromkeys((entity_key(object_text), object_text)))
