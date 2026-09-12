"""How the memory decides that two names are the same entity.

``entity_key`` is identity: case folded and spacing collapsed, nothing
looser. ``join_match`` is the one rule every join follows to decide whether
a claim's object names the thing another claim's subject is about: the
keys must be equal, and an object that cannot name one thing (prose, a
quotation, a pronoun) never joins, nor does a value whose case can change
its meaning ('MB' against 'mb'): stored subjects are folded, so even an
identical spelling cannot show the case was the same. Only a recorded
identity decision may join such a value. The
entity projection's classifier decides the same cases, so the graph a
person sees and the one retrieval walks agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..core.validation import entity_key
from ..entities.classify import join_block_reason

__all__ = ["JoinMatch", "entity_key", "join_match", "lookup_keys"]



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
    if join_block_reason(object_text) is not None:
        return None
    return JoinMatch(key, "literal" if object_text == subject_text else "normalised")


def lookup_keys(object_text: str) -> tuple[str, ...]:
    """The subject spellings to look up for an object: its key and its exact
    text, or nothing when the join rule would refuse every match."""
    if not object_text.strip() or join_block_reason(object_text) is not None:
        return ()
    return tuple(dict.fromkeys((entity_key(object_text), object_text)))
