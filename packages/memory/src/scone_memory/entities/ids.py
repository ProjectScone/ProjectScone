"""Stable, space-scoped identifiers for entities, relations and attributes.

An id is a hash of the space and what the item is, so the same ledger gives
the same ids on every store and every run, and the same name in two spaces
never shares an id. Any module can compute an entity's id from a name with
no I/O.
"""

from __future__ import annotations

import hashlib

ENTITY_ID_SCHEME = "scone.entity/1"


def _digest(*parts: str) -> str:
    # The ledger can hold a lone surrogate, which UTF-8 cannot encode.
    # surrogatepass gives such text bytes to hash and leaves every valid
    # string's bytes, and so its id, exactly as UTF-8 has them.
    return hashlib.sha256("\x1f".join(parts).encode("utf-8", "surrogatepass")).hexdigest()[:24]


def key_id(space: str, key: str) -> str:
    return "ent:" + _digest(ENTITY_ID_SCHEME, space, key)


def relation_id(space: str, subject_id: str, predicate: str, object_id: str) -> str:
    return "rel:" + _digest("scone.relation/1", space, subject_id, predicate, object_id)


def implied_id(space: str, subject_id: str, predicate: str, object_id: str) -> str:
    """An edge nobody claimed, which follows from ones they did. Its own
    prefix and its own scheme, so that it can never be read as the id of a
    claim, even when it joins the same two things by the same predicate."""
    return "imp:" + _digest("scone.implied/1", space, subject_id, predicate, object_id)


def attribute_id(space: str, entity_id: str, predicate: str, value: str) -> str:
    return "att:" + _digest("scone.attribute/1", space, entity_id, predicate, value)
