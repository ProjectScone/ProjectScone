"""What sort of thing an entity is, as far as the predicates around it suggest.

Hints only: a person is whoever works somewhere, an organisation is where
someone works, a place is where something is based. A hint is never shown as
certain; disagreeing hints give a conflict rather than a guess, and every
inferred kind names the facts that suggested it.

A code graph's entities are hinted by their shape, since the code readers
name them by it and a predicate alone cannot tell a file from the class
it holds: a name whose last segment carries a suffix a reader knows is a
**file** (``pkg/a.py``, ``README.md``); a file, a colon and a name is a
**declaration** (``pkg/a.py:Thing.run``); a name with neither that a file
imports is a **module** (``typing``, ``github.com/gorilla/mux``). A module
a manifest also depends on is the package it comes from, so ``module``
beside ``product`` resolves to ``product`` rather than a conflict. Every
one of these names the facts it was read from, like any hint.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Literal

from .classify import is_declaration_name

EntityKind = Literal["person", "organisation", "place", "project", "product", "event", "concept",
                     "file", "declaration", "module"]
KindStatus = Literal["decided", "inferred", "unknown", "conflict"]
KIND_HINTS_VERSION = "kinds/2"
#: A file's last segment carries one of these when no code reader knows it.
_TEXT_SUFFIXES = frozenset((".md", ".markdown", ".rst", ".txt", ".json", ".toml", ".yaml", ".yml", ".sql",
                            ".cfg", ".ini", ".xml", ".csv"))
_IMPORTS = frozenset(("imports", "imports_when_called", "imports_for_types"))

_AS_SUBJECT: dict[str, EntityKind] = {predicate: "person" for predicate in """works_at worked_at works_for
    worked_for employed_by studied_at studies_at lives_in lived_in born_in born_on married_to reports_to
    knows met manages mentor_of friend_of sibling_of parent_of child_of colleague_of joined""".split()}
_AS_OBJECT: dict[str, EntityKind] = {
    **{predicate: "organisation" for predicate in """works_at worked_at works_for worked_for employed_by
        studied_at studies_at member_of founded acquired invested_in customer_of supplier_of
        subsidiary_of""".split()},
    **{predicate: "place" for predicate in """lives_in lived_in based_in located_in located_at
        headquartered_in born_in moved_to visited""".split()},
    **{predicate: "person" for predicate in """knows met married_to reports_to managed_by mentored_by
        friend_of sibling_of parent_of child_of colleague_of""".split()},
    **{predicate: "project" for predicate in "works_on contributes_to leads maintains".split()},
    **{predicate: "product" for predicate in "uses built_with depends_on runs_on runs_with connects_to".split()},
    **{predicate: "event" for predicate in "attended attends hosts organised organized spoke_at".split()},
}


def hint(predicate: str, role: Literal["subject", "object"]) -> EntityKind | None:
    table = _AS_SUBJECT if role == "subject" else _AS_OBJECT
    return table.get(predicate.replace(" ", "_"))


@lru_cache(maxsize=4096)
def is_file_name(text: str) -> bool:
    """Whether a name is a file's: its last segment carries a suffix a code
    reader knows or a text file has, and it holds no space, no colon and
    no scheme. A ratio (``1/2``), a time (``12:30``) or a URL is not.
    Cached: a file is named once per fact about it."""
    if not text or ":" in text or any(c.isspace() for c in text) or text.endswith("/"):
        return False
    name = text.rsplit("/", 1)[-1]
    dot = name.rfind(".")
    if dot <= 0 or dot == len(name) - 1:
        return False
    from ..ingestion.code import code_language

    return code_language(name) is not None or name[dot:].lower() in _TEXT_SUFFIXES


def code_kind(label: str, predicate: str, role: Literal["subject", "object"]) -> EntityKind | None:
    """The kind a code entity's shape gives it, or None when the shape says
    nothing: a file, a declaration (a file, a colon and a name), or a
    module when a file imports it."""
    text = label.strip()
    if ":" in text:
        path, name = text.split(":", 1)
        # The name after the colon is a chain of identifiers, as a reader
        # writes it; `README.md: Getting Started` is a sentence about a file.
        return "declaration" if name and not name[0].isspace() and is_declaration_name(name) and is_file_name(path) else None
    if is_file_name(text):
        return "file"
    if role == "object" and predicate.replace(" ", "_") in _IMPORTS and text and not any(c.isspace() for c in text):
        return "module"
    return None


def infer_kind(hints: Iterable[tuple[EntityKind, int]]) -> tuple[EntityKind | None, KindStatus, tuple[int, ...]]:
    """Resolve hints into a kind, its status and up to eight supporting fact ids."""
    found = sorted(set(hints), key=lambda item: (item[1], item[0]))
    kinds = sorted({kind for kind, _ in found})
    if kinds == ["module", "product"]:
        # A module a manifest also depends on is the package it comes from.
        kinds = ["product"]
    if not kinds:
        return None, "unknown", ()
    # The first fact behind each kind comes first, so a conflict's bounded
    # evidence always shows every side of it.
    leading = sorted(next(fact_id for kind, fact_id in found if kind == wanted) for wanted in kinds)
    basis = tuple(dict.fromkeys((*leading, *(fact_id for _, fact_id in found))))[:8]
    if len(kinds) > 1:
        return None, "conflict", basis
    return kinds[0], "inferred", basis
