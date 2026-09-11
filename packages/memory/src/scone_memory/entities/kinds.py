"""What sort of thing an entity is, as far as the predicates around it suggest.

Hints only: a person is whoever works somewhere, an organisation is where
someone works, a place is where something is based. A hint is never shown as
certain; disagreeing hints give a conflict rather than a guess, and every
inferred kind names the facts that suggested it.
"""

from __future__ import annotations

from typing import Iterable, Literal

EntityKind = Literal["person", "organisation", "place", "project", "product", "event", "concept"]
KindStatus = Literal["decided", "inferred", "unknown", "conflict"]
KIND_HINTS_VERSION = "kinds/1"

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
    **{predicate: "product" for predicate in "uses built_with depends_on runs_on".split()},
    **{predicate: "event" for predicate in "attended attends hosts organised organized spoke_at".split()},
}


def hint(predicate: str, role: Literal["subject", "object"]) -> EntityKind | None:
    table = _AS_SUBJECT if role == "subject" else _AS_OBJECT
    return table.get(predicate.replace(" ", "_"))


def infer_kind(hints: Iterable[tuple[EntityKind, int]]) -> tuple[EntityKind | None, KindStatus, tuple[int, ...]]:
    """Resolve hints into a kind, its status and up to eight supporting fact ids."""
    found = sorted(set(hints), key=lambda item: (item[1], item[0]))
    kinds = {kind for kind, _ in found}
    if not kinds:
        return None, "unknown", ()
    basis = tuple(dict.fromkeys(fact_id for _, fact_id in found))[:8]
    if len(kinds) > 1:
        return None, "conflict", basis
    return next(iter(kinds)), "inferred", basis
