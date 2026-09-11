"""Views over an entity projection: what a person or a model is shown.

A view never adds anything the projection did not derive from the ledger.
It chooses which facts count (by status and time), keeps only relations and
attributes that at least one counted fact supports, restricts each to those
facts, ranks entities by how many counted claims they take part in, and
applies a budget. Everything it leaves out is counted, so a bounded view can
never pass for the whole graph.

Status modes:

- ``current``: facts that hold at ``as_of`` (the ledger's interval), not excluded;
- ``history``: every fact that ever held and had begun by ``as_of``, not excluded;
- ``proposed``: proposals awaiting review, not excluded;
- ``all``: everything, excluded facts included, for governance.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Literal, Mapping, Sequence

from ..core.timeutil import parse_rfc3339
from .classify import CLASSIFIER_VERSION
from .ids import ENTITY_ID_SCHEME
from .kinds import KIND_HINTS_VERSION
from .project import Entity, EntityProjection, FactRole, PROJECTION_VERSION

StatusMode = Literal["current", "history", "proposed", "all"]
STATUS_MODES: tuple[StatusMode, ...] = ("current", "history", "proposed", "all")
VIEW_SCHEMA_VERSION = 1


def counts(status: str, excluded: bool, valid_from: str, valid_until: str | None, mode: StatusMode,
           when: datetime) -> bool:
    """Whether a fact with these fields counts in a status mode at ``when``."""
    if mode == "all":
        return status != "declined"
    if excluded:
        return False
    if mode == "proposed":
        return status == "proposed"
    if status not in ("active", "closed") or parse_rfc3339(valid_from) > when:
        return False
    if mode == "history":
        return True
    return valid_until is None or parse_rfc3339(valid_until) > when


def _passes(role: FactRole, mode: StatusMode, when: datetime) -> bool:
    return counts(role.status, role.excluded, role.valid_from, role.valid_until, mode, when)


def support(roles: Sequence[FactRole]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for role in roles:
        counts["facts"] += 1
        counts[role.status] += 1
        counts["excluded"] += role.excluded
        counts[role.grounding] += 1
        counts[role.origin] += 1
    return {name: counts[name] for name in ("facts", "active", "closed", "proposed", "excluded", "quoted",
                                             "unquoted", "unsourced", "stated", "extracted", "inferred")}


def projection_meta(projection: EntityProjection) -> dict[str, object]:
    return {"version": PROJECTION_VERSION, "classifier": CLASSIFIER_VERSION, "kinds": KIND_HINTS_VERSION,
            "id_scheme": ENTITY_ID_SCHEME, "digest": projection.digest, "revision": projection.revision}


class _Counted:
    """The projection seen through one status mode and moment."""

    def __init__(self, projection: EntityProjection, mode: StatusMode, as_of: str) -> None:
        when = parse_rfc3339(as_of)
        self.roles = {role.fact_id: role for role in projection.roles if _passes(role, mode, when)}
        self.relations = [(relation, [self.roles[f] for f in relation.fact_ids if f in self.roles])
                          for relation in projection.relations]
        self.relations = [(relation, roles) for relation, roles in self.relations if roles]
        self.attributes = [(attribute, [self.roles[f] for f in attribute.fact_ids if f in self.roles])
                           for attribute in projection.attributes]
        self.attributes = [(attribute, roles) for attribute, roles in self.attributes if roles]
        self.score: Counter[str] = Counter()
        for role in self.roles.values():
            self.score[role.subject_id] += 1
            if role.object_id is not None:
                self.score[role.object_id] += 1
        self.entities = sorted((entity for entity in projection.entities if self.score[entity.entity_id]),
                               key=lambda entity: (-self.score[entity.entity_id], entity.entity_id))


def entity_record(entity: Entity, score: int) -> dict[str, object]:
    return {"id": entity.entity_id, "key": entity.key, "label": entity.label,
            "names": [{"text": form.text, "count": form.count} for form in entity.surface_forms],
            "kind": entity.kind, "kind_status": entity.kind_status, "kind_basis": list(entity.kind_basis),
            "flags": list(entity.flags), "claims": score}


def _reasons(coverage: Mapping[str, object]) -> list[str]:
    found = coverage.get("reasons", [])
    return [str(reason) for reason in found] if isinstance(found, list) else []


def knowledge_view(projection: EntityProjection, *, mode: StatusMode, as_of: str, limit: int,
                   attribute_limit: int, coverage: Mapping[str, object]) -> dict[str, object]:
    counted = _Counted(projection, mode, as_of)
    shown = counted.entities[:limit]
    ids = {entity.entity_id for entity in shown}
    relations = [(relation, roles) for relation, roles in counted.relations
                 if relation.subject_id in ids and relation.object_id in ids]
    attributes = [(attribute, roles) for attribute, roles in counted.attributes if attribute.entity_id in ids]
    reasons = _reasons(coverage)
    if len(shown) < len(counted.entities):
        reasons.append("entity_limit")
    if len(attributes) > attribute_limit:
        reasons.append("attribute_limit")
    return {
        "schema_version": VIEW_SCHEMA_VERSION, "space": projection.space, "projection": projection_meta(projection),
        "filters": {"status": mode, "as_of": as_of},
        "entities": [entity_record(entity, counted.score[entity.entity_id]) for entity in shown],
        "relations": [{"id": relation.relation_id, "subject_id": relation.subject_id,
                       "predicate": relation.predicate, "object_id": relation.object_id,
                       "fact_ids": [role.fact_id for role in roles], "support": support(roles),
                       "first_valid_from": min(role.valid_from for role in roles),
                       "last_valid_until": None if any(role.valid_until is None for role in roles)
                       else max(role.valid_until for role in roles if role.valid_until)}
                      for relation, roles in relations],
        "attributes": [{"id": attribute.attribute_id, "entity_id": attribute.entity_id,
                        "predicate": attribute.predicate, "value": attribute.value,
                        "literal_kind": attribute.literal_kind, "fact_ids": [role.fact_id for role in roles],
                        "support": support(roles)}
                       for attribute, roles in attributes[:attribute_limit]],
        "coverage": {**{key: value for key, value in coverage.items() if key != "reasons"},
                     "entities_total": len(counted.entities), "entities_shown": len(shown),
                     "relations_total": len(counted.relations), "relations_shown": len(relations),
                     "attributes_total": len(counted.attributes),
                     "attributes_shown": min(len(attributes), attribute_limit),
                     "truncated": bool(reasons), "reasons": reasons},
    }


def entity_listing(projection: EntityProjection, *, mode: StatusMode, as_of: str, limit: int, query: str | None,
                   coverage: Mapping[str, object]) -> dict[str, object]:
    counted = _Counted(projection, mode, as_of)
    matching = counted.entities
    if query:
        needle = query.casefold()
        matching = [entity for entity in matching
                    if needle in entity.key or any(needle in form.text.casefold() for form in entity.surface_forms)]
    reasons = _reasons(coverage)
    if len(matching) > limit:
        reasons.append("entity_limit")
    return {
        "schema_version": VIEW_SCHEMA_VERSION, "space": projection.space, "projection": projection_meta(projection),
        "filters": {"status": mode, "as_of": as_of, "q": query},
        "entities": [entity_record(entity, counted.score[entity.entity_id]) for entity in matching[:limit]],
        "coverage": {**{key: value for key, value in coverage.items() if key != "reasons"},
                     "entities_total": len(matching), "entities_shown": min(len(matching), limit),
                     "truncated": bool(reasons), "reasons": reasons},
    }
