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
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

from ..core.timeutil import parse_rfc3339
from .classify import CLASSIFIER_VERSION
from .ids import ENTITY_ID_SCHEME
from .kinds import KIND_HINTS_VERSION
from .project import Entity, EntityProjection, FactRole, PROJECTION_VERSION, merged_periods

if TYPE_CHECKING:
    from .usage import Usage

StatusMode = Literal["current", "history", "proposed", "all"]
#: Which way a seeded walk follows a relation: out from its subject to its
#: object, in from its object to its subject (what depends on the seed), or
#: both.
WalkDirection = Literal["both", "out", "in"]
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
        # A relation that follows from others holds only while every claim
        # under it does, so all of them must pass the filter, not just one.
        self.implied = [(item, [self.roles[f] for f in item.fact_ids if f in self.roles])
                        for item in projection.implied]
        self.implied = [(item, roles) for item, roles in self.implied if len(roles) == len(item.fact_ids)]
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


def _walk(counted: "_Counted", seeds: Sequence[str], limit: int, hub_degree: int,
          direction: WalkDirection = "both", hops: int | None = None) -> tuple[list[Entity], dict[str, int], int, bool, bool]:
    """Entities reached from the seeds, breadth first over relations in
    ``direction``, each entity's neighbours in order of the facts behind the
    relation, for at most ``hops`` steps. An entity with more than
    ``hub_degree`` relations is shown but not walked through unless it is a
    seed (its degree counts both ways, whatever the direction). Returns the
    entities (at most ``limit``), each one's hop from the nearest seed, how
    many hubs were not walked through, whether the walk reached more than
    it shows, and whether it stopped at ``hops`` with more to reach."""
    by_id = {entity.entity_id: entity for entity in counted.entities}
    degree: dict[str, int] = {}
    following_of: dict[str, list[tuple[int, str]]] = {}
    for relation, roles in counted.relations:
        if relation.subject_id == relation.object_id:
            continue
        for near, far, way in ((relation.subject_id, relation.object_id, "out"),
                               (relation.object_id, relation.subject_id, "in")):
            degree[near] = degree.get(near, 0) + 1
            if direction in ("both", way):
                following_of.setdefault(near, []).append((len(roles), far))
    starts = [seed for seed in dict.fromkeys(seeds) if seed in by_id]

    def walkable(entity_id: str) -> bool:
        return entity_id in starts or degree.get(entity_id, 0) <= hub_degree

    hop_of = {seed: 0 for seed in starts}
    order, frontier, hubs, step = list(starts), list(starts), 0, 0
    while frontier and len(order) <= limit and (hops is None or step < hops):
        step += 1
        following: list[str] = []
        for entity_id in frontier:
            if not walkable(entity_id):
                hubs += 1
                continue
            for _support, far in sorted(following_of.get(entity_id, []), key=lambda item: (-item[0], item[1])):
                if far not in hop_of:
                    hop_of[far] = step
                    following.append(far)
        order += following
        frontier = following
    # Stopped by the hop limit only if a step more would have reached more.
    stopped = hops is not None and step == hops and any(
        far not in hop_of for entity_id in frontier if walkable(entity_id) for _, far in following_of.get(entity_id, []))
    return [by_id[entity_id] for entity_id in order[:limit]], hop_of, hubs, len(order) > limit, stopped


def knowledge_view(projection: EntityProjection, *, mode: StatusMode, as_of: str, limit: int,
                   attribute_limit: int, coverage: Mapping[str, object], seeds: Sequence[str] = (),
                   hub_degree: int = 64, offset: int = 0, direction: WalkDirection = "both",
                   hops: int | None = None, usage: "Usage | None" = None) -> dict[str, object]:
    """Entities, the relations among them and their values. Without seeds,
    the entities ranked by the claims they take part in, ``limit`` from
    ``offset``; with seeds, those reached from them breadth first, following
    relations in ``direction`` for at most ``hops`` steps, each entity with
    its ``hop`` from the nearest seed."""
    counted = _Counted(projection, mode, as_of)
    reasons = _reasons(coverage)
    more = False
    hop_of: dict[str, int] = {}
    if seeds:
        shown, hop_of, hubs, cut, stopped = _walk(counted, seeds, limit, hub_degree, direction, hops)
        if hubs:
            reasons.append("hub_skipped")
        if stopped:
            reasons.append("hop_limit")
        if cut:
            reasons.append("entity_limit")
        elif len(shown) < len(counted.entities):
            reasons.append("outside_walk")  # entities the walk never reached from its seeds
    else:
        shown = counted.entities[offset:offset + limit]
        more = offset + limit < len(counted.entities)
        if len(shown) < len(counted.entities):
            reasons.append("entity_limit")
    ids = {entity.entity_id for entity in shown}
    relations = [(relation, roles) for relation, roles in counted.relations
                 if relation.subject_id in ids and relation.object_id in ids]
    implied = [(item, roles) for item, roles in counted.implied
               if item.subject_id in ids and item.object_id in ids]
    attributes = [(attribute, roles) for attribute, roles in counted.attributes if attribute.entity_id in ids]
    if len(attributes) > attribute_limit:
        reasons.append("attribute_limit")
    # How many recalls returned a fact each entity or relation stands on.
    entity_uses: Counter[str] = Counter()
    relation_uses: Counter[str] = Counter()
    if usage is not None and usage.available:
        from .usage import recalled_by_entity

        entity_uses = recalled_by_entity(usage, counted.roles)
        relation_of = {fact_id: relation.relation_id for relation, _ in counted.relations for fact_id in relation.fact_ids}
        for returned in usage.returned:
            relation_uses.update({relation_of[fact_id] for fact_id in returned if fact_id in relation_of})

    def used(counts: Counter[str], key: str) -> dict[str, object]:
        if usage is None:
            return {}
        return {"recalled": counts[key] if usage.available else None}

    return {
        "schema_version": VIEW_SCHEMA_VERSION, "space": projection.space, "projection": projection_meta(projection),
        "filters": {"status": mode, "as_of": as_of,
                    **({"seeds": list(dict.fromkeys(seeds)), "hub_degree": hub_degree, "direction": direction,
                        "hops": hops} if seeds else {})},
        "entities": [{**entity_record(entity, counted.score[entity.entity_id]),
                      **({"hop": hop_of[entity.entity_id]} if seeds else {}), **used(entity_uses, entity.entity_id)}
                     for entity in shown],
        "relations": [{"id": relation.relation_id, "subject_id": relation.subject_id,
                       "predicate": relation.predicate, "object_id": relation.object_id,
                       "fact_ids": [role.fact_id for role in roles], "support": support(roles),
                       # The stretches it held over, so that first and last
                       # cannot be read as one unbroken spell.
                       "periods": [list(period) for period in merged_periods(
                           [(role.valid_from, role.valid_until) for role in roles])],
                       "first_valid_from": min(role.valid_from for role in roles),
                       "last_valid_until": None if any(role.valid_until is None for role in roles)
                       else max(role.valid_until for role in roles if role.valid_until),
                       **used(relation_uses, relation.relation_id)}
                      for relation, roles in relations],
        # Kept apart from the relations above: what follows from claims is
        # never listed as a claim. Each says what it was worked out from.
        "implied": [{"id": item.relation_id, "subject_id": item.subject_id, "predicate": item.predicate,
                     "object_id": item.object_id, "fact_ids": [role.fact_id for role in roles],
                     "support": support(roles), "follows": item.follows,
                     "follows_from": list(item.follows_from),
                     # The stretches the claims under it actually shared,
                     # as the projection worked them out. A first and last
                     # moment taken from the facts here would span a gap
                     # that no claim does.
                     "periods": [list(period) for period in item.periods],
                     "first_valid_from": item.first_valid_from, "last_valid_until": item.last_valid_until}
                    for item, roles in implied],
        "attributes": [{"id": attribute.attribute_id, "entity_id": attribute.entity_id,
                        "predicate": attribute.predicate, "value": attribute.value,
                        "literal_kind": attribute.literal_kind, "fact_ids": [role.fact_id for role in roles],
                        "support": support(roles)}
                       for attribute, roles in attributes[:attribute_limit]],
        "coverage": {**{key: value for key, value in coverage.items() if key != "reasons"},
                     "entities_total": len(counted.entities), "entities_shown": len(shown),
                     "relations_total": len(counted.relations), "relations_shown": len(relations),
                     # `is None`, never truthiness: RelationMeanings() is falsy, so
                     # an explicitly empty vocabulary serialised as null and a
                     # reader could not tell "there are none" from "nothing said".
                     "meanings": (None if projection.meanings is None
                                  else projection.meanings.record()),
                     "vocabulary_source": projection.vocabulary_source,
                     "vocabulary_why": projection.vocabulary_why,
                     "implied_total": len(counted.implied), "implied_shown": len(implied),
                     **({"implied_capped": True} if projection.implied_capped else {}),
                     "attributes_total": len(counted.attributes),
                     "attributes_shown": min(len(attributes), attribute_limit),
                     "truncated": bool(reasons), "reasons": reasons,
                     **({"next_offset": offset + limit} if more else {}),
                     **({"usage": usage.record()} if usage is not None else {})},
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
                     # Said on every view built from a projection, not only
                     # the detailed one: a caller cannot tell why two answers
                     # about one space differ unless each says which
                     # vocabulary it was built under.
                     "vocabulary_source": projection.vocabulary_source,
                     "vocabulary_why": projection.vocabulary_why,
                     "truncated": bool(reasons), "reasons": reasons},
    }


class SeedRefused(Exception):
    """A seed name no walk can start from: ``status`` 404 or 409, and the
    answer to give, with the read's coverage so a capped read is never
    taken for absence or a complete list."""

    def __init__(self, status: int, answer: dict[str, object]) -> None:
        super().__init__(answer["error"])
        self.status, self.answer = status, answer


def walk_seeds(projection: EntityProjection, names: Sequence[str],
               coverage: Mapping[str, object]) -> list[str]:
    """The entity ids a seeded walk starts from, one per name; an unknown
    or ambiguous name raises ``SeedRefused``."""
    from .query import resolve
    from .read import read_record

    seeds: list[str] = []
    for name in names:
        found = resolve(projection, name)
        if found.status == "resolved":
            seeds.append(found.candidates[0].entity_id)
            continue
        complete, read = read_record(dict(coverage))
        if found.status == "ambiguous":
            raise SeedRefused(409, {
                "error": f"{name!r} could mean several entities", "name": name,
                "candidates": [{"id": c.entity_id, "key": c.key, "label": c.label} for c in found.candidates],
                "candidates_total": found.total,
                "truncated": found.total > len(found.candidates) or not complete, "complete": complete,
                "coverage": read})
        where = "" if complete else " in the facts read; the read was capped, so it may exist"
        raise SeedRefused(404, {"error": f"no entity is named {name!r}{where}", "name": name,
                                "complete": complete, "coverage": read})
    return seeds
