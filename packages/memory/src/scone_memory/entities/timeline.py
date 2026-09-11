"""One entity's claims in valid time: what held when, and what replaced it.

Every fact the entity takes part in, as subject, as object or through its
values, becomes an item. Items are grouped into lanes by role and predicate
and ordered by when each held (valid_from, then id), never by when it was
written, so a backfilled fact lands where it belongs. Supersession
(``superseded_by``) and stored links between the entity's facts are
relations. ``holds_at_as_of`` marks the facts that held at the chosen
moment. Every item is re-read with its quote checked; if the space changed
while the timeline was read, it is read again, and ``consistent`` says
whether a still read was reached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ..core.timeutil import format_rfc3339, parse_rfc3339
from .grounding import checked_facts
from .project import EntityProjection, FactRole
from .query import resolve
from .read import load_projection
from .view import counts, projection_meta

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

MAX_ITEMS = 500
#: Links read per item; one more tells whether any were left out.
_LINKS_PER_ITEM = 16
_ATTEMPTS = 3


class TimelineEntityAmbiguous(Exception):
    def __init__(self, candidates: list[dict[str, str]], total: int, read: dict[str, object]) -> None:
        super().__init__("the name could mean several entities")
        self.candidates, self.total, self.read = candidates, total, read


class TimelineEntityMissing(Exception):
    def __init__(self, name: str, read: dict[str, object]) -> None:
        super().__init__(name)
        self.read = read


def _role(role: FactRole, entity_id: str) -> Literal["subject", "object", "value"]:
    if role.subject_id == entity_id:
        return "value" if role.object_id is None else "subject"
    return "object"


async def _once(engine: "MemoryEngine", space: str, name: str, *, when: str,
                limit: int) -> tuple[dict[str, object], EntityProjection]:
    projection, read = await load_projection(engine, space, mode="all", as_of=when)
    found = resolve(projection, name, limit=20)
    if found.status == "ambiguous":
        raise TimelineEntityAmbiguous([{"id": c.entity_id, "key": c.key, "label": c.label} for c in found.candidates],
                                      found.total, read)
    if found.status == "not_found":
        raise TimelineEntityMissing(name, read)
    entity_id = found.candidates[0].entity_id
    entities = {entity.entity_id: entity for entity in projection.entities}
    roles = [role for role in projection.roles if entity_id in (role.subject_id, role.object_id)]
    newest = sorted(roles, key=lambda role: (role.valid_from, role.fact_id), reverse=True)
    shown = sorted(newest[:limit], key=lambda role: (role.valid_from, role.fact_id))
    found_reasons = read.get("reasons")
    reasons = [str(reason) for reason in found_reasons] if isinstance(found_reasons, list) else []
    if len(roles) > limit:
        reasons.append("item_limit")
    moment = parse_rfc3339(when)
    reread = {int(str(fact["fact_id"])): fact
              for fact in await checked_facts(engine.documents, space, [role.fact_id for role in shown])}
    items: list[dict[str, object]] = []
    lanes: dict[str, list[int]] = {}
    for role in shown:
        fact = reread.get(role.fact_id)
        if fact is None:
            if "stale_evidence" not in reasons:
                reasons.append("stale_evidence")
            continue
        kind = _role(role, entity_id)
        far_id = role.object_id if kind == "subject" else role.subject_id if kind == "object" else None
        until = None if fact["valid_until"] is None else str(fact["valid_until"])
        items.append({
            "fact_id": role.fact_id, "role": kind, "subject": fact["subject"], "predicate": fact["predicate"],
            "object": fact["object"],
            "far": None if far_id is None else {"id": far_id, "key": entities[far_id].key, "label": entities[far_id].label},
            "valid_from": fact["valid_from"], "valid_until": until, "status": fact["status"],
            "excluded": fact["excluded"], "origin": fact["origin"],
            "superseded_by": fact["superseded_by"], "source_episode_id": fact["source_episode_id"],
            "grounding": fact["grounding"], "quote": fact["quote"] if fact["grounding"] == "quote_verified" else None,
            "holds_at_as_of": counts(str(fact["status"]), bool(fact["excluded"]), str(fact["valid_from"]), until,
                                     "current", moment)})
        lanes.setdefault(f"{kind}:{role.predicate}", []).append(role.fact_id)
    shown_ids = {int(str(item["fact_id"])) for item in items}
    relations: list[dict[str, object]] = [
        {"kind": "superseded_by", "from_fact": item["fact_id"], "to_fact": item["superseded_by"]}
        for item in items if item["superseded_by"] in shown_ids]
    reader = getattr(engine.documents, "fact_links_from", None)
    if not callable(reader):
        reasons.append("links_unavailable")
    else:
        seen: set[int] = set()
        for fact_id in sorted(shown_ids):
            links = await reader(space, fact_id, _LINKS_PER_ITEM + 1)
            if len(links) > _LINKS_PER_ITEM and "links_cut" not in reasons:
                reasons.append("links_cut")
            for link in links[:_LINKS_PER_ITEM]:
                if (link.link_id not in seen and link.space == space and link.from_fact in shown_ids
                        and link.to_fact in shown_ids):
                    seen.add(link.link_id)
                    relations.append({"kind": link.kind, "link_id": link.link_id, "from_fact": link.from_fact,
                                      "to_fact": link.to_fact})
    order = {"subject": 0, "value": 1, "object": 2}
    entity = entities[entity_id]
    return {
        "schema_version": 1, "space": space, "projection": projection_meta(projection), "as_of": when,
        "entity": {"id": entity.entity_id, "key": entity.key, "label": entity.label, "kind": entity.kind},
        "lanes": [{"id": lane, "role": lane.split(":", 1)[0], "predicate": lane.split(":", 1)[1], "items": ids}
                  for lane, ids in sorted(lanes.items(), key=lambda pair: (order[pair[0].split(":", 1)[0]], pair[0]))],
        "items": items,
        "relations": sorted(relations, key=lambda r: (str(r["kind"]), int(str(r["from_fact"])), int(str(r["to_fact"])))),
        "coverage": {"items_total": len(roles), "items_shown": len(items), "truncated": bool(reasons),
                     "reasons": reasons, "read": read},
    }, projection


async def timeline_view(engine: "MemoryEngine", space: str, name: str, *, as_of: str | None = None,
                        limit: int = 200) -> dict[str, object]:
    """The timeline of the entity ``name`` means, read between two matching
    revisions when the space holds still long enough."""
    if not 1 <= limit <= MAX_ITEMS:
        raise ValueError(f"limit must be 1..{MAX_ITEMS}")
    when = format_rfc3339(parse_rfc3339(as_of if as_of is not None else engine.clock()))
    view: dict[str, object] = {}
    for _attempt in range(_ATTEMPTS):
        view, projection = await _once(engine, space, name, when=when, limit=limit)
        if await engine.revision(space) == projection.revision:
            view["consistent"] = True
            return view
    view["consistent"] = False
    coverage = view["coverage"]
    assert isinstance(coverage, dict)
    coverage["reasons"] = [*coverage["reasons"], "ledger_changed_during_read"]
    coverage["truncated"] = True
    return view
