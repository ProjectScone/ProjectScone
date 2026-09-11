"""HTTP routes for entities and the knowledge map.

``GET /v1/graph/knowledge`` draws a space's entities, the relations between
them and what is known about them; ``GET /v1/entities`` lists entities. Both
read the ledger into a projection per request (never on a recall path), are
scoped by the caller's key, and carry the projection's version and digest so
a client can tell when ids or classifications changed. Every relation and
attribute lists the facts behind it, with their status, grounding and
origin; ``coverage`` counts everything a bounded view left out. The contract is
described in docs/retrieval-and-storage.md under "Knowledge graph".
"""

from __future__ import annotations

import base64
import json
from typing import Callable, Literal, Optional

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from ..core.errors import InvalidInput, NotFound
from ..core.timeutil import format_rfc3339, parse_rfc3339
from ..entities.analysis import GraphAnalysis, analyze_projection
from ..entities.context import MAX_NAME, MAX_NAMES, MAX_QUESTION, ContextLimits, graph_context
from ..entities.sources import sources_view
from ..entities.timeline import TimelineEntityAmbiguous, TimelineEntityMissing, timeline_view
from ..entities.grounding import checked_facts
from ..entities.export import ExportFormat, export_graph
from ..entities.project import EntityProjection, Relation
from ..entities.query import Resolution, neighbourhood, paths_between, resolve
from ..entities.read import load_projection, read_record
from ..entities.schema import MAX_BYTES, MAX_BYTES_LIMIT, MAX_PREDICATES, schema_record
from ..entities.report import build_report, render_markdown
from ..entities.service import ProjectionBuilding
from ..entities.view import StatusMode, entity_listing, entity_record, knowledge_view, projection_meta, support
from ..memory.engine import MemoryEngine
from .responses import LedgerJSONResponse


class Support(BaseModel):
    facts: int
    active: int
    closed: int
    proposed: int
    excluded: int
    quoted: int
    unquoted: int
    unsourced: int
    stated: int
    extracted: int
    inferred: int


class Name(BaseModel):
    text: str
    count: int


class EntityOut(BaseModel):
    id: str
    key: str
    label: str
    names: list[Name]
    kind: Optional[str]
    kind_status: Literal["decided", "inferred", "unknown", "conflict"]
    kind_basis: list[int]
    flags: list[str]
    #: Claims in this view the entity takes part in, as subject or object.
    claims: int


class RelationOut(BaseModel):
    id: str
    subject_id: str
    predicate: str
    object_id: str
    fact_ids: list[int]
    support: Support
    first_valid_from: str
    last_valid_until: Optional[str]


class AttributeOut(BaseModel):
    id: str
    entity_id: str
    predicate: str
    value: str
    literal_kind: Optional[str]
    fact_ids: list[int]
    support: Support


class ProjectionMeta(BaseModel):
    version: str
    classifier: str
    kinds: str
    id_scheme: str
    digest: str
    revision: int


class Filters(BaseModel):
    status: StatusMode
    as_of: str
    #: The entity ids a seeded view walked out from, and the hub degree it
    #: walked under.
    seeds: Optional[list[str]] = None
    hub_degree: Optional[int] = None


class ListFilters(Filters):
    q: Optional[str]


class Coverage(BaseModel):
    facts_read: int
    #: Facts that count in this view's status mode and moment; the view is
    #: projected from these alone.
    facts_counted: int
    facts_limit: int
    #: "paged" when the store paged its ledger newest first, else "unpaged".
    read_mode: Optional[Literal["paged", "unpaged"]] = None
    entities_total: int
    entities_shown: int
    relations_total: Optional[int] = None
    relations_shown: Optional[int] = None
    attributes_total: Optional[int] = None
    attributes_shown: Optional[int] = None
    truncated: bool
    reasons: list[str]
    #: Pass as ``cursor`` for the next page of the ranking; absent on the last.
    next_cursor: Optional[str] = None


class GroupingCommunity(BaseModel):
    id: str
    label: str
    #: Shown entities in this community; the community may hold more.
    members: list[str]
    size: int


class GroupingImportance(BaseModel):
    entity_id: str
    community_id: str
    degree: int
    pagerank: float
    betweenness: float
    participation: float


class AnalysisCoverage(BaseModel):
    """What the analysis covered. Separate from the view's coverage: a view
    can show every entity while the analysis behind it was capped, or its
    betweenness estimated from a sample of sources."""
    entities_total: int
    entities_analysed: int
    isolated_entities: int
    truncated: bool
    reasons: list[str]
    #: "exact", or "sampled:N" when estimated from N sampled sources.
    betweenness: str
    betweenness_estimated: bool
    levels: int
    #: The modularity resolution the communities were found at.
    resolution: float = 1.0


class Groupings(BaseModel):
    """Computed from the view's recorded relations: analysis, never facts."""
    basis: Literal["computed"]
    method: str
    modularity: float
    membership: dict[str, str]
    communities: list[GroupingCommunity]
    importance: list[GroupingImportance]
    coverage: AnalysisCoverage


class KnowledgeView(BaseModel):
    schema_version: int
    space: str
    projection: ProjectionMeta
    filters: Filters
    entities: list[EntityOut]
    relations: list[RelationOut]
    attributes: list[AttributeOut]
    coverage: Coverage
    groupings: Optional[Groupings] = None


class EntityList(BaseModel):
    schema_version: int
    space: str
    projection: ProjectionMeta
    filters: ListFilters
    entities: list[EntityOut]
    coverage: Coverage


def _moment(engine: MemoryEngine, as_of: Optional[str]) -> str:
    if as_of is None:
        return engine.clock()
    try:
        return format_rfc3339(parse_rfc3339(as_of))
    except ValueError:
        raise InvalidInput("as_of must be an RFC 3339 timestamp") from None


def _groupings(analysis: GraphAnalysis, view: dict[str, object]) -> dict[str, object]:
    shown = {str(entity["id"]) for entity in view["entities"]}  # type: ignore[attr-defined]
    membership = {item.entity_id: item.community_id for item in analysis.importance if item.entity_id in shown}
    return {
        "basis": "computed", "method": analysis.version, "modularity": analysis.modularity, "membership": membership,
        "communities": [{"id": community.community_id, "label": community.label, "size": len(community.members),
                         "members": [member for member in community.members if member in shown]}
                        for community in analysis.communities if any(member in shown for member in community.members)],
        "importance": [{"entity_id": item.entity_id, "community_id": item.community_id, "degree": item.degree,
                        "pagerank": item.pagerank, "betweenness": item.betweenness, "participation": item.participation}
                       for item in analysis.importance if item.entity_id in shown],
        "coverage": analysis.coverage.record(),
    }


def _read_reasons(coverage: dict[str, object]) -> list[str]:
    found = coverage.get("reasons")
    return [str(reason) for reason in found] if isinstance(found, list) else []


_read = read_record


def _candidates(found: Resolution) -> dict[str, object]:
    return {"candidates": [{"id": c.entity_id, "key": c.key, "label": c.label} for c in found.candidates],
            "candidates_total": found.total, "truncated": found.total > len(found.candidates)}


def _write_cursor(offset: int, digest: str) -> str:
    return base64.urlsafe_b64encode(json.dumps({"v": 1, "offset": offset, "digest": digest}).encode()).decode()


def _read_cursor(cursor: str) -> tuple[int, str]:
    """The offset and projection digest a cursor names, or a 422."""
    try:
        found = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        offset, digest = found["offset"], found["digest"]
        if found.get("v") != 1 or type(offset) is not int or offset < 0 or not isinstance(digest, str):
            raise ValueError(cursor)
    except (ValueError, KeyError, TypeError) as error:
        raise InvalidInput("cursor is not one this server issued") from error
    return offset, digest


def _names(projection: EntityProjection) -> dict[str, dict[str, str]]:
    return {entity.entity_id: {"id": entity.entity_id, "label": entity.label, "key": entity.key}
            for entity in projection.entities}


def mount_entity_routes(app: FastAPI, engine: MemoryEngine, space_for: Callable[..., object]) -> None:
    @app.exception_handler(ProjectionBuilding)
    async def _building(_: Request, error: ProjectionBuilding) -> LedgerJSONResponse:
        return LedgerJSONResponse({"error": str(error), "code": "projection_building"}, status_code=503,
                            headers={"Retry-After": "1"})

    @app.get("/v1/graph/knowledge", response_model=KnowledgeView, response_model_exclude_unset=True)
    async def get_knowledge(
        status: StatusMode = "current", as_of: Optional[str] = None,
        limit: int = Query(default=150, ge=1, le=1000), attribute_limit: int = Query(default=300, ge=0, le=5000),
        groupings: bool = False, resolution: float = Query(default=1.0, gt=0, le=10),
        seed: list[str] = Query(default=[]), hub_degree: int = Query(default=64, ge=1, le=100_000),
        cursor: Optional[str] = Query(default=None, max_length=512), space: str = Depends(space_for),
    ) -> dict[str, object] | LedgerJSONResponse:
        """Entities, the relations between them and their values, as the ledger records them.

        Without ``seed``, the entities ranked by the claims they take part
        in, a page at a time: ``coverage.next_cursor`` asks for the next, and
        a cursor from before the graph changed is refused with 409. With
        ``seed`` (names or ids), the entities reached from them breadth
        first in both directions, hubs past ``hub_degree`` shown but not
        walked through. With ``groupings=true`` a separate, computed block
        assigns the shown entities to communities, found at ``resolution``,
        and scores their importance."""
        when = _moment(engine, as_of)
        projection, coverage = await load_projection(engine, space, mode=status, as_of=when)
        offset = 0
        if cursor is not None:
            offset, digest = _read_cursor(cursor)
            if digest != projection.digest:
                return LedgerJSONResponse(status_code=409, content={
                    "error": "the graph changed since this cursor was issued; start again without it",
                    "code": "cursor_stale"})
        if len(seed) > 24:
            raise InvalidInput("at most 24 seeds")
        seeds: list[str] = []
        for name in seed:
            found = resolve(projection, name)
            if found.status == "ambiguous":
                complete, read = _read(coverage)
                listed = _candidates(found)
                return LedgerJSONResponse(status_code=409, content={
                    "error": f"{name!r} could mean several entities", "name": name, **listed,
                    "truncated": bool(listed["truncated"]) or not complete, "complete": complete, "coverage": read})
            if found.status == "not_found":
                complete, read = _read(coverage)
                where = "" if complete else " in the facts read; the read was capped, so it may exist"
                return LedgerJSONResponse(status_code=404, content={"error": f"no entity is named {name!r}{where}",
                                                              "name": name, "complete": complete, "coverage": read})
            seeds.append(found.candidates[0].entity_id)
        view = knowledge_view(projection, mode=status, as_of=when, limit=limit, attribute_limit=attribute_limit,
                              coverage=coverage, seeds=seeds, hub_degree=hub_degree, offset=offset)
        page = view["coverage"]
        assert isinstance(page, dict)
        next_offset = page.pop("next_offset", None)
        if isinstance(next_offset, int):
            page["next_cursor"] = _write_cursor(next_offset, projection.digest)
        if groupings:
            view["groupings"] = _groupings(analyze_projection(projection, resolution=resolution), view)
        return view

    @app.get("/v1/graph/report", response_model=None)
    async def get_report(
        status: StatusMode = "current", as_of: Optional[str] = None,
        format: Literal["json", "markdown"] = "json", resolution: float = Query(default=1.0, gt=0, le=10),
        exclude_hubs: Optional[float] = Query(default=None, ge=50, le=100), space: str = Depends(space_for),
    ) -> dict[str, object] | PlainTextResponse:
        """The space's communities, central entities, surprising connections and
        questions worth asking, computed from recorded facts and citing them.
        ``resolution`` sets how fine the communities are; ``exclude_hubs``
        leaves entities above that degree percentile out of the central ranking."""
        when = _moment(engine, as_of)
        projection, coverage = await load_projection(engine, space, mode=status, as_of=when)
        view = knowledge_view(projection, mode=status, as_of=when, limit=1, attribute_limit=0, coverage=coverage)
        report = build_report(projection, analyze_projection(projection, resolution=resolution),
                              meta=view["projection"], filters={"status": status, "as_of": when},  # type: ignore[arg-type]
                              coverage=coverage, exclude_hubs=exclude_hubs)
        if format == "markdown":
            return PlainTextResponse(render_markdown(report), media_type="text/markdown; charset=utf-8")
        return report

    @app.get("/v1/graph/schema")
    async def get_schema(
        status: StatusMode = "current", as_of: Optional[str] = None,
        limit: int = Query(default=200, ge=1, le=MAX_PREDICATES),
        max_bytes: int = Query(default=MAX_BYTES, ge=1_024, le=MAX_BYTES_LIMIT), space: str = Depends(space_for),
    ) -> dict[str, object]:
        """What the space's graph is made of: the kinds its entities have, the
        predicates its facts use and which kinds each joins, most used first.
        Read it to know what the graph could be asked. ``max_bytes`` bounds
        the listed predicates' JSON; a long predicate is clipped and marked."""
        return await schema_record(engine, space, status=status, as_of=_moment(engine, as_of), limit=limit,
                                   max_bytes=max_bytes)

    @app.get("/v1/graph/export", response_model=None)
    async def get_export(
        format: ExportFormat = "json", status: StatusMode = "current", as_of: Optional[str] = None,
        space: str = Depends(space_for),
    ) -> Response:
        """The view's whole graph as a file another tool reads: node-link JSON,
        GraphML, dynamic GEXF, Cypher, CSV, JSON-LD, an Obsidian vault or a wiki
        an agent can crawl. The file says which
        projection it holds and, when the read was capped, that it is partial."""
        when = _moment(engine, as_of)
        projection, coverage = await load_projection(engine, space, mode=status, as_of=when)
        reasons = coverage.get("reasons") or []
        about = {"status": status, "as_of": when, "coverage": {**coverage, "truncated": bool(reasons)}}
        exported = export_graph(projection, format, about=about)
        return Response(exported.body, media_type=exported.media_type, headers={
            "Content-Disposition": f'attachment; filename="{exported.filename}"',
            "X-Scone-Projection-Digest": projection.digest, "X-Scone-Truncated": "true" if reasons else "false",
            "X-Scone-Space": space, "X-Scone-Projection-Revision": str(projection.revision),
            "X-Scone-Status": status, "X-Scone-As-Of": when})

    @app.get("/v1/graph/context")
    async def get_context(
        names: list[str] = Query(default=[]),
        q: Optional[str] = Query(default=None, min_length=1, max_length=MAX_QUESTION),
        max_hops: int = Query(default=2, ge=1, le=4), max_bytes: int = Query(default=8_000, ge=512, le=64_000),
        status: StatusMode = "current", as_of: Optional[str] = None, space: str = Depends(space_for),
    ) -> dict[str, object]:
        """What the graph records around some names, or the entities a question
        names, as one line per item for a model: coverage first, then the
        entities, paths between them, relations by hop and values, each citing
        facts re-read now. Cut between lines to ``max_bytes``."""
        if not names and q is None:
            raise InvalidInput("give names or q")
        if len(names) > MAX_NAMES or any(not 1 <= len(name) <= MAX_NAME for name in names):
            raise InvalidInput(f"names: at most {MAX_NAMES}, each 1..{MAX_NAME} characters")
        when = _moment(engine, as_of)
        packet = await graph_context(engine, space, names=names, question=q, status=status, as_of=when,
                                     limits=ContextLimits(max_bytes=max_bytes, max_hops=max_hops))
        return packet.record(space, status, when)

    @app.get("/v1/graph/sources")
    async def get_sources(
        episode: int = Query(ge=1), max_chunks: int = Query(default=64, ge=1, le=1000),
        max_claims: int = Query(default=200, ge=1, le=1000), space: str = Depends(space_for),
    ) -> dict[str, object]:
        """One source followed through: its sections, chunks, the claims quoting
        it with exact byte spans, the entities they name, and mentions of known
        entities kept apart. A missing episode is a 404, a forgotten one a 410."""
        return await sources_view(engine, space, episode, max_chunks=max_chunks, max_claims=max_claims)

    @app.get("/v1/graph/timeline", response_model=None)
    async def get_timeline(
        entity: str = Query(min_length=1, max_length=200), as_of: Optional[str] = None,
        limit: int = Query(default=200, ge=1, le=500), space: str = Depends(space_for),
    ) -> dict[str, object] | LedgerJSONResponse:
        """One entity's facts in valid time, in lanes by role and predicate,
        with supersession and stored links between them, and a marker for
        what held at ``as_of``."""
        when = _moment(engine, as_of)
        try:
            return await timeline_view(engine, space, entity, as_of=when, limit=limit)
        except TimelineEntityAmbiguous as ambiguous:
            return LedgerJSONResponse(status_code=409, content=ambiguous.record(entity))
        except TimelineEntityMissing as missing:
            return LedgerJSONResponse(status_code=404, content=missing.record(entity))

    @app.get("/v1/entities/resolve")
    async def get_resolved(
        name: str = Query(min_length=1, max_length=200), status: StatusMode = "current", as_of: Optional[str] = None,
        limit: int = Query(default=20, ge=1, le=200), space: str = Depends(space_for),
    ) -> dict[str, object]:
        """Which entity a name or id means. When it could mean several, the
        first ``limit`` candidates and how many there are; it never guesses."""
        when = _moment(engine, as_of)
        projection, coverage = await load_projection(engine, space, mode=status, as_of=when)
        complete, read = _read(coverage)
        found = resolve(projection, name, limit=limit)
        state = "not_found_in_read" if found.status == "not_found" and not complete else found.status
        return {"status": state, "tier": found.tier, **_candidates(found), "complete": complete, "coverage": read}

    @app.get("/v1/graph/path", response_model=None)
    async def get_path(
        from_: str = Query(alias="from", min_length=1, max_length=200), to: str = Query(min_length=1, max_length=200),
        max_hops: int = Query(default=4, ge=1, le=8), limit: int = Query(default=3, ge=1, le=20),
        hub_degree: int = Query(default=200, ge=2, le=100_000), status: StatusMode = "current",
        as_of: Optional[str] = None, space: str = Depends(space_for),
    ) -> dict[str, object] | LedgerJSONResponse:
        """How two entities connect: up to ``limit`` shortest routes, each hop
        with its relation's direction and the facts behind it."""
        when = _moment(engine, as_of)
        projection, coverage = await load_projection(engine, space, mode=status, as_of=when)
        complete, read = _read(coverage)
        ends = []
        for name in (from_, to):
            found = resolve(projection, name)
            if found.status == "not_found":
                where = "" if complete else " in the facts read; the read was capped, so it may exist"
                return LedgerJSONResponse(status_code=404, content={
                    "error": f"no entity is named {name!r}{where}", "name": name, "complete": complete, "coverage": read})
            if found.status == "ambiguous":
                return LedgerJSONResponse(status_code=409, content={
                    "error": f"{name!r} could mean several entities", "name": name, **_candidates(found),
                    "complete": complete, "coverage": read})
            ends.append(found.candidates[0].entity_id)
        result = paths_between(projection, ends[0], ends[1], max_hops=max_hops, limit=limit, hub_degree=hub_degree)
        names = _names(projection)
        state = "not_connected_in_read" if result.status == "disconnected" and not complete else result.status
        return {"schema_version": 1, "space": space, "projection": projection_meta(projection),
                "filters": {"status": status, "as_of": when},
                "policy": {"max_hops": max_hops, "limit": limit, "hub_degree": hub_degree},
                "status": state, "complete": complete, "coverage": read, "from": names[ends[0]], "to": names[ends[1]],
                "hubs_skipped": [names[hub] for hub in result.hubs_skipped], "truncated": result.truncated,
                "paths": [{"entities": [names[entity] for entity in path.entity_ids],
                           "hops": [{"relation_id": hop.relation_id, "subject": names[hop.subject_id],
                                     "predicate": hop.predicate, "object": names[hop.object_id],
                                     "direction": hop.direction, "fact_ids": list(hop.fact_ids)} for hop in path.hops]}
                          for path in result.paths]}

    @app.get("/v1/entities", response_model=EntityList)
    async def get_entities(
        status: StatusMode = "current", as_of: Optional[str] = None, q: Optional[str] = Query(default=None, max_length=200),
        limit: int = Query(default=100, ge=1, le=1000), space: str = Depends(space_for),
    ) -> dict[str, object]:
        """Entities ranked by the claims they take part in, optionally filtered by name."""
        when = _moment(engine, as_of)
        projection, coverage = await load_projection(engine, space, mode=status, as_of=when)
        return entity_listing(projection, mode=status, as_of=when, limit=limit, query=q, coverage=coverage)

    @app.get("/v1/entities/{entity_id}", response_model=None)
    async def get_entity(
        entity_id: str, status: StatusMode = "current", as_of: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=500), space: str = Depends(space_for),
    ) -> dict[str, object] | LedgerJSONResponse:
        """One entity: its relations in both directions grouped by predicate, its
        values, and every fact behind them re-read with its quote checked now.

        The page is built from one read of the ledger and then re-reads its
        facts. A write between the two would leave them disagreeing, so the
        page is read again; ``consistent`` is false only if the space kept
        changing through every attempt."""
        when = _moment(engine, as_of)
        for _attempt in range(_PAGE_ATTEMPTS):
            projection, coverage = await load_projection(engine, space, mode=status, as_of=when)
            page = await _entity_page(engine, space, projection, entity_id, status=status, when=when, limit=limit,
                                      coverage=coverage)
            if await engine.revision(space) == projection.revision:
                page["consistent"] = True
                return _page_or_missing(page, entity_id)
        page["consistent"] = False
        page_coverage = page["coverage"]
        assert isinstance(page_coverage, dict)
        page_coverage["reasons"] = [*page_coverage["reasons"], "ledger_changed_during_read"]
        return _page_or_missing(page, entity_id)


_PAGE_ATTEMPTS = 3


def _page_or_missing(page: dict[str, object], entity_id: str) -> dict[str, object] | LedgerJSONResponse:
    if page.get("entity") is not None:
        return page
    where = "" if page["complete"] else " in the facts read; the read was capped, so it may exist"
    return LedgerJSONResponse(status_code=404, content={"error": f"no entity {entity_id!r} in this view{where}",
                                                  "complete": page["complete"], "coverage": page["coverage"]})


async def _entity_page(engine: MemoryEngine, space: str, projection: EntityProjection, entity_id: str, *,
                       status: StatusMode, when: str, limit: int, coverage: dict[str, object]) -> dict[str, object]:
    complete, read = _read(coverage)
    found = neighbourhood(projection, entity_id, limit=limit)
    if found is None:
        return {"entity": None, "complete": complete, "coverage": read}
    names = _names(projection)
    roles = {role.fact_id: role for role in projection.roles}

    def grouped(relations: tuple[Relation, ...], far: Literal["subject", "object"]) -> list[dict[str, object]]:
        groups: dict[str, list[dict[str, object]]] = {}
        for relation in relations:
            groups.setdefault(relation.predicate, []).append({
                "relation_id": relation.relation_id,
                far: names[relation.subject_id if far == "subject" else relation.object_id],
                "fact_ids": list(relation.fact_ids),
                "support": support([roles[fact_id] for fact_id in relation.fact_ids if fact_id in roles])})
        return [{"predicate": predicate, "relations": items} for predicate, items in groups.items()]

    cited = sorted({fact_id for relation in (*found.outgoing, *found.incoming) for fact_id in relation.fact_ids}
                   | {fact_id for attribute in found.attributes for fact_id in attribute.fact_ids})
    reasons = [*_read_reasons(coverage), *(["relation_limit"] if found.truncated else [])]
    return {"schema_version": 1, "space": space, "projection": projection_meta(projection),
            "filters": {"status": status, "as_of": when}, "entity": entity_record(found.entity, found.claims),
            "outgoing": grouped(found.outgoing, "object"), "incoming": grouped(found.incoming, "subject"),
            "attributes": [{"predicate": attribute.predicate, "value": attribute.value,
                            "literal_kind": attribute.literal_kind, "fact_ids": list(attribute.fact_ids)}
                           for attribute in found.attributes],
            "facts": await checked_facts(engine.documents, space, cited), "complete": complete,
            "coverage": {**read, "relations_total": found.relations_total,
                         "relations_shown": len(found.outgoing) + len(found.incoming),
                         "truncated": bool(reasons), "reasons": reasons}}
