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

from ..core.errors import InvalidInput
from ..core.timeutil import format_rfc3339, parse_rfc3339
from ..entities.analysis import GraphAnalysis, analyze_projection
from ..entities.context import MAX_NAME, MAX_NAMES, MAX_QUESTION, ContextLimits, graph_context
from ..entities.sources import sources_view
from ..entities.vocabulary import read_vocabulary
from ..entities.timeline import TimelineEntityAmbiguous, TimelineEntityMissing, timeline_view
from ..entities.grounding import checked_facts
from ..entities.duplicates import (DEFAULT_MIN_SCORE, DEFAULT_PAIRS, MAX_BYTES as DUPLICATES_BYTES, MAX_PAIRS,
                                    likely_duplicates)
from ..entities.health import (DEFAULT_EXAMPLES as HEALTH_EXAMPLES, MAX_BYTES as HEALTH_BYTES, MAX_EXAMPLES,
                               graph_health)
from ..entities.changes import DEFAULT_CHANGES, MAX_BYTES as CHANGES_BYTES, MAX_CHANGES, ChangesError, graph_changes
from ..entities.overview import (DEFAULT_COMMUNITIES, DEFAULT_FACTS_EACH, MAX_BYTES as OVERVIEW_BYTES,
                                  MAX_COMMUNITIES, MAX_FACTS_EACH, graph_overview)
from ..entities.match import DEFAULT_ROWS, MAX_BYTES as MATCH_BYTES, MAX_ROWS, MIN_BYTES, MAX_WHERE, MatchQueryError, graph_match
from ..entities.export import ExportFormat, export_graph
from ..entities.project import EntityProjection, Relation
from ..entities.query import Resolution, neighbourhood, paths_between, resolve
from ..entities.read import load_projection, read_record
from ..entities.schema import MAX_BYTES, MAX_BYTES_LIMIT, MAX_PREDICATES, schema_record
from ..entities.report import render_markdown, report_record
from ..entities.service import ProjectionBuilding
from ..entities.usage import recall_usage
from ..entities.view import SeedRefused, StatusMode, WalkDirection, walk_seeds, entity_listing, entity_record, knowledge_view, projection_meta, support
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
    #: In a seeded view, the steps from the nearest seed (0 for a seed).
    hop: Optional[int] = None
    #: With ``usage``, the recalls that returned one of its facts; None when
    #: the engine keeps no events.
    recalled: Optional[int] = None


class RelationOut(BaseModel):
    id: str
    subject_id: str
    predicate: str
    object_id: str
    fact_ids: list[int]
    support: Support
    #: The stretches of valid time it held over, half-open. A claim made
    #: of two spells with a gap between them did not hold in the gap, so
    #: the two fields below are its first beginning and last ending, not
    #: one unbroken spell.
    periods: list[list[Optional[str]]] = []
    first_valid_from: str
    last_valid_until: Optional[str]
    #: With ``usage``, the recalls that returned one of its facts.
    recalled: Optional[int] = None


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
    #: How a seeded walk followed relations, and how many steps it allowed
    #: (None: as far as it could reach).
    direction: Optional[Literal["both", "out", "in"]] = None
    hops: Optional[int] = None


class ListFilters(Filters):
    q: Optional[str]


class ImpliedOut(BaseModel):
    """An edge nobody claimed, which follows from ones they did. Its own
    model, as it is its own kind of thing: the id carries an ``imp:``
    prefix, and it names the claims and relations it rests on."""

    id: str
    subject_id: str
    predicate: str
    object_id: str
    fact_ids: list[int]
    support: Support
    #: "inverse", "symmetric" or "transitive".
    follows: str
    #: The stated relations it was worked out from, in order.
    follows_from: list[str]
    #: Every stretch of valid time the claims under it shared, half-open.
    periods: list[list[Optional[str]]] = []
    first_valid_from: str
    last_valid_until: Optional[str] = None


def _provenance(held) -> dict[str, object]:
    """Where the vocabulary in force came from, for a coverage block.

    On the wire and not only in process: a reader who cannot see this
    cannot tell why two answers about one space differ, and a claim that
    every answer says where its vocabulary came from is not true of an
    answer that does not carry it.
    """
    return {"vocabulary_source": held.source, "vocabulary_why": held.why}


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
    #: The vocabulary the implications were worked out under, with the
    #: bounds it applied (max_steps, max_implied, max_walked). None when
    #: no meanings are in force and the graph holds only claims; an empty
    #: object when the vocabulary in force says there are none, which is a
    #: different answer.
    meanings: Optional[dict[str, object]] = None
    #: Where that vocabulary came from: ``space`` when the space holds its
    #: own, ``process`` when it is this process's configuration, ``none``
    #: when there is none. Two readers of one space that disagree can only
    #: discover it if the answer says which.
    vocabulary_source: Optional[str] = None
    #: Why, in words -- including the saved revision when the space holds
    #: one, so two answers built under different revisions are tellable
    #: apart.
    vocabulary_why: Optional[str] = None
    implied_total: Optional[int] = None
    implied_shown: Optional[int] = None
    #: Present only when the walk stopped before it had followed
    #: everything that follows.
    implied_capped: Optional[bool] = None
    truncated: bool
    reasons: list[str]
    #: Pass as ``cursor`` for the next page of the ranking; absent on the last.
    next_cursor: Optional[str] = None
    #: With ``usage``, which recalls were counted: how many, since when and
    #: the oldest read, whether more were left unread, what the event log
    #: keeps, which events could not be counted, and whether the engine
    #: keeps events at all.
    usage: Optional[dict[str, object]] = None


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
    #: What follows from the relations under the space's vocabulary, kept
    #: apart from them. Empty when no meanings are configured.
    implied: list[ImpliedOut] = []
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
        direction: WalkDirection = "both", hops: Optional[int] = Query(default=None, ge=1, le=8),
        usage: bool = False, usage_since: Optional[str] = None,
        cursor: Optional[str] = Query(default=None, max_length=512), space: str = Depends(space_for),
    ) -> dict[str, object] | LedgerJSONResponse:
        """Entities, the relations between them and their values, as the ledger records them.

        Without ``seed``, the entities ranked by the claims they take part
        in, a page at a time: ``coverage.next_cursor`` asks for the next, and
        a cursor from before the graph changed is refused with 409. With
        ``seed`` (names or ids), the entities reached from them breadth
        first, following relations in ``direction`` (``in`` finds what
        depends on a seed) for at most ``hops`` steps, hubs past
        ``hub_degree`` shown but not walked through. With ``groupings=true`` a separate, computed block
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
        if not seed and (direction != "both" or hops is not None):
            raise InvalidInput("direction and hops shape a seeded walk; give a seed")
        try:
            seeds = walk_seeds(projection, seed, coverage)
        except SeedRefused as refused:
            return LedgerJSONResponse(status_code=refused.status, content=refused.answer)
        recalls = None
        if usage or usage_since is not None:
            recalls = await recall_usage(engine, space, since=None if usage_since is None
                                         else _moment(engine, usage_since))
        view = knowledge_view(projection, mode=status, as_of=when, limit=limit, attribute_limit=attribute_limit,
                              coverage=coverage, seeds=seeds, hub_degree=hub_degree, offset=offset,
                              direction=direction, hops=hops, usage=recalls)
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
        exclude_hubs: Optional[float] = Query(default=None, ge=50, le=100), usage: bool = False,
        usage_since: Optional[str] = None, space: str = Depends(space_for),
    ) -> dict[str, object] | PlainTextResponse:
        """The space's communities, central entities, surprising connections and
        questions worth asking, computed from recorded facts and citing them.
        ``resolution`` sets how fine the communities are; ``exclude_hubs``
        leaves entities above that degree percentile out of the central ranking."""
        report = await report_record(engine, space, status=status, as_of=_moment(engine, as_of), resolution=resolution,
                                     exclude_hubs=exclude_hubs, usage=usage,
                                     usage_since=None if usage_since is None else _moment(engine, usage_since))
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
        GraphML, dynamic GEXF, Cypher, CSV, JSON-LD, an Obsidian vault (with a
        canvas of its notes), a wiki an agent can crawl, a Mermaid chart, an
        SVG drawing, an Obsidian canvas or one interactive page. The file says
        which projection it holds and, when the read was capped, that it is
        partial."""
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
        similar: bool = False, min_similarity: Optional[float] = Query(default=None, ge=-1, le=1),
        status: StatusMode = "current", as_of: Optional[str] = None, space: str = Depends(space_for),
    ) -> dict[str, object]:
        """What the graph records around some names, or the entities a question
        names, as one line per item for a model: coverage first, then the
        entities, paths between them, relations by hop and values, each citing
        facts re-read now. Cut between lines to ``max_bytes``. With
        ``similar``, a question also seeds up to three entities it resembles
        by vector, each marked with its score."""
        if not names and q is None:
            raise InvalidInput("give names or q")
        if len(names) > MAX_NAMES or any(not 1 <= len(name) <= MAX_NAME for name in names):
            raise InvalidInput(f"names: at most {MAX_NAMES}, each 1..{MAX_NAME} characters")
        when = _moment(engine, as_of)
        packet = await graph_context(engine, space, names=names, question=q, status=status, as_of=when,
                                     limits=ContextLimits(max_bytes=max_bytes, max_hops=max_hops), similar=similar,
                                     min_similarity=min_similarity)
        return packet.record(space, status, when)

    @app.get("/v1/graph/match")
    async def get_match(
        where: str = Query(min_length=2, max_length=MAX_WHERE),
        returns: Optional[list[str]] = Query(default=None),
        limit: int = Query(default=DEFAULT_ROWS, ge=1, le=MAX_ROWS),
        together: bool = True, follows: bool = False,
        max_bytes: int = Query(default=MATCH_BYTES, ge=MIN_BYTES, le=MAX_BYTES_LIMIT),
        status: StatusMode = "current", as_of: Optional[str] = None, space: str = Depends(space_for),
    ) -> dict[str, object]:
        """A structured question: ``where`` is a JSON array of 1 to 6 triple
        patterns, ``{"subject", "predicate", "object"}``, where a term that
        starts with ``?`` is a variable. Answers every way the patterns hold
        together, as rows over the variables returned (every one by default), each
        citing its facts re-read now; in history, only facts that held at
        one moment are joined unless ``together`` is false. ``returns``
        names the variables to answer with. With ``follows``, what follows
        from the claims under the space's vocabulary is matched too, and a
        row that used one says which meaning it followed; without it, only
        claims are matched."""
        try:
            patterns = json.loads(where)
        except ValueError:
            raise InvalidInput("where must be a JSON array of patterns") from None
        when = _moment(engine, as_of)
        try:
            result = await graph_match(engine, space, patterns, returns=returns, limit=limit, status=status,
                                       as_of=when, together=together, follows=follows, max_bytes=max_bytes)
        except MatchQueryError as refused:
            raise InvalidInput(str(refused)) from None
        return result.record(space, status=status, as_of=when, together=together, limit=limit, follows=follows)

    @app.get("/v1/graph/overview")
    async def get_overview(
        q: Optional[str] = Query(default=None, min_length=1, max_length=MAX_QUESTION),
        limit: int = Query(default=DEFAULT_COMMUNITIES, ge=1, le=MAX_COMMUNITIES),
        facts: int = Query(default=DEFAULT_FACTS_EACH, ge=0, le=MAX_FACTS_EACH),
        resolution: float = Query(default=1.0, gt=0, le=10),
        max_bytes: int = Query(default=OVERVIEW_BYTES, ge=MIN_BYTES, le=MAX_BYTES_LIMIT),
        status: StatusMode = "current", as_of: Optional[str] = None, space: str = Depends(space_for),
    ) -> dict[str, object]:
        """The graph at a glance, for a question about the whole of it: each
        community's size, kinds, predicates, central entities and a few of
        its facts, cited and re-read now. With ``q``, the communities it
        concerns come first, each saying what it matched."""
        when = _moment(engine, as_of)
        found = await graph_overview(engine, space, question=q, limit=limit, facts_each=facts, status=status,
                                     as_of=when, resolution=resolution, max_bytes=max_bytes)
        return found.record(space, status=status, as_of=when, question=q)

    @app.get("/v1/graph/changes")
    async def get_changes(
        since: str = Query(min_length=1, max_length=64), until: Optional[str] = Query(default=None, max_length=64),
        limit: int = Query(default=DEFAULT_CHANGES, ge=1, le=MAX_CHANGES),
        max_bytes: int = Query(default=CHANGES_BYTES, ge=MIN_BYTES, le=MAX_BYTES_LIMIT),
        space: str = Depends(space_for),
    ) -> dict[str, object]:
        """What changed in the graph between ``since`` and ``until`` (now by
        default): claims that moved from one object to another, relations
        that began and ended, values that changed and entities that came
        and went, each citing its facts re-read at its own moment."""
        try:
            found = await graph_changes(engine, space, since=since, until=until, limit=limit, max_bytes=max_bytes)
        except ChangesError as refused:
            raise InvalidInput(str(refused)) from None
        return found.record(space)

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

    @app.get("/v1/graph/health")
    async def get_health(
        limit: int = Query(default=HEALTH_EXAMPLES, ge=1, le=MAX_EXAMPLES,
                           description="Examples shown for each concern."),
        max_bytes: int = Query(default=HEALTH_BYTES, ge=MIN_BYTES, le=MAX_BYTES_LIMIT),
        status: StatusMode = "current", as_of: Optional[str] = None, space: str = Depends(space_for),
    ) -> dict[str, object]:
        """What in the graph wants attention: claims resting on nothing,
        kinds that disagree or are missing, entities nothing links to,
        predicates used once, and names that may be one thing. Each concern
        is counted with examples. It reads and changes nothing."""
        when = _moment(engine, as_of)
        found = await graph_health(engine, space, limit=limit, status=status, as_of=when, max_bytes=max_bytes)
        return found.record(space, status=status, as_of=when)

    @app.get("/v1/entities/duplicates")
    async def get_duplicates(
        limit: int = Query(default=DEFAULT_PAIRS, ge=1, le=MAX_PAIRS),
        min_score: float = Query(default=DEFAULT_MIN_SCORE, ge=0, le=1),
        max_bytes: int = Query(default=DUPLICATES_BYTES, ge=MIN_BYTES, le=MAX_BYTES_LIMIT),
        status: StatusMode = "current", as_of: Optional[str] = None, space: str = Depends(space_for),
    ) -> dict[str, object]:
        """Pairs of entities that may be one thing under two names, most likely
        first, each saying why: the same name once titles are set aside,
        shared words or letters, initials, neighbours in common. Suggestions
        only; nothing is merged."""
        when = _moment(engine, as_of)
        found = await likely_duplicates(engine, space, limit=limit, min_score=min_score, status=status, as_of=when,
                                        max_bytes=max_bytes)
        return found.record(space, status=status, as_of=when)

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
                   | {fact_id for attribute in found.attributes for fact_id in attribute.fact_ids}
                   | {fact_id for item in found.follows for fact_id in item.fact_ids})
    reasons = [*_read_reasons(coverage),
                *(["relation_limit"] if (len(found.outgoing) + len(found.incoming) < found.relations_total
                                         or len(found.attributes) == limit) else []),
                *(["follows_limit"] if len(found.follows) < found.follows_total else []),
                # The walk that worked out what follows stopped early, so
                # this page is missing implications it cannot name.
                *(["implied_capped"] if projection.implied_capped else [])]
    return {"schema_version": 1, "space": space, "projection": projection_meta(projection),
            "filters": {"status": status, "as_of": when}, "entity": entity_record(found.entity, found.claims),
            "outgoing": grouped(found.outgoing, "object"), "incoming": grouped(found.incoming, "subject"),
            "attributes": [{"predicate": attribute.predicate, "value": attribute.value,
                            "literal_kind": attribute.literal_kind, "fact_ids": list(attribute.fact_ids)}
                           for attribute in found.attributes],
            # What nobody said, in its own place, each saying what it was
            # worked out from so a reader can check the claims behind it.
            "follows": [{"relation_id": item.relation_id, "predicate": item.predicate,
                         "subject": names[item.subject_id], "object": names[item.object_id],
                         "follows": item.follows, "follows_from": list(item.follows_from),
                         "periods": [list(period) for period in item.periods],
                         "first_valid_from": item.first_valid_from, "last_valid_until": item.last_valid_until,
                         "fact_ids": list(item.fact_ids),
                         "support": support([roles[fact_id] for fact_id in item.fact_ids if fact_id in roles])}
                        for item in found.follows],
            "facts": await checked_facts(engine.documents, space, cited), "complete": complete,
            "coverage": {**read, "relations_total": found.relations_total,
                         "relations_shown": len(found.outgoing) + len(found.incoming),
                         "follows_total": found.follows_total, "follows_shown": len(found.follows),
                         # `is None`, never truthiness: RelationMeanings() is
                         # falsy, so an explicitly empty vocabulary serialised
                         # as null and a reader could not tell "there are none"
                         # from "nothing was said".
                         "meanings": (None if projection.meanings is None
                                      else projection.meanings.record()),
                         **_provenance(await read_vocabulary(engine, space)),
                         **({"implied_capped": True} if projection.implied_capped else {}),
                         "truncated": bool(reasons), "reasons": reasons}}
