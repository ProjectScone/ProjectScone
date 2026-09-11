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

from typing import Callable, Literal, Optional

from fastapi import Depends, FastAPI, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ..core.errors import InvalidInput
from ..core.timeutil import format_rfc3339, parse_rfc3339
from ..entities.analysis import GraphAnalysis, analyze_projection
from ..entities.read import load_projection
from ..entities.report import build_report, render_markdown
from ..entities.view import StatusMode, entity_listing, knowledge_view
from ..memory.engine import MemoryEngine


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


class ListFilters(Filters):
    q: Optional[str]


class Coverage(BaseModel):
    facts_read: int
    #: Facts that count in this view's status mode and moment; the view is
    #: projected from these alone.
    facts_counted: int
    facts_limit: int
    entities_total: int
    entities_shown: int
    relations_total: Optional[int] = None
    relations_shown: Optional[int] = None
    attributes_total: Optional[int] = None
    attributes_shown: Optional[int] = None
    truncated: bool
    reasons: list[str]


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


class Groupings(BaseModel):
    """Computed from the view's recorded relations: analysis, never facts."""
    basis: Literal["computed"]
    method: str
    modularity: float
    membership: dict[str, str]
    communities: list[GroupingCommunity]
    importance: list[GroupingImportance]


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
    }


def mount_entity_routes(app: FastAPI, engine: MemoryEngine, space_for: Callable[..., object]) -> None:
    @app.get("/v1/graph/knowledge", response_model=KnowledgeView, response_model_exclude_unset=True)
    async def get_knowledge(
        status: StatusMode = "current", as_of: Optional[str] = None,
        limit: int = Query(default=150, ge=1, le=1000), attribute_limit: int = Query(default=300, ge=0, le=5000),
        groupings: bool = False, space: str = Depends(space_for),
    ) -> dict[str, object]:
        """Entities, the relations between them and their values, as the ledger records them.

        With ``groupings=true`` a separate, computed block assigns the shown
        entities to communities and scores their importance."""
        when = _moment(engine, as_of)
        projection, coverage = await load_projection(engine, space, mode=status, as_of=when)
        view = knowledge_view(projection, mode=status, as_of=when, limit=limit, attribute_limit=attribute_limit,
                              coverage=coverage)
        if groupings:
            view["groupings"] = _groupings(analyze_projection(projection), view)
        return view

    @app.get("/v1/graph/report", response_model=None)
    async def get_report(
        status: StatusMode = "current", as_of: Optional[str] = None,
        format: Literal["json", "markdown"] = "json", space: str = Depends(space_for),
    ) -> dict[str, object] | PlainTextResponse:
        """The space's communities, central entities, surprising connections and
        questions worth asking, computed from recorded facts and citing them."""
        when = _moment(engine, as_of)
        projection, coverage = await load_projection(engine, space, mode=status, as_of=when)
        view = knowledge_view(projection, mode=status, as_of=when, limit=1, attribute_limit=0, coverage=coverage)
        report = build_report(projection, analyze_projection(projection), meta=view["projection"],  # type: ignore[arg-type]
                              filters={"status": status, "as_of": when}, coverage=coverage)
        if format == "markdown":
            return PlainTextResponse(render_markdown(report), media_type="text/markdown; charset=utf-8")
        return report

    @app.get("/v1/entities", response_model=EntityList)
    async def get_entities(
        status: StatusMode = "current", as_of: Optional[str] = None, q: Optional[str] = Query(default=None, max_length=200),
        limit: int = Query(default=100, ge=1, le=1000), space: str = Depends(space_for),
    ) -> dict[str, object]:
        """Entities ranked by the claims they take part in, optionally filtered by name."""
        when = _moment(engine, as_of)
        projection, coverage = await load_projection(engine, space, mode=status, as_of=when)
        return entity_listing(projection, mode=status, as_of=when, limit=limit, query=q, coverage=coverage)
