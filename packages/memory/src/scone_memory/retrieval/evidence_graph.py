"""Bounded explanation of one recall, using retained records only.

Retrieval and source membership are not factual relations. Facts and links
come only from the ledger; this reader never expands to neighboring facts,
extracts claims, calls a model, or writes memory. Source strings are labels,
not trusted URLs. The caller must pass the same validated scope as recall.
"""
from __future__ import annotations

import re

from typing import Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field, JsonValue

from ..core.models import Chunk, Episode, Fact, FactLink, RecallResult
from ..core.ports import TextFilter
from ..core.validation import entity_key
from ..entities.classify import classify_object, reference_flag
from ..entities.ids import ENTITY_ID_SCHEME, key_id
from ..entities.project import classification_context, quoted_form
from ..memory.engine import check_space

MAX_CHUNKS = 24
MAX_FACTS = 16
MAX_SOURCES = 40
MAX_LINKS = 48
PREVIEW_CHARS = 480
MAX_CONCEPTS = 32
MAX_MENTIONS = 96


class EvidenceDocuments(Protocol):
    async def get_chunks(self, space: str, chunk_ids: Sequence[int]) -> list[Chunk]: ...
    async def get_episode(self, space: str, episode_id: int) -> Episode | None: ...
    async def get_fact(self, space: str, fact_id: int) -> Fact | None: ...


@runtime_checkable
class BoundedFactLinks(Protocol):
    async def fact_links_between(self, space: str, fact_ids: Sequence[int], limit: int) -> list[FactLink]: ...


class EvidenceNode(BaseModel):
    id: str
    kind: Literal["query", "chunk", "episode", "claim", "concept"]
    label: str
    ts: str | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)


EdgeKind = Literal["returned", "chunked_into", "source_of", "extends", "derived_from", "contradicts", "supports", "superseded_by", "mentions", "asserts", "relation"]


class EvidenceEdge(BaseModel):
    source: str
    target: str
    kind: EdgeKind
    label: str
    data: dict[str, JsonValue] = Field(default_factory=dict)


class QueryEvidenceGraph(BaseModel):
    nodes: list[EvidenceNode] = Field(default_factory=list)
    edges: list[EvidenceEdge] = Field(default_factory=list)
    truncated: bool = False
    provenance_missing: int = 0
    provenance_omitted: int = 0
    notices: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


def _narrowed(scope: TextFilter) -> bool:
    return bool(scope.tags or scope.where or scope.conditions is not None
                or any(value is not None for value in (scope.as_of, scope.kind, scope.source_prefix, scope.since, scope.until)))


def _matches(episode: Episode, scope: TextFilter) -> bool:
    return (all(tag in episode.tags for tag in scope.tags)
            and all(episode.metadata.get(key) == value for key, value in scope.where.items())
            and (scope.conditions is None or bool(scope.conditions.matches(episode.metadata)))
            and (scope.kind is None or episode.kind == scope.kind)
            and (scope.source_prefix is None or (episode.source is not None and episode.source.startswith(scope.source_prefix)))
            and (scope.since is None or episode.created_at >= scope.since)
            and (scope.until is None or episode.created_at <= scope.until)
            and (scope.as_of is None or episode.created_at <= scope.as_of))


class _Builder:
    def __init__(self, documents: EvidenceDocuments, space: str, scope: TextFilter, exclude_session_id: str | None = None) -> None:
        self.exclude_session_id = exclude_session_id
        self.documents = documents
        self.space = space
        self.scope = scope
        self.graph = QueryEvidenceGraph()
        self.sources: dict[int, tuple[Episode | None, str]] = {}
        self.omitted: set[int] = set()
        self.ids: set[str] = set()

    def notice(self, text: str) -> None:
        if text not in self.graph.notices:
            self.graph.notices.append(text)

    def node(self, node: EvidenceNode) -> None:
        if node.id not in self.ids:
            self.ids.add(node.id)
            self.graph.nodes.append(node)

    def edge(self, source: str, target: str, kind: EdgeKind, label: str,
             data: dict[str, JsonValue]) -> None:
        if source in self.ids and target in self.ids:
            self.graph.edges.append(EvidenceEdge(source=source, target=target, kind=kind, label=label, data=data))

    async def source(self, episode_id: int) -> tuple[Episode | None, str]:
        if episode_id in self.sources:
            return self.sources[episode_id]
        if len(self.sources) >= MAX_SOURCES:
            self.omitted.add(episode_id)
            self.graph.provenance_omitted = len(self.omitted)
            self.graph.truncated = True
            return None, "omitted"
        episode = await self.documents.get_episode(self.space, episode_id)
        if episode is None or episode.space != self.space or episode.episode_id != episode_id:
            self.graph.provenance_missing += 1
            result: tuple[Episode | None, str] = (None, "missing")
        elif (not _matches(episode, self.scope) or (self.exclude_session_id is not None
                and (episode.metadata.get("session_id") == self.exclude_session_id or episode.source == self.exclude_session_id))):
            self.notice("Some retained evidence is outside the query scope and is not shown.")
            result = (None, "out_of_scope")
        else:
            result = (episode, "retained")
        self.sources[episode_id] = result
        return result

    def source_node(self, episode: Episode) -> None:
        self.node(EvidenceNode(id=f"episode:{episode.episode_id}", kind="episode",
                               label=(episode.source or f"Source {episode.episode_id}")[:PREVIEW_CHARS],
                               ts=episode.created_at,
                               data={"episode_id": episode.episode_id, "source": episode.source,
                                     "preview": episode.content[:PREVIEW_CHARS], "kind": episode.kind}))

    async def provenance(self, episode_id: int | None, quote: str | None) -> tuple[Episode | None, str, str | None]:
        if episode_id is None:
            return None, "unstated", None
        episode, status = await self.source(episode_id)
        if episode is None:
            return None, status, None
        if not quote:
            return episode, "unquoted", None
        if quote not in episode.content:
            self.notice("A stored quote no longer matches its retained source and is not shown as evidence.")
            return episode, "unquoted", None
        return episode, "retained", quote[:PREVIEW_CHARS]


async def build_query_evidence_graph(documents: EvidenceDocuments, space: str, query: str,
                                     result: RecallResult, *, scope: TextFilter | None = None,
                                     exclude_session_id: str | None = None) -> QueryEvidenceGraph:
    """Read at most 24 chunks, 16 facts, 40 sources, and 49 stored links.

    The extra link detects truncation. Unsupported backends are explicit;
    the unbounded legacy fact_links API is deliberately never called.
    Returned content is capped to 480-character previews. No graph text
    belongs in a model prompt; it is an inspection artifact for the user.
    """
    check_space(space)
    builder = _Builder(documents, space, scope or TextFilter(), exclude_session_id)
    graph = builder.graph
    builder.node(EvidenceNode(id="query:current", kind="query", label=query[:PREVIEW_CHARS],
                              data={"event_id": result.event_id}))
    graph.truncated = len(result.items) > MAX_CHUNKS or len(result.facts) + len(result.history) > MAX_FACTS
    if result.degraded:
        builder.notice("Recall reported degraded retrieval; this graph covers only returned evidence.")
    items = result.items[:MAX_CHUNKS]
    chunks = {chunk.chunk_id: chunk for chunk in await documents.get_chunks(space, [item.chunk_id for item in items])} if items else {}
    for item in items:
        chunk = chunks.get(item.chunk_id)
        if chunk is None or chunk.space != space or chunk.episode_id != item.episode_id or chunk.text != item.text:
            builder.notice("Some returned chunks are no longer retained unchanged and are not shown.")
            continue
        episode, _ = await builder.source(chunk.episode_id)
        if episode is None:
            continue
        if chunk.start < 0 or chunk.end < chunk.start or episode.content.encode()[chunk.start:chunk.end] != chunk.text.encode():
            builder.notice("Some chunk offsets do not match retained source bytes and are not shown.")
            continue
        builder.source_node(episode)
        cid = f"chunk:{chunk.chunk_id}"
        builder.node(EvidenceNode(id=cid, kind="chunk", label=chunk.text[:100], ts=chunk.created_at,
                                  data={"chunk_id": chunk.chunk_id, "episode_id": chunk.episode_id,
                                        "start": chunk.start, "end": chunk.end, "text": chunk.text[:PREVIEW_CHARS],
                                        "score": item.score, "similarity": item.similarity,
                                        "lanes": {key: value for key, value in item.lanes.items()}}))
        builder.edge("query:current", cid, "returned", "Returned by recall", {"category": "retrieval"})
        builder.edge(f"episode:{episode.episode_id}", cid, "chunked_into", "Contains source bytes",
                     {"category": "membership"})

    facts: dict[int, Fact] = {}
    for returned in (result.facts + result.history)[:MAX_FACTS]:
        if returned.space != space or returned.fact_id in facts:
            continue
        fact = await documents.get_fact(space, returned.fact_id)
        if fact is None or fact.space != space or fact != returned or fact.excluded or not fact.in_ledger:
            builder.notice("Some returned claims are no longer retained unchanged and are not shown.")
            continue
        episode, status, quote = await builder.provenance(fact.source_episode_id, fact.quote)
        if _narrowed(builder.scope) and episode is None:
            builder.notice("Claims without retained sources inside the query scope are not shown.")
            continue
        fid = f"claim:{fact.fact_id}"
        facts[fact.fact_id] = fact
        builder.node(EvidenceNode(id=fid, kind="claim", label=f"{fact.subject} {fact.predicate} {fact.object}"[:PREVIEW_CHARS],
                                  ts=fact.valid_from,
                                  data={"fact_id": fact.fact_id, "subject": fact.subject[:PREVIEW_CHARS],
                                        "predicate": fact.predicate[:PREVIEW_CHARS], "object": fact.object[:PREVIEW_CHARS],
                                        "origin": fact.origin, "status": fact.status, "source_episode_id": fact.source_episode_id,
                                        "quote": quote, "grounded": quote is not None, "provenance_status": status,
                                        "valid_from": fact.valid_from, "valid_until": fact.valid_until,
                                        "confidence": fact.confidence,
                                        "record_complete": all(len(value) <= PREVIEW_CHARS for value in
                                            (fact.subject, fact.predicate, fact.object, fact.quote or ""))}))
        builder.edge("query:current", fid, "returned", "Returned claim history" if returned in result.history else "Returned claim",
                     {"category": "retrieval"})
        if episode:
            builder.source_node(episode)
            builder.edge(f"episode:{episode.episode_id}", fid, "source_of", "Recorded claim source",
                         {"category": "provenance", "quote": quote, "verified": quote is not None})

    for fact in facts.values():
        if fact.superseded_by in facts:
            builder.edge(f"claim:{fact.fact_id}", f"claim:{fact.superseded_by}", "superseded_by", "Superseded by",
                         {"category": "fact_relation", "provenance_status": "unstated"})
    if facts and isinstance(documents, BoundedFactLinks):
        links = await documents.fact_links_between(space, sorted(facts), MAX_LINKS + 1)
        graph.truncated = graph.truncated or len(links) > MAX_LINKS
        for link in links[:MAX_LINKS]:
            if link.space != space or link.from_fact not in facts or link.to_fact not in facts:
                continue
            episode, status, quote = await builder.provenance(link.source_episode_id, link.quote)
            if _narrowed(builder.scope) and link.source_episode_id is not None and episode is None:
                builder.notice("Stored relations with evidence outside the query scope are not shown.")
                continue
            if episode:
                builder.source_node(episode)
            builder.edge(f"claim:{link.from_fact}", f"claim:{link.to_fact}", link.kind, link.kind.replace("_", " "),
                         {"category": "fact_relation", "link_id": link.link_id,
                          "source_episode_id": link.source_episode_id, "quote": quote, "provenance_status": status,
                          "record_complete": len(link.quote or "") <= PREVIEW_CHARS})
    elif facts:
        builder.notice("This store does not support bounded relationship reads; stored fact links were not inspected.")
    if graph.truncated:
        builder.notice("Graph limits omitted some returned records or stored relations.")
    if graph.provenance_missing:
        builder.notice("Some recorded source episodes are no longer retained; their quotes are not shown.")
    _add_concepts(graph, space)
    graph.counts = {kind: sum(node.kind == kind for node in graph.nodes) for kind in sorted({node.kind for node in graph.nodes})}
    return graph


def _mentioned(form: str, text: str) -> re.Match[str] | None:
    """Where a recorded spelling occurs as a whole word. Short and all-caps
    names must match their own case: 'AI' is not 'ai', 'May' is not 'may'."""
    strict = len(form) <= 3 or (form.isupper() and len(form) <= 5)
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(token) for token in form.split()) + r"(?!\w)"
    return re.search(pattern, text, 0 if strict else re.IGNORECASE)


def _add_concepts(graph: QueryEvidenceGraph, space: str) -> None:
    """Draw the entities retained claims are about, and where chunks mention them.

    Subjects are entities; an object is one only when the entity classifier
    says it names a thing, so dates, amounts and prose stay on their claim.
    Node ids are the entity ids the knowledge map uses, so a person can pivot
    from recall evidence to the whole-space graph. Nothing comes from
    typography: a chunk mentions a concept only where a spelling some claim
    recorded for it appears.
    """
    claims = [claim for claim in graph.nodes
              if claim.kind == "claim" and claim.data.get("provenance_status") == "retained"
              and claim.data.get("record_complete") is True
              and all(isinstance(claim.data.get(key), str) for key in ("subject", "predicate", "object"))]
    context = classification_context((str(claim.data["subject"]), str(claim.data["object"])) for claim in claims)
    concepts: dict[str, EvidenceNode] = {}
    spellings: dict[str, set[str]] = {}
    concept_truncated = False
    added_edges: list[EvidenceEdge] = []

    def concept(key: str, spelling: str, label: str, basis: str) -> EvidenceNode | None:
        nonlocal concept_truncated
        if not key or len(spelling) > PREVIEW_CHARS:
            return None
        if key not in concepts:
            if len(concepts) >= MAX_CONCEPTS:
                graph.truncated = True
                concept_truncated = True
                return None
            flag = reference_flag(key)
            concepts[key] = EvidenceNode(id=key_id(space, key), kind="concept", label=label, data={
                "name": label, "basis": basis, "key": key, "entity_id": key_id(space, key),
                "id_scheme": ENTITY_ID_SCHEME, "flags": [] if flag is None else [flag]})
        spellings.setdefault(key, set()).add(spelling)
        return concepts[key]

    for claim in claims:
        subject, predicate, obj = (str(claim.data[key]) for key in ("subject", "predicate", "object"))
        quote = claim.data.get("quote")
        subject_key = entity_key(subject)
        spelled = quoted_form(subject_key, quote if isinstance(quote, str) else None) or subject.strip()
        left = concept(subject_key, spelled, spelled, "retained_claim")
        classification = classify_object(obj, predicate, context)
        right = (concept(entity_key(obj), obj.strip(), obj.strip(), "retained_claim")
                 if classification.object_class == "entity" else None)
        if left is None:
            continue
        added_edges.append(EvidenceEdge(source=claim.id, target=left.id, kind="asserts", label="subject",
                                        data={"category": "assertion", "role": "subject", "fact_id": claim.data["fact_id"]}))
        if right is None:
            continue
        added_edges.append(EvidenceEdge(source=claim.id, target=right.id, kind="asserts", label="object",
                                        data={"category": "assertion", "role": "object", "fact_id": claim.data["fact_id"],
                                              "classification": classification.basis}))
        added_edges.append(EvidenceEdge(source=left.id, target=right.id, kind="relation", label=predicate,
            data={"category": "fact_relation", "predicate": predicate, "fact_id": claim.data["fact_id"],
                  "source_episode_id": claim.data["source_episode_id"], "quote": claim.data["quote"],
                  "origin": claim.data["origin"], "provenance_status": "retained"}))

    mentions = 0
    for chunk in (node for node in graph.nodes if node.kind == "chunk"):
        text = chunk.data.get("text")
        if not isinstance(text, str):
            continue
        for key, found in concepts.items():
            if reference_flag(key) is not None:
                continue
            occurrence = next((hit for form in sorted(spellings[key]) if (hit := _mentioned(form, text))), None)
            if occurrence is None:
                continue
            if mentions >= MAX_MENTIONS:
                graph.truncated = True
                concept_truncated = True
                break
            added_edges.append(EvidenceEdge(source=chunk.id, target=found.id, kind="mentions",
                                            label="Mentions a recorded name",
                                            data={"category": "mention", "quote": occurrence.group(),
                                                  "match": "exact_case" if occurrence.group() in spellings[key] else "folded"}))
            mentions += 1
    graph.nodes.extend(concepts.values())
    graph.edges.extend(added_edges)
    if concept_truncated and "Concept or mention limits may omit part of this view." not in graph.notices:
        graph.notices.append("Concept or mention limits may omit part of this view.")
