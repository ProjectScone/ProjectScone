"""Recall orchestration over explicit storage, embedding and evidence ports.

The engine constructs a request runtime from its current configuration. This
module owns lane execution, source verification, ranking and result assembly;
ingestion and fact lifecycle operations are independent of this component.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import time
from typing import TYPE_CHECKING, Optional

from ..core.errors import InvalidInput
from ..core.models import Episode, QueryEntity, RecallItem, RecallResult, RerankTrace
from ..core.ports import DocumentStore, Embedder, Event, VectorIndex, TextFilter
from ..core.validation import (KINDS, MAX_LIMIT, MAX_QUERY, MAX_SOURCE,
    check_space, normalise_metadata, normalise_tags, normalise_time)
from . import fact_recall, fusion
from .entity_lane import ENTITY_WEIGHT, entity_lane
from .episode_scope import episode_fits
from .filters import parse_filter
if TYPE_CHECKING:
    from ..entities.project import EntityProjection
from .reranking import (Reranker, RerankCandidate, candidate_is_retained,
    rerank_candidates, validate_candidate_limit, validate_rerank_options)

LANE_DEPTH = 4
UNFILTERED_DEPTH = 25

EventEmitter = Callable[[str, str, dict[str, object]], Awaitable[Event | None]]
QueryEvidence = Callable[[str], dict[str, object]]


@dataclass(frozen=True)
class RecallRuntime:
    """Retrieval ports and policy captured at dispatch; callbacks remain live."""

    documents: DocumentStore
    vectors: VectorIndex
    embedder: Embedder
    clock: Callable[[], str]
    emit: EventEmitter
    query_for_evidence: QueryEvidence
    candidate_limit: int | None = None
    reranker: Reranker | None = None
    rerank_limit: int = 32
    rerank_max_bytes: int = 64000
    rerank_timeout: float = 1.0
    contextual_embeddings: bool = False
    demote_restated: bool = True
    similarity_floor: float | None = None
    #: Why stored vectors cannot be compared with this embedder's, if so.
    vector_block: str | None = None


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 3)


async def recall(
    runtime: RecallRuntime,
    space: str,
    query: str,
    limit: int = 5,
    as_of: Optional[str] = None,
    tags: Sequence[str] = (),
    where: Mapping[str, str] | None = None,
    history: bool = False,
    kind: Optional[str] = None,
    source_prefix: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    conditions: Mapping[str, object] | None = None,
    candidate_limit: int | None = None,
    rerank: bool = True,
    graph_boost: bool = False,
    entity_projection: "EntityProjection | None" = None,
    entity_unavailable: str | None = None,
) -> RecallResult:
    """``history`` (research experiment 3) also returns, for every
    subject and predicate among the matched facts, the closed facts that
    held before: what changed, when, and why. Off by default; the
    reader gets only what holds at ``as_of`` unless asked for the chain.

    ``kind``, ``source_prefix``, ``since`` and ``until`` narrow by the
    episode's kind, literal source prefix and inclusive timestamps.
    Native SQLite and in-memory lexical lanes apply these before their
    candidate limit, so out-of-scope records cannot crowd out lexical
    matches. Vector candidates and stores that ignore these fields are
    still postfiltered within a bounded window; semantic-only matches
    and such fallback stores have no completeness guarantee. SQLite
    metadata conditions that require a Python recheck can also underfill
    the window. ``tags`` and ``where`` are applied in both lanes.
    ``as_of`` remains fact validity and the lanes' upper time bound.

    ``candidate_limit`` optionally sets each lane's candidate depth
    independently of returned ``limit``; None uses the configured depth
    or the legacy limit-based depth. A host-supplied reranker is optional.
    It sees only verified retained scoped passages within separate count,
    UTF-8 payload and cooperative async time budgets. Scores remain ranking
    signals, not confidence. A failed reranker retains baseline ordering;
    ``rerank=False`` explicitly disables the configured adapter."""
    check_space(space)
    query = query.strip()
    if not query or len(query) > MAX_QUERY:
        raise InvalidInput(f"query must be 1..={MAX_QUERY} chars")
    limit = max(1, min(limit, MAX_LIMIT))
    candidate_limit = validate_candidate_limit(runtime.candidate_limit if candidate_limit is None else candidate_limit)
    if type(rerank) is not bool:
        raise InvalidInput("rerank must be a boolean")
    active_reranker = runtime.reranker if rerank else None
    if active_reranker is not None:
        validate_rerank_options(runtime.rerank_limit, runtime.rerank_max_bytes, runtime.rerank_timeout)
    boundary = normalise_time(as_of) if as_of else None
    clean_tags = normalise_tags(tags)
    clean_where = normalise_metadata(where or {})
    if kind is not None and kind not in KINDS:
        raise InvalidInput(f"kind must be one of {KINDS}, got {kind!r}")
    if source_prefix is not None and len(source_prefix) > MAX_SOURCE:
        raise InvalidInput(f"source_prefix must be at most {MAX_SOURCE} chars")
    since_at = normalise_time(since) if since else None
    until_at = normalise_time(until) if until else None
    # Refused before any lane runs: a filter that cannot mean anything
    # must not come back as an answer drawn from everything.

    narrow_by = parse_filter(conditions) if conditions is not None else None
    # A store that cannot take the filter returns its best few, all of
    # which the filter may then reject, and the search comes back
    # empty while the memory that answers it sits outside the window.
    in_store = narrow_by is None or bool(getattr(runtime.documents, "narrows_metadata", False))
    narrowing = (kind is not None or source_prefix is not None or since_at is not None
                 or until_at is not None or narrow_by is not None)
    depth = candidate_limit if candidate_limit is not None else limit * LANE_DEPTH * (1 if in_store else UNFILTERED_DEPTH)
    degraded: list[str] = []
    started = time.perf_counter()
    latency: dict[str, float] = {}
    evidence = {
        **runtime.query_for_evidence(query),
        "limit": limit,
        "as_of": boundary,
        "tags": list(clean_tags),
        "where": clean_where,
        "embedder": runtime.embedder.id,
        "contextual_embeddings": runtime.contextual_embeddings,
        "similarity_floor": runtime.similarity_floor,
        "narrow": {"kind": kind, "source_prefix": source_prefix, "since": since_at, "until": until_at,
                   "conditions": dict(conditions) if conditions is not None else None,
                   "conditions_in_store": None if narrow_by is None else in_store},
    }

    vector_lane: list[tuple[int, float]] = []
    if runtime.vector_block is not None:
        degraded.append(f"vectors: {runtime.vector_block}")
    else:
        try:
            t0 = time.perf_counter()
            [qvec] = await runtime.embedder.embed([query])
            latency["embed"] = _ms(t0)
            t0 = time.perf_counter()
            vector_lane = await runtime.vectors.search(space, qvec, depth, boundary, clean_tags, clean_where)
            latency["vector"] = _ms(t0)
        except Exception as e:  # noqa: BLE001 - the lane is reported, not hidden
            degraded.append(f"vectors: {type(e).__name__}: {e}")

    if candidate_limit is not None or active_reranker is not None:
        vector_lane = vector_lane[:depth]

    text_lane: list[tuple[int, float]] = []
    try:
        t0 = time.perf_counter()
        text_lane = await runtime.documents.search_text(
            space, query, depth,
            TextFilter(as_of=boundary, tags=clean_tags, where=clean_where, conditions=narrow_by,
                       kind=kind, source_prefix=source_prefix, since=since_at, until=until_at),
        )
        latency["text"] = _ms(t0)
    except Exception as e:  # noqa: BLE001
        degraded.append(f"text: {type(e).__name__}: {e}")

    if candidate_limit is not None or active_reranker is not None:
        text_lane = text_lane[:depth]

    if len(degraded) == 2:
        latency["total"] = _ms(started)
        await runtime.emit(space, "recall", {**evidence, "degraded": degraded, "latency_ms": latency,
                                           "error": "both lanes failed"})
        raise RuntimeError("both recall lanes failed: " + "; ".join(degraded))

    # The entity lane, only when asked for: passages naming the question's
    # entities or their neighbours, under the same filter as the text lane.
    entity_hits: list[tuple[int, float]] = []
    query_entities: list[QueryEntity] = []
    if graph_boost:
        if entity_projection is None:
            degraded.append(f"entity: {entity_unavailable or 'index_unavailable'}")
        else:
            try:
                t0 = time.perf_counter()
                entity_hits, query_entities = await entity_lane(
                    runtime.documents, entity_projection, space, query, depth,
                    TextFilter(as_of=boundary, tags=clean_tags, where=clean_where, conditions=narrow_by,
                               kind=kind, source_prefix=source_prefix, since=since_at, until=until_at))
                latency["entity"] = _ms(t0)
            except Exception as e:  # noqa: BLE001 - the lane is reported, not hidden
                degraded.append(f"entity: {type(e).__name__}: {e}")
        if candidate_limit is not None or active_reranker is not None:
            entity_hits = entity_hits[:depth]

    # An index that ranks without a cosine (a bridged store with an
    # unknown score) reports NaN: its order counts for fusion, but no
    # similarity is shown and no confidence is judged from it.
    similarity = {cid: (None if math.isnan(score) else score) for cid, score in vector_lane}
    # The best cosine the vector lane saw, before fusion, is the
    # confidence signal. A degraded vector lane cannot judge (None); a
    # lane that ran and found nothing is as weak as evidence gets.
    top_similarity = round(vector_lane[0][1], 6) if vector_lane and not math.isnan(vector_lane[0][1]) else None
    vector_ran = not any(d.startswith("vectors:") for d in degraded)
    low_confidence: Optional[bool] = None
    if runtime.similarity_floor is not None and vector_ran and (top_similarity is not None or not vector_lane):
        low_confidence = top_similarity is None or top_similarity < runtime.similarity_floor
    ranks = {
        "vector": {cid: i + 1 for i, (cid, _) in enumerate(vector_lane)},
        "text": {cid: i + 1 for i, (cid, _) in enumerate(text_lane)},
    }
    if graph_boost:
        ranks["entity"] = {cid: i + 1 for i, (cid, _) in enumerate(entity_hits)}
    fused = (fusion.rrf([vector_lane, text_lane, entity_hits], weights=[1.0, 1.0, ENTITY_WEIGHT]) if graph_boost
             else fusion.rrf([vector_lane, text_lane]))
    chunks = {c.chunk_id: c for c in await runtime.documents.get_chunks(space, list(fused))}
    now = runtime.clock()
    items = [
        fusion.Fused(cid, score + fusion.recency_boost(chunks[cid].created_at, now), similarity.get(cid))
        for cid, score in fused.items()
        if cid in chunks
    ]
    items = fusion.order(items)
    episode_of = {cid: c.episode_id for cid, c in chunks.items()}
    capped_items = fusion.cap_per_episode(items, episode_of)
    baseline_allowed = {item.chunk_id for item in capped_items}
    if active_reranker is None:
        items = capped_items
    episodes: dict[int, Episode] = {}
    if narrowing:
        # Read every candidate's episode and keep the ones that fit,
        # before the limit is applied; the window is the lanes' depth.
        for item in items:
            eid = chunks[item.chunk_id].episode_id
            if eid not in episodes:
                found = await runtime.documents.get_episode(space, eid)
                if found is not None:
                    episodes[eid] = found
        items = [
            item for item in items
            if (episode := episodes.get(chunks[item.chunk_id].episode_id)) is not None
            and episode_fits(episode, kind, source_prefix, since_at, until_at, narrow_by)
        ]
    scope = TextFilter(as_of=boundary, tags=clean_tags, where=clean_where, conditions=narrow_by,
                       kind=kind, source_prefix=source_prefix, since=since_at, until=until_at)
    if active_reranker is not None:
        # Stores/indexes may be stale or ignore optional filters. No adapter
        # sees text until its original source, bytes and scope are verified.
        retained = []
        for item in items:
            chunk = chunks[item.chunk_id]
            episode = episodes.get(chunk.episode_id)
            if episode is None:
                episode = await runtime.documents.get_episode(space, chunk.episode_id)
                if episode is not None:
                    episodes[chunk.episode_id] = episode
            if episode is not None and candidate_is_retained(chunk, episode, space, scope):
                retained.append(item)
        items = retained
    candidate_items = items
    items = [item for item in items if item.chunk_id in baseline_allowed][:limit]
    if runtime.demote_restated:
        items = fusion.demote_restated(
            items,
            {cid: c.text for cid, c in chunks.items()},
            {cid: c.created_at for cid, c in chunks.items()},
        )
    normalization_base = items[0].score if items else 0.0
    items = fusion.normalise(items)
    rerank_scores: dict[int, float] = {}
    rerank_trace: RerankTrace | None = None
    if active_reranker is not None:
        # Preserve the legacy denominator, including restatement demotion,
        # for both returned baseline items and additional candidates.
        normalized = {
            item.chunk_id: fusion.Fused(item.chunk_id,
                item.score / normalization_base if normalization_base > 0 else item.score,
                item.similarity)
            for item in candidate_items
        }
        baseline_ids = [item.chunk_id for item in items]
        candidate_order = baseline_ids + [item.chunk_id for item in candidate_items if item.chunk_id not in baseline_ids]
        candidates = tuple(RerankCandidate(
            chunk_id=cid, episode_id=chunks[cid].episode_id, text=chunks[cid].text,
            source=episodes[chunks[cid].episode_id].source, created_at=chunks[cid].created_at,
            baseline_score=normalized[cid].score, similarity=normalized[cid].similarity,
            lanes=tuple((lane, rank[cid]) for lane, rank in ranks.items() if cid in rank),
        ) for cid in candidate_order)
        outcome = await rerank_candidates(active_reranker, query, candidates,
            limit=runtime.rerank_limit, max_bytes=runtime.rerank_max_bytes, timeout=runtime.rerank_timeout)
        rerank_trace = outcome.trace
        latency["rerank"] = outcome.trace.duration_ms
        if outcome.failure is not None:
            degraded.append(f"rerank: {outcome.failure}")
        if outcome.trace.status == "applied":
            rerank_scores = outcome.scores
            ranked = list(outcome.ordered_ids) + [cid for cid in candidate_order if cid not in rerank_scores]
            items = fusion.cap_per_episode([normalized[cid] for cid in ranked], episode_of)[:limit]
        # A cooperative adapter can yield while a source is forgotten or
        # replaced. Never emit a stale passage after that asynchronous step.
        fresh = {chunk.chunk_id: chunk for chunk in await runtime.documents.get_chunks(space, [item.chunk_id for item in items])} if items else {}
        surviving = []
        for item in items:
            fresh_chunk = fresh.get(item.chunk_id)
            if fresh_chunk is None or fresh_chunk != chunks[item.chunk_id]:
                continue
            episode = await runtime.documents.get_episode(space, fresh_chunk.episode_id)
            if episode is not None and candidate_is_retained(fresh_chunk, episode, space, scope):
                episodes[fresh_chunk.episode_id] = episode
                surviving.append(item)
        if len(surviving) < len(items):
            degraded.append("rerank: retained_sources_changed")
        items = surviving

    for item in items:
        eid = chunks[item.chunk_id].episode_id
        if eid not in episodes:
            found = await runtime.documents.get_episode(space, eid)
            if found is not None:
                episodes[eid] = found

    result_items = []
    for item in items:
        chunk = chunks[item.chunk_id]
        episode = episodes.get(chunk.episode_id)
        result_items.append(
            RecallItem(
                chunk_id=chunk.chunk_id,
                episode_id=chunk.episode_id,
                text=chunk.text,
                score=round(item.score, 6),
                rerank_score=rerank_scores.get(item.chunk_id),
                similarity=None if item.similarity is None else round(item.similarity, 6),
                lanes={lane: r[chunk.chunk_id] for lane, r in ranks.items() if chunk.chunk_id in r},
                created_at=chunk.created_at,
                source=episode.source if episode else None,
                tags=episode.tags if episode else (),
                metadata=dict(episode.metadata) if episode else {},
            )
        )
    fact_scope = scope if narrowing or clean_tags or clean_where else None
    facts = await fact_recall.facts_for_query(runtime.documents, space, query, boundary or now, scope=fact_scope, degraded=degraded)
    previous = await fact_recall.history_for(runtime.documents, space, facts, boundary or now, scope=fact_scope) if history else []
    counts = await runtime.documents.counts(space)
    result = RecallResult(
        items=result_items,
        rerank=rerank_trace,
        facts=facts,
        history=previous,
        degraded=degraded,
        entities=query_entities,
        top_similarity=top_similarity,
        low_confidence=low_confidence,
        returned_bytes=sum(len(i.text.encode()) for i in result_items),
        space_bytes=counts.bytes,
    )
    latency["total"] = _ms(started)
    if candidate_limit is not None:
        evidence["candidate_limit"] = candidate_limit
    if rerank_trace is not None:
        evidence["rerank"] = rerank_trace.model_dump(mode="json")
    event = await runtime.emit(space, "recall", {
        **evidence,
        "latency_ms": latency,
        "degraded": degraded,
        # Only the returned items are recorded; candidates the lanes
        # saw but fusion dropped are not, so coverage is "returned".
        "items": [
            {"chunk_id": i.chunk_id, "episode_id": i.episode_id, "score": i.score,
             "similarity": i.similarity, "lanes": i.lanes,
             **({"rerank_score": i.rerank_score} if i.rerank_score is not None else {})}
            for i in result_items
        ],
        "items_coverage": "returned",
        "lane_candidates": {"vector": len(vector_lane), "text": len(text_lane)},
        "top_similarity": top_similarity,
        "low_confidence": low_confidence,
        "facts": len(facts),
        "fact_ids": [f.fact_id for f in facts],
        "returned_bytes": result.returned_bytes,
        "space_bytes": result.space_bytes,
    })
    if event is not None:
        result.event_id = event.event_id
    return result
