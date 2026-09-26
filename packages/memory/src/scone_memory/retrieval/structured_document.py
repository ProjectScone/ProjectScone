"""Experimental vector-first retrieval with Jev-directed section addresses.

Callers provide current authorized snapshots. A separate index namespace binds
vectors to source bytes and embedder identity. Existing MemoryEngine collections
are never migrated or switched by importing this module.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import hashlib
import math
import time
from typing import Literal, Sequence

from ..backends.memory import InMemoryVectorIndex
from ..core.embedding import embed_queries
from ..core.ports import Embedder, VectorIndex, VectorPoint
from ..ingestion.chunker import byte_spans, chunk_spans
from ..ingestion.vectors import validated_vectors
from .section_routing import FetchChooser, FetchDecision, RouteResult, SectionRouter, SectionSnapshot

Mode = Literal['auto', 'flat_vector', 'section_vector', 'original']
Routing = Literal['hierarchy', 'vector_candidates']


@dataclass(frozen=True)
class SectionEvidence:
    section_id: str
    text: str
    start: int
    end: int
    score: float = 0


@dataclass(frozen=True)
class StructuredResult:
    mode: str
    reason: str
    evidence: tuple[SectionEvidence, ...]
    baseline: tuple[SectionEvidence, ...]
    route: RouteResult | None
    fetch: FetchDecision | None
    query_embedding_ms: float
    vector_search_ms: float
    fetch_decision_ms: float
    total_ms: float


def _bounded(items: Sequence[SectionEvidence], limit: int, max_bytes: int) -> tuple[SectionEvidence, ...]:
    result: list[SectionEvidence] = []
    remaining = max_bytes
    for item in items[:limit]:
        raw = item.text.encode()[:remaining]
        text = raw.decode(errors='ignore')
        if text:
            result.append(replace(item, text=text, end=item.start + len(text.encode())))
            remaining -= len(text.encode())
        if remaining <= 0:
            break
    return tuple(result)


def _fuse(broad: Sequence[SectionEvidence], scoped: Sequence[SectionEvidence]) -> list[SectionEvidence]:
    scores: dict[tuple[int, int], float] = {}
    items: dict[tuple[int, int], SectionEvidence] = {}
    for lane in (broad, scoped):
        for rank, item in enumerate(lane, 1):
            key = (item.start, item.end)
            items[key] = item
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
    return [replace(items[key], score=scores[key]) for key in sorted(scores,
            key=lambda k: (-scores[k], k))]


class StructuredDocumentIndex:
    def __init__(self, snapshot: SectionSnapshot, embedder: Embedder, vectors: VectorIndex,
                 space: str, passages: dict[int, SectionEvidence]) -> None:
        self.snapshot, self.embedder, self.vectors = snapshot, embedder, vectors
        self._embedder_id = embedder.id
        self._dimension = embedder.dim
        self._nodes = {n.id: n for n in snapshot.nodes}
        self.space, self.passages = space, passages
        self._depth = {n.id: self._ancestors(n.id) for n in snapshot.nodes}

    def _ancestors(self, identifier: str) -> tuple[str, ...]:
        ancestors: list[str] = []
        current: str | None = identifier
        while current is not None:
            ancestors.append(current)
            current = self._nodes[current].parent_id
        return tuple(reversed(ancestors))

    @classmethod
    async def build(cls, snapshot: SectionSnapshot, embedder: Embedder, *,
                    vectors: VectorIndex | None = None, chunk_size: int = 700) -> StructuredDocumentIndex:
        if type(chunk_size) is not int or not 120 <= chunk_size <= 8000:
            raise ValueError('chunk size must be 120..8000')
        # The native parser is the sole authority for byte spans and hierarchy.
        if SectionSnapshot.from_markdown(snapshot.scope, snapshot.document_id, snapshot.content) != snapshot:
            raise ValueError('snapshot does not match source structure')
        identity = (snapshot.scope, snapshot.document_id, snapshot.version, embedder.id, chunk_size)
        space = 'section-experiment:' + hashlib.sha256(repr(identity).encode()).hexdigest()
        passages: dict[int, SectionEvidence] = {}
        raw = snapshot.content.encode()
        first_child: dict[str, int] = {}
        for node in snapshot.nodes:
            if node.parent_id is not None:
                first_child.setdefault(node.parent_id, node.start)
        for node in snapshot.nodes:
            own_end = first_child.get(node.id, node.end)
            own_text = raw[node.start:own_end].decode()
            for span in byte_spans(own_text, chunk_spans(own_text, chunk_size)):
                start, end = node.start + span.start, node.start + span.end
                identifier = int.from_bytes(hashlib.sha256(f'{space}:{start}:{end}'.encode()).digest()[:8], 'big') & ((1 << 63) - 1)
                if identifier in passages:
                    raise ValueError('passage identity collision')
                text = raw[start:end].decode()
                passages[identifier] = SectionEvidence(node.id, text, start, end)
        index = cls(snapshot, embedder, vectors if vectors is not None else InMemoryVectorIndex(), space, passages)
        records = list(passages.items())
        for offset in range(0, len(records), 128):
            batch = records[offset:offset + 128]
            encoded = validated_vectors(await embedder.embed([p.text for _, p in batch]),
                                        len(batch), index._dimension)
            index._dimension = len(encoded[0])
            await index.vectors.ensure(index._dimension)
            points = [VectorPoint(identifier, space, 0, '2026-09-24T00:00:00Z', vector,
                metadata={f'section_{depth}': ancestor for depth, ancestor in enumerate(index._depth[p.section_id])})
                for (identifier, p), vector in zip(batch, encoded, strict=True)]
            await index.vectors.upsert(points)
        return index

    async def _search(self, vector: Sequence[float], limit: int,
                      section_ids: tuple[str, ...] = ()) -> list[SectionEvidence]:
        lanes: list[list[tuple[int, float]]] = []
        for section_id in section_ids or ('',):
            where = ({f'section_{len(self._depth[section_id]) - 1}': section_id} if section_id else None)
            lanes.append(await self.vectors.search(self.space, vector, limit, where=where))
        found: dict[int, float] = {}
        for lane in lanes:
            for identifier, score in lane:
                if identifier not in self.passages or not math.isfinite(score):
                    raise ValueError('vector index returned an invalid passage')
                passage = self.passages[identifier]
                if section_ids and not any(s in self._depth[passage.section_id] for s in section_ids):
                    continue
                found[identifier] = max(found.get(identifier, -math.inf), score)
        return [replace(self.passages[key], score=found[key]) for key in sorted(found,
                key=lambda key: (-found[key], self.passages[key].start))[:limit]]

    def _originals(self, route: RouteResult) -> list[SectionEvidence]:
        by_id = {node.id: node for node in self.snapshot.nodes}
        result: list[SectionEvidence] = []
        ranked = sorted(zip(route.section_ids, route.scores, strict=True), key=lambda pair: -pair[1])
        for identifier, score in ranked:
            node = by_id[identifier]
            if any(node.start < item.end and item.start < node.end for item in result):
                continue
            result.append(SectionEvidence(identifier, self.snapshot.original(identifier), node.start, node.end, score))
        return result

    async def retrieve(self, query: str, snapshot: SectionSnapshot, router: SectionRouter,
                       fetch_chooser: FetchChooser, *, mode: Mode = 'auto', limit: int = 5,
                       max_bytes: int = 8000, routing: Routing = 'hierarchy') -> StructuredResult:
        if (snapshot != self.snapshot or self.embedder.id != self._embedder_id):
            raise ValueError('snapshot or embedder changed; rebuild the document index')
        if (mode not in ('auto', 'flat_vector', 'section_vector', 'original')
                or routing not in ('hierarchy', 'vector_candidates')
                or type(limit) is not int or not 1 <= limit <= 64
                or type(max_bytes) is not int or not 1 <= max_bytes <= 128000
                or not query.strip() or len(query.encode()) > 8000):
            raise ValueError('invalid structured retrieval bounds')
        started = time.perf_counter()
        encoded_ms = 0.0

        async def encode() -> list[float]:
            nonlocal encoded_ms
            before = time.perf_counter()
            encoded = validated_vectors(await embed_queries(self.embedder, [query]), 1, self._dimension)[0]
            encoded_ms = (time.perf_counter() - before) * 1000
            return encoded

        route: RouteResult | None = None
        if mode == 'flat_vector' or not self.passages or routing == 'vector_candidates':
            vector = await encode() if self.passages else []
        else:
            async with asyncio.TaskGroup() as group:
                encoded_task = group.create_task(encode())
                route_task = group.create_task(router.route(query, snapshot))
            vector, route = encoded_task.result(), route_task.result()
        before = time.perf_counter()
        broad = await self._search(vector, max(32, limit)) if self.passages else []
        vector_ms = (time.perf_counter() - before) * 1000
        baseline = _bounded(broad, limit, max_bytes)
        if routing == 'vector_candidates' and mode != 'flat_vector' and broad:
            route = await router.route(query, snapshot, candidate_section_ids=tuple(
                dict.fromkeys(passage.section_id for passage in broad)))
        effective, reason = 'flat_vector', 'flat_requested' if mode == 'flat_vector' else 'empty_document'
        evidence, fetch, fetch_ms = baseline, None, 0.0
        if route is not None:
            reason = route.reason
        elif mode != 'flat_vector' and self.passages and routing == 'vector_candidates':
            reason = 'no_vector_candidates'
        if route is not None and route.reason == 'routed':
            originals = self._originals(route)
            effective, reason = mode, 'explicit_mode'
            if mode == 'auto':
                before = time.perf_counter()
                try:
                    async with asyncio.timeout(router.timeout):
                        fetch = await fetch_chooser.choose_fetch(query, tuple(
                            (' / '.join(snapshot.path(p.section_id)) or '(whole document)', len(p.text.encode()))
                            for p in originals), max_bytes)
                    modes = ('original', 'section_vector', 'flat_vector')
                    probabilities = fetch.probabilities
                    if (fetch.mode not in modes or len(probabilities) != 3
                            or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                                   for p in probabilities)
                            or not math.isclose(sum(probabilities), 1, abs_tol=.005)
                            or probabilities[modes.index(fetch.mode)] < max(probabilities)
                            or not fetch.model.strip()
                            or any(type(t) is not int or t < 0 for t in (fetch.input_tokens, fetch.output_tokens))):
                        raise ValueError('invalid fetch decision')
                    effective, reason = fetch.mode, 'jev_selected'
                except Exception:
                    effective, reason = 'flat_vector', 'fetch_decision_failed'
                fetch_ms = (time.perf_counter() - before) * 1000
            if effective == 'original':
                if len(originals) > limit or sum(len(p.text.encode()) for p in originals) > max_bytes:
                    effective, reason = 'section_vector', 'original_exceeds_budget'
                else:
                    evidence = tuple(originals)
            if effective == 'section_vector':
                before = time.perf_counter()
                try:
                    async with asyncio.timeout(router.timeout):
                        scoped = await self._search(vector, max(32, limit),
                                                    tuple(p.section_id for p in originals))
                    evidence = _bounded(_fuse(broad, scoped), limit, max_bytes)
                except Exception:
                    effective, reason = 'flat_vector', 'scoped_search_failed'
                    evidence = baseline
                vector_ms += (time.perf_counter() - before) * 1000
        return StructuredResult(effective, reason, evidence, baseline, route, fetch,
            encoded_ms, vector_ms, fetch_ms, (time.perf_counter() - started) * 1000)
