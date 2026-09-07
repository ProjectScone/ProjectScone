"""Documents, vectors and evidence in Elasticsearch (8.x).

Indices under one prefix: ``<prefix>_episodes``, ``_chunks`` (the lexical
lane: a ``text`` field scored by BM25 through ``match``, with the
episode's tags and metadata copied onto every chunk so scope filters
need no join), ``_facts``, ``_vectors`` (a ``dense_vector`` with cosine
similarity under approximate kNN, filtered inside the kNN clause),
``_events``, ``_counters`` (integer ids, bumped by a script) and
``_meta`` (the schema stamp). Tags are ``keyword`` arrays, metadata is a
``flattened`` object, timestamps are the engine's RFC 3339 strings kept
as keywords, which order lexicographically.

Elasticsearch is near-real-time: a write is visible after a refresh.
Every write here asks for one (``refresh=True``) so a recall right after
a remember sees it, which is what the contract requires; pass
``refresh=False`` for bulk loads where visibility can lag a second.

Elasticsearch reports kNN cosine as (1 + cos) / 2; the port speaks
cosine, so 2 * score - 1. Needs ``pip install 'scone-memory[elasticsearch]'``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from ..core.errors import SconeError
from ..retrieval.lexical import tokenize
from ..core.models import Chunk, Episode, Fact
from ..core.ports import DuplicateEvent, Event, NewChunk, NewEpisode, NewEvent, NewFact, SpaceCounts, TextFilter, VectorPoint
from ..core.timeutil import epoch_seconds, format_rfc3339, now_rfc3339, parse_rfc3339
from .validation import validate_vector

#: Shared spec 3.6. Pre-release: an index set another build wrote is
#: refused, not migrated.
SCHEMA_VERSION = 6


class SchemaMismatch(SconeError):
    pass


def _client(url: str, api_key: Optional[str], client):
    if client is not None:
        return client
    try:
        from elasticsearch import AsyncElasticsearch
    except ImportError as e:  # pragma: no cover
        raise ImportError("Elasticsearch stores need elasticsearch>=8: pip install 'scone-memory[elasticsearch]'") from e
    # A refresh per write on a small node can take a moment; wait for it
    # and retry a timed-out request rather than fail the operation.
    options = {"request_timeout": 30, "retry_on_timeout": True, "max_retries": 3}
    return AsyncElasticsearch(url, api_key=api_key, **options) if api_key else AsyncElasticsearch(url, **options)


def _same_payload(a, b) -> bool:
    return json.dumps(dict(a), sort_keys=True, default=str) == json.dumps(dict(b), sort_keys=True, default=str)


class Shared:
    """The client and index prefix the three stores share, plus the
    counter that hands out integer ids."""

    def __init__(self, url: str, prefix: str, api_key: Optional[str], client, refresh: bool) -> None:
        self.client = _client(url, api_key, client)
        self.prefix = prefix
        self.refresh = refresh
        self.users = 0

    def index(self, name: str) -> str:
        return f"{self.prefix}_{name}"

    async def ensure_index(self, name: str, mappings: dict) -> None:
        if not await self.client.indices.exists(index=self.index(name)):
            await self.client.indices.create(index=self.index(name), mappings=mappings)

    async def next_id(self, counter: str) -> int:
        # Concurrent bumps conflict on the document version (and on the
        # first upsert); the update API re-reads and reapplies the script.
        response = await self.client.update(
            index=self.index("counters"), id=counter, script={"source": "ctx._source.seq += 1"}, upsert={"seq": 1}, source=True,
            refresh=self.refresh, retry_on_conflict=20,
        )
        return int(response["get"]["_source"]["seq"])

    async def release(self) -> None:
        self.users = max(0, self.users - 1)
        if self.users == 0:
            await self.client.close()


def _episode(doc: Mapping) -> Episode:
    return Episode(
        episode_id=doc["episode_id"], space=doc["space"], kind=doc["kind"], content=doc["content"], content_hash=doc["content_hash"],
        source=doc.get("source"), tags=tuple(doc.get("tags", [])), metadata=dict(doc.get("meta", {})), created_at=doc["created_at"],
        ingested_at=doc["ingested_at"],
    )


def _chunk(doc: Mapping) -> Chunk:
    return Chunk(
        chunk_id=doc["chunk_id"], episode_id=doc["episode_id"], space=doc["space"], ordinal=doc["ordinal"], start=doc["start"],
        end=doc["end"], text=doc["text"], created_at=doc["created_at"],
    )


def _fact(doc: Mapping) -> Fact:
    return Fact(
        fact_id=doc["fact_id"], space=doc["space"], subject=doc["subject"], predicate=doc["predicate"], object=doc["object"],
        confidence=doc["confidence"], valid_from=doc["valid_from"], valid_until=doc.get("valid_until"), status=doc["status"],
        closed_reason=doc.get("closed_reason"), source_episode_id=doc.get("source_episode_id"), origin=doc.get("origin", "stated"),
        excluded_reason=doc.get("excluded_reason"), superseded_by=doc.get("superseded_by"), quote=doc.get("quote"),
    )


def _event(doc: Mapping) -> Event:
    return Event(
        event_id=doc["event_id"], ts=doc["ts"], space=doc["space"], kind=doc["kind"], payload=json.loads(doc["payload"]),
        schema_version=doc["schema_version"], dedup_key=doc.get("dedup_key"),
    )


def _filters(space: str, as_of: Optional[str], tags: Sequence[str], where: Mapping[str, str] | None, time_field: str, time_value) -> list[dict]:
    """The bool filter clauses every lane applies: space, time bound,
    every tag present, every metadata pair equal."""
    clauses: list[dict] = [{"term": {"space": space}}]
    if as_of:
        clauses.append({"range": {time_field: {"lte": time_value}}})
    for tag in tags:
        clauses.append({"term": {"tags": tag}})
    for key, value in (where or {}).items():
        clauses.append({"term": {f"meta.{key}": value}})
    return clauses


KEYWORD = {"type": "keyword"}
LONG = {"type": "long"}

EPISODE_MAPPINGS = {"properties": {
    "episode_id": LONG, "space": KEYWORD, "kind": KEYWORD, "content": {"type": "text", "index": False}, "content_hash": KEYWORD,
    "source": KEYWORD, "tags": KEYWORD, "meta": {"type": "flattened"}, "created_at": KEYWORD, "ingested_at": KEYWORD,
    "bytes": LONG,  # UTF-8 length of content, stored at insert so counts() is a sum, not a script
}}
CHUNK_MAPPINGS = {"properties": {
    "chunk_id": LONG, "episode_id": LONG, "space": KEYWORD, "ordinal": LONG, "start": LONG, "end": LONG,
    "text": {"type": "text"}, "created_at": KEYWORD, "tags": KEYWORD, "meta": {"type": "flattened"},
}}
FACT_MAPPINGS = {"properties": {
    "fact_id": LONG, "space": KEYWORD, "subject": KEYWORD, "predicate": KEYWORD, "object": KEYWORD, "confidence": {"type": "double"},
    "valid_from": KEYWORD, "valid_until": KEYWORD, "status": KEYWORD, "closed_reason": {"type": "text", "index": False},
    "source_episode_id": LONG, "origin": KEYWORD, "superseded_by": LONG, "excluded_reason": {"type": "text", "index": False},
    "quote": {"type": "text", "index": False},
}}
EVENT_MAPPINGS = {"properties": {
    "event_id": LONG, "ts": KEYWORD, "space": KEYWORD, "kind": KEYWORD, "schema_version": LONG,
    "payload": {"type": "text", "index": False}, "dedup_key": KEYWORD,
}}


class ElasticsearchDocumentStore:
    name = "elasticsearch"

    def __init__(
        self, url: str = "http://localhost:9200", prefix: str = "scone", api_key: Optional[str] = None, client=None,
        refresh: bool = True, shared: Optional[Shared] = None,
    ) -> None:
        self.shared = shared or Shared(url, prefix, api_key, client, refresh)
        self.shared.users += 1

    @property
    def client(self):
        return self.shared.client

    def _idx(self, name: str) -> str:
        return self.shared.index(name)

    def vectors(self) -> "ElasticsearchVectorIndex":
        return ElasticsearchVectorIndex(shared=self.shared)

    def events(self, max_age_days: Optional[float] = None, clock: Callable[[], str] = now_rfc3339) -> "ElasticsearchEventLog":
        return ElasticsearchEventLog(max_age_days=max_age_days, clock=clock, shared=self.shared)

    async def open(self) -> "ElasticsearchDocumentStore":
        await self.shared.ensure_index("episodes", EPISODE_MAPPINGS)
        await self.shared.ensure_index("chunks", CHUNK_MAPPINGS)
        await self.shared.ensure_index("facts", FACT_MAPPINGS)
        await self.shared.ensure_index("meta", {"properties": {"value": KEYWORD}})
        await self.shared.ensure_index("inflight", {"properties": {"space": KEYWORD, "content_hash": KEYWORD}})
        await self.check_schema()
        return self

    async def schema_version(self) -> int:
        if await self.client.exists(index=self._idx("meta"), id="schema_version"):
            return int((await self.client.get(index=self._idx("meta"), id="schema_version"))["_source"]["value"])
        count = (await self.client.count(index=self._idx("episodes")))["count"]
        return 1 if count else SCHEMA_VERSION

    async def check_schema(self) -> None:
        version = await self.schema_version()
        if version != SCHEMA_VERSION:
            raise SchemaMismatch(
                f"indices {self.shared.prefix!r}_* hold schema v{version}; this build writes v{SCHEMA_VERSION}. "
                "scone-memory is pre-release and does not migrate: export with the build that wrote them, "
                "delete the indices, and import again."
            )
        await self.client.index(index=self._idx("meta"), id="schema_version", document={"value": str(SCHEMA_VERSION)}, refresh=self.shared.refresh)

    async def drop(self) -> None:
        # Wildcards are refused by default (action.destructive_requires_name),
        # so the indices are named one by one.
        names = [self._idx(n) for n in ("episodes", "chunks", "facts", "meta", "counters", "vectors", "events", "inflight")]
        await self.client.indices.delete(index=",".join(names), ignore_unavailable=True)

    async def close(self) -> None:
        await self.shared.release()

    async def _search(self, name: str, **kwargs) -> list[dict]:
        response = await self.client.search(index=self._idx(name), **kwargs)
        return [hit["_source"] | {"_score": hit.get("_score")} for hit in response["hits"]["hits"]]

    async def insert_episode(self, new: NewEpisode) -> Episode:
        episode_id = await self.shared.next_id("episodes")
        doc = {
            "episode_id": episode_id, "space": new.space, "kind": new.kind, "content": new.content, "content_hash": new.content_hash,
            "source": new.source, "tags": list(new.tags), "meta": dict(new.metadata), "created_at": new.created_at, "ingested_at": new.ingested_at,
            "bytes": len(new.content.encode("utf-8")),
        }
        await self.client.index(index=self._idx("episodes"), id=str(episode_id), document=doc, refresh=self.shared.refresh)
        return _episode(doc)

    async def episode_by_hash(self, space: str, content_hash: str) -> Optional[Episode]:
        hits = await self._search("episodes", query={"bool": {"filter": [{"term": {"space": space}}, {"term": {"content_hash": content_hash}}]}}, size=1)
        return _episode(hits[0]) if hits else None

    async def _doc(self, name: str, doc_id: str) -> Optional[dict]:
        """One document by id, or None. A single get, not exists-then-get:
        a concurrent delete between the two would turn None into a 404."""
        from elasticsearch import NotFoundError

        try:
            return (await self.client.get(index=self._idx(name), id=doc_id))["_source"]
        except NotFoundError:
            return None

    async def get_episode(self, space: str, episode_id: int) -> Optional[Episode]:
        doc = await self._doc("episodes", str(episode_id))
        return _episode(doc) if doc is not None and doc["space"] == space else None

    #: Rows per page when walking an episode's chunks; Elasticsearch caps a
    #: single page at index.max_result_window (10,000 by default).
    PAGE = 10_000

    async def _chunk_ids(self, episode_id: int) -> list[int]:
        ids: list[int] = []
        after: Optional[list] = None
        while True:
            kwargs: dict = {"query": {"term": {"episode_id": episode_id}}, "size": self.PAGE, "sort": [{"chunk_id": "asc"}], "source": ["chunk_id"]}
            if after is not None:
                kwargs["search_after"] = after
            response = await self.client.search(index=self._idx("chunks"), **kwargs)
            hits = response["hits"]["hits"]
            ids.extend(int(h["_source"]["chunk_id"]) for h in hits)
            if len(hits) < self.PAGE:
                return ids
            after = hits[-1]["sort"]

    async def delete_episode(self, space: str, episode_id: int) -> list[int]:
        if await self.get_episode(space, episode_id) is None:
            return []
        removed = await self._chunk_ids(episode_id)
        await self.client.delete_by_query(index=self._idx("chunks"), query={"term": {"episode_id": episode_id}}, refresh=True)
        from elasticsearch import NotFoundError

        try:
            await self.client.delete(index=self._idx("episodes"), id=str(episode_id), refresh=self.shared.refresh)
        except NotFoundError:
            pass  # a concurrent delete got there first; the chunks are gone either way
        return removed

    async def insert_chunks(self, new: Sequence[NewChunk]) -> list[Chunk]:
        if not new:
            return []
        from elasticsearch.helpers import async_bulk

        # Scope fields ride on every chunk so the lexical lane filters
        # without a join; read once per episode in the batch.
        scopes: dict[int, tuple[list[str], dict]] = {}
        for episode_id in {n.episode_id for n in new}:
            episode = await self.get_episode(new[0].space, episode_id)
            scopes[episode_id] = (list(episode.tags), dict(episode.metadata)) if episode else ([], {})
        out, actions = [], []
        for n in new:
            chunk_id = await self.shared.next_id("chunks")
            tags, meta = scopes[n.episode_id]
            doc = {"chunk_id": chunk_id, **n.__dict__, "tags": tags, "meta": meta}
            actions.append({"_index": self._idx("chunks"), "_id": str(chunk_id), "_source": doc})
            out.append(_chunk(doc))
        await async_bulk(self.client, actions, refresh=self.shared.refresh)
        return out

    async def get_chunks(self, space: str, chunk_ids: Sequence[int]) -> list[Chunk]:
        if not chunk_ids:
            return []
        hits = await self._search("chunks", query={"bool": {"filter": [{"term": {"space": space}}, {"terms": {"chunk_id": list(chunk_ids)}}]}}, size=len(chunk_ids))
        return [_chunk(h) for h in hits]

    async def chunks_of(self, space: str, episode_id: int) -> list[Chunk]:
        out: list[Chunk] = []
        after: Optional[list] = None
        while True:
            kwargs: dict = {
                "query": {"bool": {"filter": [{"term": {"space": space}}, {"term": {"episode_id": episode_id}}]}},
                "size": self.PAGE, "sort": [{"ordinal": "asc"}],
            }
            if after is not None:
                kwargs["search_after"] = after
            response = await self.client.search(index=self._idx("chunks"), **kwargs)
            hits = response["hits"]["hits"]
            out.extend(_chunk(h["_source"]) for h in hits)
            if len(hits) < self.PAGE:
                return out
            after = hits[-1]["sort"]

    async def mark_inflight(self, space: str, content_hash: str) -> None:
        await self.client.index(
            index=self._idx("inflight"), id=f"{space}:{content_hash}", document={"space": space, "content_hash": content_hash},
            refresh=self.shared.refresh,
        )

    async def clear_inflight(self, space: str, content_hash: str) -> None:
        from elasticsearch import NotFoundError

        try:
            await self.client.delete(index=self._idx("inflight"), id=f"{space}:{content_hash}", refresh=self.shared.refresh)
        except NotFoundError:
            pass

    async def inflight(self) -> list[tuple[str, str]]:
        hits = await self._search("inflight", query={"match_all": {}}, size=10_000, sort=[{"space": "asc"}, {"content_hash": "asc"}])
        return [(h["space"], h["content_hash"]) for h in hits]

    async def search_text(self, space: str, query: str, limit: int, filter: TextFilter) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        body = {"bool": {
            "must": [{"match": {"text": {"query": " ".join(terms), "operator": "or"}}}],
            "filter": _filters(space, filter.as_of, filter.tags, filter.where, "created_at", filter.as_of),
        }}
        hits = await self._search("chunks", query=body, size=limit, sort=[{"_score": "desc"}, {"chunk_id": "asc"}], source=["chunk_id"])
        return [(int(h["chunk_id"]), float(h["_score"])) for h in hits]

    async def recent_episodes(self, space: str, limit: int) -> list[Episode]:
        hits = await self._search("episodes", query={"term": {"space": space}}, size=min(limit, 10_000),
                                  sort=[{"created_at": "desc"}, {"episode_id": "desc"}])
        return [_episode(h) for h in hits]

    async def page_episodes(self, space, before, limit, kind):
        filters = [{"term": {"space": space}}]
        if before is not None:
            filters.append({"range": {"episode_id": {"lt": before}}})
        if kind is not None:
            filters.append({"term": {"kind": kind}})
        hits = await self._search("episodes", query={"bool": {"filter": filters}},
                                  size=limit, sort=[{"episode_id": "desc"}])
        return [_episode(h) for h in hits]

    async def counts(self, space: str) -> SpaceCounts:
        counts = SpaceCounts()
        response = await self.client.search(
            index=self._idx("episodes"), query={"term": {"space": space}}, size=0, track_total_hits=True,
            aggs={"tags": {"terms": {"field": "tags", "size": 10_000}}, "bytes": {"sum": {"field": "bytes"}}},
        )
        counts.episodes = int(response["hits"]["total"]["value"])
        counts.bytes = int(response["aggregations"]["bytes"]["value"])
        for bucket in response["aggregations"]["tags"]["buckets"]:
            counts.tags[bucket["key"]] = int(bucket["doc_count"])
        counts.chunks = int((await self.client.count(index=self._idx("chunks"), query={"term": {"space": space}}))["count"])
        return counts

    async def insert_fact(self, new: NewFact) -> Fact:
        fact_id = await self.shared.next_id("facts")
        doc = {"fact_id": fact_id, **new.__dict__}
        await self.client.index(index=self._idx("facts"), id=str(fact_id), document=doc, refresh=self.shared.refresh)
        return _fact(doc)

    async def update_fact(self, fact: Fact) -> None:
        if await self.get_fact(fact.space, fact.fact_id) is None:
            raise KeyError(fact.fact_id)
        await self.client.update(
            index=self._idx("facts"), id=str(fact.fact_id), refresh=self.shared.refresh,
            doc={
                "object": fact.object, "confidence": fact.confidence, "valid_from": fact.valid_from, "valid_until": fact.valid_until,
                "status": fact.status, "closed_reason": fact.closed_reason, "origin": fact.origin, "excluded_reason": fact.excluded_reason,
                "superseded_by": fact.superseded_by,
            },
        )

    async def get_fact(self, space: str, fact_id: int) -> Optional[Fact]:
        doc = await self._doc("facts", str(fact_id))
        return _fact(doc) if doc is not None and doc["space"] == space else None

    async def list_facts(self, space: str, include_closed: bool) -> list[Fact]:
        filters: list[dict] = [{"term": {"space": space}}]
        if not include_closed:
            filters.append({"term": {"status": "active"}})
        hits = await self._search("facts", query={"bool": {"filter": filters}}, size=10_000, sort=[{"fact_id": "asc"}])
        return [_fact(h) for h in hits]

    async def facts_for(self, space: str, subject: str, predicate: str) -> list[Fact]:
        filters = [{"term": {"space": space}}, {"term": {"subject": subject}}, {"term": {"predicate": predicate}}]
        hits = await self._search("facts", query={"bool": {"filter": filters}}, size=10_000, sort=[{"fact_id": "asc"}])
        return [_fact(h) for h in hits]

    async def bump_revision(self, space: str) -> int:
        return await self.shared.next_id(f"revision:{space}")

    async def revision(self, space: str) -> int:
        if not await self.client.exists(index=self._idx("counters"), id=f"revision:{space}"):
            return 0
        return int((await self.client.get(index=self._idx("counters"), id=f"revision:{space}"))["_source"]["seq"])


class ElasticsearchVectorIndex:
    """Chunk vectors as a dense_vector under approximate kNN with the
    scope filters inside the kNN clause. The width is fixed by the mapping
    when the index is created; another width is refused."""

    name = "elasticsearch"

    def __init__(
        self, url: str = "http://localhost:9200", prefix: str = "scone", api_key: Optional[str] = None, client=None,
        refresh: bool = True, shared: Optional[Shared] = None,
    ) -> None:
        self.shared = shared or Shared(url, prefix, api_key, client, refresh)
        self.shared.users += 1
        self.dim: Optional[int] = None

    @property
    def client(self):
        return self.shared.client

    @property
    def index(self) -> str:
        return self.shared.index("vectors")

    async def ensure(self, dim: int) -> None:
        if await self.client.indices.exists(index=self.index):
            mapping = await self.client.indices.get_mapping(index=self.index)
            existing = int(mapping[self.index]["mappings"]["properties"]["embedding"]["dims"])
            if existing != dim:
                raise ValueError(f"index {self.index!r} holds {existing}-d vectors, embedder makes {dim}-d")
        else:
            await self.client.indices.create(index=self.index, mappings={"properties": {
                "chunk_id": LONG, "space": KEYWORD, "episode_id": LONG, "created_at": KEYWORD, "created_ts": {"type": "double"},
                "tags": KEYWORD, "meta": {"type": "flattened"},
                # Plain HNSW over float32: the 8.x default (int8_hnsw)
                # quantises vectors and reports a cosine of 0.998 for an
                # identical text, which would leak into the abstention gate.
                "embedding": {"type": "dense_vector", "dims": dim, "index": True, "similarity": "cosine", "index_options": {"type": "hnsw"}},
            }})
        self.dim = dim

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        from elasticsearch.helpers import async_bulk

        for point in points:
            validate_vector(point.vector, self.dim)
        actions = [
            {"_index": self.index, "_id": str(p.chunk_id), "_source": {
                "chunk_id": p.chunk_id, "space": p.space, "episode_id": p.episode_id, "created_at": p.created_at,
                "created_ts": epoch_seconds(p.created_at), "tags": list(p.tags), "meta": dict(p.metadata),
                "embedding": list(map(float, p.vector)),
            }}
            for p in points
        ]
        await async_bulk(self.client, actions, refresh=self.shared.refresh)

    async def search(
        self,
        space: str,
        vector: Sequence[float],
        limit: int,
        as_of: Optional[str] = None,
        tags: tuple[str, ...] = (),
        where: Mapping[str, str] | None = None,
    ) -> list[tuple[int, float]]:
        validate_vector(vector, self.dim)
        knn = {
            "field": "embedding", "query_vector": list(map(float, vector)), "k": int(limit), "num_candidates": max(50, int(limit) * 10),
            "filter": {"bool": {"filter": _filters(space, as_of, tags, where, "created_ts", epoch_seconds(as_of) if as_of else None)}},
        }
        response = await self.client.search(index=self.index, knn=knn, size=int(limit), source=["chunk_id"])
        # Elasticsearch scores cosine kNN as (1 + cos) / 2; the port speaks cosine.
        ranked = [(int(h["_source"]["chunk_id"]), 2.0 * float(h["_score"]) - 1.0) for h in response["hits"]["hits"]]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        if not chunk_ids:
            return
        await self.client.delete_by_query(index=self.index, query={"terms": {"chunk_id": [int(c) for c in chunk_ids]}}, refresh=True)

    async def drop(self) -> None:
        await self.client.indices.delete(index=self.index, ignore_unavailable=True)

    async def close(self) -> None:
        await self.shared.release()


class ElasticsearchEventLog:
    """Evidence in an ``events`` index. A keyed event's document id is
    ``<space>:<dedup_key>`` written with op_type create, so the second
    append with the same key conflicts in the server and the payload
    comparison decides same-or-conflict. Retention is a sweep by the
    log's own clock, on open and every ``SWEEP_EVERY`` appends."""

    name = "elasticsearch"
    SWEEP_EVERY = 500

    def __init__(
        self, url: str = "http://localhost:9200", prefix: str = "scone", api_key: Optional[str] = None, client=None,
        refresh: bool = True, max_age_days: Optional[float] = None, clock: Callable[[], str] = now_rfc3339,
        shared: Optional[Shared] = None,
    ) -> None:
        self.shared = shared or Shared(url, prefix, api_key, client, refresh)
        self.shared.users += 1
        self.max_age_days = max_age_days
        self.clock = clock
        self._appends = 0

    @property
    def client(self):
        return self.shared.client

    @property
    def index(self) -> str:
        return self.shared.index("events")

    async def open(self) -> "ElasticsearchEventLog":
        await self.shared.ensure_index("events", EVENT_MAPPINGS)
        await self.sweep()
        return self

    async def drop(self) -> None:
        await self.client.indices.delete(index=self.index, ignore_unavailable=True)

    async def close(self) -> None:
        await self.shared.release()

    async def append(self, new: NewEvent) -> Event:
        from elasticsearch import ConflictError

        event_id = await self.shared.next_id("events")
        doc = {
            "event_id": event_id, "ts": new.ts, "space": new.space, "kind": new.kind, "schema_version": new.schema_version,
            "payload": json.dumps(dict(new.payload), ensure_ascii=False, default=str), "dedup_key": new.dedup_key,
        }
        doc_id = f"{new.space}:{new.dedup_key}" if new.dedup_key is not None else str(event_id)
        try:
            await self.client.index(index=self.index, id=doc_id, document=doc, op_type="create", refresh=self.shared.refresh)
        except ConflictError:
            existing = _event((await self.client.get(index=self.index, id=doc_id))["_source"])
            if _same_payload(existing.payload, new.payload):
                return existing
            raise DuplicateEvent(existing) from None
        self._appends += 1
        if self.max_age_days is not None and self._appends % self.SWEEP_EVERY == 0:
            await self.sweep()
        return Event(event_id=event_id, **new.__dict__)

    async def sweep(self) -> int:
        if self.max_age_days is None:
            return 0
        cutoff = parse_rfc3339(self.clock()).timestamp() - self.max_age_days * 86400
        cutoff_text = format_rfc3339(datetime.fromtimestamp(cutoff, tz=timezone.utc))
        response = await self.client.delete_by_query(index=self.index, query={"range": {"ts": {"lt": cutoff_text}}}, refresh=True)
        return int(response["deleted"])

    async def get(self, space: str, event_id: int) -> Optional[Event]:
        response = await self.client.search(
            index=self.index, query={"bool": {"filter": [{"term": {"space": space}}, {"term": {"event_id": event_id}}]}}, size=1
        )
        hits = response["hits"]["hits"]
        return _event(hits[0]["_source"]) if hits else None

    async def query(
        self, space: str, kind: Optional[str] = None, since: Optional[str] = None, limit: int = 100, after_id: Optional[int] = None
    ) -> list[Event]:
        filters: list[dict] = [{"term": {"space": space}}]
        if kind:
            filters.append({"term": {"kind": kind}})
        if since:
            filters.append({"range": {"ts": {"gte": format_rfc3339(parse_rfc3339(since))}}})
        if after_id is not None:
            filters.append({"range": {"event_id": {"gt": after_id}}})
        order = "asc" if after_id is not None else "desc"
        response = await self.client.search(index=self.index, query={"bool": {"filter": filters}}, size=min(limit, 10_000), sort=[{"event_id": order}])
        return [_event(h["_source"]) for h in response["hits"]["hits"]]
