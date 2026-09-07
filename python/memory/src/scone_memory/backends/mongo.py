"""Documents in MongoDB through pymongo's async client.

Integer ids come from a ``counters`` collection so the HTTP surface stays
compatible with the shared client, which reads ``episode_id`` and
``fact_id`` as integers. The lexical lane is a ``$text`` index on chunk
text; its score is not BM25, but only its order reaches the fusion, so
ranking agrees with the reference implementation in every contract test.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from ..core.errors import SconeError
from ..retrieval.lexical import tokenize
from ..core.models import Chunk, Episode, Fact
from ..core.ports import NewChunk, NewEpisode, NewFact, SpaceCounts, TextFilter

#: Shared spec 3.6. Pre-release: a database another build wrote is
#: refused, not migrated.
SCHEMA_VERSION = 6  # 6: facts carry quote


class SchemaMismatch(SconeError):
    pass


def _episode(doc: Mapping) -> Episode:
    return Episode(
        episode_id=doc["_id"],
        space=doc["space"],
        kind=doc["kind"],
        content=doc["content"],
        content_hash=doc["content_hash"],
        source=doc.get("source"),
        tags=tuple(doc.get("tags", [])),
        metadata=dict(doc.get("metadata", {})),
        created_at=doc["created_at"],
        ingested_at=doc["ingested_at"],
    )


def _chunk(doc: Mapping) -> Chunk:
    return Chunk(
        chunk_id=doc["_id"],
        episode_id=doc["episode_id"],
        space=doc["space"],
        ordinal=doc["ordinal"],
        start=doc["start"],
        end=doc["end"],
        text=doc["text"],
        created_at=doc["created_at"],
    )


def _fact(doc: Mapping) -> Fact:
    return Fact(
        fact_id=doc["_id"],
        space=doc["space"],
        subject=doc["subject"],
        predicate=doc["predicate"],
        object=doc["object"],
        confidence=doc["confidence"],
        valid_from=doc["valid_from"],
        valid_until=doc.get("valid_until"),
        status=doc["status"],
        closed_reason=doc.get("closed_reason"),
        source_episode_id=doc.get("source_episode_id"),
        origin=doc.get("origin", "stated"),
        excluded_reason=doc.get("excluded_reason"),
        superseded_by=doc.get("superseded_by"),
        quote=doc.get("quote"),
    )


class MongoDocumentStore:
    name = "mongo"

    def __init__(self, url: str, database: str = "scone", client=None) -> None:
        try:
            from pymongo import AsyncMongoClient
        except ImportError as e:  # pragma: no cover
            raise ImportError("MongoDocumentStore needs pymongo>=4.13: pip install 'scone-memory[mongo]'") from e
        self.client = client or AsyncMongoClient(url)
        self.db = self.client[database]
        self.episodes = self.db["episodes"]
        self.chunks = self.db["chunks"]
        self.facts = self.db["facts"]
        self.counters = self.db["counters"]
        self.revisions = self.db["revisions"]
        self.meta = self.db["meta"]
        self.inflight_marks = self.db["inflight"]

    async def open(self) -> "MongoDocumentStore":
        await self.episodes.create_index([("space", 1), ("content_hash", 1)], unique=True)
        await self.episodes.create_index([("space", 1), ("created_at", -1)])
        await self.chunks.create_index([("episode_id", 1)])
        await self.chunks.create_index([("space", 1), ("created_at", 1)])
        await self.chunks.create_index([("text", "text")])
        await self.facts.create_index([("space", 1), ("subject", 1), ("predicate", 1)])
        await self.inflight_marks.create_index([("space", 1), ("content_hash", 1)], unique=True)
        await self.check_schema()
        return self

    async def schema_version(self) -> int:
        doc = await self.meta.find_one({"_id": "schema"})
        if doc:
            return int(doc["version"])
        has_rows = await self.episodes.find_one({}, {"_id": 1}) is not None
        return 1 if has_rows else SCHEMA_VERSION

    async def check_schema(self) -> None:
        version = await self.schema_version()
        if version != SCHEMA_VERSION:
            raise SchemaMismatch(
                f"database {self.db.name!r} holds schema v{version}; this build writes v{SCHEMA_VERSION}. "
                "scone-memory is pre-release and does not migrate: export with the build that wrote it, "
                "drop the database, and import again."
            )
        await self.meta.replace_one({"_id": "schema"}, {"_id": "schema", "version": SCHEMA_VERSION}, upsert=True)

    async def drop(self) -> None:
        await self.client.drop_database(self.db.name)

    async def close(self) -> None:
        await self.client.close()

    async def _next_id(self, name: str) -> int:
        doc = await self.counters.find_one_and_update(
            {"_id": name}, {"$inc": {"seq": 1}}, upsert=True, return_document=True
        )
        return int(doc["seq"])

    async def insert_episode(self, new: NewEpisode) -> Episode:
        episode_id = await self._next_id("episodes")
        doc = {
            "_id": episode_id,
            "space": new.space,
            "kind": new.kind,
            "content": new.content,
            "content_hash": new.content_hash,
            "source": new.source,
            "tags": list(new.tags),
            "metadata": dict(new.metadata),
            "created_at": new.created_at,
            "ingested_at": new.ingested_at,
        }
        await self.episodes.insert_one(doc)
        return _episode(doc)

    async def episode_by_hash(self, space: str, content_hash: str) -> Optional[Episode]:
        doc = await self.episodes.find_one({"space": space, "content_hash": content_hash})
        return _episode(doc) if doc else None

    async def get_episode(self, space: str, episode_id: int) -> Optional[Episode]:
        doc = await self.episodes.find_one({"_id": episode_id, "space": space})
        return _episode(doc) if doc else None

    async def delete_episode(self, space: str, episode_id: int) -> list[int]:
        if await self.episodes.find_one({"_id": episode_id, "space": space}) is None:
            return []
        removed = [doc["_id"] async for doc in self.chunks.find({"episode_id": episode_id}, {"_id": 1})]
        await self.chunks.delete_many({"episode_id": episode_id})
        await self.episodes.delete_one({"_id": episode_id})
        return removed

    async def insert_chunks(self, new: Sequence[NewChunk]) -> list[Chunk]:
        out = []
        docs = []
        for n in new:
            chunk_id = await self._next_id("chunks")
            doc = {"_id": chunk_id, **n.__dict__}
            docs.append(doc)
            out.append(_chunk(doc))
        if docs:
            await self.chunks.insert_many(docs)
        return out

    async def get_chunks(self, space: str, chunk_ids: Sequence[int]) -> list[Chunk]:
        if not chunk_ids:
            return []
        cursor = self.chunks.find({"space": space, "_id": {"$in": list(chunk_ids)}})
        return [_chunk(doc) async for doc in cursor]

    async def chunks_of(self, space: str, episode_id: int) -> list[Chunk]:
        cursor = self.chunks.find({"space": space, "episode_id": episode_id}).sort("ordinal", 1)
        return [_chunk(doc) async for doc in cursor]

    async def mark_inflight(self, space: str, content_hash: str) -> None:
        await self.inflight_marks.update_one(
            {"space": space, "content_hash": content_hash}, {"$setOnInsert": {"space": space, "content_hash": content_hash}}, upsert=True
        )

    async def clear_inflight(self, space: str, content_hash: str) -> None:
        await self.inflight_marks.delete_one({"space": space, "content_hash": content_hash})

    async def inflight(self) -> list[tuple[str, str]]:
        cursor = self.inflight_marks.find({}).sort([("space", 1), ("content_hash", 1)])
        return [(doc["space"], doc["content_hash"]) async for doc in cursor]

    async def search_text(
        self, space: str, query: str, limit: int, filter: TextFilter
    ) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        match: dict = {"$text": {"$search": " ".join(terms)}, "space": space}
        if filter.as_of:
            match["created_at"] = {"$lte": filter.as_of}
        pipeline: list[dict] = [{"$match": match}, {"$addFields": {"score": {"$meta": "textScore"}}}]
        if filter.tags or filter.where:
            pipeline.append(
                {"$lookup": {"from": "episodes", "localField": "episode_id", "foreignField": "_id", "as": "episode"}}
            )
            pipeline.append({"$unwind": "$episode"})
            if filter.tags:
                pipeline.append({"$match": {"episode.tags": {"$all": list(filter.tags)}}})
            for key, value in filter.where.items():
                pipeline.append({"$match": {f"episode.metadata.{key}": value}})
        pipeline += [{"$sort": {"score": -1, "_id": 1}}, {"$limit": limit}, {"$project": {"score": 1}}]
        return [(doc["_id"], float(doc["score"])) async for doc in await self.chunks.aggregate(pipeline)]

    async def recent_episodes(self, space: str, limit: int) -> list[Episode]:
        cursor = self.episodes.find({"space": space}).sort([("created_at", -1), ("_id", -1)]).limit(limit)
        return [_episode(doc) async for doc in cursor]

    async def page_episodes(self, space, before, limit, kind):
        query = {"space": space}
        if before is not None:
            query["_id"] = {"$lt": before}
        if kind is not None:
            query["kind"] = kind
        cursor = self.episodes.find(query).sort("_id", -1).limit(limit)
        return [_episode(doc) async for doc in cursor]

    async def counts(self, space: str) -> SpaceCounts:
        counts = SpaceCounts()
        pipeline = [
            {"$match": {"space": space}},
            {"$group": {"_id": None, "n": {"$sum": 1}, "b": {"$sum": {"$binarySize": "$content"}}}},
        ]
        async for doc in await self.episodes.aggregate(pipeline):
            counts.episodes, counts.bytes = doc["n"], doc["b"]
        counts.chunks = await self.chunks.count_documents({"space": space})
        tag_pipeline = [
            {"$match": {"space": space}},
            {"$unwind": "$tags"},
            {"$group": {"_id": "$tags", "n": {"$sum": 1}}},
        ]
        async for doc in await self.episodes.aggregate(tag_pipeline):
            counts.tags[doc["_id"]] = doc["n"]
        return counts

    async def insert_fact(self, new: NewFact) -> Fact:
        fact_id = await self._next_id("facts")
        doc = {"_id": fact_id, **new.__dict__}
        await self.facts.insert_one(doc)
        return _fact(doc)

    async def update_fact(self, fact: Fact) -> None:
        result = await self.facts.update_one(
            {"_id": fact.fact_id, "space": fact.space},
            {
                "$set": {
                    "object": fact.object,
                    "confidence": fact.confidence,
                    "valid_from": fact.valid_from,
                    "valid_until": fact.valid_until,
                    "status": fact.status,
                    "closed_reason": fact.closed_reason,
                    "origin": fact.origin,
                    "excluded_reason": fact.excluded_reason,
                    "superseded_by": fact.superseded_by,
                }
            },
        )
        if result.matched_count == 0:
            raise KeyError(fact.fact_id)

    async def get_fact(self, space: str, fact_id: int) -> Optional[Fact]:
        doc = await self.facts.find_one({"_id": fact_id, "space": space})
        return _fact(doc) if doc else None

    async def list_facts(self, space: str, include_closed: bool) -> list[Fact]:
        query: dict = {"space": space}
        if not include_closed:
            query["status"] = "active"
        return [_fact(doc) async for doc in self.facts.find(query).sort("_id", 1)]

    async def facts_for(self, space: str, subject: str, predicate: str) -> list[Fact]:
        cursor = self.facts.find({"space": space, "subject": subject, "predicate": predicate}).sort("_id", 1)
        return [_fact(doc) async for doc in cursor]

    async def bump_revision(self, space: str) -> int:
        doc = await self.revisions.find_one_and_update(
            {"_id": space}, {"$inc": {"revision": 1}}, upsert=True, return_document=True
        )
        return int(doc["revision"])

    async def revision(self, space: str) -> int:
        doc = await self.revisions.find_one({"_id": space})
        return int(doc["revision"]) if doc else 0
