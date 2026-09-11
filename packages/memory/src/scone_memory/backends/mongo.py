"""Documents in MongoDB through pymongo's async client.

Integer ids come from a ``counters`` collection so the HTTP surface stays
compatible with the shared client, which reads ``episode_id`` and
``fact_id`` as integers. The lexical lane is a ``$text`` index on chunk
text; its score is not BM25, but only its order reaches the fusion, so
ranking agrees with the reference implementation in every contract test.
"""

from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence

from ..core.affirmations import Affirmation, NewAffirmation, read_links, stored_links
from ..core.errors import SconeError
from ..retrieval.lexical import tokenize
from ..core.models import Chunk, Episode, Fact, FactLink, Tombstone
from ..core.ports import DeletedSpace, NewChunk, NewEpisode, NewFact, NewFactLink, NewTombstone, SpaceCounts, TextFilter

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


def _tombstone(doc: Mapping) -> Tombstone:
    return Tombstone(space=doc["space"], episode_id=int(doc["episode_id"]), content_hash=doc["content_hash"],
                     forgotten_at=doc["forgotten_at"], reason=doc.get("reason"))


def _fact_link(doc: Mapping) -> FactLink:
    return FactLink(link_id=int(doc["_id"]), space=doc["space"], from_fact=int(doc["from_fact"]), to_fact=int(doc["to_fact"]),
                    kind=doc["kind"], created_at=doc["created_at"], source_episode_id=doc.get("source_episode_id"), quote=doc.get("quote"))


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
        self._fact_links = self.db["fact_links"]
        self._affirmations = self.db["fact_affirmations"]
        self.tombstones = self.db["tombstones"]
        self.counters = self.db["counters"]
        self.revisions = self.db["revisions"]
        self.meta = self.db["meta"]
        self.inflight_marks = self.db["inflight"]

    async def open(self) -> "MongoDocumentStore":
        await self.episodes.create_index([("space", 1), ("content_hash", 1)], unique=True)
        await self.episodes.create_index([("space", 1), ("created_at", -1)])
        await self.chunks.create_index([("episode_id", 1)])
        await self.chunks.create_index([("space", 1), ("episode_id", 1), ("ordinal", 1), ("_id", 1)])
        await self.chunks.create_index([("space", 1), ("created_at", 1)])
        await self.chunks.create_index([("text", "text")])
        await self.facts.create_index([("space", 1), ("subject", 1), ("predicate", 1)])
        await self.facts.create_index([("space", 1), ("subject", 1), ("_id", 1)])
        await self.facts.create_index([("space", 1), ("_id", 1)])
        await self.facts.create_index([("space", 1), ("source_episode_id", 1), ("_id", 1)])
        await self._fact_links.create_index([("space", 1), ("from_fact", 1), ("to_fact", 1), ("kind", 1)], unique=True)
        await self._fact_links.create_index([("space", 1), ("to_fact", 1)])
        await self._fact_links.create_index([("space", 1), ("from_fact", 1), ("_id", 1)])
        await self._fact_links.create_index([("space", 1), ("to_fact", 1), ("_id", 1)])
        await self._affirmations.create_index([("space", 1), ("fact_id", 1), ("valid_from", 1)], unique=True)
        await self.tombstones.create_index([("space", 1), ("episode_id", 1)], unique=True)
        await self.tombstones.create_index([("space", 1), ("content_hash", 1)])
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
        if doc is None:
            raise SconeError("MongoDB counter update returned no document")
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

    async def delete_space(self, space: str, erased_at: str) -> DeletedSpace:
        chunk_ids = tuple(sorted([doc["_id"] async for doc in self.chunks.find({"space": space}, {"_id": 1})]))
        gone = DeletedSpace(
            chunk_ids=chunk_ids,
            episodes=await self.episodes.count_documents({"space": space}),
            facts=await self.facts.count_documents({"space": space}),
            links=await self._fact_links.count_documents({"space": space}),
            tombstones=await self.tombstones.count_documents({"space": space}),
        )
        for collection in (self.chunks, self.episodes, self._fact_links, self._affirmations, self.facts, self.tombstones,
                           self.inflight_marks):
            await collection.delete_many({"space": space})
        await self.revisions.delete_one({"_id": space})
        await self.db["erased_spaces"].replace_one({"_id": space}, {"_id": space, "erased_at": erased_at}, upsert=True)
        return gone

    async def space_deleted(self, space: str) -> Optional[str]:
        doc = await self.db["erased_spaces"].find_one({"_id": space})
        return doc["erased_at"] if doc else None

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

    async def page_chunks(self, space: str, episode_id: int, *, start_ordinal: int, limit: int) -> list[Chunk]:
        from ..core.chunk_window import validate_chunk_window
        validate_chunk_window(episode_id, start_ordinal, limit)
        cursor = self.chunks.find({'space':space, 'episode_id':episode_id, 'ordinal':{'$gte':start_ordinal}})
        return [_chunk(doc) async for doc in cursor.sort([('ordinal', 1), ('_id', 1)]).limit(limit)]

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
        if (filter.tags or filter.where or filter.conditions is not None
                or any(value is not None for value in (filter.kind, filter.source_prefix, filter.since, filter.until))):
            pipeline.append(
                {"$lookup": {"from": "episodes", "localField": "episode_id", "foreignField": "_id", "as": "episode"}}
            )
            pipeline.append({"$unwind": "$episode"})
            scope: dict[str, object] = {"episode.space": space}
            if filter.kind is not None:
                scope["episode.kind"] = filter.kind
            if filter.source_prefix is not None:
                # Mongo regex is case-sensitive; escape every metacharacter.
                # Even an empty prefix requires an actual string source.
                prefix = re.escape(filter.source_prefix).replace("\x00", r"\x00")
                scope["episode.source"] = {"$type": "string", "$regex": "^" + prefix}
            times: dict[str, str] = {}
            if filter.since is not None:
                times["$gte"] = filter.since
            if filter.until is not None:
                times["$lte"] = filter.until
            if times:
                scope["episode.created_at"] = times
            if filter.tags:
                scope["episode.tags"] = {"$all": list(filter.tags)}
            for key, value in filter.where.items():
                scope[f"episode.metadata.{key}"] = value
            pipeline.append({"$match": scope})
        pipeline.append({"$sort": {"score": -1, "_id": 1}})
        projection = {"score": 1}
        if filter.conditions is None:
            pipeline.append({"$limit": limit})
        else:
            # Settle parsed numeric/negation semantics exactly before limiting.
            # Stream only ranked IDs, scores and metadata, not all chunk text.
            projection["episode.metadata"] = 1
        pipeline.append({"$project": projection})
        cursor = await self.chunks.aggregate(pipeline, batchSize=100)
        found: list[tuple[int, float]] = []
        try:
            async for doc in cursor:
                if filter.conditions is not None and not filter.conditions.matches(doc["episode"].get("metadata", {})):
                    continue
                found.append((doc["_id"], float(doc["score"])))
                if len(found) >= limit:
                    break
        finally:
            await cursor.close()
        return found

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
        pipeline: list[dict[str, object]] = [
            {"$match": {"space": space}},
            {"$group": {"_id": None, "n": {"$sum": 1}, "b": {"$sum": {"$binarySize": "$content"}}}},
        ]
        async for doc in await self.episodes.aggregate(pipeline):
            counts.episodes, counts.bytes = doc["n"], doc["b"]
        counts.chunks = await self.chunks.count_documents({"space": space})
        tag_pipeline: list[dict[str, object]] = [
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

    async def page_facts(self, space: str, before_id: int | None, limit: int) -> list[Fact]:
        """Newest first below the cursor, on the (space, _id) index."""
        from ..core.graph_read import ledger_page_limit
        cap = ledger_page_limit(limit)
        if not cap:
            return []  # MongoDB limit(0) would remove the bound.
        query: dict[str, object] = {'space': space}
        if before_id is not None:
            query['_id'] = {'$lt': before_id}
        cursor = self.facts.find(query).sort('_id', -1).limit(cap)
        return [_fact(doc) async for doc in cursor]

    async def facts_for_graph(self, space: str, source_episode_id: int | None, limit: int) -> list[Fact]:
        from ..core.graph_read import graph_fact_read_limit
        cap = graph_fact_read_limit(source_episode_id, limit)
        if not cap:
            return []  # MongoDB limit(0) would remove the bound.
        query: dict[str, object] = {'space':space}
        if source_episode_id is not None:
            query['source_episode_id'] = source_episode_id
        cursor = self.facts.find(query).sort('_id', 1).limit(cap)
        return [_fact(doc) async for doc in cursor]

    async def facts_for(self, space: str, subject: str, predicate: str) -> list[Fact]:
        cursor = self.facts.find({"space": space, "subject": subject, "predicate": predicate}).sort("_id", 1)
        return [_fact(doc) async for doc in cursor]

    async def facts_by_subject(self, space: str, subject: str, limit: int) -> list[Fact]:
        cap = max(0, min(limit, 129))
        if not cap:
            return []  # MongoDB's limit(0) means unbounded.
        cursor = self.facts.find({"space": space, "subject": subject}).sort("_id", 1).limit(cap)
        return [_fact(doc) async for doc in cursor]

    async def record_tombstone(self, new: NewTombstone) -> Tombstone:
        from pymongo.errors import DuplicateKeyError

        try:
            await self.tombstones.insert_one(dict(new.__dict__))
        except DuplicateKeyError:
            pass
        doc = await self.tombstones.find_one({"space": new.space, "episode_id": new.episode_id})
        if doc is None:
            raise SconeError("MongoDB tombstone disappeared during insert")
        return _tombstone(doc)

    async def tombstone(self, space: str, episode_id: int) -> Optional[Tombstone]:
        doc = await self.tombstones.find_one({"space": space, "episode_id": episode_id})
        return _tombstone(doc) if doc else None

    async def tombstone_by_hash(self, space: str, content_hash: str) -> Optional[Tombstone]:
        doc = await self.tombstones.find_one({"space": space, "content_hash": content_hash}, sort=[("episode_id", -1)])
        return _tombstone(doc) if doc else None

    async def list_tombstones(self, space: str) -> list[Tombstone]:
        return [_tombstone(doc) async for doc in self.tombstones.find({"space": space}).sort("episode_id", 1)]

    async def add_affirmation(self, new: NewAffirmation) -> Affirmation:
        from pymongo.errors import DuplicateKeyError

        key = {"space": new.space, "fact_id": new.fact_id, "valid_from": new.valid_from}
        existing = await self._affirmations.find_one(key)
        if existing is None:
            doc = {"_id": await self._next_id("fact_affirmations"), **new.__dict__, "links": stored_links(new.links)}
            try:
                await self._affirmations.insert_one(doc)
                return _affirmation(doc)
            except DuplicateKeyError:
                existing = await self._affirmations.find_one(key)
        if existing is None:
            raise SconeError("MongoDB affirmation disappeared during insert")
        return _affirmation(existing)

    async def affirmations(self, space: str, fact_id: int) -> list[Affirmation]:
        cursor = self._affirmations.find({"space": space, "fact_id": fact_id}).sort([("valid_from", 1), ("_id", 1)])
        return [_affirmation(doc) async for doc in cursor]

    async def drop_affirmations(self, space: str, affirmation_ids: Sequence[int]) -> None:
        if affirmation_ids:
            await self._affirmations.delete_many({"space": space, "_id": {"$in": list(affirmation_ids)}})

    async def space_affirmations(self, space: str) -> list[Affirmation]:
        return [_affirmation(doc) async for doc in self._affirmations.find({"space": space}).sort("_id", 1)]

    async def insert_fact_link(self, new: NewFactLink) -> FactLink:
        from pymongo.errors import DuplicateKeyError

        key = {"space": new.space, "from_fact": new.from_fact, "to_fact": new.to_fact, "kind": new.kind}
        existing = await self._fact_links.find_one(key)
        if existing is None:
            doc = {"_id": await self._next_id("fact_links"), **new.__dict__}
            try:
                await self._fact_links.insert_one(doc)
                return _fact_link(doc)
            except DuplicateKeyError:
                existing = await self._fact_links.find_one(key)
        if existing is None:
            raise SconeError("MongoDB fact link disappeared during insert")
        return _fact_link(existing)

    async def fact_links(self, space: str, fact_id: int) -> list[FactLink]:
        cursor = self._fact_links.find({"space": space, "$or": [{"from_fact": fact_id}, {"to_fact": fact_id}]}).sort("_id", 1)
        return [_fact_link(doc) async for doc in cursor]

    async def fact_links_from(self, space: str, fact_id: int, limit: int) -> list[FactLink]:
        """Read both incident directions, retaining stored direction and ID order."""
        cap = max(0, min(limit, 129))
        if not cap:
            return []
        cursor = self._fact_links.find({"space": space,
            "$or": [{"from_fact": fact_id}, {"to_fact": fact_id}]}).sort("_id", 1).limit(cap)
        return [_fact_link(doc) async for doc in cursor]

    async def get_fact_link(self, space: str, link_id: int) -> FactLink | None:
        doc = await self._fact_links.find_one({"space": space, "_id": link_id})
        return _fact_link(doc) if doc is not None else None

    async def fact_links_between(self, space: str, fact_ids: Sequence[int], limit: int) -> list[FactLink]:
        """Bounded induced graph over returned facts; never expand to neighbors."""
        wanted = list(dict.fromkeys(fact_ids[:16]))
        cap = max(0, min(limit, 49))
        if not wanted or not cap:
            return []
        cursor = self._fact_links.find({"space": space, "from_fact": {"$in": wanted},
            "to_fact": {"$in": wanted}}).sort("_id", 1).limit(cap)
        return [_fact_link(doc) async for doc in cursor]

    async def bump_revision(self, space: str) -> int:
        doc = await self.revisions.find_one_and_update(
            {"_id": space}, {"$inc": {"revision": 1}}, upsert=True, return_document=True
        )
        if doc is None:
            raise SconeError("MongoDB revision update returned no document")
        return int(doc["revision"])

    async def revision(self, space: str) -> int:
        doc = await self.revisions.find_one({"_id": space})
        return int(doc["revision"]) if doc else 0


def _affirmation(doc: Mapping) -> Affirmation:
    return Affirmation(affirmation_id=doc["_id"], space=doc["space"], fact_id=doc["fact_id"],
                       valid_from=doc["valid_from"], recorded_at=doc["recorded_at"],
                       confidence=doc.get("confidence", 1.0), source_episode_id=doc.get("source_episode_id"),
                       origin=doc.get("origin", "stated"), quote=doc.get("quote"), links=read_links(doc.get("links", [])))
