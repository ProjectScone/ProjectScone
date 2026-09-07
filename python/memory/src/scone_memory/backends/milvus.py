"""Vectors in Milvus: one collection keyed by chunk id with a COSINE
index, and the scope fields as scalar columns so every filter is a Milvus
filter expression evaluated with the search: ``space`` (VARCHAR),
``created_ts`` (DOUBLE), ``tags`` (an ARRAY of VARCHAR matched with
``array_contains``) and ``meta`` (a JSON field matched by key). String
literals in expressions are JSON-encoded (double quotes, backslash
escapes), so a tag or value shaped like an expression is data.

``MilvusVectorIndex("./milvus.db")`` runs Milvus Lite in-process (a
local path ending in .db); ``MilvusVectorIndex("http://host:19530")``
talks to a server. Milvus reports COSINE as similarity already. The client is
synchronous, so calls run in a worker thread. Needs
``pip install 'scone-memory[milvus]'``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Mapping, Optional, Sequence

from ..core.ports import VectorPoint
from ..core.timeutil import epoch_seconds
from .validation import validate_vector


def literal(value: str) -> str:
    """A Milvus expression string literal."""
    return json.dumps(value, ensure_ascii=False)


class MilvusVectorIndex:
    name = "milvus"

    def __init__(self, uri: str, collection: str = "scone_chunks", token: Optional[str] = None, client=None) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as e:  # pragma: no cover
            raise ImportError("MilvusVectorIndex needs pymilvus: pip install 'scone-memory[milvus]'") from e
        self.client = client or (MilvusClient(uri, token=token) if token else MilvusClient(uri))
        self.collection = collection
        self.dim: Optional[int] = None

    async def ensure(self, dim: int) -> None:
        from pymilvus import DataType

        def _ensure():
            if self.client.has_collection(self.collection):
                fields = self.client.describe_collection(self.collection)["fields"]
                existing = int(next(f for f in fields if f["name"] == "embedding")["params"]["dim"])
                if existing != dim:
                    raise ValueError(f"collection {self.collection!r} holds {existing}-d vectors, embedder makes {dim}-d")
                return
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("chunk_id", DataType.INT64, is_primary=True)
            schema.add_field("space", DataType.VARCHAR, max_length=64)
            schema.add_field("episode_id", DataType.INT64)
            schema.add_field("created_at", DataType.VARCHAR, max_length=40)
            schema.add_field("created_ts", DataType.DOUBLE)
            schema.add_field("tags", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=256)
            schema.add_field("meta", DataType.JSON)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
            index = self.client.prepare_index_params()
            index.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
            self.client.create_collection(self.collection, schema=schema, index_params=index)

        await asyncio.to_thread(_ensure)
        self.dim = dim

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        for point in points:
            validate_vector(point.vector, self.dim)
        rows = [
            {
                "chunk_id": p.chunk_id, "space": p.space, "episode_id": p.episode_id, "created_at": p.created_at,
                "created_ts": epoch_seconds(p.created_at), "tags": list(p.tags), "meta": dict(p.metadata),
                "embedding": list(map(float, p.vector)),
            }
            for p in points
        ]
        await asyncio.to_thread(self.client.upsert, self.collection, rows)

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
        clauses = [f"space == {literal(space)}"]
        if as_of:
            clauses.append(f"created_ts <= {epoch_seconds(as_of)!r}")
        for tag in tags:
            clauses.append(f"array_contains(tags, {literal(tag)})")
        for key, value in (where or {}).items():
            clauses.append(f"meta[{literal(key)}] == {literal(value)}")

        def _search():
            return self.client.search(
                self.collection, data=[list(map(float, vector))], limit=int(limit), filter=" and ".join(clauses),
                output_fields=["chunk_id"], search_params={"metric_type": "COSINE"},
            )

        [hits] = await asyncio.to_thread(_search)
        ranked = [(int(h["entity"]["chunk_id"]), float(h["distance"])) for h in hits]  # COSINE: distance is the similarity
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        if not chunk_ids:
            return
        await asyncio.to_thread(self.client.delete, self.collection, ids=[int(c) for c in chunk_ids])

    async def drop(self) -> None:
        def _drop():
            if self.client.has_collection(self.collection):
                self.client.drop_collection(self.collection)

        await asyncio.to_thread(_drop)

    async def close(self) -> None:
        await asyncio.to_thread(self.client.close)
