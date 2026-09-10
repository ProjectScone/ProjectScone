"""Vectors in Redis with the RediSearch module (redis-stack). One hash per
chunk under ``<prefix>:<chunk_id>``, one index over the prefix with an
HNSW cosine vector field, and the scope fields as TAG and NUMERIC fields
so every filter is part of the KNN query: ``space`` and ``tags`` as tags,
metadata pairs as ``key=value`` tags, ``created_ts`` numeric. Tag values
are joined with the unit separator (U+001F), which no tag or metadata
value contains, and every non-alphanumeric character in a query value is
backslash-escaped, so a value shaped like query syntax is data.

``RedisVectorIndex(url)`` talks to a server; needs
``pip install 'scone-memory[redis]'``. Redis reports cosine distance;
the port speaks similarity, so 1 - d.
"""

from __future__ import annotations

import re
import struct
from typing import Mapping, Optional, Sequence

from ..core.ports import VectorPoint
from ..core.timeutil import epoch_seconds
from .validation import validate_vector

SEP = "\x1f"


def escape(value: str) -> str:
    """A RediSearch tag literal: every non-word character escaped."""
    return re.sub(r"([^A-Za-z0-9_])", r"\\\1", value)


def pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


class RedisVectorIndex:
    name = "redis"

    def __init__(self, url: str = "redis://localhost:6379", prefix: str = "scone_chunks", client=None) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as e:  # pragma: no cover
            raise ImportError("RedisVectorIndex needs redis: pip install 'scone-memory[redis]'") from e
        self.client = client or aioredis.from_url(url)
        self.prefix = prefix
        self.index = f"{prefix}_idx"
        self.dim: Optional[int] = None

    def _key(self, chunk_id: int) -> str:
        return f"{self.prefix}:{int(chunk_id)}"

    async def ensure(self, dim: int) -> None:
        from redis.commands.search.field import NumericField, TagField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType
        from redis.exceptions import ResponseError

        try:
            info = await self.client.ft(self.index).info()
        except ResponseError:
            info = None
        if info is not None:
            existing = _vector_dim(info)
            if existing is not None and existing != dim:
                raise ValueError(f"index {self.index!r} holds {existing}-d vectors, embedder makes {dim}-d")
        else:
            await self.client.ft(self.index).create_index(
                [
                    TagField("space"),
                    NumericField("created_ts"),
                    TagField("tags", separator=SEP),
                    TagField("meta", separator=SEP),
                    VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"}),
                ],
                definition=IndexDefinition(prefix=[f"{self.prefix}:"], index_type=IndexType.HASH),
            )
        self.dim = dim

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        for point in points:
            validate_vector(point.vector, self.dim)
        async with self.client.pipeline(transaction=False) as pipe:
            for p in points:
                pipe.hset(
                    self._key(p.chunk_id),
                    mapping={
                        "chunk_id": p.chunk_id,
                        "space": p.space,
                        "episode_id": p.episode_id,
                        "created_at": p.created_at,
                        "created_ts": epoch_seconds(p.created_at),
                        "tags": SEP.join(p.tags),
                        "meta": SEP.join(f"{k}={v}" for k, v in p.metadata.items()),
                        "embedding": pack(list(map(float, p.vector))),
                    },
                )
            await pipe.execute()

    async def search(
        self,
        space: str,
        vector: Sequence[float],
        limit: int,
        as_of: Optional[str] = None,
        tags: tuple[str, ...] = (),
        where: Mapping[str, str] | None = None,
    ) -> list[tuple[int, float]]:
        from redis.commands.search.query import Query

        validate_vector(vector, self.dim)
        clauses = [f"@space:{{{escape(space)}}}"]
        if as_of:
            clauses.append(f"@created_ts:[-inf {epoch_seconds(as_of)!r}]")
        for tag in tags:
            clauses.append(f"@tags:{{{escape(tag)}}}")
        for key, value in (where or {}).items():
            clauses.append(f"@meta:{{{escape(f'{key}={value}')}}}")
        query = (
            Query(f"({' '.join(clauses)})=>[KNN {int(limit)} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("chunk_id", "score")
            .paging(0, int(limit))
            .dialect(2)
        )
        result = await self.client.ft(self.index).search(query, query_params={"vec": pack(list(map(float, vector)))})
        ranked = [(int(doc.chunk_id), 1.0 - float(doc.score)) for doc in result.docs]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        if not chunk_ids:
            return
        await self.client.delete(*(self._key(c) for c in chunk_ids))

    async def drop(self) -> None:
        from redis.exceptions import ResponseError

        try:
            await self.client.ft(self.index).dropindex(delete_documents=True)
        except ResponseError:
            pass

    async def close(self) -> None:
        await self.client.aclose()


def _vector_dim(info: Mapping) -> Optional[int]:
    """The DIM of the embedding field from FT.INFO, whose attribute list is
    a nested sequence of name/value pairs."""
    for attr in info.get("attributes", []):
        items = [x.decode() if isinstance(x, bytes) else x for x in attr]
        if "embedding" in items and "DIM" in items:
            return int(items[items.index("DIM") + 1])
        if "embedding" in items and "dim" in items:
            return int(items[items.index("dim") + 1])
    return None
