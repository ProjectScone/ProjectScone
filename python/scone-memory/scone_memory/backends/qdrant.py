"""Vectors in Qdrant. One collection, cosine distance, chunk id as the
point id, and the scope fields (space, created_at, tags, metadata) in the
payload so every filter the engine applies runs server-side.

``QdrantVectorIndex(":memory:")`` uses the client's local mode, which
runs the same code path without a server; the contract tests use it.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from ..ports import VectorPoint
from ..timeutil import epoch_seconds


class QdrantVectorIndex:
    name = "qdrant"

    def __init__(
        self,
        url: str = ":memory:",
        collection: str = "scone_chunks",
        api_key: Optional[str] = None,
        client=None,
    ) -> None:
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as e:  # pragma: no cover
            raise ImportError("QdrantVectorIndex needs qdrant-client: pip install 'scone-memory[qdrant]'") from e
        if client is not None:
            self.client = client
        elif url == ":memory:":
            self.client = AsyncQdrantClient(":memory:")
        else:
            self.client = AsyncQdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.dim: Optional[int] = None

    async def ensure(self, dim: int) -> None:
        from qdrant_client import models

        if await self.client.collection_exists(self.collection):
            info = await self.client.get_collection(self.collection)
            existing = info.config.params.vectors.size  # type: ignore[union-attr]
            if existing != dim:
                raise ValueError(f"collection {self.collection!r} holds {existing}-d vectors, embedder makes {dim}-d")
        else:
            await self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            for field, schema in (
                ("space", models.PayloadSchemaType.KEYWORD),
                ("created_ts", models.PayloadSchemaType.FLOAT),
                ("tags", models.PayloadSchemaType.KEYWORD),
            ):
                await self.client.create_payload_index(self.collection, field, schema)
        self.dim = dim

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        from qdrant_client import models

        if not points:
            return
        await self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=p.chunk_id,
                    vector=list(p.vector),
                    payload={
                        "space": p.space,
                        "episode_id": p.episode_id,
                        "created_at": p.created_at,
                        "created_ts": epoch_seconds(p.created_at),
                        "tags": list(p.tags),
                        "meta": dict(p.metadata),
                    },
                )
                for p in points
            ],
        )

    async def search(
        self,
        space: str,
        vector: Sequence[float],
        limit: int,
        as_of: Optional[str] = None,
        tags: tuple[str, ...] = (),
        where: Mapping[str, str] | None = None,
    ) -> list[tuple[int, float]]:
        from qdrant_client import models

        must: list = [models.FieldCondition(key="space", match=models.MatchValue(value=space))]
        if as_of:
            must.append(models.FieldCondition(key="created_ts", range=models.Range(lte=epoch_seconds(as_of))))
        for tag in tags:
            must.append(models.FieldCondition(key="tags", match=models.MatchValue(value=tag)))
        for key, value in (where or {}).items():
            must.append(models.FieldCondition(key=f"meta.{key}", match=models.MatchValue(value=value)))
        response = await self.client.query_points(
            self.collection,
            query=list(vector),
            query_filter=models.Filter(must=must),
            limit=limit,
            with_payload=False,
        )
        ranked = [(int(p.id), float(p.score)) for p in response.points]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        from qdrant_client import models

        if not chunk_ids:
            return
        await self.client.delete(
            self.collection, points_selector=models.PointIdsList(points=[int(c) for c in chunk_ids])
        )

    async def drop(self) -> None:
        await self.client.delete_collection(self.collection)

    async def close(self) -> None:
        await self.client.close()
