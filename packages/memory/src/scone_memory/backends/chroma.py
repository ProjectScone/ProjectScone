"""Vectors in Chroma. One collection with cosine space, the chunk id as
the record id, and the scope fields in the record metadata so every
filter runs inside Chroma: ``space`` and ``created_ts`` as scalars,
``tags`` as a string array (omitted when empty; Chroma refuses empty
arrays) matched with ``$contains``, and each metadata pair as
``meta_<key>``.

``ChromaVectorIndex()`` is an in-process ephemeral index;
``ChromaVectorIndex(path=...)`` persists to a directory;
``ChromaVectorIndex(url=...)`` talks to a server. Chroma's embedded
client is synchronous, so calls run in a worker thread.
"""

from __future__ import annotations

import asyncio
from typing import Mapping, Optional, Sequence

from ..core.ports import VectorPoint
from ..core.timeutil import epoch_seconds
from .validation import validate_vector


class ChromaVectorIndex:
    name = "chroma"

    def __init__(
        self,
        path: Optional[str] = None,
        url: Optional[str] = None,
        collection: str = "scone_chunks",
        client=None,
    ) -> None:
        try:
            import chromadb
        except ImportError as e:  # pragma: no cover
            raise ImportError("ChromaVectorIndex needs chromadb: pip install 'scone-memory[chroma]'") from e
        if client is not None:
            self.client = client
        elif url:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            self.client = chromadb.HttpClient(host=parsed.hostname or "localhost", port=parsed.port or 8000, ssl=parsed.scheme == "https")
        elif path:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.EphemeralClient()
        self.collection_name = collection
        self.collection = None
        self.dim: Optional[int] = None

    async def ensure(self, dim: int) -> None:
        def _ensure():
            # Chroma fixes a collection's width only once data arrives; the
            # width is recorded in the collection metadata at creation so an
            # empty or reopened collection still refuses another embedder.
            col = self.client.get_or_create_collection(
                self.collection_name, configuration={"hnsw": {"space": "cosine"}}, metadata={"scone_dim": dim}
            )
            stored = (col.metadata or {}).get("scone_dim")
            if stored is not None and int(stored) != dim:
                raise ValueError(f"collection {self.collection_name!r} holds {stored}-d vectors, embedder makes {dim}-d")
            return col

        self.collection = await asyncio.to_thread(_ensure)
        self.dim = dim

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        for point in points:
            validate_vector(point.vector, self.dim)
        metadatas = []
        for p in points:
            meta: dict = {"space": p.space, "episode_id": p.episode_id, "created_at": p.created_at, "created_ts": epoch_seconds(p.created_at)}
            if p.tags:
                meta["tags"] = list(p.tags)
            for key, value in p.metadata.items():
                meta[f"meta_{key}"] = value
            metadatas.append(meta)
        await asyncio.to_thread(
            self.collection.upsert,
            ids=[str(p.chunk_id) for p in points],
            embeddings=[list(map(float, p.vector)) for p in points],
            metadatas=metadatas,
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
        validate_vector(vector, self.dim)
        clauses: list[dict] = [{"space": {"$eq": space}}]
        if as_of:
            clauses.append({"created_ts": {"$lte": epoch_seconds(as_of)}})
        for tag in tags:
            clauses.append({"tags": {"$contains": tag}})
        for key, value in (where or {}).items():
            clauses.append({f"meta_{key}": {"$eq": value}})
        condition = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        if await asyncio.to_thread(self.collection.count) == 0:
            return []  # Chroma refuses to query an empty collection
        response = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[list(map(float, vector))],
            n_results=limit,
            where=condition,
            include=["distances"],
        )
        ids, distances = response["ids"][0], response["distances"][0]
        # Chroma reports cosine distance, 1 - cos; the port speaks similarity.
        ranked = [(int(i), 1.0 - float(d)) for i, d in zip(ids, distances)]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        if not chunk_ids:
            return
        await asyncio.to_thread(self.collection.delete, ids=[str(int(c)) for c in chunk_ids])

    async def drop(self) -> None:
        await asyncio.to_thread(self.client.delete_collection, self.collection_name)

    async def close(self) -> None:
        return None
