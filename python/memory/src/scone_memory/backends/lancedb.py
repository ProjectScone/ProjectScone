"""Vectors in LanceDB, an embedded columnar store on local disk (or object
storage). One table with the chunk id as the key, cosine distance, and
the scope fields as columns so every filter is a SQL predicate LanceDB
evaluates before the vector search: ``space`` and ``created_ts`` as
scalars, ``tags`` and ``meta`` (as ``key=value`` strings) as string
arrays matched with ``array_has``. String literals in those predicates
are quoted with single quotes doubled, so a tag or value containing a
quote is data, not syntax.

``LanceDBVectorIndex(path)`` opens or creates a database directory. The
client is synchronous, so calls run in a worker thread.
"""

from __future__ import annotations

import asyncio
from typing import Mapping, Optional, Sequence

from ..ports import VectorPoint
from ..timeutil import epoch_seconds
from .validation import validate_vector


def quote(value: str) -> str:
    """A SQL string literal: single quotes doubled, nothing else special."""
    return "'" + value.replace("'", "''") + "'"


class LanceDBVectorIndex:
    name = "lancedb"

    def __init__(self, path: str, table: str = "scone_chunks", connection=None) -> None:
        try:
            import lancedb
        except ImportError as e:  # pragma: no cover
            raise ImportError("LanceDBVectorIndex needs lancedb: pip install 'scone-memory[lancedb]'") from e
        self.db = connection if connection is not None else lancedb.connect(path)
        self.table_name = table
        self.table = None
        self.dim: Optional[int] = None

    async def ensure(self, dim: int) -> None:
        import pyarrow as pa

        def _existing_tables() -> list[str]:
            listed = self.db.list_tables()
            return list(getattr(listed, "tables", listed))  # a response object in 0.38, a plain list before

        def _ensure():
            if self.table_name in _existing_tables():
                table = self.db.open_table(self.table_name)
                existing = table.schema.field("vector").type.list_size
                if existing != dim:
                    raise ValueError(f"table {self.table_name!r} holds {existing}-d vectors, embedder makes {dim}-d")
                return table
            schema = pa.schema([
                pa.field("chunk_id", pa.int64()),
                pa.field("space", pa.string()),
                pa.field("episode_id", pa.int64()),
                pa.field("created_at", pa.string()),
                pa.field("created_ts", pa.float64()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("meta", pa.list_(pa.string())),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ])
            return self.db.create_table(self.table_name, schema=schema)

        self.table = await asyncio.to_thread(_ensure)
        self.dim = dim

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        for point in points:
            validate_vector(point.vector, self.dim)
        rows = [
            {
                "chunk_id": p.chunk_id,
                "space": p.space,
                "episode_id": p.episode_id,
                "created_at": p.created_at,
                "created_ts": epoch_seconds(p.created_at),
                "tags": list(p.tags),
                "meta": [f"{k}={v}" for k, v in p.metadata.items()],
                "vector": list(map(float, p.vector)),
            }
            for p in points
        ]

        def _upsert():
            self.table.merge_insert("chunk_id").when_matched_update_all().when_not_matched_insert_all().execute(rows)

        await asyncio.to_thread(_upsert)

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
        predicates = [f"space = {quote(space)}"]
        if as_of:
            predicates.append(f"created_ts <= {epoch_seconds(as_of)!r}")
        for tag in tags:
            predicates.append(f"array_has(tags, {quote(tag)})")
        for key, value in (where or {}).items():
            predicates.append(f"array_has(meta, {quote(f'{key}={value}')})")

        def _search():
            query = self.table.search(list(map(float, vector))).metric("cosine").where(" AND ".join(predicates), prefilter=True)
            return query.limit(limit).to_list()

        rows = await asyncio.to_thread(_search)
        # LanceDB reports cosine distance, 1 - cos; the port speaks similarity.
        ranked = [(int(r["chunk_id"]), 1.0 - float(r["_distance"])) for r in rows]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        if not chunk_ids:
            return
        ids = ", ".join(str(int(c)) for c in chunk_ids)
        await asyncio.to_thread(self.table.delete, f"chunk_id IN ({ids})")

    async def drop(self) -> None:
        await asyncio.to_thread(self.db.drop_table, self.table_name)

    async def close(self) -> None:
        return None
