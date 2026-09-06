"""Any LangChain VectorStore as a VectorIndex: the long tail of vector
databases (Pinecone, Weaviate, FAISS, PGVector, Azure Search, Vertex,
OpenSearch, and the rest) through one adapter, with what each store can
and cannot promise stated rather than assumed.

Three things vary between stores and the bridge makes each explicit:

- **Vectors.** A VectorStore embeds text itself. The bridge hands the
  store a placeholder text per chunk and an ``Embeddings`` object
  (``bridge.embeddings``) that returns the engine's vector for that
  placeholder, so the store must be constructed with it; a store with
  ``add_embeddings`` gets the vectors directly instead.
- **Filters.** Filter syntax differs per store, so scope filtering
  needs a ``filter_builder(space, as_of_ts, tags, where)`` that returns
  the store's own filter for those conditions. Without one the bridge
  over-fetches (``overfetch`` times the limit, unfiltered) and filters
  on the metadata it wrote. That is exact whenever it returns ``limit``
  matches, or the store ran out of candidates: anything past the window
  scored lower than everything in it. When the window fills with
  out-of-scope candidates and fewer than ``limit`` matches were found,
  the bridge raises rather than return a list that may miss better
  matches; the engine reports the lane as degraded and the other lane
  answers. Give the store a filter_builder to remove the window.
- **Scores.** ``score`` says what the store's number means:
  ``"cosine_similarity"`` (used as is), ``"cosine_distance"`` (1 - d),
  ``"unit_l2_distance"`` (squared or plain L2 between unit vectors, which is
  2 - 2 cos for the squared form; only the squared form is converted, as
  ``1 - d*d/2``), or ``"unknown"``: the store's order is kept for fusion,
  every similarity is NaN, and the engine shows none and judges no
  confidence from it. Naming a score wrongly makes the abstention gate
  wrong; when in doubt, say unknown.

Needs ``pip install 'scone-memory[langchain]'``; the store's own package
on top.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Callable, Mapping, Optional, Sequence

from ..ports import VectorPoint
from ..timeutil import epoch_seconds
from .validation import validate_vector

SCORES = ("cosine_similarity", "cosine_distance", "unit_l2_squared", "unknown")


class WindowExhausted(RuntimeError):
    """The unfiltered window filled with out-of-scope candidates before
    ``limit`` matches were found; the answer might miss better matches."""


class PrecomputedEmbeddings:
    """An ``Embeddings`` that hands back the vectors the bridge registered
    for placeholder texts. Queries never go through it: the bridge
    searches by vector."""

    def __init__(self) -> None:
        self.pending: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return [self.pending[t] for t in texts]
        except KeyError as e:
            raise RuntimeError(f"no vector registered for placeholder {e.args[0]!r}; add through the bridge, not the store") from None

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("the bridge searches by vector; the store must not embed queries")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class LangChainVectorIndex:
    name = "langchain"

    def __init__(
        self,
        store: Any = None,
        *,
        score: str = "unknown",
        filter_builder: Optional[Callable[[str, Optional[float], tuple[str, ...], Mapping[str, str]], Any]] = None,
        overfetch: int = 10,
        max_window: int = 2000,
        embeddings: Optional[PrecomputedEmbeddings] = None,
    ) -> None:
        if score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}, got {score!r}")
        self.embeddings = embeddings or PrecomputedEmbeddings()
        self.store = store
        self.score = score
        self.filter_builder = filter_builder
        self.overfetch = max(1, int(overfetch))
        self.max_window = max(1, int(max_window))
        self.dim: Optional[int] = None

    def bind(self, store: Any) -> "LangChainVectorIndex":
        """Attach the store after construction, for stores that must be
        built with ``self.embeddings`` first."""
        self.store = store
        return self

    async def ensure(self, dim: int) -> None:
        if self.store is None:
            raise ValueError("LangChainVectorIndex needs a store: pass one or call bind(store)")
        if not hasattr(self.store, "similarity_search_with_score_by_vector"):
            raise TypeError(f"{type(self.store).__name__} has no similarity_search_with_score_by_vector; the bridge cannot search it")
        if self.dim is not None and self.dim != dim:
            raise ValueError(f"bridge holds {self.dim}-d vectors, embedder makes {dim}-d")
        self.dim = dim

    @staticmethod
    def metadata(p: VectorPoint) -> dict:
        meta: dict = {"chunk_id": p.chunk_id, "space": p.space, "episode_id": p.episode_id, "created_at": p.created_at,
                      "created_ts": epoch_seconds(p.created_at), "tags": list(p.tags)}
        for key, value in p.metadata.items():
            meta[f"meta_{key}"] = value
        return meta

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        for point in points:
            validate_vector(point.vector, self.dim)
        ids = [str(p.chunk_id) for p in points]
        metadatas = [self.metadata(p) for p in points]
        # An existing id is replaced, not duplicated: delete first, since
        # not every store treats add with a known id as an update.
        await self._delete(ids)
        if hasattr(self.store, "add_embeddings"):
            pairs = [(f"chunk:{p.chunk_id}", list(map(float, p.vector))) for p in points]
            await asyncio.to_thread(self.store.add_embeddings, text_embeddings=pairs, metadatas=metadatas, ids=ids)
            return
        placeholders = [f"chunk:{p.chunk_id}" for p in points]
        for placeholder, p in zip(placeholders, points):
            self.embeddings.pending[placeholder] = list(map(float, p.vector))
        try:
            await self.store.aadd_texts(placeholders, metadatas=metadatas, ids=ids)
        finally:
            for placeholder in placeholders:
                self.embeddings.pending.pop(placeholder, None)

    def _similarity(self, raw: float) -> float:
        if self.score == "cosine_similarity":
            return float(raw)
        if self.score == "cosine_distance":
            return 1.0 - float(raw)
        if self.score == "unit_l2_squared":
            return 1.0 - float(raw) / 2.0
        return math.nan

    @staticmethod
    def _matches(meta: Mapping, space: str, as_of_ts: Optional[float], tags: tuple[str, ...], where: Mapping[str, str]) -> bool:
        if meta.get("space") != space:
            return False
        if as_of_ts is not None and float(meta.get("created_ts", math.inf)) > as_of_ts:
            return False
        have = set(meta.get("tags") or ())
        if any(t not in have for t in tags):
            return False
        return all(meta.get(f"meta_{k}") == v for k, v in where.items())

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
        where = dict(where or {})
        as_of_ts = epoch_seconds(as_of) if as_of else None
        query = list(map(float, vector))
        if self.filter_builder is not None:
            flt = self.filter_builder(space, as_of_ts, tags, where)
            hits = await asyncio.to_thread(self.store.similarity_search_with_score_by_vector, query, int(limit), filter=flt)
            return self._rank(hits)
        window = min(int(limit) * self.overfetch, self.max_window)
        hits = await asyncio.to_thread(self.store.similarity_search_with_score_by_vector, query, window)
        kept = [(doc, raw) for doc, raw in hits if self._matches(doc.metadata, space, as_of_ts, tags, where)]
        if len(kept) < limit and len(hits) >= window:
            raise WindowExhausted(
                f"{len(hits)} candidates scanned, {len(kept)} in scope, {limit} asked: matches past the window may score lower "
                "but exist; give the store a filter_builder"
            )
        return self._rank(kept[: int(limit)])

    def _rank(self, hits: Sequence[tuple[Any, float]]) -> list[tuple[int, float]]:
        ranked = [(int(doc.metadata.get("chunk_id", doc.id)), self._similarity(raw)) for doc, raw in hits]
        if self.score != "unknown":
            ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked  # unknown scores: the store's order stands

    async def _delete(self, ids: Sequence[str]) -> None:
        if ids:
            await self.store.adelete(ids=list(ids))

    async def delete(self, chunk_ids: Sequence[int]) -> None:
        await self._delete([str(int(c)) for c in chunk_ids])

    async def close(self) -> None:
        return None
