"""Vectors are only compared with vectors from the same embedder.

A vector index used to record only the width of its vectors. Reopening a
store under a different embedder of the same width, or under the same
embedder with different token rules, compared new query vectors with old
stored ones and ranked noise: a store written by the ASCII-only hash
embedder and reopened under the Unicode tokenizer scored its own note 0.0.
Indexes that can record who wrote their vectors now do, and the engine
refuses to compare vectors it cannot vouch for.
"""
from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from pathlib import Path
from typing import Sequence

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, InvalidInput, MemoryEngine
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex


class AsciiHash:
    """The hash embedder as it was before the Unicode tokenizer."""

    id = "hash-256"
    dim = 256

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.casefold()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                vec[int.from_bytes(digest[:4], "little") % self.dim] += 1.0 if digest[4] & 1 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class Model:
    """A deterministic stand-in for a model embedder: not cheap to rebuild."""

    dim = 256

    def __init__(self, name: str) -> None:
        self.id = name
        self._inner = HashEmbedder(256)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._inner.embed(texts)


async def open_sqlite(path: Path, embedder, **options) -> MemoryEngine:
    return await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), embedder, **options).open()


def forget_writer(path: Path) -> None:
    """Make the store look like one written before writers were recorded."""
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM vector_meta WHERE key='embedder'")
    conn.commit()
    conn.close()


async def self_score(engine: MemoryEngine, space: str, text: str) -> float:
    [vector] = await engine.embedder.embed([text])
    hits = await engine.vectors.search(space, vector, limit=1)
    return hits[0][1] if hits else 0.0


async def test_hash_vectors_from_older_token_rules_are_rebuilt_on_open(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    old = await open_sqlite(path, AsciiHash())
    await old.remember("review", "Café Zürich")
    await old.close()
    forget_writer(path)
    upgraded = await open_sqlite(path, HashEmbedder())
    try:
        assert await self_score(upgraded, "review", "Café Zürich") == pytest.approx(1.0)
        assert upgraded.vector_identity.state == "rebuilt"
        result = await upgraded.recall("review", "Zürich")
        assert not any(reason.startswith("vectors:") for reason in result.degraded)
    finally:
        await upgraded.close()


async def test_a_recorded_writer_that_differs_is_rebuilt_when_rebuilding_is_cheap(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    old = await open_sqlite(path, AsciiHash())
    await old.remember("review", "Café Zürich")
    await old.close()
    upgraded = await open_sqlite(path, HashEmbedder())
    try:
        assert upgraded.vector_identity.state == "rebuilt"
        assert await self_score(upgraded, "review", "Café Zürich") == pytest.approx(1.0)
    finally:
        await upgraded.close()
    again = await open_sqlite(path, HashEmbedder())
    try:
        assert again.vector_identity.state == "verified"
    finally:
        await again.close()


async def test_another_model_disables_the_vector_lane_instead_of_mixing(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    await first.remember("space", "The Lisbon office opens in May")
    await first.close()
    second = await open_sqlite(path, Model("model-b"))
    try:
        assert second.vector_identity.state == "mismatch"
        result = await second.recall("space", "Lisbon office")
        assert any(reason.startswith("vectors:") and "embedder" in reason for reason in result.degraded)
        assert [item.text for item in result.items] == ["The Lisbon office opens in May"]
    finally:
        await second.close()


async def test_rebuilding_restores_the_vector_lane_and_records_the_new_writer(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    await first.remember("space", "The Lisbon office opens in May")
    await first.remember("other", "Porto has a small team")
    await first.close()
    second = await open_sqlite(path, Model("model-b"))
    try:
        report = await second.reembed_vectors()
        assert report.spaces == ("other", "space") and report.chunks == 2
        assert second.vector_identity.state == "rebuilt"
        result = await second.recall("space", "Lisbon office")
        assert not any(reason.startswith("vectors:") for reason in result.degraded)
    finally:
        await second.close()
    third = await open_sqlite(path, Model("model-b"))
    try:
        assert third.vector_identity.state == "verified"
    finally:
        await third.close()


async def test_contextual_embeddings_are_part_of_the_writer(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    await first.remember("space", "The Lisbon office opens in May")
    await first.close()
    second = await open_sqlite(path, Model("model-a"), contextual_embeddings=True)
    try:
        assert second.vector_identity.state == "mismatch"
    finally:
        await second.close()


async def test_unrecorded_vectors_from_a_model_wait_for_an_explicit_decision(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    await first.remember("space", "The Lisbon office opens in May")
    await first.close()
    forget_writer(path)
    second = await open_sqlite(path, Model("model-a"))
    try:
        assert second.vector_identity.state == "unknown"
        assert any(reason.startswith("vectors:") for reason in (await second.recall("space", "Lisbon")).degraded)
        await second.adopt_vector_identity()
        assert second.vector_identity.state == "declared"
        assert not any(reason.startswith("vectors:") for reason in (await second.recall("space", "Lisbon")).degraded)
    finally:
        await second.close()
    # The declaration stays a declaration: reopening does not promote it.
    third = await open_sqlite(path, Model("model-a"))
    try:
        assert third.vector_identity.state == "declared"
    finally:
        await third.close()


async def test_a_recorded_different_writer_cannot_be_declared_away(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    await first.remember("space", "The Lisbon office opens in May")
    await first.close()
    second = await open_sqlite(path, Model("model-b"))
    try:
        with pytest.raises(InvalidInput, match="model-a"):
            await second.adopt_vector_identity()
        assert second.vector_identity.state == "mismatch"
    finally:
        await second.close()


async def test_a_fresh_store_records_its_writer(tmp_path: Path) -> None:
    engine = await open_sqlite(tmp_path / "memory.db", Model("model-a"))
    try:
        assert engine.vector_identity.state == "verified"
        await engine.remember("space", "The Lisbon office opens in May")
    finally:
        await engine.close()
    again = await open_sqlite(tmp_path / "memory.db", Model("model-a"))
    try:
        assert again.vector_identity.state == "verified"
    finally:
        await again.close()


async def test_semantic_duplicate_search_refuses_vectors_it_cannot_vouch_for(tmp_path: Path) -> None:
    from scone_memory.deduplication.memory import MemoryCandidateProvider
    from scone_memory.deduplication.types import DocumentRevision, PassageEmbedding
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    await first.remember("space", "The Lisbon office opens in May")
    await first.close()
    second = await open_sqlite(path, Model("model-b"))
    try:
        [vector] = await second.embedder.embed(["The Lisbon office opens in May"])
        document = DocumentRevision("space", "incoming", "1", "The Lisbon office opens in May")
        with pytest.raises(ValueError, match="embedder"):
            await MemoryCandidateProvider(second).candidates(
                document, query_embeddings=(PassageEmbedding(0, 1, tuple(vector)),), embedder_id=second.embedder.id, limit=4)
    finally:
        await second.close()


async def test_an_index_that_cannot_record_its_writer_is_reported_unverifiable() -> None:
    class Plain(InMemoryVectorIndex):
        written_by = None  # hides the recording methods
    engine = await MemoryEngine(InMemoryDocumentStore(), Plain(), HashEmbedder()).open()
    assert engine.vector_identity.state == "unverifiable"


async def test_a_rebuild_refuses_an_embedder_that_returns_the_wrong_number_of_vectors(tmp_path: Path) -> None:
    class Short(Model):
        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return (await super().embed(texts))[:-1]
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    await first.remember("space", "The Lisbon office opens in May")
    await first.close()
    second = await open_sqlite(path, Short("model-b"))
    try:
        with pytest.raises(ValueError, match="1 texts"):
            await second.reembed_vectors()
        assert second.vector_identity.state == "mismatch"
    finally:
        await second.close()
    third = await open_sqlite(path, Model("model-b"))
    try:
        assert third.vector_identity.state == "mismatch"
    finally:
        await third.close()


async def test_the_command_line_reports_and_rebuilds_the_vector_writer(tmp_path: Path) -> None:
    import io
    import json
    from scone_memory.runtime.cli import build_parser, run
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    await first.remember("space", "The Lisbon office opens in May")
    await first.close()
    second = await open_sqlite(path, Model("model-b"))
    try:
        out = io.StringIO()
        await run(build_parser().parse_args(["vectors", "--json"]), second, io.StringIO(""), out)
        shown = json.loads(out.getvalue())
        assert shown["state"] == "mismatch" and shown["recorded"] == "model-a;contextual=0"
        assert "reembed" in shown["blocked"]
        out = io.StringIO()
        await run(build_parser().parse_args(["vectors", "--reembed", "--json"]), second, io.StringIO(""), out)
        rebuilt = json.loads(out.getvalue())
        assert rebuilt["state"] == "rebuilt" and rebuilt["chunks"] == 1 and rebuilt["blocked"] is None
    finally:
        await second.close()


async def test_a_rebuild_drops_vectors_whose_chunk_is_gone(tmp_path: Path) -> None:
    from scone_memory.core.ports import VectorPoint
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    added = await first.remember("space", "The Lisbon office opens in May")
    [vector] = await first.embedder.embed(["nothing stores this passage"])
    await first.vectors.upsert([VectorPoint(chunk_id=999_999, space="space", episode_id=added.episode_id,
                                            created_at="2025-01-01T00:00:00Z", vector=vector)])
    await first.close()
    second = await open_sqlite(path, Model("model-b"))
    try:
        report = await second.reembed_vectors()
        assert report.orphans_removed == 1
        assert 999_999 not in await second.vectors.ids("space")
    finally:
        await second.close()
