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
        # The rebuild marked the index before its first write, so the failed
        # attempt leaves nothing trusted, for either writer.
        assert second.vector_identity.state == "interrupted"
    finally:
        await second.close()
    third = await open_sqlite(path, Model("model-b"))
    try:
        assert third.vector_identity.state == "interrupted"
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


class Rotated(Model):
    """A model whose vectors differ from Model's, so mixing shows in scores."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector[1:] + vector[:1] for vector in await super().embed(texts)]


BLOCKED = {"mismatch", "unknown", "mixed", "interrupted"}


async def test_writing_under_a_mismatch_marks_the_index_mixed(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    await first.remember("space", "The Lisbon office opens in May")
    await first.close()
    second = await open_sqlite(path, Rotated("model-b"))
    try:
        assert second.vector_identity.state == "mismatch"
        await second.remember("space", "The Porto office opens in June")
    finally:
        await second.close()
    back = await open_sqlite(path, Model("model-a"))
    try:
        assert back.vector_identity.state == "mixed"
        assert any(reason.startswith("vectors:") for reason in (await back.recall("space", "office")).degraded)
    finally:
        await back.close()


async def test_an_interrupted_rebuild_is_never_mistaken_for_either_writer(tmp_path: Path) -> None:
    class FailsSecondCall(Rotated):
        calls = 0

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            FailsSecondCall.calls += 1
            if FailsSecondCall.calls == 2:
                raise RuntimeError("embedding provider went away")
            return await super().embed(texts)

    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    await first.remember("space", "The Lisbon office opens in May")
    await first.remember("space", "The Porto office opens in June")
    await first.close()
    second = await open_sqlite(path, FailsSecondCall("model-b"))
    try:
        with pytest.raises(RuntimeError, match="went away"):
            await second.reembed_vectors()
        assert second.vector_identity.state == "interrupted"
    finally:
        await second.close()
    for embedder in (Model("model-a"), Rotated("model-b")):
        reopened = await open_sqlite(path, embedder)
        try:
            assert reopened.vector_identity.state == "interrupted", embedder.id
        finally:
            await reopened.close()
    finisher = await open_sqlite(path, Rotated("model-b"))
    try:
        await finisher.reembed_vectors()
    finally:
        await finisher.close()
    done = await open_sqlite(path, Rotated("model-b"))
    try:
        assert done.vector_identity.state == "verified"
    finally:
        await done.close()


async def test_an_engine_opened_before_another_rebuilt_stops_trusting_its_vectors(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    stale = await open_sqlite(path, Model("model-a"))
    await stale.remember("space", "The Lisbon office opens in May")
    rebuilder = await open_sqlite(path, Rotated("model-b"))
    try:
        await rebuilder.reembed_vectors()
        result = await stale.recall("space", "Lisbon office")
        assert any(reason.startswith("vectors:") for reason in result.degraded)
        assert stale.vector_identity.state == "mismatch"
        await stale.remember("space", "The Porto office opens in June")
    finally:
        await rebuilder.close()
        await stale.close()
    reopened = await open_sqlite(path, Rotated("model-b"))
    try:
        assert reopened.vector_identity.state == "mixed"
    finally:
        await reopened.close()


@pytest.mark.parametrize(("intruder", "finishes"), [(Model("model-a"), False), (Rotated("model-b"), True)])
async def test_a_write_during_a_rebuild_is_judged_by_who_wrote_it(tmp_path: Path, intruder: Model, finishes: bool) -> None:
    from scone_memory.memory.vector_identity import VectorWriterChanged
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    await first.remember("space", "The Lisbon office opens in May")
    await first.remember("space", "The Porto office opens in June")
    await first.close()
    other = await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), intruder).open()

    class WritesMidway(Rotated):
        calls = 0

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            WritesMidway.calls += 1
            if WritesMidway.calls == 2:
                await other.remember("space", "The Faro office opens in July")
            return await super().embed(texts)

    rebuilder = await open_sqlite(path, WritesMidway("model-b"))
    try:
        if finishes:
            await rebuilder.reembed_vectors()
            assert rebuilder.vector_identity.state == "rebuilt"
        else:
            with pytest.raises(VectorWriterChanged):
                await rebuilder.reembed_vectors()
            assert rebuilder.vector_identity.state == "mixed"
    finally:
        await rebuilder.close()
        await other.close()


async def test_semantic_duplicate_search_rereads_the_writer_before_comparing(tmp_path: Path) -> None:
    from scone_memory.deduplication.memory import MemoryCandidateProvider
    from scone_memory.deduplication.types import DocumentRevision, PassageEmbedding
    path = tmp_path / "memory.db"
    stale = await open_sqlite(path, Model("model-a"))
    await stale.remember("space", "The Lisbon office opens in May")
    rebuilder = await open_sqlite(path, Rotated("model-b"))
    try:
        await rebuilder.reembed_vectors()
        [vector] = await stale.embedder.embed(["The Lisbon office opens in May"])
        with pytest.raises(ValueError, match="embedder"):
            await MemoryCandidateProvider(stale).candidates(
                DocumentRevision("space", "incoming", "1", "The Lisbon office opens in May"),
                query_embeddings=(PassageEmbedding(0, 1, tuple(vector)),), embedder_id=stale.embedder.id, limit=4)
    finally:
        await rebuilder.close()
        await stale.close()


async def test_a_foreign_write_between_the_check_and_the_search_is_caught(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    reader = await open_sqlite(path, Model("model-a"))
    await reader.remember("space", "The Lisbon office opens in May")
    intruder = await open_sqlite(path, Rotated("model-b"))

    class WritesWhileEmbeddingTheQuery(Model):
        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            vectors = await super().embed(texts)
            if texts == ["Lisbon office"]:
                await intruder.remember("space", "The Porto office opens in June")
            return vectors

    reader.embedder = WritesWhileEmbeddingTheQuery("model-a")
    try:
        result = await reader.recall("space", "Lisbon office")
        assert any(reason.startswith("vectors:") for reason in result.degraded)
    finally:
        await intruder.close()
        await reader.close()


async def test_a_record_added_during_a_rebuild_keeps_its_vector(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    await first.remember("space", "The Lisbon office opens in May")
    await first.remember("space", "The Porto office opens in June")
    await first.close()
    other = await open_sqlite(path, Rotated("model-b"))
    added: list[int] = []

    class AddsMidway(Rotated):
        calls = 0

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            AddsMidway.calls += 1
            if AddsMidway.calls == 1:
                added.append((await other.remember("space", "The Faro office opens in July")).episode_id)
            return await super().embed(texts)

    rebuilder = await open_sqlite(path, AddsMidway("model-b"))
    try:
        report = await rebuilder.reembed_vectors()
        chunks = await rebuilder.documents.chunks_of("space", added[0])
        assert report.orphans_removed == 0
        assert {chunk.chunk_id for chunk in chunks} <= set(await rebuilder.vectors.ids("space"))
    finally:
        await rebuilder.close()
        await other.close()


async def test_a_record_forgotten_during_a_rebuild_does_not_come_back(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    first = await open_sqlite(path, Model("model-a"))
    doomed = await first.remember("space", "The Lisbon office opens in May")
    doomed_chunks = [chunk.chunk_id for chunk in await first.documents.chunks_of("space", doomed.episode_id)]
    await first.close()
    other = await open_sqlite(path, Model("model-a"))

    class ForgetsWhileEmbedding(Rotated):
        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            vectors = await super().embed(texts)
            if texts == ["The Lisbon office opens in May"]:
                await other.forget("space", doomed.episode_id)
            return vectors

    rebuilder = await open_sqlite(path, ForgetsWhileEmbedding("model-b"))
    try:
        await rebuilder.reembed_vectors()
        assert not set(doomed_chunks) & set(await rebuilder.vectors.ids("space"))
    finally:
        await rebuilder.close()
        await other.close()


async def test_a_yielding_write_override_cannot_overwrite_another_writers_mark() -> None:
    import asyncio
    from scone_memory.core.ports import VectorPoint

    class Yielding(InMemoryVectorIndex):
        async def upsert(self, points):
            await asyncio.sleep(0)
            await super().upsert(points)

    index = Yielding()
    await index.ensure(2)
    await asyncio.gather(
        index.upsert_as([VectorPoint(1, "s", 1, "2025-01-01T00:00:00Z", [1.0, 0.0])], "model-a"),
        index.upsert_as([VectorPoint(2, "s", 2, "2025-01-01T00:00:00Z", [0.0, 1.0])], "model-b"))
    assert await index.written_by() == ("mixed", "invalidated")


async def test_a_yielding_search_override_is_rechecked_after_it_returns() -> None:
    import asyncio
    from scone_memory.core.ports import VectorPoint
    from scone_memory.core.vector_writers import VectorsNotComparable

    class YieldingSearch(InMemoryVectorIndex):
        async def search(self, *args, **kwargs):
            await asyncio.sleep(0)
            return await super().search(*args, **kwargs)

    index = YieldingSearch()
    await index.ensure(2)
    await index.upsert_as([VectorPoint(1, "s", 1, "2025-01-01T00:00:00Z", [1.0, 0.0])], "model-a")
    search = asyncio.ensure_future(index.search_as("s", [1.0, 0.0], 5, writer="model-a"))
    await asyncio.sleep(0)
    await index.upsert_as([VectorPoint(2, "s", 2, "2025-01-01T00:00:00Z", [0.0, 1.0])], "model-b")
    with pytest.raises(VectorsNotComparable):
        await search


async def test_a_first_write_that_fails_leaves_no_claim_behind() -> None:
    from scone_memory.core.ports import VectorPoint

    class Failing(InMemoryVectorIndex):
        async def upsert(self, points):
            raise ConnectionError("vector store went away")

    index = Failing()
    await index.ensure(2)
    with pytest.raises(ConnectionError):
        await index.upsert_as([VectorPoint(1, "s", 1, "2025-01-01T00:00:00Z", [1.0, 0.0])], "model-a")
    assert await index.written_by() is None


async def test_a_failed_write_never_withdraws_another_pending_writers_claim() -> None:
    import asyncio
    from scone_memory.core.ports import VectorPoint

    class Gated(InMemoryVectorIndex):
        def __init__(self) -> None:
            super().__init__()
            self.gates: list[asyncio.Event] = []

        async def upsert(self, points):
            gate = asyncio.Event()
            self.gates.append(gate)
            await gate.wait()
            if points[0].chunk_id == 1:
                raise ConnectionError("first write fails")
            await super().upsert(points)

    def point(chunk: int, vector: list[float]) -> VectorPoint:
        return VectorPoint(chunk, "s", chunk, "2025-01-01T00:00:00Z", vector)

    index = Gated()
    await index.ensure(2)
    first = asyncio.ensure_future(index.upsert_as([point(1, [1.0, 0.0])], "model-a"))
    second = asyncio.ensure_future(index.upsert_as([point(2, [1.0, 0.0])], "model-a"))
    await asyncio.sleep(0)
    index.gates[0].set()
    with pytest.raises(ConnectionError):
        await first
    other = asyncio.ensure_future(index.upsert_as([point(3, [0.0, 1.0])], "model-b"))
    await asyncio.sleep(0)
    index.gates[2].set()
    await other
    index.gates[1].set()
    await second
    assert await index.written_by() == ("mixed", "invalidated")
