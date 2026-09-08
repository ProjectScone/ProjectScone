"""doctor: what references what across the stores, read-only, so a
person can see orphans before deciding anything about them."""

from __future__ import annotations

import asyncio

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.core.ports import NewFact
from scone_memory.runtime import cli


async def clean(engine):
    episode = await engine.remember("default", "Acme is headquartered in Lisbon.")
    await engine.assert_fact("default", "acme", "based_in", "lisbon", source_episode_id=episode.episode_id, quote="Lisbon")
    return episode


async def test_a_consistent_store_reports_no_orphans_and_says_what_it_looked_at():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await clean(engine)
    report = await engine.doctor("default")
    assert (report.episodes, report.chunks, report.facts, report.links, report.tombstones) == (1, 1, 1, 0, 0)
    assert report.chunks_without_episode == [] and report.vectors_without_chunk == []
    assert report.facts_citing_forgotten == [] and report.facts_citing_unknown == []
    assert report.links_with_missing_ends == [] and report.attachments_unlinked == []
    assert report.not_inspected == [] and report.healthy is True


async def test_every_kind_of_orphan_is_named_and_nothing_is_repaired():
    documents, vectors = InMemoryDocumentStore(), InMemoryVectorIndex()
    engine = await MemoryEngine(documents, vectors, HashEmbedder()).open()
    episode = await clean(engine)
    receipt = await engine.forget("default", episode.episode_id)
    cited = receipt.facts_citing[0]
    unknown = await documents.insert_fact(NewFact(space="default", subject="x", predicate="y", object="z",
                                                  valid_from="2026-01-01T00:00:00.000Z", source_episode_id=999_999))
    loose = await engine.attach("default", b"never linked", "text/plain")
    kept = await engine.remember("default", "a second note, kept")
    # Break the store by hand, the way a crash between two writes would.
    [chunk] = await documents.chunks_of("default", kept.episode_id)
    del documents._episodes[kept.episode_id]
    orphan_vector = await engine.remember("default", "a third note")
    [third] = await documents.chunks_of("default", orphan_vector.episode_id)
    del documents._chunks[third.chunk_id]

    report = await engine.doctor("default")
    assert report.facts_citing_forgotten == [cited] and report.facts_citing_unknown == [unknown.fact_id]
    assert report.attachments_unlinked == [loose.attachment_id]
    assert report.chunks_without_episode == [chunk.chunk_id]
    assert report.vectors_without_chunk == [third.chunk_id]
    assert report.tombstones == 1 and report.healthy is False and report.not_inspected == []
    assert (await engine.status("default")).episodes == report.episodes, "read-only: nothing moved"
    assert await documents.get_fact("default", unknown.fact_id) is not None


async def test_sqlite_is_fully_inspectable_and_http_reads_the_same_report(tmp_path):
    path = tmp_path / "memory.db"
    engine = await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder()).open()
    episode = await clean(engine)
    await engine.forget("default", episode.episode_id)
    report = await engine.doctor("default")
    assert report.not_inspected == [] and report.tombstones == 1 and len(report.facts_citing_forgotten) == 1
    import httpx

    app = create_app(engine, {"k": "default"}, console=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test") as c:
        manifest = (await c.get("/v1/capabilities", headers={"Authorization": "Bearer k"})).json()
        assert manifest["features"]["integrity.read"] is True
        body = (await c.get("/v1/doctor", headers={"Authorization": "Bearer k"})).json()
        assert body["facts_citing_forgotten"] == report.facts_citing_forgotten and body["healthy"] is False
        assert (await c.get("/v1/doctor")).status_code == 401
    await engine.close()


def test_the_cli_reads_the_same_report(tmp_path):
    import io
    import json

    env = {"SCONE_SQLITE_PATH": str(tmp_path / "cli.db")}

    def run(*argv, stdin=""):
        out = io.StringIO()
        return cli.main(list(argv), env=env, stdin=io.StringIO(stdin), out=out), out.getvalue()

    run("remember", stdin="Acme is headquartered in Lisbon.")
    run("assert", "acme", "based_in", "lisbon")
    code, text = run("doctor")
    assert code == 0 and text.startswith("healthy")
    run("forget", "1")
    code, text = run("doctor", "--json")
    assert code == 0 and json.loads(text)["tombstones"] == 1 and json.loads(text)["healthy"] is True, "a tombstone is a record, not an orphan"
    code, text = run("doctor")
    assert code == 0 and text.startswith("healthy") and "1 tombstone(s)" in text
