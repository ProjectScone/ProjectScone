"""The affordances a data pipeline needs: batch ingest, a blocking
facade, and a dump that moves between stores."""

from __future__ import annotations

import asyncio
import io
import json

import pytest

from scone_memory import (
    HashEmbedder,
    InMemoryDocumentStore,
    InMemoryVectorIndex,
    MemoryEngine,
    Record,
    SyncMemoryEngine,
)
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory import cli


class CountingEmbedder(HashEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return await super().embed(texts)


async def test_batch_ingest_embeds_once_per_batch_and_dedups_within_it():
    embedder = CountingEmbedder()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()
    records = [Record(f"note number {i} about the harbour") for i in range(40)] + [Record("note number 3 about the harbour")]
    added = await engine.remember_many("default", records)
    assert len(added) == 41
    assert embedder.calls == 1
    assert added[-1].deduplicated and added[-1].episode_id == added[3].episode_id
    assert sum(1 for a in added if not a.deduplicated) == 40
    assert (await engine.status("default")).revision == 1


def test_sync_facade_works_from_plain_python_and_inside_a_running_loop():
    with SyncMemoryEngine(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())) as sync:
        added = sync.remember("default", "synchronous callers exist too")
        assert sync.recall("default", "synchronous callers").items[0].episode_id == added.episode_id

        async def from_inside_a_loop():
            return sync.status("default").episodes

        assert asyncio.run(from_inside_a_loop()) == 1
    with pytest.raises(RuntimeError):
        sync.status("default")


async def test_export_then_import_moves_memory_between_stores(tmp_path):
    source = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await source.remember("default", "the harbour crane was repainted", created_at="2024-05-01", tags=["town"], metadata={"user_id": "ana"})
    await source.assert_fact("default", "ana", "lives_in", "Porto", valid_from="2022-01-01")
    await source.assert_fact("default", "ana", "lives_in", "Lisbon", valid_from="2024-03-02")
    dump = [json.loads(json.dumps(r)) async for r in source.export("default")]

    path = tmp_path / "moved.db"
    target = await MemoryEngine(SqliteDocumentStore(path), SqliteVectorIndex(path), HashEmbedder()).open()
    summary = await target.import_records("default", dump)
    assert (summary.episodes, summary.facts, summary.deduplicated) == (1, 2, 0)

    result = await target.recall("default", "harbour crane", where={"user_id": "ana"})
    assert [i.created_at[:10] for i in result.items] == ["2024-05-01"]
    assert [f.object for f in await target.facts("default")] == ["Lisbon"]
    assert [f.object for f in await target.facts("default", as_of="2023-01-01")] == ["Porto"]
    again = await target.import_records("default", dump)
    assert again.deduplicated == 1


def test_cli_round_trip_through_sqlite(tmp_path):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "cli.db")}

    def run(*argv, stdin=""):
        out = io.StringIO()
        code = cli.main(list(argv), env=env, stdin=io.StringIO(stdin), out=out)
        return code, out.getvalue()

    code, text = run("remember", "--created-at", "2024-03-02", "--tag", "life", "--meta", "user_id=mark", stdin="Moved to Lisbon in March")
    assert code == 0 and text.strip() == "remembered 1 episode(s)"
    code, text = run("assert", "mark", "lives_in", "Lisbon", "--valid-from", "2024-03-02", "--json")
    assert json.loads(text)["status"] == "active"
    code, text = run("recall", "where does mark live", "--json")
    payload = json.loads(text)
    assert [i["episode_id"] for i in payload["items"]] == [1]
    assert [f["object"] for f in payload["facts"]] == ["Lisbon"]
    code, text = run("recall", "lisbon", "--where", "user_id=someone_else", "--json")
    assert json.loads(text)["items"] == []

    code, dump = run("export")
    assert len(dump.splitlines()) == 2
    env2 = {"SCONE_SQLITE_PATH": str(tmp_path / "second.db")}
    out = io.StringIO()
    assert cli.main(["import", "--json"], env=env2, stdin=io.StringIO(dump), out=out) == 0
    assert json.loads(out.getvalue()) == {"episodes": 1, "deduplicated": 0, "facts": 1}
    out = io.StringIO()
    cli.main(["status", "--json"], env=env2, stdin=io.StringIO(), out=out)
    assert json.loads(out.getvalue())["episodes"] == 1


def test_cli_reports_engine_errors_with_exit_code_2(tmp_path):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "cli.db")}
    assert cli.main(["--space", "Bad Space", "status"], env=env, stdin=io.StringIO(), out=io.StringIO()) == 2
    assert cli.main(["forget", "42"], env=env, stdin=io.StringIO(), out=io.StringIO()) == 2


def test_cli_jsonl_batch_ingest(tmp_path):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "cli.db")}
    lines = "\n".join(json.dumps({"content": f"ticket {i} about billing", "metadata": {"session_id": "s1"}}) for i in range(5))
    out = io.StringIO()
    assert cli.main(["remember", "--jsonl", "--json"], env=env, stdin=io.StringIO(lines), out=out) == 0
    assert [json.loads(l)["chunks"] for l in out.getvalue().splitlines()] == [1, 1, 1, 1, 1]


async def test_recall_names_the_lanes_that_found_each_item():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("default", "the lighthouse keeper logs the tide twice a day")
    await engine.remember("default", "quarterly revenue exceeded expectations")
    [top, *_] = (await engine.recall("default", "lighthouse tide log", limit=1)).items
    assert top.lanes == {"vector": 1, "text": 1}


async def test_scopes_count_episodes_per_metadata_value():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("default", "a", metadata={"user_id": "alice", "agent_id": "planner"})
    await engine.remember("default", "b", metadata={"user_id": "alice", "session_id": "s1"})
    await engine.remember("default", "c", metadata={"user_id": "bob"})
    await engine.remember("default", "d")
    assert await engine.scopes("default") == {
        "agent_id": {"planner": 1},
        "session_id": {"s1": 1},
        "user_id": {"alice": 2, "bob": 1},
    }
