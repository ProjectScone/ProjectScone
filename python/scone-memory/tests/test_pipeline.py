"""The affordances a data pipeline needs: batch ingest, a blocking
facade, and a dump that moves between stores."""

from __future__ import annotations

import asyncio
import io
import json

import pytest

from scone_memory import (
    HashEmbedder,
    InvalidInput,
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
    import gc
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError):
            sync.status("default")
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)], "closed facade leaked a coroutine"


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
    assert json.loads(out.getvalue()) == {"episodes": 1, "deduplicated": 0, "facts": 1, "facts_skipped": 0}
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


class ExplodingIndex(InMemoryVectorIndex):
    async def upsert(self, points):
        raise ConnectionError("vector store went away")


class ExplodingEmbedder(HashEmbedder):
    async def embed(self, texts):
        raise TimeoutError("embedding server timed out")


async def test_a_batch_that_fails_leaves_nothing_behind():
    for engine in (
        MemoryEngine(InMemoryDocumentStore(), ExplodingIndex(), HashEmbedder()),
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), ExplodingEmbedder()),
    ):
        await engine.open()
        with pytest.raises((ConnectionError, TimeoutError)):
            await engine.remember_many("default", [Record("first"), Record("second")])
        status = await engine.status("default")
        assert (status.episodes, status.chunks, status.revision) == (0, 0, 0)
        # A retry with a working batch is not confused by the failed one.
        engine.vectors = InMemoryVectorIndex()
        engine.embedder = HashEmbedder()
        await engine.open()
        added = await engine.remember_many("default", [Record("first"), Record("second"), Record("first")])
        assert [a.deduplicated for a in added] == [False, False, True]
        assert added[2].episode_id == added[0].episode_id


def test_serve_uses_the_same_store_defaults_as_the_other_commands(monkeypatch, tmp_path):
    captured = {}
    import scone_memory.api.__main__ as serve_module

    monkeypatch.setattr(serve_module, "main", lambda settings=None: captured.setdefault("settings", settings))
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "s.db"), "SCONE_API_KEY": "k"}
    assert cli.main(["serve"], env=env, stdin=io.StringIO(), out=io.StringIO()) == 0
    assert captured["settings"] is not None
    assert (captured["settings"].documents, captured["settings"].vectors) == ("sqlite", "sqlite")
    assert captured["settings"].sqlite_path == str(tmp_path / "s.db")


async def test_chunks_store_byte_offsets_into_the_episode():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=120).open()
    content = "São Tomé ☕ " * 30
    added = await engine.remember("default", content)
    assert added.chunks > 1
    raw = content.encode()
    result = await engine.recall("default", "São Tomé", limit=10)
    seen = 0
    for item in result.items:
        [chunk] = await engine.documents.get_chunks("default", [item.chunk_id])
        assert raw[chunk.start : chunk.end].decode() == chunk.text
        seen += 1
    assert seen == min(added.chunks, 2)


async def test_import_into_a_populated_store_remaps_provenance_and_does_not_double_facts():
    source = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    src_ep = await source.remember("default", "ana moved to porto for the harbour job", created_at="2024-05-01")
    await source.assert_fact("default", "ana", "lives_in", "Porto", valid_from="2024-05-01", source_episode_id=src_ep.episode_id)
    dump = [r async for r in source.export("default")]
    assert dump[-1]["source_episode_id"] == 1

    target = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    unrelated = await target.remember("default", "an unrelated note that happens to be episode one")
    assert unrelated.episode_id == 1
    first = await target.import_records("default", dump)
    assert (first.episodes, first.facts) == (1, 1)
    [fact] = await target.facts("default")
    moved = await target.recall("default", "porto harbour job", limit=1)
    assert fact.source_episode_id == moved.items[0].episode_id != unrelated.episode_id

    again = await target.import_records("default", dump)
    assert (again.episodes, again.facts, again.deduplicated, again.facts_skipped) == (0, 0, 1, 1)
    assert len(await target.facts("default", include_closed=True)) == 1


async def test_episode_kinds_are_the_rust_vocabulary():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for kind in ("note", "file", "conversation", "observation", "connector"):
        await engine.remember("default", f"a {kind}", kind=kind)  # type: ignore[arg-type]
    for kind in ("chat", "web", "memo"):  # pre-release names are gone, not aliased
        with pytest.raises(InvalidInput):
            await engine.remember("default", f"a {kind}", kind=kind)  # type: ignore[arg-type]


def test_cli_review_flow(tmp_path):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "cli.db")}

    def run(*argv, stdin=""):
        out = io.StringIO()
        code = cli.main(list(argv), env=env, stdin=io.StringIO(stdin), out=out)
        return code, out.getvalue()

    _, text = run("assert", "mark", "lives_in", "Lisbon", "--valid-from", "2024-03-02", "--origin", "extracted", "--propose", "--json")
    proposed = json.loads(text)
    assert (proposed["status"], proposed["origin"]) == ("proposed", "extracted")
    _, text = run("review")
    assert f"#{proposed['fact_id']} [proposed] [extracted]" in text and "confidence 1.00" in text
    assert run("facts")[1].strip() == "no facts"
    _, text = run("approve", str(proposed["fact_id"]), "--json")
    assert json.loads(text)["status"] == "active"
    _, text = run("exclude", str(proposed["fact_id"]), "--reason", "private", "--json")
    assert json.loads(text)["excluded_reason"] == "private"
    assert run("facts")[1].strip() == "no facts"
    assert "excluded: private" in run("facts", "--excluded")[1]
    _, text = run("include", str(proposed["fact_id"]), "--json")
    assert json.loads(text)["excluded_reason"] is None
    assert run("decline", str(proposed["fact_id"]), "--reason", "no")[0] == 2  # not proposed any more
    assert run("review")[1].strip() == "nothing awaiting review"


def test_cli_distill_reports_a_pass_or_refuses_without_a_model(tmp_path, monkeypatch):
    env = {"SCONE_SQLITE_PATH": str(tmp_path / "d.db")}
    out = io.StringIO()
    assert cli.main(["distill"], env=env, stdin=io.StringIO(), out=out) == 2

    from scone_memory import FakeChat
    import scone_memory.config as config

    monkeypatch.setattr(config, "build_chat", lambda settings: FakeChat([json.dumps([{"subject": "ana", "predicate": "lives_in", "object": "Lisbon", "confidence": 0.9}])]))
    env2 = {"SCONE_SQLITE_PATH": str(tmp_path / "d.db"), "SCONE_CHAT_URL": "http://x", "SCONE_CHAT_MODEL": "m"}
    cli.main(["remember"], env=env2, stdin=io.StringIO("Ana moved to Lisbon."), out=io.StringIO())
    out = io.StringIO()
    assert cli.main(["distill", "--json"], env=env2, stdin=io.StringIO(), out=out) == 0
    report = json.loads(out.getvalue())
    assert (report["episodes"], report["proposed"], report["error"]) == (1, 1, None)
    out = io.StringIO()
    cli.main(["review"], env=env2, stdin=io.StringIO(), out=out)
    assert "[proposed] [extracted] ana lives_in Lisbon" in out.getvalue()

    monkeypatch.setattr(config, "build_chat", lambda settings: FakeChat(["not json"]))
    cli.main(["remember"], env=env2, stdin=io.StringIO("Bob moved to Berlin."), out=io.StringIO())
    out = io.StringIO()
    assert cli.main(["distill", "--json"], env=env2, stdin=io.StringIO(), out=out) == 1, "a failed pass exits 1"
    assert json.loads(out.getvalue())["error"].startswith("DistillError")


def test_cli_agent_hook_accepts_its_own_flags_first(tmp_path):
    """The hook is invoked as `scone-memory agent-hook --agent claude-code ...`;
    argparse REMAINDER refused a leading flag with exit 2, which would have
    made an installed hook fail on every call."""
    env = {"SCONE_API_KEY": "k", "SCONE_HOOK_PROJECTS": f"scone={tmp_path}"}
    out = io.StringIO()
    payload = json.dumps({"hook_event_name": "Stop", "session_id": "s", "cwd": str(tmp_path), "last_assistant_message": "x"})
    code = cli.main(["agent-hook", "--agent", "claude-code", "--feed", "metadata", "--server", "http://127.0.0.1:1"], env=env, stdin=io.StringIO(payload), out=out)
    assert code == 0, "the hook never blocks the agent, even when the server is down"
    assert out.getvalue().strip() == "{}"
    with pytest.raises(SystemExit) as exit_info:  # other commands still reject unknown flags
        cli.main(["status", "--bogus"], env={"SCONE_SQLITE_PATH": str(tmp_path / "x.db")}, stdin=io.StringIO(), out=io.StringIO())
    assert exit_info.value.code == 2


class SeeingEmbedder(HashEmbedder):
    """Records exactly what it was asked to embed."""

    def __init__(self):
        super().__init__()
        self.seen: list[str] = []

    async def embed(self, texts):
        self.seen.extend(texts)
        return await super().embed(texts)


async def test_contextual_embeddings_change_what_is_embedded_not_what_is_stored():
    from scone_memory.engine import contextual_prefix
    from scone_memory.ports import NewEpisode

    plain, ctx = SeeingEmbedder(), SeeingEmbedder()
    off = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), plain).open()
    on = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), ctx, contextual_embeddings=True).open()
    text = "the harbour crane was repainted"
    for engine in (off, on):
        await engine.remember("default", text, created_at="2024-05-01", source="notes://harbour", metadata={"user_id": "ana"})
    assert plain.seen == [text]
    assert ctx.seen == ["2024-05-01 | notes://harbour | user id ana\n" + text], "the prefix names date, source and scope"
    for engine in (off, on):
        [item] = (await engine.recall("default", "harbour crane", limit=1)).items
        assert item.text == text, "stored and returned text is the raw span either way"
        [chunk] = await engine.documents.get_chunks("default", [item.chunk_id])
        assert chunk.text == text and chunk.end - chunk.start == len(text.encode())
    # queries are embedded as written in both modes; only chunks get the prefix
    assert ctx.seen[-1] == "harbour crane" and plain.seen[-1] == "harbour crane"
    [ev] = await on.events.query("default", kind="recall") if on.events else [None]
    bare = NewEpisode(space="s", kind="note", content="x", content_hash="h", created_at="", ingested_at="")
    assert contextual_prefix(bare) == "", "nothing to say means no prefix"


async def test_recall_events_record_the_embedding_mode():
    from scone_memory import InMemoryEventLog

    on = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=InMemoryEventLog(), contextual_embeddings=True).open()
    await on.remember("default", "one note")
    await on.recall("default", "note")
    [ev] = await on.events.query("default", kind="recall")
    assert ev.payload["contextual_embeddings"] is True
