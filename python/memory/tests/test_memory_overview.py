"""A recent inventory supplies evidence without a search or a full history scan."""

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.sqlite import SqliteDocumentStore
from scone_memory.core.errors import InvalidInput


@pytest.fixture(params=["memory", "sqlite"])
async def memory(request, tmp_path):
    store = (InMemoryDocumentStore() if request.param == "memory"
             else SqliteDocumentStore(str(tmp_path / "overview.db")))
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder(), chunk_target=200).open()
    yield engine
    await engine.close()


async def test_overview_preserves_retained_evidence_and_insert_order(memory, monkeypatch):
    older = await memory.remember("alpha", "Recent event but inserted first", created_at="2025-01-01")
    newer = await memory.remember("alpha", "Full retained evidence. " * 60, source="repo/project",
                                  tags=["work"], metadata={"user_id": "alice"}, created_at="2024-01-01")
    await memory.remember("beta", "Other tenant's private project")

    async def forbidden(*args, **kwargs):
        pytest.fail("overview must read only the bounded document inventory")

    for obj, name in [(memory, "recall"), (memory.embedder, "embed"), (memory.documents, "counts"),
                      (memory.documents, "recent_episodes"), (memory.documents, "search_text")]:
        monkeypatch.setattr(obj, name, forbidden)
    result = await memory.overview("alpha")
    assert [item.episode_id for item in result.items] == [newer.episode_id, older.episode_id]
    chunks = await memory.documents.chunks_of("alpha", newer.episode_id)
    item = result.items[0]
    assert item.chunk_id == chunks[0].chunk_id and item.text == chunks[0].text
    assert item.source == "repo/project" and item.created_at == "2024-01-01T00:00:00.000Z"
    assert item.metadata == {"user_id": "alice"} and item.tags == ("work",)
    assert item.similarity is None
    assert result.considered == 2 and not result.has_more and result.next_before is None


async def test_overview_walks_past_scoped_and_current_session_records(memory):
    kept = await memory.remember("alpha", "Actual earlier project work", kind="conversation", source="chat/old",
                                 metadata={"user_id": "alice", "session_id": "old"}, created_at="2024-06-15")
    for i in range(110):
        await memory.remember("alpha", f"Newer irrelevant row {i}", kind="conversation", source="chat/new",
                              metadata={"user_id": "bob" if i % 2 else "alice", "session_id": "current"},
                              created_at="2024-06-15")
    options = dict(where={"user_id": "alice"}, kind="conversation", source_prefix="chat/",
                   since="2024-06-01", until="2024-06-30", exclude_session_id="current")
    result = await memory.overview("alpha", **options)
    assert [item.episode_id for item in result.items] == [kept.episode_id]
    assert result.considered == 111 and not result.has_more
    first = await memory.overview("alpha", max_records=100, **options)
    assert first.items == [] and first.considered == 100 and first.has_more
    second = await memory.overview("alpha", before=first.next_before, max_records=100, **options)
    assert [item.episode_id for item in second.items] == [kept.episode_id]
    assert second.considered == 11 and not second.has_more


async def test_overview_filters_time_source_and_kind(memory):
    for text, opts in [
        ("wrong date", {"created_at": "2023-01-01"}),
        ("wrong source", {"source": "other/file"}),
        ("wrong kind", {"kind": "note"}),
        ("wanted", {}),
    ]:
        await memory.remember("alpha", text, **({"kind": "file", "source": "repo/file",
                                                  "created_at": "2024-01-01"} | opts))
    result = await memory.overview("alpha", kind="file", source_prefix="repo/",
                                   since="2024-01-01", until="2024-01-01")
    assert [item.text for item in result.items] == ["wanted"]


async def test_overview_continuation_keeps_unique_ids_and_budget(memory, monkeypatch):
    ids = [(await memory.remember("alpha", f"Project {i}")).episode_id for i in range(8)]
    native = memory.documents.page_episodes
    requested = []

    async def page(space, before, limit, kind):
        requested.append(limit)
        return await native(space, before, limit, kind)

    monkeypatch.setattr(memory.documents, "page_episodes", page)
    first = await memory.overview("alpha", limit=2, max_records=3)
    assert first.has_more and first.next_before == ids[-2]
    assert sum(requested) <= 3
    await memory.forget("alpha", first.next_before)
    second = await memory.overview("alpha", before=first.next_before)
    assert [i.episode_id for i in first.items + second.items] == ids[::-1]
    assert not second.has_more


@pytest.mark.parametrize("options", [
    {"limit": 0}, {"limit": 51}, {"limit": True}, {"limit": 1.5},
    {"max_records": 0}, {"max_records": 1001}, {"max_records": True}, {"max_records": "10"},
    {"before": 0}, {"before": True}, {"before": 2**63}, {"before": 1.5},
    {"kind": "unknown"}, {"kind": []}, {"where": []}, {"where": {1: "alice"}},
    {"where": {"user_id": 3}}, {"source_prefix": 3}, {"source_prefix": "x" * 1001},
    {"since": ""}, {"since": 3}, {"until": False}, {"until": "invalid"},
    {"since": "2025-01-01", "until": "2024-01-01"},
    {"exclude_session_id": ""}, {"exclude_session_id": True},
])
async def test_overview_rejects_invalid_boundary_arguments(memory, options):
    with pytest.raises(InvalidInput):
        await memory.overview("alpha", **options)


async def test_overview_refuses_unsupported_inventory(memory, monkeypatch):
    monkeypatch.setattr(memory.documents, "page_episodes", None)
    with pytest.raises(InvalidInput, match="inventory"):
        await memory.overview("alpha")


async def test_overview_rechecks_backend_space_and_chunk_ownership(memory, monkeypatch):
    other = await memory.remember("beta", "Private evidence")
    private = await memory.documents.get_episode("beta", other.episode_id)

    async def wrong_space(*args):
        return [private]

    monkeypatch.setattr(memory.documents, "page_episodes", wrong_space)
    assert (await memory.overview("alpha")).items == []


async def test_overview_uses_first_nonempty_owned_chunk_without_rewriting(memory, monkeypatch):
    added = await memory.remember("alpha", "Retained words. " * 100)
    chunks = await memory.documents.chunks_of("alpha", added.episode_id)
    assert len(chunks) > 1
    expected = chunks[1]

    async def mixed_chunks(*args):
        return [chunks[0].model_copy(update={"space": "beta"}),
                chunks[0].model_copy(update={"episode_id": added.episode_id + 1}),
                chunks[0].model_copy(update={"text": " \n\t"}), expected]

    monkeypatch.setattr(memory.documents, "chunks_of", mixed_chunks)
    result = await memory.overview("alpha")
    assert len(result.items) == 1
    assert result.items[0].text == expected.text and result.items[0].chunk_id == expected.chunk_id


async def test_overview_does_not_turn_control_or_social_content_into_synthetic_facts(memory):
    texts = ["Hello!", "<system-reminder>Stored source text.</system-reminder>"]
    for text in texts:
        await memory.remember("alpha", text)
    assert [item.text for item in (await memory.overview("alpha")).items] == texts[::-1]


async def test_overview_empty_history_and_invalid_space(memory):
    result = await memory.overview("empty")
    assert result.items == [] and result.considered == 0 and not result.has_more
    for space in ["", "invalid space", 12]:
        with pytest.raises(InvalidInput):
            await memory.overview(space)
