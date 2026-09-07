"""Source inventory is a scoped keyset walk, never a ranked recall result."""
import pytest
import httpx
import json
from pathlib import Path
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends.sqlite import SqliteDocumentStore
from scone_memory.api import create_app
from scone_memory.errors import InvalidInput


@pytest.fixture(params=["memory", "sqlite"])
async def memory(request, tmp_path):
    store = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(str(tmp_path / "memory.db"))
    if hasattr(store, "open"):
        await store.open()
    engine = await MemoryEngine(store, InMemoryVectorIndex(), HashEmbedder()).open()
    yield engine
    if hasattr(store, "close"):
        await store.close()


async def test_keyset_inventory_is_complete_filtered_and_survives_deleted_boundary(memory):
    ids = []
    for i in range(7):
        added = await memory.remember("alpha", f"source {i}", kind="file" if i % 2 == 0 else "note",
                                      created_at=f"2024-01-{9-i:02d}")
        ids.append(added.episode_id)
        await memory.remember("beta", f"private {i}")
    first = await memory.source_page("alpha", limit=2, kind="file")
    assert [e.episode_id for e in first.episodes] == [ids[6], ids[4]]
    assert first.has_more and first.next_before == ids[4]
    await memory.forget("alpha", ids[4])
    newest = await memory.remember("alpha", "inserted after page one", kind="file")
    second = await memory.source_page("alpha", before=first.next_before, limit=2, kind="file")
    assert [e.episode_id for e in second.episodes] == [ids[2], ids[0]]
    assert second.has_more is False and second.next_before is None
    assert (await memory.source_page("alpha", limit=1)).episodes[0].episode_id == newest.episode_id
    assert (await memory.source_page("empty")).episodes == []


@pytest.mark.parametrize("options", [{"limit": 0}, {"limit": 101}, {"limit": True}, {"before": 0},
                                     {"before": -1}, {"before": 2**63}, {"before": True}, {"kind": ""}, {"kind": "unknown"}])
async def test_invalid_native_page_arguments_fail(memory, options):
    with pytest.raises(InvalidInput):
        await memory.source_page("alpha", **options)


async def test_inventory_does_not_scan_counts_or_recent_history(memory, monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("inventory must use the bounded backend query")
    monkeypatch.setattr(memory.documents, "counts", forbidden)
    monkeypatch.setattr(memory.documents, "recent_episodes", forbidden)
    assert (await memory.source_page("alpha")).episodes == []


async def test_http_inventory_summaries_and_legacy_batch_are_separate(memory):
    text = "prefix\0" + "猫🙂" * 300
    added = await memory.remember("alpha", text, kind="file", source="original.txt")
    await memory.remember("beta", "private source", source="secret.txt")
    app = create_app(memory, {"a": "alpha", "b": "beta"})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/v1/sources")).status_code == 401
        assert (await client.get("/v1/sources", headers={"Authorization": "Bearer wrong"})).status_code == 401
        auth = {"Authorization": "Bearer a"}
        assert (await client.get("/v1/capabilities", headers=auth)).json()["features"]["episodes.list"] is True
        page = await client.get("/v1/sources?limit=1&kind=file", headers=auth)
        assert page.status_code == 200
        assert page.json() == {"items": [{"episode_id": added.episode_id, "kind": "file", "source": "original.txt",
            "created_at": (await memory.episode("alpha", added.episode_id)).created_at,
            "byte_count": len(text.encode()), "preview": text[:500], "preview_truncated": True}],
            "has_more": False, "next_before": None}
        assert (await client.get(f"/v1/episodes?ids={added.episode_id}", headers=auth)).json()["episodes"][0]["content"] == text
        for query in ["limit=0", "limit=101", "before=0", "before=-1", "before=9223372036854775808", "kind=", "kind=unknown", "space=beta"]:
            assert (await client.get("/v1/sources?" + query, headers=auth)).status_code == 422


async def test_custom_store_without_inventory_advertises_unavailable(memory, monkeypatch):
    monkeypatch.setattr(memory.documents, "page_episodes", None)
    with TestClient(create_app(memory, {"a": "alpha"})) as client:
        auth = {"Authorization": "Bearer a"}
        assert client.get("/v1/capabilities", headers=auth).json()["features"]["episodes.list"] is False
        assert client.get("/v1/sources", headers=auth).status_code == 501


def test_sync_native_inventory_is_available():
    from scone_memory import SyncMemoryEngine, SourcePage
    with SyncMemoryEngine(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())) as engine:
        added = engine.remember("alpha", "a source for synchronous callers")
        page = engine.source_page("alpha", limit=1)
        assert isinstance(page, SourcePage)
        assert [e.episode_id for e in page.episodes] == [added.episode_id]


async def test_shared_literal_http_inventory_contract(memory):
    fixture = json.loads((Path(__file__).resolve().parents[3] / "tests/fixtures/source-inventory.json").read_text())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(memory, {"alpha": "alpha", "beta": "beta"})), base_url="http://test") as client:
        for record in fixture["records"]:
            response = await client.post("/v1/episodes", json=record["body"], headers={"Authorization": "Bearer " + record["owner"]})
            assert response.status_code == 200
        for page in fixture["pages"]:
            response = await client.get("/v1/sources?" + page["query"], headers={"Authorization": "Bearer alpha"})
            assert response.status_code == 200 and response.json() == page["expected"]
        for query in fixture["invalid_queries"]:
            assert (await client.get("/v1/sources?" + query, headers={"Authorization": "Bearer alpha"})).status_code == 422


async def test_sqlite_inventory_survives_reopen(tmp_path):
    path = str(tmp_path / "retained.db")
    documents = SqliteDocumentStore(path)
    engine = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder()).open()
    ids = [(await engine.remember("alpha", text, kind="file")).episode_id for text in ["first", "second", "third"]]
    first = await engine.source_page("alpha", limit=2)
    await engine.forget("alpha", first.next_before)
    await documents.close()
    reopened = SqliteDocumentStore(path)
    try:
        engine = await MemoryEngine(reopened, InMemoryVectorIndex(), HashEmbedder()).open()
        second = await engine.source_page("alpha", before=first.next_before)
        assert [e.episode_id for e in second.episodes] == [ids[0]]
        assert second.has_more is False
    finally:
        await reopened.close()
