"""Optional source-inventory contract; supply the usual native engine fixture."""
import pytest


async def test_source_inventory_complete_scoped_keyset_walk(engine):
    if not callable(getattr(engine.documents, "page_episodes", None)):
        pytest.skip("custom document store does not implement optional inventory")
    ids = []
    for i in range(9):
        added = await engine.remember("alpha", f"retained source {i}", kind="file" if i % 2 == 0 else "note",
                                      created_at=f"2024-01-{9-i:02d}")
        ids.append(added.episode_id)
        await engine.remember("beta", f"other space source {i}", kind="file")
    found, before = [], None
    for _ in range(5):
        page = await engine.source_page("alpha", before=before, limit=2, kind="file")
        found.extend(e.episode_id for e in page.episodes)
        if not page.has_more:
            assert page.next_before is None
            break
        assert len(page.episodes) == 2
        assert page.next_before == page.episodes[-1].episode_id
        assert before is None or page.next_before < before
        before = page.next_before
    assert found == ids[::2][::-1]
    first = await engine.source_page("alpha", limit=2, kind="file")
    await engine.forget("alpha", first.next_before)
    second = await engine.source_page("alpha", before=first.next_before, limit=2, kind="file")
    assert [e.episode_id for e in second.episodes] == [ids[4], ids[2]]
    assert (await engine.source_page("unused")).episodes == []
