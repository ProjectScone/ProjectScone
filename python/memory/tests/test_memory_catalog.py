"""Catalog reads work without an engine and preserve inventory contracts."""
from __future__ import annotations

import pytest

from scone_memory import MemoryEngine


async def test_catalog_exposes_scoped_profile_and_pending_sources(engine: MemoryEngine) -> None:
    from scone_memory.memory import catalog
    from scone_memory.memory.engine import Profile, RecentActivity

    cited = await engine.remember("alpha", "Juniper uses Polaris.", metadata={"team": "blue"})
    uncited = await engine.remember("alpha", "A later note.", metadata={"team": "blue"})
    await engine.remember("beta", "Hidden source.", metadata={"team": "red"})
    accepted = await engine.assert_fact("alpha", "Juniper", "uses", "Polaris",
        source_episode_id=cited.episode_id, quote="Juniper uses Polaris.")
    await engine.assert_fact("alpha", "Juniper", "needs", "review", proposed=True)

    result = await catalog.profile(engine.documents, "alpha", clock=engine.clock)
    assert isinstance(result, Profile)
    assert all(isinstance(item, RecentActivity) for item in result.recent)
    assert result.static_facts == [accepted]
    assert {item.episode_id for item in result.recent} == {cited.episode_id, uncited.episode_id}
    assert await catalog.pending_distillation(engine.documents, "alpha") == 1
    assert await catalog.cited_episode_ids(engine.documents, "alpha") == {cited.episode_id}
    assert await catalog.scopes(engine.documents, "alpha") == {"team": {"blue": 2}}
    assert await catalog.facts(engine.documents, "alpha") == [accepted]
    # A status selection historically takes precedence over as_of parsing.
    proposed = await catalog.facts(engine.documents, "alpha", status="proposed", as_of="unused")
    assert len(proposed) == 1 and proposed[0].status == "proposed"


async def test_catalog_filtered_walk_preserves_continuation_at_read_budget(engine: MemoryEngine) -> None:
    from scone_memory.memory import catalog

    ids = []
    for index in range(5):
        added = await engine.remember("alpha", f"source {index}", metadata={"match": "no"})
        ids.append(added.episode_id)
    page = await catalog.source_page(engine.documents, "alpha", limit=2,
        conditions={"field": "match", "is": "yes"}, walk_page=2, walk_reads=1)
    assert page.episodes == [] and page.has_more
    assert page.next_before == ids[-2]
    next_page = await catalog.source_page(engine.documents, "alpha", before=page.next_before, limit=2)
    assert [item.episode_id for item in next_page.episodes] == [ids[2], ids[1]]


async def test_catalog_metadata_reads_keep_source_order_and_tags(engine: MemoryEngine) -> None:
    from scone_memory.memory import catalog

    later = await engine.remember("alpha", "Later by source date", created_at="2025-02-02",
        metadata={"session": "s"}, tags=["z", "a"])
    earlier = await engine.remember("alpha", "Earlier by source date", created_at="2025-02-01",
        metadata={"session": "s"}, tags=["a"])
    await engine.remember("beta", "Different space", metadata={"session": "s"}, tags=["secret"])
    found = await catalog.episodes(engine.documents, "alpha", {"session": "s"})
    assert [item.episode_id for item in found] == [earlier.episode_id, later.episode_id]
    newest = await catalog.episodes(engine.documents, "alpha", {"session": "s"}, limit=1)
    assert [item.episode_id for item in newest] == [later.episode_id]
    assert list((await catalog.tags(engine.documents, "alpha")).items()) == [("a", 2), ("z", 1)]


async def test_catalog_status_reads_identity_after_revision(
    engine: MemoryEngine, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scone_memory.memory import catalog

    await engine.remember("alpha", "A retained source")
    await engine.assert_fact("alpha", "subject", "needs", "review", proposed=True)
    labels: catalog.CatalogIdentity = {
        "embedder": "before", "document_store": "before", "vector_index": "before",
    }
    revision = engine.documents.revision

    async def changed(space: str) -> int:
        nonlocal labels
        result = await revision(space)
        labels = {"embedder": "embedder-after", "document_store": "store-after", "vector_index": "vectors-after"}
        return result

    monkeypatch.setattr(engine.documents, "revision", changed)
    result = await catalog.status(engine.documents, "alpha", identity=lambda: labels)
    assert result.episodes == 1 and result.pending_review == 1
    assert (result.embedder, result.document_store, result.vector_index) == tuple(labels.values())
    assert result.embedder == "embedder-after"
