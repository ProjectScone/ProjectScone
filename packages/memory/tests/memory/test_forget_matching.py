"""Forget every source a filter selects, after seeing what that takes.

One episode can be forgotten with a receipt; a whole space can be deleted
with an impact preview; a retention policy can expire by kind and age.
What could not be done is the ordinary bulk case: forget everything
synced from one directory, or everything tagged for one client, as one
authorized operation with one preview. A caller had to page the sources
and forget them one by one, with no combined impact and no receipt for
the whole.

`forget_matching` previews by default and forgets nothing. The preview
names what the filter chose, what forgetting it would take with it, and
a digest of exactly those episodes. Forgetting requires that digest, so
a selection that changed after the preview -- a new source that matches,
one already gone -- is refused rather than forgotten. Each pass is
bounded and says how many are left.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for n in range(5):
        await engine.remember("alpha", f"note {n} from the old tree", source=f"old/{n}.md", tags=["client_x"] if n < 3 else [],
                              metadata={"sync": "old"})
    await engine.remember("alpha", "a note that stays", source="new/keep.md", tags=["client_x"], metadata={"sync": "new"})
    yield engine
    await engine.close()


async def sources(engine):
    return sorted(e.source for e in (await engine.source_page("alpha", limit=100)).episodes)


async def test_a_preview_names_what_the_filter_chose_and_forgets_nothing(memory):
    preview = await memory.forget_matching("alpha", source_prefix="old/")
    assert preview.applied is False and preview.matched == 5 and preview.selection_complete is True
    assert sorted(preview.episode_ids) == sorted(preview.episode_ids) and len(preview.episode_ids) == 5
    assert preview.chunks >= 5 and preview.forgotten == [] and preview.receipts == []
    assert preview.selection and len(preview.selection) == 64
    assert len(await sources(memory)) == 6


async def test_forgetting_needs_the_previewed_selection_and_takes_exactly_that(memory):
    preview = await memory.forget_matching("alpha", source_prefix="old/")
    done = await memory.forget_matching("alpha", source_prefix="old/", apply=True, selection=preview.selection)
    assert done.applied is True and sorted(done.forgotten) == sorted(preview.episode_ids)
    assert len(done.receipts) == 5 and done.remaining == 0
    assert await sources(memory) == ["new/keep.md"]


async def test_a_selection_that_changed_since_the_preview_is_refused_and_nothing_goes(memory):
    preview = await memory.forget_matching("alpha", source_prefix="old/")
    await memory.remember("alpha", "a late arrival", source="old/late.md")
    with pytest.raises(InvalidInput, match="selection"):
        await memory.forget_matching("alpha", source_prefix="old/", apply=True, selection=preview.selection)
    assert len(await sources(memory)) == 7


async def test_apply_without_a_selection_is_refused(memory):
    with pytest.raises(InvalidInput, match="needs the selection digest from a preview"):
        await memory.forget_matching("alpha", source_prefix="old/", apply=True)
    assert len(await sources(memory)) == 6


async def test_an_empty_filter_is_refused_because_deleting_a_space_has_its_own_route(memory):
    with pytest.raises(InvalidInput, match="filter"):
        await memory.forget_matching("alpha")


async def test_filters_combine(memory):
    by_tag = await memory.forget_matching("alpha", tags=["client_x"])
    assert by_tag.matched == 4
    both = await memory.forget_matching("alpha", tags=["client_x"], source_prefix="old/")
    assert both.matched == 3
    by_meta = await memory.forget_matching("alpha", conditions={"field": "sync", "is": "new"})
    assert by_meta.matched == 1


async def test_a_pass_is_bounded_and_says_how_many_are_left(memory):
    preview = await memory.forget_matching("alpha", source_prefix="old/", limit=2)
    assert preview.matched == 5 and len(preview.episode_ids) == 2 and preview.pass_limited is True
    first = await memory.forget_matching("alpha", source_prefix="old/", limit=2, apply=True, selection=preview.selection)
    assert len(first.forgotten) == 2 and first.remaining == 3
    again = await memory.forget_matching("alpha", source_prefix="old/", limit=2)
    assert again.matched == 3


async def test_the_claims_policy_reaches_every_forget(memory):
    [episode] = [e for e in (await memory.source_page("alpha", limit=100)).episodes if e.source == "old/0.md"]
    fact = await memory.assert_fact("alpha", "old tree", "held", "notes", source_episode_id=episode.episode_id, quote="note 0")
    preview = await memory.forget_matching("alpha", source_prefix="old/0")
    assert preview.facts_citing == 1
    done = await memory.forget_matching("alpha", source_prefix="old/0", apply=True, selection=preview.selection, with_claims="exclude")
    assert done.receipts[0].claims_excluded == [fact.fact_id]
    assert (await memory.documents.get_fact("alpha", fact.fact_id)).excluded


async def test_over_http_preview_then_apply():
    from httpx import ASGITransport, AsyncClient

    from scone_memory.api import create_app

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        for n in range(3):
            await engine.remember("alpha", f"note {n}", source=f"old/{n}.md")
        async with AsyncClient(transport=ASGITransport(app=create_app(engine, {"k": "alpha"})), base_url="http://fixture") as client:
            auth = {"authorization": "Bearer k"}
            preview = await client.post("/v1/episodes/forget-matching", json={"source_prefix": "old/"}, headers=auth)
            assert preview.status_code == 200, preview.text
            body = preview.json()
            assert body["applied"] is False and body["matched"] == 3
            refused = await client.post("/v1/episodes/forget-matching", json={"source_prefix": "old/", "apply": True}, headers=auth)
            assert refused.status_code == 422
            done = await client.post("/v1/episodes/forget-matching",
                                     json={"source_prefix": "old/", "apply": True, "selection": body["selection"]}, headers=auth)
            assert done.status_code == 200 and len(done.json()["forgotten"]) == 3
            unknown = await client.post("/v1/episodes/forget-matching", json={"source_prefix": "old/", "everything": True}, headers=auth)
            assert unknown.status_code == 422
    finally:
        await engine.close()


async def test_from_the_command_line(memory):
    import io
    import json

    from scone_memory.runtime.cli import build_parser, run

    out = io.StringIO()
    code = await run(build_parser().parse_args(["--space", "alpha", "--json", "forget-matching", "--source-prefix", "old/"]),
                     memory, io.StringIO(""), out)
    assert code == 0, out.getvalue()
    preview = json.loads(out.getvalue())
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--space", "alpha", "--json", "forget-matching", "--source-prefix", "old/",
                                                "--apply", "--selection", preview["selection"]]), memory, io.StringIO(""), out)
    assert code == 0 and len(json.loads(out.getvalue())["forgotten"]) == 5
