"""A document's summary tree: levels written from the level below, every sentence quoting down, nothing invented."""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.app import create_app
from scone_memory.core.errors import InvalidInput, NotFound
from scone_memory.providers.llm import FakeChat
from scone_memory.retrieval import summary_tree as module
from scone_memory.retrieval.summary_tree import build_summary_tree, stored_summaries
from scone_memory.retrieval.synthesis import SynthesisLimits
from scone_memory.runtime.cli import build_parser, run

PARTS = [
    "The harbour at Vellmar closes to sailing boats every November when the winter swell begins.",
    "Its lighthouse was rebuilt in 1904 after a storm took the first one down to the foundations.",
    "Pilots board arriving ships two miles out, at the red buoy, in every season of the year.",
    "The orchard on the ridge above the town grows only Bramley apples, planted in rows of forty.",
    "Picking starts in the last week of September and ends before the first frost of the year.",
    "The press in the barn makes two thousand litres of juice a season for the town's market.",
    "A narrow-gauge railway once carried the apples down to the harbour; its bed is a footpath now.",
    "The town council meets on the first Tuesday of the month in the old customs house by the quay.",
]


async def document(chunk_target=100):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=chunk_target).open()
    added = await engine.remember("s", "\n\n".join(PARTS), kind="file", source="town.md")
    return engine, added.episode_id, await engine.documents.chunks_of("s", added.episode_id)


def notes(*rows):
    return json.dumps({"notes": [{"sentence": s, "passage": p, "quote": q} for s, p, q in rows]})


def first_words(text, count=3):
    return " ".join(text.split()[:count])


def levels_of(chunks, fan_in, max_levels=5):
    """The groups the builder will form, level by level, as (ids, texts) --
    the same arithmetic the builder uses, mirrored so the script can cite
    what each group holds. A lone remainder is carried, not grouped."""
    below = [(f"chunk:{c.chunk_id}", c.text) for c in chunks]
    plan = []
    level = 0
    while len(below) > 1 and level < max_levels:
        level += 1
        groups, written = [], []
        for start in range(0, len(below), fan_in):
            group = below[start:start + fan_in]
            if len(group) == 1 and start > 0:
                written.append(group[0])
                continue
            groups.append(group)
            text = f"Level {level} group {len(groups) - 1} says {first_words(group[0][1])}."
            written.append((f"node:{level}:{len(groups) - 1}", text))
        plan.append(groups)
        below = written
    return plan


def script(plan):
    """One reply per group: a sentence citing the group's first member with its first words."""
    return [notes((f"Level {level} group {index} says {first_words(group[0][1])}.", group[0][0], first_words(group[0][1])))
            for level, groups in enumerate(plan, 1) for index, group in enumerate(groups)]


async def test_the_tree_rises_from_chunks_to_a_root_and_every_sentence_quotes_the_level_below():
    engine, episode_id, chunks = await document()
    try:
        assert len(chunks) >= 5, [c.text for c in chunks]
        plan = levels_of(chunks, 3)
        model = FakeChat(script(plan))
        tree = await build_summary_tree(engine, model, "s", episode_id, fan_in=3)
        assert (tree.chunks, tree.fan_in, tree.levels, tree.unjoined, tree.stored) == (len(chunks), 3, len(plan), False, False)
        assert [(n.level, n.index) for n in tree.nodes] == [(level, index) for level, groups in enumerate(plan, 1) for index in range(len(groups))]
        assert [n.written_from for n in tree.nodes] == [tuple(i for i, _ in group) for groups in plan for group in groups]
        root = tree.root
        assert root is not None and root.covers == tuple(c.chunk_id for c in chunks), "the root covers every chunk"
        held = {f"chunk:{c.chunk_id}": c.text for c in chunks} | {n.id: n.text for n in tree.nodes}
        for node in tree.nodes:
            for sentence in node.synthesis.sentences:
                assert sentence.citations and all(c.quote in held[c.passage_id] for c in sentence.citations), "every sentence quotes what it was written from"
                assert all(c.passage_id in node.written_from for c in sentence.citations)
        assert tree.model_calls == len(script(plan)) and tree.empty_groups == {} and tree.reasons == ()
        assert "the document say" in model.calls[0][1] and chunks[0].text in model.calls[0][1]
        assert tree.record()["nodes"][-1]["id"] == root.id and tree.record()["levels"] == len(plan)
    finally:
        await engine.close()


async def test_a_group_the_model_says_nothing_about_leaves_no_node_and_is_counted():
    engine, episode_id, chunks = await document()
    try:
        plan = levels_of(chunks, 3, max_levels=1)
        groups = plan[0]
        model = FakeChat([json.dumps({"notes": []})] + [notes(("Invented.", group[0][0], "the moon is made of cheese")) for group in groups[1:]])
        silent = await build_summary_tree(engine, model, "s", episode_id, fan_in=3, max_levels=1)
        assert silent.nodes == () and silent.levels == 0 and silent.empty_groups == {1: len(groups)}
        assert "wrote nothing about any group" in silent.reasons[0] and not silent.unjoined
        assert len(model.calls) == len(groups), "a lone remainder is carried, never summarized from itself"
        low = await build_summary_tree(engine, FakeChat(script(plan)), "s", episode_id, fan_in=3, max_levels=1)
        assert low.levels == 1 and len(low.nodes) == len(groups) and low.root is None
        assert low.unjoined and "remain at level 1" in low.reasons[0]
    finally:
        await engine.close()


async def test_stored_summaries_are_notes_that_say_what_they_are_and_notice_a_changed_document():
    engine, episode_id, chunks = await document()
    try:
        plan = levels_of(chunks, 3)
        tree = await build_summary_tree(engine, FakeChat(script(plan)), "s", episode_id, fan_in=3, store=True, model_name="fake-4b")
        assert tree.stored and all(n.episode_id for n in tree.nodes)
        root = await engine.episode("s", tree.root.episode_id)
        assert root.kind == "note" and root.source == f"town.md#summary/{tree.levels}/0"
        assert root.metadata["summary_of"] == str(episode_id) and root.metadata["summary_level"] == str(tree.levels)
        assert root.metadata["summary_covers"] == ",".join(str(c.chunk_id) for c in chunks)
        assert json.loads(root.metadata["summary_citations"]) == [[c.passage_id, c.quote] for s in tree.root.synthesis.sentences for c in s.citations]
        assert root.metadata["summary_model"] == "fake-4b" and root.metadata["summary_content_hash"] == tree.content_hash
        found = await engine.recall("s", f"Level {tree.levels} group 0 says", limit=3)
        assert any(item.episode_id == root.episode_id for item in found.items), "a summary is retrieved like any note"
        stored = await stored_summaries(engine, "s", episode_id)
        assert [(s.level, s.index) for s in stored][:1] == [(tree.levels, 0)] and not any(s.stale for s in stored)
        assert len(stored) == len(tree.nodes)
        again = await build_summary_tree(engine, FakeChat(script(plan)), "s", episode_id, fan_in=3, store=True)
        assert [n.episode_id for n in again.nodes] == [n.episode_id for n in tree.nodes], "the same tree is the same notes, not more"
        changed = await engine.remember("s", "\n\n".join(PARTS) + "\n\nA new closing line.", kind="file", source="town.md")
        assert changed.episode_id != episode_id and await stored_summaries(engine, "s", changed.episode_id) == ()
        # A summary whose recorded hash is not the document's is stale: the
        # document is what it was, so the hash on the note is what moves.
        stale_note = await engine.episode("s", tree.root.episode_id)
        await engine.remember("s", stale_note.content + " (written for an earlier text)", kind="note", source=stale_note.source,
                              metadata={**stale_note.metadata, "summary_content_hash": "0" * 64},
                              dedup_key=f"summary:{episode_id}:{tree.content_hash}:{tree.levels}:0", replace=True)
        assert [s.stale for s in await stored_summaries(engine, "s", episode_id)][0] is True
    finally:
        await engine.close()


async def test_bounds_are_stated_and_bad_asks_are_refused(monkeypatch):
    engine, episode_id, chunks = await document()
    try:
        for bad in (dict(fan_in=1), dict(fan_in=module.MAX_FAN_IN + 1), dict(max_levels=0),
                    dict(limits=SynthesisLimits(max_passages=2), fan_in=3)):
            with pytest.raises(InvalidInput):
                await build_summary_tree(engine, FakeChat(), "s", episode_id, **bad)
        with pytest.raises(NotFound):
            await build_summary_tree(engine, FakeChat(), "s", 999_999)
        monkeypatch.setattr(module, "MAX_CHUNKS", 3)
        with pytest.raises(InvalidInput, match="at most 3 chunks"):
            await build_summary_tree(engine, FakeChat(), "s", episode_id)
    finally:
        await engine.close()


async def test_the_routes_and_the_command_need_a_model_and_store_the_tree():
    engine, episode_id, chunks = await document()
    try:
        plan = levels_of(chunks, 6)
        replies = script(plan)
        with TestClient(create_app(engine, {"key-a": "s"})) as client:
            refused = client.post(f"/v1/episodes/{episode_id}/summaries", headers={"Authorization": "Bearer key-a"})
            assert refused.status_code == 501 and "no synthesis model" in refused.json()["error"]
        with TestClient(create_app(engine, {"key-a": "s"}, synthesis_factory=lambda: FakeChat(list(replies)))) as client:
            made = client.post(f"/v1/episodes/{episode_id}/summaries", headers={"Authorization": "Bearer key-a"})
            assert made.status_code == 200, made.text
            body = made.json()
            assert body["stored"] and body["levels"] == len(plan) and body["nodes"][-1]["episode_id"]
            listed = client.get(f"/v1/episodes/{episode_id}/summaries", headers={"Authorization": "Bearer key-a"}).json()
            assert [s["level"] for s in listed["summaries"]][0] == len(plan) and not listed["summaries"][0]["stale"]
        out = io.StringIO()
        with pytest.raises(InvalidInput, match="SCONE_CHAT_URL"):
            await run(build_parser().parse_args(["summarize", str(episode_id), "--dry-run"]), engine, io.StringIO(""), out)
    finally:
        await engine.close()
