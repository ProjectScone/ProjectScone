"""A summary hit, expanded to the chunks it cites: only those, never a forgotten one, and always said where it came from."""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.app import create_app
from scone_memory.core.errors import InvalidInput, SconeError
from scone_memory.core.ports import TextFilter
from scone_memory.providers.llm import FakeChat
from scone_memory.retrieval import summary_expand as module
from scone_memory.retrieval.summary_expand import expand_summaries
from scone_memory.retrieval.summary_tree import _hash, build_summary_tree, summary_key
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


async def document(**options):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=100, **options).open()
    added = await engine.remember("s", "\n\n".join(PARTS), kind="file", source="town.md")
    return engine, added.episode_id, await engine.documents.chunks_of("s", added.episode_id)


def reply(*rows):
    return json.dumps({"notes": [{"sentence": s, "passage": p, "quote": q} for s, p, q in rows]})


def opening(text, count=3):
    return " ".join(text.split()[:count])


def sentence_about(level, group, member, text):
    return f"Zorvath {level}.{group}.{member} recounts {opening(text)}."


def replies_for(chunks, fan_in=3, cite=2, max_levels=5):
    """One reply per group the builder forms, level by level (the builder's
    own arithmetic, mirrored): at level one a sentence for each of the first
    ``cite`` members quoting its opening words; above, a sentence quoting
    the whole first sentence of the first ``cite`` members, so a citation
    rests on one sentence of the node below and not on all of it."""
    below = [(f"chunk:{c.chunk_id}", c.text, opening(c.text)) for c in chunks]
    replies = []
    level = 0
    while len(below) > 1 and level < max_levels:
        level += 1
        written = []
        position = 0
        for start in range(0, len(below), fan_in):
            group = below[start:start + fan_in]
            if len(group) == 1 and start > 0:
                written.append(group[0])
                continue
            rows = [(sentence_about(level, position, m, text), ident, quote)
                    for m, (ident, text, quote) in enumerate(group[:cite])]
            replies.append(reply(*rows))
            node_text = " ".join(row[0] for row in rows)
            written.append((f"node:{level}:{position}", node_text, rows[0][0]))
            position += 1
        below = written
    return replies


async def summary_items(engine, level, index, source="town.md"):
    """The recall items of one stored summary, found by its source."""
    found = await engine.recall("s", f"Zorvath {level}.{index}", limit=50, kind="note",
                                source_prefix=f"{source}#summary/{level}/{index}")
    assert found.items, f"summary {level}/{index} was not recalled"
    return found.items


async def test_a_level_one_summary_is_followed_by_the_chunks_it_cites_and_only_those():
    engine, episode_id, chunks = await document()
    try:
        await build_summary_tree(engine, FakeChat(replies_for(chunks, max_levels=1)), "s", episode_id,
                                 fan_in=3, max_levels=1, store=True)
        hit = await summary_items(engine, 1, 0)
        summary_episode = hit[0].episode_id
        followed = await expand_summaries(engine, "s", hit, mode="follow")
        ids = [item.chunk_id for item in followed.items]
        assert ids == [hit[0].chunk_id, chunks[0].chunk_id, chunks[1].chunk_id], "the two cited chunks, in document order"
        assert chunks[2].chunk_id not in ids, "a chunk the summary covers but does not cite is not served"
        added = followed.items[1:]
        assert all(item.episode_id == episode_id and item.source == "town.md" for item in added)
        assert [item.text for item in added] == [chunks[0].text, chunks[1].text]
        assert [(item.start, item.end) for item in added] == [(chunks[0].start, chunks[0].end), (chunks[1].start, chunks[1].end)]
        assert all(item.score == hit[0].score and item.lanes == {} for item in added)
        via = added[0].via_summary
        assert via is not None and via["episode_id"] == summary_episode and via["chunk_id"] == hit[0].chunk_id
        assert (via["level"], via["index"], via["summary_of"], via["mode"]) == (1, 0, episode_id, "follow")
        quoted = opening(chunks[0].text)
        assert via["cited"] == [[0, len(quoted)]] and chunks[0].text[0:len(quoted)] == quoted, \
            "where in the chunk's own text the quote the summary rests on sits"
        assert quoted not in json.dumps(via), "offsets, never a second copy of the passage's text"
        assert followed.items[0].via_summary is None, "the summary itself did not come through a summary"
        assert (followed.summaries, followed.expanded, followed.chunks_added, followed.capped) == (1, 1, 2, 0)
        assert followed.refused == () and "expanded 1 of 1 summary hit(s) to 2 cited chunk(s)" in followed.why
        record = followed.record()
        assert record["mode"] == "follow" and record["chunks_added"] == 2 and record["by_summary"] == {
            str(summary_episode): [chunks[0].chunk_id, chunks[1].chunk_id]}

        replaced = await expand_summaries(engine, "s", hit, mode="replace")
        assert [item.chunk_id for item in replaced.items] == [chunks[0].chunk_id, chunks[1].chunk_id]
        assert all(item.via_summary["mode"] == "replace" for item in replaced.items)
    finally:
        await engine.close()


async def test_a_higher_summary_resolves_down_to_the_sentences_its_quotes_rest_on():
    engine, episode_id, chunks = await document()
    try:
        assert len(chunks) == 7, [c.text for c in chunks]
        # Seven chunks at fan_in 3: level one writes [1,2,3] and [4,5,6] and
        # carries the seventh; the root is written from node:1:0, node:1:1
        # and chunk 7, and cites the first two of them.
        tree = await build_summary_tree(engine, FakeChat(replies_for(chunks)), "s", episode_id, fan_in=3, store=True)
        assert tree.levels == 2 and tree.root is not None
        root = await summary_items(engine, 2, 0)
        opened = await expand_summaries(engine, "s", root, mode="replace")
        # node:1:0's first sentence cites chunk 1 only; node:1:1's first
        # sentence cites chunk 4 only. The second sentences' chunks (2, 5)
        # sit under the root and are not what it rests on.
        assert [item.chunk_id for item in opened.items] == [chunks[0].chunk_id, chunks[3].chunk_id]
        assert opened.reads == 5, "the root's account, the two level-one nodes it cites, and their accounts"
        assert opened.items[0].via_summary["level"] == 2 and opened.items[0].via_summary["episode_id"] == root[0].episode_id

        # The root and a level-one summary of the same document in one answer:
        # the document, its stored nodes and each account are read once, and a
        # chunk both cite is served once.
        reads = {"episode": 0, "by_key": 0, "walked": 0}
        real_episode, real_by_key, real_walk = engine.episode, engine.episode_by_key, engine.documents.recent_episodes

        async def episode(space, wanted):
            reads["episode"] += 1
            return await real_episode(space, wanted)

        async def by_key(space, key):
            reads["by_key"] += 1
            return await real_by_key(space, key)

        async def walked(*args, **kwargs):
            reads["walked"] += 1
            return await real_walk(*args, **kwargs)

        first = await summary_items(engine, 1, 0)
        engine.episode, engine.episode_by_key, engine.documents.recent_episodes = episode, by_key, walked
        try:
            both = await expand_summaries(engine, "s", [*root, *first], mode="follow")
        finally:
            engine.episode, engine.episode_by_key, engine.documents.recent_episodes = real_episode, real_by_key, real_walk
        assert [item.chunk_id for item in both.items] == [root[0].chunk_id, chunks[0].chunk_id, chunks[3].chunk_id,
                                                          first[0].chunk_id, chunks[1].chunk_id]
        # A node is one keyed read, never a walk over the space's episodes.
        assert both.already_present == 1 and both.reads == 5 and reads == {"episode": 1, "by_key": 2, "walked": 0}
    finally:
        await engine.close()


async def test_a_carried_chunk_cited_from_above_is_served_directly():
    engine, episode_id, chunks = await document()
    try:
        replies = replies_for(chunks, cite=2)
        # The root's reply: cite node:1:0's first sentence and the carried seventh chunk.
        node_text = sentence_about(1, 0, 0, chunks[0].text)
        replies[-1] = reply(("The root about the first node.", "node:1:0", node_text),
                            ("The root about the carried chunk.", f"chunk:{chunks[6].chunk_id}", opening(chunks[6].text)))
        await build_summary_tree(engine, FakeChat(replies), "s", episode_id, fan_in=3, store=True)
        opened = await expand_summaries(engine, "s", await summary_items(engine, 2, 0), mode="replace")
        assert [item.chunk_id for item in opened.items] == [chunks[0].chunk_id, chunks[6].chunk_id]
    finally:
        await engine.close()


FORGERIES: list[dict] = []


async def forged_summary(engine, episode_id, chunks, **overrides):
    """A note that says it summarizes ``episode_id``, as a migrated store or
    a hand-written note could: the metadata and account are ours to spoil."""
    source = await engine.episode("s", episode_id)
    keyed = overrides.pop("keyed", False)
    account = overrides.pop("account", {"sentences": [
        {"text": "Forged.", "citations": [{"passage": f"chunk:{chunks[0].chunk_id}", "quote": opening(chunks[0].text),
                                           "start": 0, "end": len(opening(chunks[0].text))}]}]})
    raw = overrides.pop("raw", None) or json.dumps(account).encode()
    detail = await engine.attach("s", raw, media_type="application/json", filename="summary-node.json")
    metadata = {"summary_of": str(episode_id), "summary_level": "1", "summary_index": "0", "summary_fan_in": "3",
                "summary_chunks": "3", "summary_first_chunk": str(chunks[0].chunk_id),
                "summary_last_chunk": str(chunks[2].chunk_id), "summary_content_hash": _hash(source.content),
                "summary_model": "hand", "summary_detail": detail.attachment_id} | overrides
    # Each forgery its own text, or remembering it again would be a duplicate of the first
    # (the attachment is named by its bytes, so it cannot tell two forgeries apart).
    FORGERIES.append(overrides)
    # ``keyed`` stores it under the key a built node of that place carries, so a summary above can find it.
    key = summary_key(episode_id, _hash(source.content), int(metadata["summary_fan_in"]), int(metadata["summary_level"]),
                      int(metadata["summary_index"])) if keyed else None
    added = await engine.remember("s", f"Xylocarp forged summary of the harbour, number {len(FORGERIES)}.", kind="note",
                                  source="forged#summary/1/0", metadata=metadata, attachment_ids=(detail.attachment_id,),
                                  dedup_key=key)
    found = await engine.recall("s", "Xylocarp forged summary", limit=20, kind="note", source_prefix="forged#")
    return [item for item in found.items if item.episode_id == added.episode_id]


async def test_a_summary_whose_document_changed_or_went_is_refused_and_stands_as_it_was():
    engine, episode_id, chunks = await document()
    try:
        honest = await forged_summary(engine, episode_id, chunks)
        assert [item.chunk_id for item in (await expand_summaries(engine, "s", honest)).items][1:] == [chunks[0].chunk_id]

        stale = await forged_summary(engine, episode_id, chunks, summary_content_hash="0" * 64)
        refused = await expand_summaries(engine, "s", stale, mode="replace")
        assert refused.items == tuple(stale), "a refused summary stands; nothing is served in its place"
        assert refused.expanded == 0 and [r["reason"] for r in refused.refused] == ["content_changed"]
        assert refused.refused[0]["episode_id"] == stale[0].episode_id and "no longer matches" in refused.why

        unreadable = await forged_summary(engine, episode_id, chunks, summary_detail="nope")
        assert [r["reason"] for r in (await expand_summaries(engine, "s", unreadable)).refused] == ["detail_unreadable"]
        malformed = await forged_summary(engine, episode_id, chunks, summary_of="the harbour")
        assert [r["reason"] for r in (await expand_summaries(engine, "s", malformed)).refused] == ["malformed"]
        # A document id that never existed here -- a note carried from another store -- is not a forgotten one.
        unknown = await forged_summary(engine, episode_id, chunks, summary_of="987654")
        unknown_too = await forged_summary(engine, episode_id, chunks, summary_of="987654", summary_index="1")
        stranger = await expand_summaries(engine, "s", [*unknown, *unknown], mode="replace")
        assert [r["reason"] for r in stranger.refused] == ["source_unknown"] and stranger.items == (*unknown, *unknown)
        assert "not stored here" in stranger.why and "forgotten" not in stranger.why

        from scone_memory.core.errors import SconeError

        real_episode = engine.episode

        async def failing(space, wanted):
            if wanted == episode_id:
                raise SconeError("the store did not answer")
            return await real_episode(space, wanted)

        engine.episode = failing
        unread = await expand_summaries(engine, "s", honest, mode="replace")
        engine.episode = real_episode
        assert unread.items == tuple(honest) and [r["reason"] for r in unread.refused] == ["unread"]
        assert "could not be read" in unread.why and "forgotten" not in unread.why

        await engine.forget("s", episode_id)
        gone = await expand_summaries(engine, "s", honest, mode="replace")
        assert gone.items == tuple(honest) and gone.chunks_added == 0, "a forgotten document's chunks are never served"
        assert [r["reason"] for r in gone.refused] == ["source_gone"] and "forgotten" in gone.why
        asked = []

        async def counted(space, wanted):
            asked.append(wanted)
            return await real_episode(space, wanted)

        engine.episode = counted
        twice = await expand_summaries(engine, "s", [*honest, *stale, *unknown, *unknown_too], mode="replace")
        engine.episode = real_episode
        assert [r["reason"] for r in twice.refused] == ["source_gone", "source_gone", "source_unknown", "source_unknown"]
        assert twice.chunks_added == 0 and asked == [episode_id, 987654], "a refused document is read once, and keeps its reason"
    finally:
        await engine.close()


async def test_a_cited_chunk_that_is_not_the_documents_or_does_not_hold_the_quote_is_not_served():
    engine, episode_id, chunks = await document()
    try:
        other = await engine.remember("s", "An unrelated note about nothing in the town at all.", kind="note")
        other_chunk = (await engine.documents.chunks_of("s", other.episode_id))[0]

        def cite(chunk_id, quote):
            return {"passage": f"chunk:{chunk_id}", "quote": quote, "start": 0, "end": len(quote)}

        account = {"sentences": [{"text": "Mixed.", "citations": [
            cite(chunks[1].chunk_id, opening(chunks[1].text)),
            cite(other_chunk.chunk_id, opening(other_chunk.text)),
            cite(999_999, "anything"),
            cite(chunks[2].chunk_id, "words this chunk never held"),
            {"passage": "node:1:9", "quote": "no such node", "start": 0, "end": 12},
            {"passage": "elsewhere", "quote": "x", "start": 0, "end": 1},
            {"passage": f"chunk:{chunks[3].chunk_id}"},
            {"passage": "chunk:three", "quote": "x", "start": 0, "end": 1},
            cite(chunks[0].chunk_id, opening(chunks[0].text)),
            {"passage": f"chunk:{chunks[1].chunk_id}", "quote": chunks[1].text[10:30], "start": 10, "end": 30}]}]}
        hit = await forged_summary(engine, episode_id, chunks, account=account)
        opened = await expand_summaries(engine, "s", hit, mode="follow")
        assert [item.chunk_id for item in opened.items][1:] == [chunks[0].chunk_id, chunks[1].chunk_id], \
            "served in the document's order, not the order they were cited in"
        assert opened.items[2].via_summary["cited"] == [[0, len(opening(chunks[1].text))], [10, 30]]
        assert (opened.missing, opened.unquoted, opened.unresolved) == (2, 1, 4)
        assert "not among the document's chunks" in opened.why and "does not hold the quote" in opened.why

        nothing = await forged_summary(engine, episode_id, chunks, account={"sentences": [{"text": "None.", "citations": [
            cite(999_999, "anything")]}]})
        empty = await expand_summaries(engine, "s", nothing, mode="replace")
        assert empty.items == tuple(nothing) and [r["reason"] for r in empty.refused] == ["nothing_cited"]
        for raw in (b"not json at all", json.dumps({"sentences": ["a sentence with no citations"]}).encode()):
            spoiled = await forged_summary(engine, episode_id, chunks, raw=raw)
            assert [r["reason"] for r in (await expand_summaries(engine, "s", spoiled)).refused] == ["detail_unreadable"]
    finally:
        await engine.close()


async def test_a_node_cited_from_above_is_checked_like_a_chunk_and_never_followed_sideways():
    engine, episode_id, chunks = await document()
    try:
        def cite(chunk):
            return {"passage": f"chunk:{chunk.chunk_id}", "quote": opening(chunk.text), "start": 0, "end": len(opening(chunk.text))}

        def node(level, index, quote="Xylocarp"):
            return {"passage": f"node:{level}:{index}", "quote": quote, "start": 0, "end": len(quote)}

        def one(*citations):
            return {"sentences": [{"text": "Forged.", "citations": list(citations)}]}

        await forged_summary(engine, episode_id, chunks, summary_index="0", summary_content_hash="0" * 64, account=one(cite(chunks[0])),
                             keyed=True)
        await forged_summary(engine, episode_id, chunks, summary_index="1", account=one(cite(chunks[1])), keyed=True)
        await forged_summary(engine, episode_id, chunks, summary_index="2", summary_detail="nope", keyed=True)
        # A note at 1/3 saying what a node says, but not stored under a node's key, is not found.
        await forged_summary(engine, episode_id, chunks, summary_index="3", account=one(cite(chunks[2])))
        top = await forged_summary(engine, episode_id, chunks, summary_level="2", summary_index="0",
                                   account=one(node(1, 0), node(1, 1, "Wrongful"), node(1, 1), node(1, 2), node(1, 3),
                                               {"passage": "page:1:1", "quote": "Xylocarp", "start": 0, "end": 8}))
        opened = await expand_summaries(engine, "s", top, mode="replace")
        # node:1:0 was written from other content; the first node:1:1 quote is
        # not its text; node:1:2's account cannot be read; node:1:3 is not stored
        # under a node's key; a page is not a node. Only node:1:1 is followed,
        # and a summary resting partly on citations that failed is not replaced.
        assert [item.chunk_id for item in opened.items] == [top[0].chunk_id, chunks[1].chunk_id] and opened.unresolved == 5
        assert opened.partial == (top[0].episode_id,) and opened.expanded == 1
        # The top account, four node lookups (node:1:1 once, though cited twice), and the two accounts of nodes that held.
        assert opened.reads == 7
    finally:
        await engine.close()

    engine, episode_id, chunks = await document()
    try:
        # A summary citing a node at its own level -- here, itself -- is not followed round in a circle.
        itself = await forged_summary(engine, episode_id, chunks, keyed=True, account={"sentences": [{"text": "Forged.", "citations": [
            {"passage": "node:1:0", "quote": "Xylocarp", "start": 0, "end": 8}]}]})
        circle = await expand_summaries(engine, "s", itself, mode="replace")
        assert circle.items == tuple(itself) and circle.unresolved == 1 and [r["reason"] for r in circle.refused] == ["nothing_cited"]
    finally:
        await engine.close()


async def test_the_cap_cuts_and_says_so_and_a_cut_summary_is_not_replaced():
    engine, episode_id, chunks = await document()
    try:
        await build_summary_tree(engine, FakeChat(replies_for(chunks, cite=3, max_levels=1)), "s", episode_id,
                                 fan_in=3, max_levels=1, store=True)
        first, second = await summary_items(engine, 1, 0), await summary_items(engine, 1, 1)
        both = [*first, *second]
        cut = await expand_summaries(engine, "s", both, mode="replace", max_chunks=4)
        ids = [item.chunk_id for item in cut.items]
        assert ids == [chunks[0].chunk_id, chunks[1].chunk_id, chunks[2].chunk_id,
                       second[0].chunk_id, chunks[3].chunk_id], "the cut summary stands, followed by what fit"
        assert (cut.chunks_added, cut.capped, cut.cut) == (4, 2, (second[0].episode_id,))
        assert "the cap of 4 chunk(s) left 2 cited chunk(s) out" in cut.why
        assert "stand beside the part of what they cite that fit" in cut.why and cut.expanded == 2
        full = await expand_summaries(engine, "s", both, mode="replace", max_chunks=3)
        assert [item.chunk_id for item in full.items][3:] == [second[0].chunk_id] and full.cut == (second[0].episode_id,)
        # A summary the cap left no room for added nothing, so it was not expanded, and says so.
        assert (full.expanded, full.chunks_added, full.capped) == (1, 3, 3)
        assert full.by_summary == {first[0].episode_id: (chunks[0].chunk_id, chunks[1].chunk_id, chunks[2].chunk_id)}
        assert "expanded 1 of 2 summary hit(s)" in full.why and "1 summary hit(s) stand as they were" in full.why
        assert "beside the part" not in full.why
        whole = await expand_summaries(engine, "s", both, mode="replace", max_chunks=6)
        assert whole.capped == 0 and whole.cut == () and "cap" not in whole.why
        for bad in (0, module.MAX_CHUNKS + 1, True, 2.5):
            with pytest.raises(InvalidInput):
                await expand_summaries(engine, "s", both, max_chunks=bad)
        with pytest.raises(InvalidInput):
            await expand_summaries(engine, "s", both, mode="whole")
    finally:
        await engine.close()


async def test_the_read_budget_leaves_later_summaries_as_they_were_and_counts_them(monkeypatch):
    engine, episode_id, chunks = await document()
    try:
        await build_summary_tree(engine, FakeChat(replies_for(chunks)), "s", episode_id, fan_in=3, store=True)
        root = await summary_items(engine, 2, 0)
        monkeypatch.setattr(module, "MAX_READS", 2)
        short = await expand_summaries(engine, "s", root, mode="replace")
        assert short.items == tuple(root) and short.not_read == 1 and short.reads == 2
        assert "budget of 2 read(s)" in short.why
        # The root's account, two node lookups, one node account: the fourth read is not enough.
        monkeypatch.setattr(module, "MAX_READS", 4)
        assert (await expand_summaries(engine, "s", root, mode="replace")).not_read == 1
        monkeypatch.setattr(module, "MAX_READS", 5)
        assert (await expand_summaries(engine, "s", root, mode="replace")).not_read == 0
    finally:
        await engine.close()


async def test_a_cited_chunk_already_in_the_answer_is_not_repeated_and_a_summary_expands_once():
    engine, episode_id, chunks = await document()
    try:
        await build_summary_tree(engine, FakeChat(replies_for(chunks, max_levels=1)), "s", episode_id,
                                 fan_in=3, max_levels=1, store=True)
        hit = await summary_items(engine, 1, 0)
        direct = await engine.recall("s", PARTS[1], limit=1, kind="file")
        assert direct.items[0].chunk_id == chunks[1].chunk_id
        opened = await expand_summaries(engine, "s", [*direct.items, *hit, *hit], mode="follow")
        assert [item.chunk_id for item in opened.items] == [chunks[1].chunk_id, hit[0].chunk_id, chunks[0].chunk_id, hit[0].chunk_id]
        assert opened.already_present == 1 and opened.items[0].via_summary is None and opened.summaries == 1
        replaced = await expand_summaries(engine, "s", [*hit, *hit], mode="replace")
        assert [item.chunk_id for item in replaced.items] == [chunks[0].chunk_id, chunks[1].chunk_id], \
            "a replaced summary's other occurrences go with it"
    finally:
        await engine.close()


async def test_recall_expands_on_request_and_answers_as_before_without_it():
    engine, episode_id, chunks = await document()
    try:
        await build_summary_tree(engine, FakeChat(replies_for(chunks, max_levels=1)), "s", episode_id,
                                 fan_in=3, max_levels=1, store=True)
        plain = await engine.recall("s", "Zorvath 1.0 recounts", limit=3)
        assert plain.expanded is None and all(item.via_summary is None for item in plain.items)
        dumped = json.dumps(plain.model_dump(mode="json"))
        assert "via_summary" not in dumped and '"expanded"' not in dumped
        opened = await engine.recall("s", "Zorvath 1.0 recounts", limit=3, expand_summaries="follow", expand_max_chunks=1)
        assert opened.expanded is not None and opened.expanded["mode"] == "follow" and opened.expanded["max_chunks"] == 1
        assert any(item.via_summary for item in opened.items) and opened.expanded["capped"] >= 1
        assert opened.returned_bytes == sum(len(item.text.encode()) for item in opened.items)
        import scone_memory.memory.engine as engine_module

        def no_search(*args, **kwargs):
            raise AssertionError("a mistaken request is refused before anything is searched")

        real_recall, engine_module.recall = engine_module.recall, no_search
        try:
            with pytest.raises(InvalidInput):
                await engine.recall("s", "Zorvath", expand_summaries="whole")
            with pytest.raises(InvalidInput):
                await engine.recall("s", "Zorvath", expand_summaries="follow", expand_max_chunks=0)
        finally:
            engine_module.recall = real_recall
        with pytest.raises(InvalidInput, match="expand_max_chunks"):
            await engine.recall("s", "Zorvath", expand_max_chunks=3)

        with TestClient(create_app(engine, {"key-a": "s"})) as client:
            auth = {"Authorization": "Bearer key-a"}
            body = client.get("/v1/recall", params={"q": "Zorvath 1.0 recounts", "limit": 3, "expand_summaries": "replace"},
                              headers=auth).json()
            assert body["expanded"]["mode"] == "replace" and any("via_summary" in item for item in body["items"])
            bare = client.get("/v1/recall", params={"q": "Zorvath 1.0 recounts", "limit": 3}, headers=auth).json()
            assert "expanded" not in bare and all("via_summary" not in item for item in bare["items"])
            assert '"recall.expand_summaries": true' in json.dumps(client.get("/v1/capabilities", headers=auth).json())
            assert client.get("/v1/recall", params={"q": "x", "expand_summaries": "whole"}, headers=auth).status_code in (400, 422)
            merged = client.get("/v1/recall", params={"q": "Zorvath", "expand_summaries": "follow", "merge": "true"}, headers=auth)
            assert merged.status_code in (400, 422) and "merge" in merged.text

        out = io.StringIO()
        code = await run(build_parser().parse_args(["--space", "s", "recall", "Zorvath 1.0 recounts", "--limit", "3",
                                                    "--expand-summaries", "follow", "--json"]), engine, io.StringIO(""), out)
        said = json.loads(out.getvalue())
        assert code == 0 and said["expanded"]["mode"] == "follow" and any(item.get("via_summary") for item in said["items"])
        out = io.StringIO()
        await run(build_parser().parse_args(["--space", "s", "recall", "Zorvath 1.0 recounts", "--limit", "3", "--expand-summaries", "replace",
                                             "--expand-max-chunks", "1"]), engine, io.StringIO(""), out)
        assert "summary hit(s) to" in out.getvalue() and "via summary #" in out.getvalue()
        for clash in ("--merge", "--parts"):
            with pytest.raises(InvalidInput, match="--expand-summaries cannot be combined"):
                await run(build_parser().parse_args(["--space", "s", "recall", "Zorvath", "--expand-summaries", "follow", clash]),
                          engine, io.StringIO(""), io.StringIO())
    finally:
        await engine.close()


async def test_the_phrases_a_recall_was_given_hold_for_what_a_summary_brings():
    engine, episode_id, chunks = await document()
    try:
        await build_summary_tree(engine, FakeChat(replies_for(chunks, max_levels=1)), "s", episode_id,
                                 fan_in=3, max_levels=1, store=True)
        assert "foundations" in chunks[1].text and "foundations" not in chunks[0].text
        opened = await engine.recall("s", "Zorvath 1.0 recounts", limit=10, exclude=["foundations"], expand_summaries="follow")
        assert opened.expanded["chunks_added"] >= 1 and opened.expanded["dropped_excluded"] == 1
        assert all("foundations" not in item.text for item in opened.items), "a chunk the phrases exclude does not come back through a summary"

        hit = await summary_items(engine, 1, 0)
        excluded = await expand_summaries(engine, "s", hit, mode="replace", exclude=["foundations"])
        assert [item.chunk_id for item in excluded.items] == [hit[0].chunk_id, chunks[0].chunk_id], \
            "part of what it cites does not stand for all of it, so the summary stays"
        assert (excluded.dropped_excluded, excluded.dropped_required, excluded.expanded) == (1, 0, 1)
        assert "1 cited chunk(s) were dropped by the phrases" in excluded.why
        assert excluded.partial == (hit[0].episode_id,) and "could not be followed" not in excluded.why
        assert excluded.record()["dropped_excluded"] == 1

        required = await expand_summaries(engine, "s", hit, mode="replace", require=["Bramley"])
        assert required.items == tuple(hit) and (required.dropped_required, required.expanded, required.chunks_added) == (2, 0, 0)
        assert required.record()["dropped_required"] == 2 and required.by_summary == {}
        assert "2 cited chunk(s) were dropped by the phrases" in required.why

        both = await expand_summaries(engine, "s", hit, mode="replace", require=["harbour"])
        assert [item.chunk_id for item in both.items] == [hit[0].chunk_id, chunks[0].chunk_id] and both.dropped_required == 1
        with pytest.raises(InvalidInput):
            await expand_summaries(engine, "s", hit, exclude="foundations")
    finally:
        await engine.close()


async def test_the_recall_event_lists_what_expansion_returned_so_feedback_reaches_it():
    engine, episode_id, chunks = await document(events=InMemoryEventLog())
    try:
        await build_summary_tree(engine, FakeChat(replies_for(chunks, max_levels=1)), "s", episode_id,
                                 fan_in=3, max_levels=1, store=True)
        plain = await engine.recall("s", "Zorvath 1.0 recounts", limit=3)
        summaries = [item.chunk_id for item in plain.items if "summary_of" in item.metadata]
        result = await engine.recall("s", "Zorvath 1.0 recounts", limit=3, expand_summaries="replace")
        event = await engine.events.get("s", result.event_id)
        assert [one["chunk_id"] for one in event.payload["items"]] == [item.chunk_id for item in result.items]
        assert event.payload["returned_bytes"] == result.returned_bytes and event.payload["expanded"] == result.expanded
        assert [one.get("via_summary") for one in event.payload["items"]] == [
            item.via_summary["episode_id"] if item.via_summary else None for item in result.items]
        brought = [item.chunk_id for item in result.items if item.via_summary]
        replaced = [chunk_id for chunk_id in summaries if chunk_id not in {item.chunk_id for item in result.items}]
        assert brought and replaced
        await engine.feedback("s", result.event_id, brought[0], True)
        with pytest.raises(InvalidInput, match="not returned"):
            await engine.feedback("s", result.event_id, replaced[0], True)

        # Whether the phrases left the answer short is about what the lanes returned, not what a summary added.
        short = await engine.recall("s", "Zorvath 1.0 recounts", limit=3, exclude=["harbour"], candidate_limit=2,
                                    expand_summaries="follow")
        assert short.expanded["chunks_added"] >= 2 and len(short.items) >= 3
        assert short.phrases is not None and short.phrases.short and "1 of 3 came back" in short.phrases.why
    finally:
        await engine.close()


CODE = '''def harbour_depth(tide):
    """Metres of water at the quay when the tide is in."""
    return 4.5 + tide


def lighthouse_range(height):
    """Nautical miles the rebuilt light carries on a clear night."""
    return 1.17 * height ** 0.5


def orchard_rows(trees):
    """Rows of forty Bramley trees on the ridge above the town."""
    return trees // 40
'''


async def test_a_chunk_brought_by_a_summary_says_where_in_the_file_it_is_as_a_direct_hit_does():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=100).open()
    try:
        added = await engine.remember("s", CODE, kind="file", source="mill.py")
        chunks = await engine.documents.chunks_of("s", added.episode_id)
        assert len(chunks) == 3
        # A code chunk's opening words run across a line break, so each sentence quotes a first line whole.
        firsts = [chunk.text.splitlines()[0] for chunk in chunks]
        written = reply(*[(sentence_about(1, 0, m, firsts[m]), f"chunk:{chunks[m].chunk_id}", firsts[m]) for m in range(2)])
        await build_summary_tree(engine, FakeChat([written]), "s", added.episode_id, fan_in=3, max_levels=1, store=True)
        opened = await expand_summaries(engine, "s", await summary_items(engine, 1, 0, source="mill.py"), mode="replace")
        assert [item.chunk_id for item in opened.items] == [chunks[0].chunk_id, chunks[1].chunk_id]
        for item in opened.items:
            direct = (await engine.recall("s", item.text, limit=1, kind="file")).items[0]
            assert direct.chunk_id == item.chunk_id and direct.declaration is not None
            assert (item.first_line, item.last_line, item.declaration) == (direct.first_line, direct.last_line, direct.declaration)
        assert opened.items[1].first_line == 6 and opened.items[1].declaration == "lighthouse_range"
    finally:
        await engine.close()


async def test_a_document_forgotten_while_expansion_reads_serves_nothing_of_it():
    engine, episode_id, chunks = await document()
    try:
        await build_summary_tree(engine, FakeChat(replies_for(chunks, max_levels=1)), "s", episode_id,
                                 fan_in=3, max_levels=1, store=True)
        first, second = await summary_items(engine, 1, 0), await summary_items(engine, 1, 1)
        real = engine.attachment
        read: list[str] = []

        async def attachment(space, attachment_id):
            got = await real(space, attachment_id)
            read.append(attachment_id)
            if len(read) == 2:
                # Another request forgets the document while the second summary's account is read,
                # after the first summary's citations were already followed and checked.
                await engine.forget("s", episode_id)
            return got

        engine.attachment = attachment
        try:
            opened = await expand_summaries(engine, "s", [*first, *second], mode="replace")
        finally:
            engine.attachment = real
        assert await engine.documents.chunks_of("s", episode_id) == []
        assert opened.items == (*first, *second) and opened.chunks_added == 0 and opened.expanded == 0, \
            "no chunk of a document forgotten during the call is served, whichever summary's read it happened in"
        assert [r["reason"] for r in opened.refused] == ["source_gone", "source_gone"] and "forgotten" in opened.why
        assert [r["episode_id"] for r in opened.refused] == [first[0].episode_id, second[0].episode_id]
    finally:
        await engine.close()

    engine, episode_id, chunks = await document()
    try:
        await build_summary_tree(engine, FakeChat(replies_for(chunks, max_levels=1)), "s", episode_id,
                                 fan_in=3, max_levels=1, store=True)
        hit, second = await summary_items(engine, 1, 0), await summary_items(engine, 1, 1)
        real_get = engine.documents.get_chunks

        async def changed(space, ids):
            # One chunk of the document reads differently now; the document is still there.
            return [chunk.model_copy(update={"text": chunk.text.upper()}) if chunk.chunk_id == chunks[0].chunk_id else chunk
                    for chunk in await real_get(space, ids)]

        async def failing(space, ids):
            raise SconeError("the store did not answer")

        engine.documents.get_chunks = changed
        try:
            moved = await expand_summaries(engine, "s", [*hit, *second], mode="replace")
        finally:
            engine.documents.get_chunks = real_get
        assert moved.items == (*hit, *second) and moved.chunks_added == 0, \
            "a document that changed serves none of its chunks, not only the one that moved"
        assert [r["reason"] for r in moved.refused] == ["changed_while_read", "changed_while_read"]
        assert "changed while" in moved.why
        engine.documents.get_chunks = failing
        try:
            unread = await expand_summaries(engine, "s", hit, mode="replace")
        finally:
            engine.documents.get_chunks = real_get
        assert unread.items == tuple(hit) and [r["reason"] for r in unread.refused] == ["unread"]
        assert [item.chunk_id for item in (await expand_summaries(engine, "s", hit, mode="replace")).items] == [
            chunks[0].chunk_id, chunks[1].chunk_id]

        # A second document, forgotten while the first one's reason is read again: the last read
        # before the answer is the one that finds nothing moved.
        mill = await engine.remember("s", "\n\n".join(reversed(PARTS)), kind="file", source="mill.md")
        mill_chunks = await engine.documents.chunks_of("s", mill.episode_id)
        await build_summary_tree(engine, FakeChat(replies_for(mill_chunks, max_levels=1)), "s", mill.episode_id,
                                 fan_in=3, max_levels=1, store=True)
        milled = await summary_items(engine, 1, 0, source="mill.md")
        real_episode = engine.episode
        asked: list[int] = []

        async def episode(space, wanted):
            asked.append(wanted)
            if wanted == episode_id and asked.count(episode_id) == 2:
                await engine.forget("s", mill.episode_id)
            return await real_episode(space, wanted)

        firsts: list[int] = []

        async def changed_once(space, ids):
            firsts.append(1)
            return await (changed if len(firsts) == 1 else real_get)(space, ids)

        engine.episode, engine.documents.get_chunks = episode, changed_once
        try:
            both = await expand_summaries(engine, "s", [*hit, *milled], mode="replace")
        finally:
            engine.episode, engine.documents.get_chunks = real_episode, real_get
        assert both.items == (*hit, *milled) and both.chunks_added == 0
        assert [r["reason"] for r in both.refused] == ["changed_while_read", "source_gone"]
    finally:
        await engine.close()


async def test_what_a_summary_brings_holds_to_the_scope_the_recall_was_given():
    engine, episode_id, chunks = await document()
    try:
        await build_summary_tree(engine, FakeChat(replies_for(chunks, max_levels=1)), "s", episode_id,
                                 fan_in=3, max_levels=1, store=True)
        noted = await engine.recall("s", "Zorvath 1.0 recounts", limit=3, kind="note", expand_summaries="follow")
        assert noted.items and all("summary_of" in item.metadata and item.via_summary is None for item in noted.items), \
            "a recall of notes does not hand back a file's chunks through a summary"
        assert noted.expanded["dropped_scope"] >= 2 and (noted.expanded["chunks_added"], noted.expanded["expanded"]) == (0, 0)
        assert "outside the recall's scope" in noted.expanded["why"]
        prefixed = await engine.recall("s", "Zorvath 1.0 recounts", limit=3, source_prefix="town.md#summary",
                                       expand_summaries="replace")
        assert prefixed.items and all(item.source.startswith("town.md#summary") for item in prefixed.items), \
            "a summary whose chunks are out of scope stands as it was, even under replace"
        assert prefixed.expanded["dropped_scope"] >= 2
        open_scope = await engine.recall("s", "Zorvath 1.0 recounts", limit=3, expand_summaries="follow")
        assert open_scope.expanded["chunks_added"] >= 2 and open_scope.expanded["dropped_scope"] == 0

        hit = await summary_items(engine, 1, 0)
        tagged = await expand_summaries(engine, "s", hit, mode="replace", scope=TextFilter(tags=("elsewhere",)))
        assert tagged.items == tuple(hit) and tagged.dropped_scope == 2 and tagged.record()["dropped_scope"] == 2
        assert [item.chunk_id for item in (await expand_summaries(engine, "s", hit, mode="replace", scope=TextFilter())).items] == [
            chunks[0].chunk_id, chunks[1].chunk_id]
    finally:
        await engine.close()


async def test_a_summary_some_of_whose_citations_fail_is_not_replaced_by_the_rest():
    engine, episode_id, chunks = await document()
    try:
        def cite(chunk, quote=None):
            said = quote or opening(chunk.text)
            return {"passage": f"chunk:{chunk.chunk_id}", "quote": said, "start": 0, "end": len(said)}

        account = {"sentences": [{"text": "A.", "citations": [cite(chunks[0])]},
                                 {"text": "B.", "citations": [cite(chunks[1], "words it never held")]},
                                 {"text": "C.", "citations": [{"passage": "node:0:5", "quote": "x", "start": 0, "end": 1}]}]}
        hit = await forged_summary(engine, episode_id, chunks, account=account)
        kept = await expand_summaries(engine, "s", hit, mode="replace")
        assert [item.chunk_id for item in kept.items] == [hit[0].chunk_id, chunks[0].chunk_id], \
            "part of what a summary cites does not stand for all of it"
        assert (kept.unquoted, kept.unresolved, kept.expanded) == (1, 1, 1) and kept.partial == (hit[0].episode_id,)
        assert kept.record()["partial"] == [hit[0].episode_id]
        assert "1 summary hit(s) rest partly on citations that could not be followed" in kept.why

        honest = await forged_summary(engine, episode_id, chunks, account={"sentences": [account["sentences"][0]]})
        whole = await expand_summaries(engine, "s", honest, mode="replace")
        assert [item.chunk_id for item in whole.items] == [chunks[0].chunk_id] and whole.partial == ()
        assert "could not be followed" not in whole.why
        # In one answer, each summary is judged on its own citations, and one recalled twice is followed once.
        mixed = await expand_summaries(engine, "s", [*hit, *hit, *honest], mode="replace")
        assert [item.chunk_id for item in mixed.items] == [hit[0].chunk_id, chunks[0].chunk_id, hit[0].chunk_id]
        assert (mixed.unquoted, mixed.unresolved, mixed.partial, mixed.already_present) == (1, 1, (hit[0].episode_id,), 1)
    finally:
        await engine.close()


async def test_a_chunk_brought_by_a_summary_carries_the_superseded_mark_a_direct_hit_does():
    engine, episode_id, chunks = await document()
    try:
        newer = await engine.remember("s", "The harbour at Vellmar now stays open all winter.", kind="note", source="newer.md")
        await engine.assert_fact("s", "vellmar harbour", "closes_in", "November", valid_from="2024-01-01T00:00:00Z",
                                 source_episode_id=episode_id)
        await engine.assert_fact("s", "vellmar harbour", "closes_in", "never", valid_from="2025-01-01T00:00:00Z",
                                 source_episode_id=newer.episode_id)
        await build_summary_tree(engine, FakeChat(replies_for(chunks, max_levels=1)), "s", episode_id,
                                 fan_in=3, max_levels=1, store=True)
        direct = await engine.recall("s", PARTS[0], limit=3, kind="file")
        assert direct.items and all(item.superseded for item in direct.items if item.episode_id == episode_id)
        opened = await engine.recall("s", "Zorvath 1.0 recounts", limit=3, expand_summaries="follow")
        brought = [item for item in opened.items if item.via_summary is not None]
        assert brought and all(item.episode_id == episode_id and item.superseded for item in brought), \
            "a retired claim is marked whether it came back directly or through a summary"
    finally:
        await engine.close()


async def test_a_window_is_refused_with_expansion_since_it_would_move_what_the_cited_spans_index():
    engine, episode_id, chunks = await document()
    try:
        with TestClient(create_app(engine, {"key-a": "s"})) as client:
            auth = {"Authorization": "Bearer key-a"}
            for widening in ({"window": 40}, {"window_unit": "sentences"}):
                refused = client.get("/v1/recall", params={"q": "harbour", "expand_summaries": "follow", **widening}, headers=auth)
                assert refused.status_code == 422 and "window" in refused.json()["error"], refused.text
                assert client.get("/v1/recall", params={"q": "harbour", **widening}, headers=auth).status_code == 200
        for widening in (["--window", "40"], ["--window-unit", "sentences"]):
            with pytest.raises(InvalidInput, match="--expand-summaries cannot be combined"):
                await run(build_parser().parse_args(["--space", "s", "recall", "harbour", "--expand-summaries", "follow", *widening]),
                          engine, io.StringIO(""), io.StringIO())
    finally:
        await engine.close()
