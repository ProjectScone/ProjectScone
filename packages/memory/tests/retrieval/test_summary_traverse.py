"""Recall by descending stored summary trees: the best branches, bounded, down to chunks, each with its path."""
from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api.app import create_app
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.core.errors import InvalidInput, SconeError
from scone_memory.core.validation import MAX_SOURCE
from scone_memory.providers.llm import FakeChat
from scone_memory.retrieval import summary_traverse as module
from scone_memory.retrieval.summary_traverse import traverse_summaries
from scone_memory.retrieval.summary_tree import build_summary_tree, stored_summaries
from scone_memory.runtime.cli import build_parser, run

LIBRARY = {
    "vellmar.md": ("Vellmar", [
        ("harbour", ["The harbour at Vellmar closes to sailing boats every November, when the winter swell runs in across the outer bar from the west.",
                     "Its lighthouse was rebuilt in 1904 after a storm took the first tower down to its footings, and its lamp now turns on a mercury bath.",
                     "Pilots board arriving ships two miles out at the red buoy in every season, and their cutter is moored by the fish market steps."]),
        ("orchard", ["The orchard on the ridge grows only Bramley apples, planted in rows of forty on terraces that were cut by hand a century ago.",
                     "Picking starts in the last week of September and ends before the first frost, and the school children get two days off lessons.",
                     "A cider press in the tithe barn turns out two thousand litres a season, sold at the Saturday stall beside the old church wall."]),
        ("council", ["The council meets on the first Tuesday of each month in the old customs house, and any resident may speak for three minutes.",
                     "Seven members are elected every four years; the chair rotates each spring, and a tied vote is settled by a name drawn from a hat.",
                     "Minutes are pinned in the post office window within a week and kept in bound ledgers that go back without a gap to the year 1871."]),
    ]),
    "brannock.md": ("Brannock", [
        ("river", ["The river Brann drives the mill wheel through a stone leat that was dug in 1760 and relined with brick after the flood of 1952.",
                   "In a dry August the flow falls below what the wheel needs, so the sluice is closed at night to let the pond fill up for the morning.",
                   "Otters returned to the lower weir in 2011, and anglers are asked to keep well away from the holt under the alders by the footbridge."]),
        ("machinery", ["The machinery is turned by an overshot wheel of fourteen feet, driving two pairs of French burr stones through wooden gearing.",
                       "Cogs of apple wood are replaced every twelve years; the last set was cut by a wheelwright from Dunmore using templates from 1890.",
                       "A sack hoist lifts grain to the top floor, where it falls by its own weight through chutes into the hoppers above the stones."]),
        ("visitors", ["Visitors may tour the mill on Sundays from Easter to October, and the miller grinds a batch of wholemeal flour for sale at noon.",
                      "School groups book through the parish office, and each child leaves with a paper bag of flour and a drawing of the great wheel.",
                      "The tea room in the old drying kiln serves scones made from the mill's own flour, and dogs are welcome in the cobbled yard outside."]),
    ]),
}


def reply(*rows):
    return json.dumps({"notes": [{"sentence": s, "passage": p, "quote": q} for s, p, q in rows]})


def opening(text, count=3):
    return " ".join(text.split()[:count])


def replies_for(place, sections, chunks, empty=()):
    """What a model would answer for a tree of fan_in 3 over three sections of three chunks: a
    sentence per chunk naming the section's theme and place, then a root sentence per section.
    A section in ``empty`` is one the model wrote nothing about."""
    replies, written = [], []
    for group, (theme, _) in enumerate(sections):
        if theme in empty:
            replies.append(reply())
            continue
        rows = [(f"The {theme} of {place}, part {member}: {opening(chunk.text)}.", f"chunk:{chunk.chunk_id}", opening(chunk.text))
                for member, chunk in enumerate(chunks[group * 3:group * 3 + 3])]
        replies.append(reply(*rows))
        written.append((group, theme, rows[0][0]))
    replies.append(reply(*[(f"{place} keeps a record of its {theme}.", f"node:1:{group}", first)
                           for group, theme, first in written]))
    return replies


class Counting(HashEmbedder):
    """The engine's embedder, counting every call it answers."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []

    async def embed(self, texts):
        self.calls.append(len(texts))
        return await super().embed(texts)


class Constant(HashEmbedder):
    """An embedder that cannot tell any two texts apart."""

    async def embed(self, texts):
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


async def library(embedder=None, empty=(), index=None, documents_store=None):
    engine = await MemoryEngine(documents_store or InMemoryDocumentStore(), index or InMemoryVectorIndex(),
                                embedder or HashEmbedder(), chunk_target=170).open()
    documents = {}
    for source, (place, sections) in LIBRARY.items():
        parts = [part for _, texts in sections for part in texts]
        added = await engine.remember("s", "\n\n".join(parts), kind="file", source=source, tags=(place.lower(),))
        chunks = await engine.documents.chunks_of("s", added.episode_id)
        assert [chunk.text.strip() for chunk in chunks] == parts, "one chunk per part"
        await build_summary_tree(engine, FakeChat(replies_for(place, sections, chunks, empty)), "s", added.episode_id,
                                 fan_in=3, store=True)
        documents[place] = (added.episode_id, chunks)
    return engine, documents


async def nodes_of(engine, episode_id):
    return {(row.level, row.index): row for row in await stored_summaries(engine, "s", episode_id)}


async def test_the_best_branch_is_descended_to_its_chunks_and_each_says_the_path_that_led_there():
    embedder = Counting()
    engine, documents = await library(embedder)
    try:
        vellmar, chunks = documents["Vellmar"]
        nodes = await nodes_of(engine, vellmar)
        embedder.calls.clear()
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        assert [item.chunk_id for item in found.items][0] == chunks[0].chunk_id
        assert sorted(item.chunk_id for item in found.items) == [chunk.chunk_id for chunk in chunks[:3]], \
            "the harbour section's chunks and nothing else"
        assert embedder.calls == [1] and (found.embed_calls, found.embedded_texts) == (1, 1), \
            "the question is embedded once; every node and chunk is scored by the vectors already stored"
        assert found.vectors == {"index": 3, "embedded": 0}
        first = found.items[0]
        assert (first.episode_id, first.source, first.text, first.start, first.end) == (
            vellmar, "vellmar.md", chunks[0].text, chunks[0].start, chunks[0].end)
        assert first.similarity is not None and first.similarity > 0 and first.first_line == 1
        path = first.via_tree["path"]
        assert first.via_tree["summary_of"] == vellmar
        assert [(step["episode_id"], step["level"], step["index"], step["rank"]) for step in path] == [
            (nodes[(2, 0)].episode_id, 2, 0, 1), (nodes[(1, 0)].episode_id, 1, 0, 1)], "root, then the harbour section"
        assert all(isinstance(step["similarity"], float) and "text_rank" not in step for step in path)
        assert [step["pool"] for step in found.steps] == [2, 3], "two roots, then the three sections of the chosen one"
        assert found.steps[0]["chosen"] == [nodes[(2, 0)].episode_id]
        assert found.documents == tuple(sorted(episode for episode, _ in documents.values()))
        assert (found.leaves, found.cut_by_limit, found.cut_by_depth, found.stale, found.missing, found.unresolved) == (3, 0, 0, 0, 0, 0)
        assert found.walked == len(await engine.documents.recent_episodes("s", 100))
        assert "descended 2 level(s) of 2 document tree(s) to 3 chunk(s)" in found.why
        record = json.loads(json.dumps(found.record()))
        assert record["items"][0]["via_tree"]["path"][1]["index"] == 0 and record["branching"] == 1
    finally:
        await engine.close()


async def test_a_summary_stored_as_several_chunks_scores_as_its_best_chunk():
    engine, documents = await library()
    try:
        vellmar, _ = documents["Vellmar"]
        filler = ("Tallow candles, woollen mittens and brass buttons were counted in the storeroom ledger by the clerk "
                  "every spring and every autumn.")
        pointed = ("The harbour of Vellmar keeps its harbour pilots, and the harbour lighthouse at Vellmar turns every night "
                   "above the harbour steps.")
        assert 120 < len(filler) < 168 and 120 < len(pointed) < 168
        top = await forge(engine, vellmar, level=3, index=0, written_from=["node:2:0"], text=filler + "\n\n" + pointed)
        owned = [chunk.chunk_id for chunk in await engine.documents.chunks_of("s", top)]
        [question] = await engine.embedder.embed(["harbour of Vellmar"])
        cosines = [module._cosine(question, vector) for vector in (await engine.vectors.vectors_of("s", owned)).values()]
        assert len(owned) == 2 and round(max(cosines), 6) != round(min(cosines), 6), "a summary stored as two chunks"
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        assert found.items[0].via_tree["path"][0] == {"episode_id": top, "level": 3, "index": 0, "rank": 1,
                                                      "similarity": round(max(cosines), 6)}
    finally:
        await engine.close()


async def test_branching_bounds_the_branches_kept_at_every_level_across_documents():
    engine, documents = await library()
    try:
        vellmar, chunks = documents["Vellmar"]
        wide = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=2)
        assert [len(step["chosen"]) for step in wide.steps] == [2, 2] and [step["pool"] for step in wide.steps] == [2, 6], \
            "both roots are kept, then the two best of their six sections, not two per root"
        assert len(wide.items) == 6 and {item.chunk_id for item in wide.items[:3]} == {chunk.chunk_id for chunk in chunks[:3]}
        assert [item.via_tree["path"][-1]["rank"] for item in wide.items] == [1, 1, 1, 2, 2, 2]
        assert [item.score for item in wide.items] == [round(1 / (1 + place), 6) for place in range(6)]

        ordered = await traverse_summaries(engine, "s", "harbour of Vellmar and the customs house", branching=3)
        assert [item.via_tree["path"][-1]["rank"] for item in ordered.items] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
        assert {item.chunk_id for item in ordered.items[:3]} == {chunk.chunk_id for chunk in chunks[:3]}, \
            "the chunks of the branch kept first come first"
        assert max(item.similarity for item in ordered.items[3:]) > min(item.similarity for item in ordered.items[:3]), \
            "a lower branch holds a chunk that scores better on its own, or the order proves nothing"
        assert {item.via_tree["path"][1]["episode_id"] for item in wide.items} == set(wide.steps[1]["chosen"])
        narrow = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        assert len(narrow.items) == 3
    finally:
        await engine.close()


async def test_the_limit_and_the_depth_cut_and_say_so():
    engine, documents = await library()
    try:
        _, chunks = documents["Vellmar"]
        short = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1, limit=2)
        assert len(short.items) == 2 and short.items[0].chunk_id == chunks[0].chunk_id
        assert (short.leaves, short.cut_by_limit) == (3, 1) and "the limit of 2 left 1 chunk(s) out" in short.why
        shallow = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1, max_depth=1)
        assert shallow.items == () and shallow.cut_by_depth == 3 and len(shallow.steps) == 1
        assert "the depth of 1 left 3 summary node(s) unscored" in shallow.why
        assert (await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1, max_depth=2)).cut_by_depth == 0
    finally:
        await engine.close()


async def forge(engine, document, *, level, index, written_from, content_hash=None, text=None, chunks=None, of=None,
                fan_in=None):
    """A summary note written by hand: the metadata a build writes, with another place, account or hash."""
    nodes = await nodes_of(engine, document)
    model = (await engine.episode("s", nodes[(1, 0)].episode_id)).metadata
    detail = await engine.attach("s", json.dumps({"id": f"node:{level}:{index}", "written_from": written_from,
                                                  "covers": [], "sentences": []}).encode(), media_type="application/json")
    metadata = {**model, "summary_level": str(level), "summary_index": str(index), "summary_detail": detail.attachment_id,
                **({"summary_content_hash": content_hash} if content_hash is not None else {}),
                **({"summary_chunks": str(chunks)} if chunks is not None else {}),
                **({"summary_of": str(of)} if of is not None else {}),
                **({"summary_fan_in": str(fan_in)} if fan_in is not None else {})}
    added = await engine.remember("s", text or f"Forged node {level} {index} about the harbour of Vellmar.", kind="note",
                                  source=f"vellmar.md#summary/{level}/{index}", metadata=metadata)
    return added.episode_id


async def test_a_stale_node_is_never_descended_and_a_named_child_that_is_not_there_is_counted():
    engine, documents = await library()
    try:
        vellmar, chunks = documents["Vellmar"]
        nodes = await nodes_of(engine, vellmar)
        stale = await forge(engine, vellmar, level=3, index=0, written_from=["node:2:0"], content_hash="0" * 64)
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        assert found.stale == 1 and "1 summary node(s) were written from other content" in found.why
        assert all(stale not in [step["episode_id"] for step in item.via_tree["path"]] for item in found.items)
        assert found.items[0].via_tree["path"][0]["episode_id"] == nodes[(2, 0)].episode_id, \
            "the top of the tree is the highest node written from this content, not the highest note"
        await engine.forget("s", stale)

        stale_child = await forge(engine, vellmar, level=1, index=5, written_from=[f"chunk:{chunks[0].chunk_id}"],
                                  content_hash="0" * 64, text="Forged stale harbour harbour harbour of Vellmar.")
        top = await forge(engine, vellmar, level=3, index=0, chunks=40,
                          written_from=["node:1:5", "node:2:0", "chunk:987654", "node:2:9", "node:3:0", "node:x:0",
                                        "note:2:0", "chunk:abc", "node:2:0:1"])
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        assert found.items and found.items[0].via_tree["path"][0]["episode_id"] == top
        assert all(stale_child not in [step["episode_id"] for step in item.via_tree["path"]] for item in found.items), \
            "a child written from other content is not scored, even when its words fit best"
        assert found.steps[1]["pool"] == 1, "only the root written from this content was a candidate below the top"
        assert (found.stale, found.missing, found.unresolved) == (1, 1, 7), \
            "a stale node, one never stored, itself, names that are not one, another kind, a chunk id that is not one"
        assert found.uncovered == 0, "a top that claims more chunks than the document has leaves none uncovered, not fewer than none"
        assert "1 chunk(s) named below a summary are not the document's" in found.why
        assert "7 node(s) named below a summary are not stored from this content" in found.why
        await engine.forget("s", top)

        other_shape = await forge(engine, vellmar, level=3, index=0, fan_in=4, written_from=["node:2:0"])
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1, episode_ids=[vellmar])
        assert found.steps[0]["chosen"] == [other_shape] and found.items == () and found.unresolved == 1, \
            "a node names nodes of its own tree's shape, not another's at the same place"

        plain = await engine.remember("s", "A note someone summarized by hand, long ago, from other words.", kind="note")
        await forge(engine, vellmar, level=1, index=0, of=plain.episode_id, content_hash="1" * 64,
                    written_from=[f"chunk:{chunks[0].chunk_id}"], text="Forged summary of a plain note.")
        await forge(engine, vellmar, level=1, index=1, of="x", written_from=[], text="A summary of nothing that names no episode.")
        refused = await traverse_summaries(engine, "s", "harbour", episode_ids=[plain.episode_id])
        assert refused.items == () and refused.refused == ({"episode_id": plain.episode_id, "reason": "content_changed"},)
        assert refused.stale == 1 and "every summary they hold was written from other content" in refused.why
    finally:
        await engine.close()


async def test_a_node_whose_account_cannot_be_read_is_not_descended_and_is_counted():
    engine, documents = await library()
    try:
        vellmar, _ = documents["Vellmar"]
        real = engine.attachment

        async def attachment(space, attachment_id):
            if len(read) == 1:
                raise SconeError("the blob store did not answer")
            read.append(attachment_id)
            return await real(space, attachment_id)

        read: list[str] = []
        engine.attachment = attachment
        try:
            found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        finally:
            engine.attachment = real
        assert found.items == () and found.unresolved == 1 and "could not be read" in found.why
        broken = await forge(engine, vellmar, level=3, index=0, written_from="chunk:1")
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1, episode_ids=[vellmar])
        assert found.items == () and found.unresolved == 1, "an account that does not list what it was written from"
        assert found.steps[0]["chosen"] == [broken]
        await engine.forget("s", broken)
        numbered = await forge(engine, vellmar, level=3, index=0, written_from=["node:2:0", 7])
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1, episode_ids=[vellmar])
        assert found.steps[0]["chosen"] == [numbered] and found.items == () and found.unresolved == 1, \
            "an account listing something other than names is not read in part"
    finally:
        await engine.close()


async def test_a_forgotten_document_serves_nothing_and_its_tree_is_refused_with_the_reason():
    engine, documents = await library()
    try:
        vellmar, _ = documents["Vellmar"]
        brannock, brannock_chunks = documents["Brannock"]
        await engine.forget("s", vellmar)
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        assert found.items and all(item.episode_id == brannock for item in found.items)
        assert found.refused == () and found.documents == (brannock,) and found.unlisted_trees == 1, \
            "a tree left behind by a forgotten document is counted on an unscoped call, not read"
        assert "1 summary tree(s) name a document the space's listing did not return" in found.why
        named = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1, episode_ids=[vellmar])
        assert named.refused == ({"episode_id": vellmar, "reason": "source_gone"},) and named.unlisted_trees == 0
        assert "1 document tree(s) were refused: their document is forgotten" in named.why
        unknown = await traverse_summaries(engine, "s", "harbour", episode_ids=[424242])
        assert unknown.items == () and unknown.refused == ({"episode_id": 424242, "reason": "source_unknown"},)
    finally:
        await engine.close()


async def test_a_document_forgotten_or_changed_while_the_tree_is_read_serves_nothing_of_it():
    engine, documents = await library()
    try:
        vellmar, chunks = documents["Vellmar"]
        real = engine.attachment

        async def attachment(space, attachment_id):
            got = await real(space, attachment_id)
            if not forgotten:
                # Another request forgets the document once its root's account has been read.
                forgotten.append(await engine.forget("s", vellmar))
            return got

        forgotten: list[object] = []

        engine.attachment = attachment
        try:
            found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        finally:
            engine.attachment = real
        assert all(item.episode_id != vellmar for item in found.items), "no chunk of a document forgotten mid-walk"
        assert {"episode_id": vellmar, "reason": "source_gone"} in found.refused

        brannock, brannock_chunks = documents["Brannock"]
        river = (await nodes_of(engine, brannock))[(1, 0)].episode_id

        async def forgets_a_summary(space, attachment_id):
            got = await real(space, attachment_id)
            if not gone:
                # The river section's note is forgotten after the walk listed it: its chunks and vectors went.
                gone.append(await engine.forget("s", river))
            return got

        gone: list[object] = []
        engine.attachment = forgets_a_summary
        try:
            found = await traverse_summaries(engine, "s", "machinery of Brannock", branching=1, episode_ids=[brannock])
        finally:
            engine.attachment = real
        assert found.vectors == {"index": 2, "embedded": 1}, "a candidate with no stored chunk puts its step on embeddings"
        assert found.items and all(item.episode_id == brannock for item in found.items)

        real_walk = engine.documents.recent_episodes

        async def before_brannock(space, limit):
            return [episode for episode in await real_walk(space, limit) if episode.episode_id != brannock]

        engine.documents.recent_episodes = before_brannock
        try:
            late = await traverse_summaries(engine, "s", "river of Brannock", episode_ids=[brannock])
        finally:
            engine.documents.recent_episodes = real_walk
        assert late.items == () and late.refused == ({"episode_id": brannock, "reason": "unlisted"},), \
            "a document the listing left out had no tree seen whole"
        assert "their document was not in the space's listing" in late.why

        async def then_forgotten(space, limit):
            listed = await real_walk(space, limit)
            await engine.forget("s", brannock)
            return listed

        engine.documents.recent_episodes = then_forgotten
        try:
            went = await traverse_summaries(engine, "s", "river of Brannock", episode_ids=[brannock])
        finally:
            engine.documents.recent_episodes = real_walk
        assert went.unlisted == 0, "an episode forgotten after the listing leaves nothing unlisted, not fewer than nothing"
        assert went.items == () and went.refused == ({"episode_id": brannock, "reason": "source_gone"},), \
            "a document the walk listed and that went before it was read"
    finally:
        await engine.close()

    engine, documents = await library()
    try:
        vellmar, chunks = documents["Vellmar"]
        brannock, _ = documents["Brannock"]
        real_get = engine.documents.get_chunks

        async def changed(space, ids):
            return [chunk.model_copy(update={"text": chunk.text.upper()}) if chunk.chunk_id == chunks[1].chunk_id else chunk
                    for chunk in await real_get(space, ids)]

        engine.documents.get_chunks = changed
        try:
            moved = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=2)
        finally:
            engine.documents.get_chunks = real_get
        assert moved.items == () and moved.refused == ({"episode_id": vellmar, "reason": "changed_while_read"},), \
            "a document one of whose chunks moved serves none of them"

        wide = await traverse_summaries(engine, "s", "Vellmar Brannock harbour river", branching=2, limit=50)
        assert {item.episode_id for item in wide.items} == {vellmar, brannock}
        engine.documents.get_chunks = changed
        try:
            kept = await traverse_summaries(engine, "s", "Vellmar Brannock harbour river", branching=2, limit=50)
        finally:
            engine.documents.get_chunks = real_get
        assert kept.items and {item.episode_id for item in kept.items} == {brannock}, \
            "the other document's chunks still stand, read again after the refused one went"

        async def failing(space, ids):
            raise SconeError("the store did not answer")

        engine.documents.get_chunks = failing
        try:
            unread = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        finally:
            engine.documents.get_chunks = real_get
        assert unread.items == () and unread.refused == ({"episode_id": vellmar, "reason": "unread"},)
        assert "could not be read" in unread.why
    finally:
        await engine.close()


async def test_the_text_lane_can_choose_a_branch_the_vectors_cannot_tell_apart():
    engine, documents = await library(Constant())
    try:
        brannock, chunks = documents["Brannock"]
        blind = await traverse_summaries(engine, "s", "Brannock machinery overshot wheel", branching=1)
        assert blind.items and blind.items[0].episode_id != brannock, "equal vectors keep the first root"
        read = await traverse_summaries(engine, "s", "Brannock machinery overshot wheel", branching=1, text=True)
        assert read.items[0].chunk_id == chunks[3].chunk_id
        assert sorted(item.chunk_id for item in read.items) == [chunk.chunk_id for chunk in chunks[3:6]]
        path = read.items[0].via_tree["path"]
        assert [step["text_rank"] for step in path] == [1, 1] and read.items[0].via_tree["text_rank"] == 1
        assert read.items[-1].via_tree["text_rank"] is None, "a chunk BM25 does not match has no text rank"
        assert read.text is True and read.record()["text"] is True
    finally:
        await engine.close()


class NoStoredVectors(InMemoryVectorIndex):
    vectors_of = None  # an index that cannot hand back what it stored


async def test_a_pool_without_every_stored_vector_is_embedded_and_counted(monkeypatch):
    embedder = Counting()
    engine, documents = await library(embedder, index=NoStoredVectors())
    try:
        _, chunks = documents["Vellmar"]
        embedder.calls.clear()
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        assert embedder.calls == [1, 2, 3, 3], "the question, then each pool in one call: roots, sections, chunks"
        assert (found.embed_calls, found.embedded_texts, found.vectors) == (4, 9, {"index": 0, "embedded": 3})
        assert sorted(item.chunk_id for item in found.items) == [chunk.chunk_id for chunk in chunks[:3]]
    finally:
        await engine.close()

    engine, documents = await library(embedder)
    try:
        _, chunks = documents["Vellmar"]
        await engine.vectors.delete([chunks[2].chunk_id])
        embedder.calls.clear()
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        assert embedder.calls == [1, 3] and found.vectors == {"index": 2, "embedded": 1}, \
            "one missing vector puts its whole pool on one scale, embedded"
        monkeypatch.setattr(MemoryEngine, "vector_block", property(lambda self: "written by another embedder"))
        embedder.calls.clear()
        blocked = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        assert blocked.vectors == {"index": 0, "embedded": 3} and embedder.calls == [1, 2, 3, 3], \
            "vectors another embedder wrote are never compared with this one's"
    finally:
        await engine.close()


async def test_the_documents_traversed_are_the_named_ones_that_the_filters_keep(monkeypatch):
    engine, documents = await library()
    try:
        vellmar, _ = documents["Vellmar"]
        brannock, _ = documents["Brannock"]
        named = await traverse_summaries(engine, "s", "harbour of Vellmar", episode_ids=[brannock, brannock])
        assert named.documents == (brannock,) and all(item.episode_id == brannock for item in named.items)
        assert named.steps[0]["pool"] == 1, "a document named twice is descended once"
        later = await traverse_summaries(engine, "s", "harbour", since="2999-01-01T00:00:00Z")
        assert later.documents == () and later.out_of_scope == 2
        earlier = await traverse_summaries(engine, "s", "harbour", until="2001-01-01T00:00:00Z")
        assert earlier.documents == () and earlier.out_of_scope == 2
        prefixed = await traverse_summaries(engine, "s", "river of Brannock", source_prefix="vellmar")
        assert prefixed.documents == (vellmar,) and prefixed.out_of_scope == 1
        tagged = await traverse_summaries(engine, "s", "harbour of Vellmar", tags=["brannock"])
        assert tagged.documents == (brannock,) and tagged.out_of_scope == 1
        nothing = await traverse_summaries(engine, "s", "harbour", kind="note")
        assert nothing.documents == () and nothing.items == () and nothing.out_of_scope == 2
        assert nothing.embed_calls == 0 and nothing.why.startswith("no document with a summary tree is in scope"), \
            "nothing to score spends no embedding"
        plain = await engine.remember("s", "A short note with no summary tree at all.", kind="note")
        untreed = await traverse_summaries(engine, "s", "harbour", episode_ids=[plain.episode_id, vellmar])
        assert untreed.untreed == (plain.episode_id,) and untreed.documents == (vellmar,)
        assert "1 document(s) named have no summary tree" in untreed.why
        monkeypatch.setattr(module, "MAX_CANDIDATES", 2)
        with pytest.raises(InvalidInput, match="a step of the descent has 3 candidates"):
            await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        monkeypatch.setattr(module, "MAX_CANDIDATES", 3)
        assert (await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)).leaves == 3
        monkeypatch.undo()
        monkeypatch.setattr(module, "MAX_DOCUMENTS", 1)
        with pytest.raises(InvalidInput, match="2 documents with summary trees are in scope"):
            await traverse_summaries(engine, "s", "harbour")
        assert (await traverse_summaries(engine, "s", "harbour", episode_ids=[vellmar])).documents == (vellmar,)
    finally:
        await engine.close()


async def test_chunks_no_summary_covers_are_counted_since_no_descent_reaches_them():
    engine, documents = await library(empty=("orchard",))
    try:
        vellmar, _ = documents["Vellmar"]
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", episode_ids=[vellmar])
        assert found.uncovered == 3 and "3 chunk(s) are under no summary the descent starts from" in found.why
    finally:
        await engine.close()


async def test_an_empty_space_and_a_sqlite_store_answer_as_the_memory_one_does(tmp_path):
    for documents, vectors in ((InMemoryDocumentStore(), InMemoryVectorIndex()),
                               (SqliteDocumentStore(str(tmp_path / "empty.db")), SqliteVectorIndex(str(tmp_path / "empty.db")))):
        engine = await MemoryEngine(documents, vectors, HashEmbedder()).open()
        try:
            found = await traverse_summaries(engine, "s", "harbour")
            assert (found.items, found.walked, found.unlisted, found.embed_calls) == ((), 0, 0, 0)
            assert found.why.startswith("no document with a summary tree is in scope")
        finally:
            await engine.close()
    path = str(tmp_path / "library.db")
    engine, documents = await library(documents_store=SqliteDocumentStore(path), index=SqliteVectorIndex(path))
    try:
        _, chunks = documents["Vellmar"]
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        assert sorted(item.chunk_id for item in found.items) == [chunk.chunk_id for chunk in chunks[:3]]
        assert found.vectors == {"index": 3, "embedded": 0} and found.embed_calls == 1, "SQLite hands its vectors back"
    finally:
        await engine.close()


async def test_episodes_the_listing_leaves_out_are_counted_since_a_tree_among_them_is_not_found():
    engine, documents = await library()
    try:
        vellmar, _ = documents["Vellmar"]
        real_walk = engine.documents.recent_episodes

        async def capped(space, limit):
            return (await real_walk(space, limit))[:-2]  # a store that lists fewer than it holds, newest first

        engine.documents.recent_episodes = capped
        try:
            short = await traverse_summaries(engine, "s", "harbour of Vellmar")
        finally:
            engine.documents.recent_episodes = real_walk
        assert short.unlisted == 2 and vellmar not in short.documents
        assert short.refused == () and short.unlisted_trees == 1, "a tree whose document the listing left out is counted"
        assert "leaves its tree behind, and the listing left episodes out)" in short.why
        assert f"the store listed {short.walked} episode(s) and holds {short.walked + 2}" in short.why
        assert (await traverse_summaries(engine, "s", "harbour of Vellmar")).unlisted == 0
    finally:
        await engine.close()


async def test_a_mistaken_request_is_refused_before_anything_is_read():
    engine, _ = await library()
    try:
        real = engine.documents.recent_episodes

        async def no_walk(*args, **kwargs):
            raise AssertionError("refused before the walk")

        engine.documents.recent_episodes = no_walk
        try:
            for options in ({"branching": 0}, {"branching": module.MAX_BRANCHING + 1}, {"branching": True},
                            {"max_depth": 0}, {"max_depth": module.MAX_DEPTH + 1}, {"limit": 0}, {"limit": 51},
                            {"text": "yes"}, {"episode_ids": []}, {"episode_ids": ["1"]}, {"kind": "diary"},
                            {"source_prefix": "x" * (MAX_SOURCE + 1)}):
                with pytest.raises(InvalidInput):
                    await traverse_summaries(engine, "s", "harbour", **options)
            with pytest.raises(InvalidInput):
                await traverse_summaries(engine, "s", "   ")
        finally:
            engine.documents.recent_episodes = real
    finally:
        await engine.close()


async def test_the_engine_http_and_cli_surfaces_answer_alike():
    engine, documents = await library()
    try:
        vellmar, chunks = documents["Vellmar"]
        defaults = await engine.tree_recall("s", "harbour of Vellmar")
        assert (defaults.limit, defaults.branching, defaults.max_depth, defaults.text) == (
            module.DEFAULT_LIMIT, module.DEFAULT_BRANCHING, module.MAX_DEPTH, False)
        direct = await engine.tree_recall("s", "harbour of Vellmar", branching=1)
        assert [item.chunk_id for item in direct.items] == [
            item.chunk_id for item in (await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)).items]
        plain = await engine.recall("s", "harbour of Vellmar", limit=3)
        assert "via_tree" not in json.dumps(plain.model_dump(mode="json")), "recall answers as it did before"

        with TestClient(create_app(engine, {"key-a": "s"})) as client:
            auth = {"Authorization": "Bearer key-a"}
            body = client.get("/v1/recall/tree", params={"q": "harbour of Vellmar", "branching": 1, "episode_id": [vellmar]},
                              headers=auth).json()
            assert [item["chunk_id"] for item in body["items"]] == [item.chunk_id for item in direct.items]
            assert body["items"][0]["via_tree"]["summary_of"] == vellmar and body["branching"] == 1
            assert body["documents"] == [vellmar] and "why" in body
            assert client.get("/v1/recall/tree", params={"q": "x", "branching": 0}, headers=auth).status_code in (400, 422)
            assert '"recall.tree": true' in json.dumps(client.get("/v1/capabilities", headers=auth).json())

        out = io.StringIO()
        code = await run(build_parser().parse_args(["--space", "s", "--json", "tree-recall", "harbour of Vellmar",
                                                    "--branching", "1", "--episode", str(vellmar)]),
                         engine, io.StringIO(""), out)
        said = json.loads(out.getvalue())
        assert code == 0 and [item["chunk_id"] for item in said["items"]] == [item.chunk_id for item in direct.items]
        out = io.StringIO()
        await run(build_parser().parse_args(["--space", "s", "tree-recall", "harbour of Vellmar", "--branching", "1",
                                             "--limit", "2", "--text", "--max-depth", "2", "--source-prefix", "vellmar"]),
                  engine, io.StringIO(""), out)
        printed = out.getvalue()
        assert "descended 2 level(s)" in printed and f"#{vellmar}" in printed and "via level 2 #" in printed
        assert chunks[0].text[:40] in printed
    finally:
        await engine.close()


async def test_trees_forgotten_documents_leave_behind_cost_no_read_and_no_refusal_on_an_unscoped_call(monkeypatch):
    engine, documents = await library()
    try:
        vellmar, _ = documents["Vellmar"]
        brannock, _ = documents["Brannock"]
        await engine.forget("s", vellmar)
        await engine.forget("s", brannock)
        monkeypatch.setattr(module, "MAX_DOCUMENTS", 1)
        reads: list[int] = []
        real_episode = engine.episode

        async def counting(space, episode_id, *args, **kwargs):
            reads.append(episode_id)
            return await real_episode(space, episode_id, *args, **kwargs)

        engine.episode = counting
        scoped = await traverse_summaries(engine, "s", "harbour", kind="note")
        assert (scoped.refused, scoped.unlisted_trees, scoped.out_of_scope, reads) == ((), 2, 0, []), \
            "two trees whose documents are gone: counted, never read, never refused, and no bound spent on them"
        assert scoped.why.startswith("no document with a summary tree is in scope")
        assert json.loads(json.dumps(scoped.record()))["unlisted_trees"] == 2
        assert "2 summary tree(s) name a document the space's listing did not return" in scoped.why
        named = await traverse_summaries(engine, "s", "harbour", episode_ids=[vellmar])
        assert named.refused == ({"episode_id": vellmar, "reason": "source_gone"},) and reads == [vellmar], \
            "a document named is still read and refused with its reason"
        assert not named.why.startswith("no document with a summary tree is in scope")
    finally:
        await engine.close()


async def test_a_replaced_document_leaves_a_count_not_a_growing_list_of_refusals():
    engine, _ = await library()
    try:
        place, sections = LIBRARY["vellmar.md"]
        parts = [part for _, texts in sections for part in texts]
        for version in range(3):
            added = await engine.remember("s", "\n\n".join(parts) + f"\n\nRevision {version} of the notes was filed by the clerk.",
                                          kind="file", source="vellmar.md", dedup_key="vellmar.md", replace=True)
            chunks = await engine.documents.chunks_of("s", added.episode_id)
            await build_summary_tree(engine, FakeChat(replies_for(place, sections, chunks[:9]) + [reply()] * 4), "s",
                                     added.episode_id, fan_in=3, store=True)
        found = await traverse_summaries(engine, "s", "harbour of Vellmar", kind="conversation")
        assert found.refused == () and found.unlisted_trees == 2 and found.out_of_scope == 3, \
            "two replaced versions left trees behind; the library's two documents and the latest version are out of scope"
    finally:
        await engine.close()


CARRIED = [
    "The harbour at Vellmar closes to sailing boats every November, when the winter swell runs in across the outer bar from the west.",
    "Its lighthouse was rebuilt in 1904 after a storm took the first tower down to its footings, and its lamp now turns on a mercury bath.",
    "Pilots board arriving ships two miles out at the red buoy in every season, and their cutter is moored by the fish market steps.",
    "The orchard on the ridge grows only Bramley apples, planted in rows of forty on terraces that were cut by hand a century ago.",
    "Picking starts in the last week of September and ends before the first frost, and the school children get two days off lessons.",
    "A cider press in the tithe barn turns out two thousand litres a season, sold at the Saturday stall beside the old church wall.",
    "Harbour dues at Vellmar harbour are paid to the harbour master of Vellmar, who keeps the harbour ledger at Vellmar harbour office.",
]


async def test_a_chunk_carried_up_beside_summaries_competes_with_them_and_is_not_a_leaf_for_free():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=170).open()
    try:
        added = await engine.remember("s", "\n\n".join(CARRIED), kind="file", source="vellmar.md")
        chunks = await engine.documents.chunks_of("s", added.episode_id)
        assert [chunk.text.strip() for chunk in chunks] == CARRIED
        harbour = [(f"The harbour of Vellmar: {opening(c.text)}.", f"chunk:{c.chunk_id}", opening(c.text)) for c in chunks[0:3]]
        orchard = [(f"The orchard: {opening(c.text)}.", f"chunk:{c.chunk_id}", opening(c.text)) for c in chunks[3:6]]
        root = [("Vellmar keeps a harbour.", "node:1:0", harbour[0][0]), ("Vellmar keeps an orchard.", "node:1:1", orchard[0][0]),
                ("And harbour dues.", f"chunk:{chunks[6].chunk_id}", opening(chunks[6].text))]
        tree = await build_summary_tree(engine, FakeChat([reply(*harbour), reply(*orchard), reply(*root)]), "s",
                                        added.episode_id, fan_in=3, store=True)
        assert [node.written_from for node in tree.nodes if node.level == 2] == [("node:1:0", "node:1:1", f"chunk:{chunks[6].chunk_id}")]

        apples = await traverse_summaries(engine, "s", "orchard Bramley apples", branching=1)
        assert apples.steps[1]["pool"] == 3, "the two sections and the chunk carried up beside them"
        assert [item.chunk_id for item in apples.items] == [chunk.chunk_id for chunk in chunks[3:6]], \
            "the chosen section's chunks, and not the carried chunk the step did not keep"
        assert apples.steps[1]["chunks"] == [] and apples.leaves == 3

        dues = await traverse_summaries(engine, "s", "harbour dues paid to the harbour master", branching=1, limit=3)
        assert dues.steps[1]["chosen"] == [] and dues.steps[1]["chunks"] == [chunks[6].chunk_id], \
            "the carried chunk scored best at its step and was kept in place of a section"
        assert [item.chunk_id for item in dues.items] == [chunks[6].chunk_id] and (dues.leaves, dues.cut_by_limit) == (1, 0)
        assert dues.items[0].via_tree["rank"] == 1 and [step["level"] for step in dues.items[0].via_tree["path"]] == [2]

        shallow = await traverse_summaries(engine, "s", "orchard Bramley apples", branching=1, max_depth=1)
        assert shallow.cut_by_depth == 3 and shallow.items == (), "the carried chunk was reached and not scored too"
    finally:
        await engine.close()


SHORT = [
    "Kelp gathering on the Orrin shore happens at the lowest spring tides of the year, by families with carts.",
    "Orrin kelp is burned in stone pits to make soda ash that was once sold to glassworks across the water.",
    "The Orrin kelp season ends in August, when the storms start to throw the weed high above the reach of carts.",
]
TALL = [
    "Glassworks at Tamsey melted soda ash and sand in a brick cone furnace that burned coal day and night.",
    "Tamsey glassworks blowers made green bottles for the cider trade, hundreds of them in a single shift.",
    "The Tamsey glassworks cone was pulled down in 1931 and its bricks went into the new harbour wall.",
    "Tamsey market is held on Thursdays in the square, with stalls of fish, bread and woollen goods.",
    "Tamsey market tolls were paid to the lord of the manor until the town bought the rights in 1880.",
    "The Tamsey market cross was rebuilt after a cart ran into it during the fair of 1902.",
    "Tamsey chapel was built by the quarrymen with stone they carried down from the hill on Sundays.",
    "The Tamsey chapel organ came second hand from a church in the city and still needs two to pump it.",
    "Tamsey chapel choir sings at the harvest supper every October in the hall behind the chapel.",
]


async def test_branch_first_holds_across_trees_of_different_depths():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=90).open()
    try:
        short = await engine.remember("s", "\n\n".join(SHORT), kind="file", source="orrin.md")
        short_chunks = await engine.documents.chunks_of("s", short.episode_id)
        assert len(short_chunks) == 3
        await build_summary_tree(engine, FakeChat([reply(*[(f"Orrin kelp: {opening(c.text)}.", f"chunk:{c.chunk_id}", opening(c.text))
                                                          for c in short_chunks])]), "s", short.episode_id, fan_in=3, store=True)
        tall = await engine.remember("s", "\n\n".join(TALL), kind="file", source="tamsey.md")
        tall_chunks = await engine.documents.chunks_of("s", tall.episode_id)
        assert len(tall_chunks) == 9
        replies, firsts = [], []
        themes = ["glassworks soda ash", "market", "chapel"]
        for group, theme in enumerate(themes):
            rows = [(f"Tamsey {theme}: {opening(c.text)}.", f"chunk:{c.chunk_id}", opening(c.text))
                    for c in tall_chunks[group * 3:group * 3 + 3]]
            replies.append(reply(*rows))
            firsts.append(rows[0][0])
        replies.append(reply(*[(f"Tamsey keeps its {theme}.", f"node:1:{group}", firsts[group]) for group, theme in enumerate(themes)]))
        assert (await build_summary_tree(engine, FakeChat(replies), "s", tall.episode_id, fan_in=3, store=True)).levels == 2

        def ranks(item):
            return [step["rank"] for step in item.via_tree["path"]]

        short_first = await traverse_summaries(engine, "s", "Orrin kelp soda ash glassworks", branching=2, limit=3)
        assert [ranks(item) for item in short_first.items] == [[1], [1], [1]], \
            "the one-level tree kept first serves all its chunks before a chunk under the tree kept second"
        assert {item.episode_id for item in short_first.items} == {short.episode_id}

        tall_first = await traverse_summaries(engine, "s", "Tamsey glassworks soda ash and the market", branching=2)
        assert [ranks(item) for item in tall_first.items] == [[1, 1]] * 3 + [[1, 2]] * 3 + [[2]] * 3, \
            "every chunk under the tree kept first, its sections in their order, before the tree kept second"
        assert max(item.similarity for item in tall_first.items[6:]) > min(item.similarity for item in tall_first.items[3:6]), \
            "the tree kept second holds a chunk that scores better on its own, or the order proves nothing"
    finally:
        await engine.close()


async def test_a_summary_on_the_returned_path_forgotten_during_the_walk_refuses_its_document():
    engine, documents = await library()
    try:
        vellmar, _ = documents["Vellmar"]
        nodes = await nodes_of(engine, vellmar)
        harbour = nodes[(1, 0)].episode_id
        standing = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        assert len(standing.items) == 3 and standing.refused == (), "summaries still there confirm their paths"
        real = engine.attachment

        async def rebuilt(space, attachment_id):
            got = await real(space, attachment_id)
            if attachment_id == nodes[(1, 0)].detail and not done:
                done.append(await engine.forget("s", harbour))  # a rebuild forgets the old nodes
            return got

        done: list[object] = []
        engine.attachment = rebuilt
        try:
            found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        finally:
            engine.attachment = real
        assert found.items == () and found.refused == ({"episode_id": vellmar, "reason": "tree_changed"},), \
            "no item comes back through a summary that is gone"
        assert "a summary on the path to their chunks was forgotten or changed while the tree was read" in found.why
    finally:
        await engine.close()

    engine, documents = await library(index=NoStoredVectors())
    try:
        vellmar, _ = documents["Vellmar"]
        nodes = await nodes_of(engine, vellmar)
        real = engine.attachment

        async def early(space, attachment_id):
            got = await real(space, attachment_id)
            if attachment_id == nodes[(1, 0)].detail and not done:
                done.append(await engine.forget("s", nodes[(1, 0)].episode_id))  # before its chunks were ever read
            return got

        done = []
        engine.attachment = early
        try:
            found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1, episode_ids=[vellmar])
        finally:
            engine.attachment = real
        assert found.steps[1]["chosen"] == [nodes[(1, 0)].episode_id], "the forgotten section is still kept by its words"
        assert found.items == () and found.refused == ({"episode_id": vellmar, "reason": "tree_changed"},), \
            "a summary whose chunks were gone when first read confirms nothing"
    finally:
        await engine.close()


async def test_a_document_refused_at_the_last_read_leaves_the_receipt_as_well_as_the_answer():
    engine, documents = await library()
    try:
        vellmar, _ = documents["Vellmar"]
        brannock, _ = documents["Brannock"]
        real_get = engine.documents.get_chunks

        async def gone(space, ids):
            engine.documents.get_chunks = real_get
            await engine.forget("s", vellmar)
            return await real_get(space, ids)

        engine.documents.get_chunks = gone
        try:
            found = await traverse_summaries(engine, "s", "harbour of Vellmar", branching=1)
        finally:
            engine.documents.get_chunks = real_get
        assert found.items == () and found.refused == ({"episode_id": vellmar, "reason": "source_gone"},)
        assert (found.documents, found.leaves, found.cut_by_limit) == ((brannock,), 0, 0)
        assert "of 1 document tree(s) to 0 chunk(s)" in found.why
    finally:
        await engine.close()

    engine, documents = await library()
    try:
        vellmar, _ = documents["Vellmar"]
        brannock, _ = documents["Brannock"]
        both = await traverse_summaries(engine, "s", "Vellmar Brannock harbour river", branching=2, limit=3)
        assert {item.episode_id for item in both.items} != {vellmar, brannock} and both.cut_by_limit == 3, \
            "the leaves beyond the limit hold the other document"

        async def failing(space, ids):
            raise SconeError("the store did not answer")

        engine.documents.get_chunks = failing
        unread = await traverse_summaries(engine, "s", "Vellmar Brannock harbour river", branching=2, limit=3)
        assert unread.items == () and {one["episode_id"] for one in unread.refused} == {vellmar, brannock}
        assert all(one["reason"] == "unread" for one in unread.refused)
        assert (unread.documents, unread.leaves, unread.cut_by_limit) == ((), 0, 0), \
            "a confirming read that failed confirms no document's chunks, returned or cut"
    finally:
        await engine.close()
