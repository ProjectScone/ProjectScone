"""A chunk is embedded with the headings it was cut from under.

A chunk cut from a long document loses what the document said it was
about: "within 30 days" under "## Refund policy" in "# Chapter 4" is,
once cut, just "within 30 days", and a question about the refund policy
has nothing in that chunk's vector to find. The date and source prefix
already goes in front of what is embedded; this adds the path of
headings above the chunk, outermost first. For code, the same slot takes
the file and the declarations the chunk sits in -- `code_context` was
written for exactly that and had no caller.

Only what is embedded changes. Stored text is never touched (I1), so a
recall still returns the exact excerpt. Opt-in, because it changes
vectors; bounded, and the receipt says when the bound bit.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion import batch

BODY = ("The shop takes returns of anything unopened, and it says so on the receipt and on the wall behind the "
        "counter, in letters large enough to read from the door. ") * 20
DOC = "# Chapter 4\n\nThe chapter begins here.\n\n## Refund policy\n\n" + BODY + "\n\nwithin 30 days of purchase.\n"
CODE = ("import os\n\n\nclass Beta:\n    def method(self):\n        " + "value = os.getcwd()\n        " * 30
        + "return value\n\n\ndef alpha():\n    return 2\n")


class Recording(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.inputs: list[str] = []

    async def embed(self, texts):
        self.inputs.extend(texts)
        return await super().embed(texts)


async def engine_with(**options):
    embedder = Recording()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder, **options).open()
    return memory, embedder


async def test_a_chunk_under_headings_is_embedded_with_their_path_and_stored_unchanged():
    memory, embedder = await engine_with(heading_context=True)
    try:
        added = await memory.remember("default", DOC, source="guide.md")
        chunks = await memory.documents.chunks_of("default", added.episode_id)
        assert len(chunks) >= 2
        last = chunks[-1]
        assert last.text == DOC.encode()[last.start:last.end].decode(), "stored text is the source, unchanged"
        [seen] = [text for text in embedder.inputs if text.endswith(last.text)]
        assert seen.startswith("Chapter 4 > Refund policy\n"), seen[:60]
        first = [text for text in embedder.inputs if text.endswith(chunks[0].text)][0]
        assert first.startswith("Chapter 4\n") or first == chunks[0].text, first[:40]
        assert added.embedding_context == {"mode": "headings", "chunks_with_context": added.embedding_context["chunks_with_context"],
                                           "context_bytes": added.embedding_context["context_bytes"], "context_cut": 0}
        assert added.embedding_context["chunks_with_context"] >= 1 and added.embedding_context["context_bytes"] > 0
    finally:
        await memory.close()


async def test_off_by_default_and_the_inputs_are_exactly_as_before():
    memory, embedder = await engine_with()
    try:
        added = await memory.remember("default", DOC, source="guide.md")
        chunks = await memory.documents.chunks_of("default", added.episode_id)
        assert embedder.inputs == [chunk.text for chunk in chunks]
        assert added.embedding_context is None
    finally:
        await memory.close()


async def test_a_code_chunk_is_embedded_with_its_file_and_declarations():
    memory, embedder = await engine_with(heading_context=True)
    try:
        added = await memory.remember("default", CODE, kind="file", source="pkg/a.py")
        chunks = await memory.documents.chunks_of("default", added.episode_id)
        inside = [chunk for chunk in chunks if "value = os.getcwd()" in chunk.text]
        assert inside
        [seen] = [text for text in embedder.inputs if text.endswith(inside[0].text)]
        assert seen.startswith("pkg/a.py | Beta"), seen[:60]
    finally:
        await memory.close()


async def test_a_long_path_keeps_its_innermost_headings_and_says_it_was_cut(monkeypatch):
    monkeypatch.setattr(batch, "MAX_HEADING_CONTEXT_BYTES", 20)
    memory, embedder = await engine_with(heading_context=True)
    try:
        added = await memory.remember("default", DOC, source="guide.md")
        chunks = await memory.documents.chunks_of("default", added.episode_id)
        [seen] = [text for text in embedder.inputs if text.endswith(chunks[-1].text)]
        line = seen.split("\n", 1)[0]
        assert line == "Refund policy" and len(line.encode()) <= 20, line
        assert added.embedding_context["context_cut"] >= 1
    finally:
        await memory.close()


async def test_recovery_embeds_the_same_way():
    """A crash between storing the episode and cutting it leaves the episode
    row and its inflight mark; recovery must embed with the same context,
    or a recovered chunk's vector differs from an unrecovered one's."""
    from scone_memory.ingestion.batch import validated_record
    from scone_memory.ingestion.records import Record

    memory, embedder = await engine_with(heading_context=True)
    try:
        new = validated_record("default", Record(DOC, source="guide.md"), memory.clock())
        await memory.documents.mark_inflight("default", new.content_hash)
        episode = await memory.documents.insert_episode(new)
        report = await memory.recover()
        assert report.completed == 1, report
        chunks = await memory.documents.chunks_of("default", episode.episode_id)
        [seen] = [text for text in embedder.inputs if text.endswith(chunks[-1].text)]
        assert seen.startswith("Chapter 4 > Refund policy\n"), seen[:60]
    finally:
        await memory.close()


async def test_the_setting_reaches_every_engine():
    from scone_memory.runtime.config import ENGINE_SETTINGS, Settings, build_engine, build_in_process_engine

    settings = Settings.from_env({"SCONE_HEADING_CONTEXT": "1"})
    assert settings.heading_context is True and Settings.from_env({}).heading_context is False
    assert "heading_context" in ENGINE_SETTINGS
    assert (await build_in_process_engine(settings, HashEmbedder())).heading_context is True
    built = await build_engine(settings)
    try:
        assert built.heading_context is True
    finally:
        await built.close()


async def test_replace_refuses_when_the_setting_changes_while_preparing(monkeypatch):
    """Vectors from one setting beside vectors from the other would compare
    as if they meant the same thing. The configuration a replacement is
    prepared under includes this setting, like every other that changes
    what is embedded."""
    from scone_memory.core.errors import InvalidInput
    from scone_memory.ingestion.records import Record

    memory, _ = await engine_with(heading_context=True)
    try:
        await memory.replace("default", Record(DOC, source="guide.md", dedup_key="guide"))
        original = batch.embed_pending

        async def flip(*args, **kwargs):
            memory.heading_context = False
            return await original(*args, **kwargs)

        monkeypatch.setattr(batch, "embed_pending", flip)
        with pytest.raises(InvalidInput, match="configuration changed"):
            await memory.replace("default", Record(DOC + "\nMore.", source="guide.md", dedup_key="guide"))
    finally:
        await memory.close()
