"""A file's own headings decide where it is cut and what its chunks are embedded under.

A Word heading, an OpenDocument ``text:h`` and an HTML ``h2`` reach the
stored content as plain lines: the level lived in a style or a tag, and
the readers now keep it beside the text as ``heading_level``. The cutter
and the heading path read only the text, so for those files structure
chunking found no structure and the heading path found no headings. Here
both read the headings the document said it had, from the manifest kept
with the episode, the same way ingestion and recovery alike.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import batch
from scone_memory.ingestion.document_outline import Heading, outline
from scone_memory.ingestion.files import ingest_document
from scone_memory.ingestion.formats.types import DocumentSegment
from scone_memory.ingestion.structure_chunks import structured_spans

PARA = ("The shop takes back anything unopened, and says so on the receipt and on the wall behind the "
        "counter, in letters large enough to read from the door. ") * 4
HTML = (f"<h1>Terms</h1><p>{PARA}</p><h2>Refunds</h2><p>{PARA}</p><h2>Shipping</h2><p>{PARA}</p>").encode()
TARGET = 800


class Recording(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.inputs: list[str] = []

    async def embed(self, texts):
        self.inputs.extend(texts)
        return await super().embed(texts)


async def engine_with(**options):
    embedder = Recording()
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder, chunk_target=TARGET,
                                **options).open()
    return memory, embedder


def test_the_outline_is_each_heading_segments_byte_offset_level_and_title():
    segments = (DocumentSegment(text="Terms", locator="line:1", metadata={"heading_level": "1"}),
                DocumentSegment(text="Café prices.", locator="line:2"),
                DocumentSegment(text="Refunds\nand returns", locator="line:3", metadata={"heading_level": "2"}),
                DocumentSegment(text="Not a level", locator="line:4", metadata={"heading_level": "x"}))
    content = "\n\n".join(segment.text for segment in segments).encode()
    found = outline(segments)
    assert found == (Heading(0, 1, "Terms"), Heading(content.index(b"Refunds"), 2, "Refunds and returns"))


def test_the_cutter_cuts_at_the_documents_headings_and_counts_them():
    content = "\n\n".join(["Terms", PARA, "Refunds", PARA, "Shipping", PARA])
    encoded = content.encode()
    headings = (Heading(0, 1, "Terms"), Heading(encoded.index(b"Refunds"), 2, "Refunds"),
                Heading(encoded.index(b"Shipping"), 2, "Shipping"))
    plain = structured_spans(content, TARGET)
    assert plain.units == 0, "the text alone carries no structure"
    cut = structured_spans(content, TARGET, headings=headings)
    starts = [content[span.start:span.end].split("\n", 1)[0] for span in cut.spans]
    assert starts == ["Terms", "Refunds", "Shipping"], starts
    assert cut.document_headings == 3 and cut.record()["document_headings"] == 3
    assert "document_headings" not in plain.record()
    # Asked to use the file's headings and given none, the receipt says none were used.
    assert structured_spans(content, TARGET, headings=()).record()["document_headings"] == 0
    # Past the unit bound, it counts the headings read, not the headings given.
    bounded = structured_spans(content, TARGET, headings=headings, units_max=2)
    assert bounded.capped and bounded.document_headings == 2


def test_the_heading_path_reads_the_documents_headings():
    content = "\n\n".join(["Terms", PARA, "Refunds", PARA, "Shipping", PARA])
    encoded = content.encode()
    refunds, shipping = encoded.index(b"Refunds"), encoded.index(b"Shipping")
    headings = (Heading(0, 1, "Terms"), Heading(refunds, 2, "Refunds"), Heading(shipping, 2, "Shipping"))
    spans = [(0, refunds), (refunds, shipping), (shipping, len(encoded))]
    lines, cut = batch.context_lines(content, "attachment:x", spans, TARGET, headings)
    assert lines == ["Terms", "Terms > Refunds", "Terms > Shipping"] and cut == 0, lines
    assert batch.context_lines(content, "attachment:x", spans, TARGET)[0] == ["", "", ""]


async def test_an_imported_file_is_cut_at_its_headings_and_embedded_under_them():
    memory, embedder = await engine_with(structure_aware=True, heading_context=True)
    try:
        result = await ingest_document(memory, "default", HTML, filename="terms.html")
        chunks = await memory.documents.chunks_of("default", result.added.episode_id)
        assert [chunk.text.split("\n", 1)[0] for chunk in chunks] == ["Terms", "Refunds", "Shipping"]
        assert result.added.structure is not None and result.added.structure["document_headings"] == 3
        assert result.added.embedding_context is not None and result.added.embedding_context["chunks_with_context"] == 3
        refunds = [text for text in embedder.inputs if text.endswith(chunks[1].text)]
        assert refunds and refunds[0].startswith("Terms > Refunds\n"), refunds[:1]
    finally:
        await memory.close()


async def test_a_file_without_heading_marks_and_a_plain_episode_are_cut_as_before():
    memory, _ = await engine_with(structure_aware=True)
    try:
        plain = "\n\n".join(["Terms", PARA, "Refunds", PARA])
        added = await memory.remember("default", plain, source="terms.txt")
        chunks = await memory.documents.chunks_of("default", added.episode_id)
        assert [chunk.text for chunk in chunks] == [plain[s.start:s.end] for s in structured_spans(plain, TARGET).spans]
        assert "document_headings" not in (added.structure or {})
    finally:
        await memory.close()


async def test_recovery_cuts_and_embeds_a_file_exactly_as_an_uninterrupted_import():
    """A crash between storing the episode and its chunks leaves the row, its inflight mark and the
    retained manifest. Recovery must read the same headings, or the recovered cut differs."""
    memory, embedder = await engine_with(structure_aware=True, heading_context=True)
    try:
        whole = await ingest_document(memory, "default", HTML, filename="terms.html")
        expected_chunks = [c.text for c in await memory.documents.chunks_of("default", whole.added.episode_id)]
        expected_inputs = list(embedder.inputs)
    finally:
        await memory.close()

    class Crash(BaseException):
        pass

    memory, embedder = await engine_with(structure_aware=True, heading_context=True)
    try:
        inserting = memory.documents.insert_chunks

        async def crash(*args, **kwargs):
            raise Crash()
        memory.documents.insert_chunks = crash
        with pytest.raises(Crash):
            await ingest_document(memory, "default", HTML, filename="terms.html")
        memory.documents.insert_chunks = inserting
        embedder.inputs.clear()
        report = await memory.recover()
        assert report.completed == 1, report
        [episode] = await memory.documents.recent_episodes("default", 5)
        assert [c.text for c in await memory.documents.chunks_of("default", episode.episode_id)] == expected_chunks
        assert embedder.inputs == expected_inputs
    finally:
        await memory.close()


async def test_a_file_episode_whose_manifest_does_not_match_its_text_is_refused():
    memory, _ = await engine_with(structure_aware=True)
    try:
        whole = await ingest_document(memory, "default", HTML, filename="terms.html")
        episode = await memory.episode("default", whole.added.episode_id)
        with pytest.raises(InvalidInput, match="heading"):
            await memory.remember("default", episode.content + " altered", kind="file", source=episode.source,
                                  metadata={key: episode.metadata[key] for key in
                                            ("document_format", "document_original", "document_manifest")})
    finally:
        await memory.close()
