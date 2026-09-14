"""One chunk per unit the file's reader declared: a page, a slide, a row, a record.

A slide deck, a spreadsheet and a scanned report already say where one
thing ends and the next begins -- the reader names every segment by its
page, slide or row -- and length, structure and semantic cuts all ignore
that: a chunk of a deck could hold the end of one slide and the start of
the next, a chunk of a table three rows and half of a fourth. Asked for
``chunking="unit"``, an imported file is cut where its units are, one
chunk each, and a unit longer than the target is split by length with the
receipt saying so. Units come from the manifest kept with the episode, so
recovery cuts the same way.
"""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import BuiltinDocumentParser, ingest_document
from scone_memory.ingestion.documents import ingest_pdf
from scone_memory.ingestion.document_outline import source_units
from scone_memory.ingestion.formats.types import DocumentSegment
from scone_memory.ingestion.pdf import ParsedPdf, PdfPage
from scone_memory.ingestion.structure import SourceUnit
from scone_memory.ingestion.unit_chunks import unit_spans
from scone_memory.ingestion.chunker import chunk_spans

ROWS = b"name,city,note\n" + b"".join(f"Person {n},City {n},likes the number {n}\n".encode() for n in range(1, 6))


def segment(locator: str, text: str) -> DocumentSegment:
    return DocumentSegment(text=text, locator=locator)


def test_units_are_read_from_the_locators_the_reader_gave():
    segments = (segment("paragraph:1", "Intro."), segment("paragraph:2", "More intro."),
                segment("slide:1/shape:1", "Title"), segment("slide:1/notes/paragraph:1", "Say hello"),
                segment("slide:2/shape:1", "Next"), segment("table:1/row:1", "a: 1"), segment("table:1/row:2", "a: 2"),
                segment("sheet:Q1/cell:B7", "7"), segment("sheet:Q1/cell:C7", "8"), segment("sheet:Q1/cell:B8", "9"),
                segment("line:4#/name", "/name: Ada"), segment("line:4#/city", "/city: Rome"),
                segment("audio:0/segment:3/seconds:1.0-2.0", "hello"), segment("line:9", "plain text line"))
    content = "\n\n".join(s.text for s in segments).encode()
    found = source_units(segments)
    assert [(unit.label, content[unit.start:unit.end].decode()) for unit in found] == [
        ("text", "Intro.\n\nMore intro."), ("slide:1", "Title\n\nSay hello"), ("slide:2", "Next"),
        ("table:1/row:1", "a: 1"), ("table:1/row:2", "a: 2"), ("sheet:Q1/row:7", "7\n\n8"), ("sheet:Q1/row:8", "9"),
        ("line:4", "/name: Ada\n\n/city: Rome"), ("audio:0/segment:3", "hello"), ("text", "plain text line")]


def test_each_unit_is_one_chunk_and_a_long_one_is_split_by_length_and_counted():
    # Units are placed in bytes and chunks in code points: "é" is where the two differ.
    content = "Pagé one.\n\n" + "word " * 60 + "\n\nPage three."
    encoded = content.encode()
    second = encoded.index(b"word")
    units = (SourceUnit(0, len("Pagé one.".encode()), "page:1"), SourceUnit(second, second + 300, "page:2"),
             SourceUnit(len(encoded) - 11, len(encoded), "page:3"))
    cut = unit_spans(content, units, 100)
    texts = [content[span.start:span.end] for span in cut.spans]
    assert texts[0] == "Pagé one." and texts[-1] == "Page three."
    long = content[content.index("word"):content.index("word") + 300]
    assert texts[1:-1] == [long[s.start:s.end] for s in chunk_spans(long, 100)] and len(texts) > 3
    assert cut.record() == {"chunks": len(texts), "units": 3, "split_units": 1, "by_size": len(texts) - 3,
                            "kinds": {"page": 3}}
    # Text no unit holds still moves the next chunk's code-point offset by its characters.
    skipped = unit_spans("é\n\nab", (SourceUnit(len("é\n\n".encode()), len("é\n\nab".encode()), "row:1"),), 10)
    assert [("é\n\nab")[span.start:span.end] for span in skipped.spans] == ["ab"]
    with pytest.raises(ValueError):
        unit_spans("abcdef", (SourceUnit(0, 4, "row:1"), SourceUnit(2, 6, "row:2")), 10)
    with pytest.raises(ValueError):
        unit_spans("abc", (SourceUnit(0, 9, "row:1"),), 10)


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=2000).open()
    try:
        yield engine
    finally:
        await engine.close()


async def test_an_imported_table_is_cut_one_row_per_chunk_and_the_receipt_says_so(memory):
    result = await ingest_document(memory, "default", ROWS, filename="people.csv", chunking="unit")
    chunks = await memory.documents.chunks_of("default", result.added.episode_id)
    assert [chunk.text for chunk in chunks] == [f"name: Person {n}\ncity: City {n}\nnote: likes the number {n}"
                                                for n in range(1, 6)]
    assert result.added.chunking == "unit"
    assert result.added.structure == {"units": 5, "split_units": 0, "by_size": 0, "kinds": {"row": 5}}


async def test_the_same_table_cut_by_length_is_one_chunk(memory):
    result = await ingest_document(memory, "default", ROWS, filename="people.csv")
    assert len(await memory.documents.chunks_of("default", result.added.episode_id)) == 1


class Pages:
    async def parse(self, data, limits):
        text = "First page.\n\n\n\nThird page."
        size = dict(width_points=600., height_points=800., rotation=0)
        return ParsedPdf(text=text, parser="pages", pages=(
            PdfPage(number=1, start=0, end=11, empty=False, **size), PdfPage(number=2, start=13, end=13, empty=True, **size),
            PdfPage(number=3, start=15, end=26, empty=False, **size)))


async def test_a_pdf_is_cut_one_page_per_chunk_through_either_import(memory):
    pdf = await ingest_pdf(memory, "default", b"%PDF-1.7 pages", filename="report.pdf", parser=Pages(), chunking="unit")
    assert [c.text for c in await memory.documents.chunks_of("default", pdf.added.episode_id)] == ["First page.", "Third page."]
    assert pdf.added.structure == {"units": 2, "split_units": 0, "by_size": 0, "kinds": {"page": 2}}
    generic = await ingest_document(memory, "other", b"%PDF-1.7 pages", filename="report.pdf",
                                    parser=BuiltinDocumentParser(pdf_parser=Pages()), chunking="unit")
    assert [c.text for c in await memory.documents.chunks_of("other", generic.added.episode_id)] == ["First page.", "Third page."]


async def test_a_pdf_episode_whose_manifest_does_not_match_its_text_is_refused(memory):
    pdf = await ingest_pdf(memory, "default", b"%PDF-1.7 pages", filename="report.pdf", parser=Pages())
    episode = await memory.episode("default", pdf.added.episode_id)
    with pytest.raises(InvalidInput, match="PDF outline"):
        await memory.remember("default", episode.content + " altered", kind="file", source=episode.source,
                              chunking="unit", metadata={key: episode.metadata[key] for key in ("pdf_original", "pdf_manifest")})


async def test_unit_chunking_is_refused_for_a_record_without_units(memory):
    with pytest.raises(InvalidInput, match="unit"):
        await memory.remember("default", "Just prose.\n\nMore prose.", chunking="unit")
    with pytest.raises(InvalidInput, match="unit"):
        await ingest_document(memory, "default", b"# Title\n\nProse only.\n", filename="notes.md", chunking="unit")
    assert await memory.documents.recent_episodes("default", 5) == []


async def test_recovery_cuts_a_file_by_its_units_again():
    class Crash(BaseException):
        pass

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=2000).open()
    try:
        inserting = engine.documents.insert_chunks

        async def crash(*args, **kwargs):
            raise Crash()
        engine.documents.insert_chunks = crash
        with pytest.raises(Crash):
            await ingest_document(engine, "default", ROWS, filename="people.csv", chunking="unit")
        engine.documents.insert_chunks = inserting
        report = await engine.recover()
        assert report.completed == 1, report
        [episode] = await engine.documents.recent_episodes("default", 5)
        assert len(await engine.documents.chunks_of("default", episode.episode_id)) == 5
    finally:
        await engine.close()


async def test_over_http_a_file_import_can_ask_for_units(memory):
    import httpx
    from scone_memory.api import create_app

    transport = httpx.ASGITransport(app=create_app(memory, {"k": "default"}))
    async with httpx.AsyncClient(transport=transport, base_url="http://test", headers={"authorization": "Bearer k"}) as client:
        stored = await client.post("/v1/attachments", content=ROWS, headers={"content-type": "text/csv", "x-filename": "people.csv"})
        assert stored.status_code == 200, stored.text
        answer = await client.post("/v1/documents", json={"attachment_id": stored.json()["attachment_id"],
                                                          "chunking": "unit"})
        assert answer.status_code == 200, answer.text
        assert answer.json()["added"]["structure"]["units"] == 5
        wrong = await client.post("/v1/documents", json={"attachment_id": stored.json()["attachment_id"],
                                                         "chunking": "pages"})
        assert wrong.status_code == 400
