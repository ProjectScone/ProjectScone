"""A PDF text layer that decodes to nothing a reader can use.

A font without a usable Unicode map extracts as private-use code points,
``(cid:N)`` runs or replacement characters: the page is not empty, so OCR
in ``missing_text`` mode never looked at it, and the garbage was indexed as
the page's text. The reference checks extracted text for private-use and
``(cid:)`` runs and falls back to OCR. Here a page is unreadable when at
least ``MIN_UNREADABLE`` of its visible characters, and ``GARBLED_SHARE``
of them, are such characters. OCR in its default mode recognizes such a
page; without OCR the text is kept, never deleted, and the page is named
as unreadable in the document's metadata and its coverage.
"""

from io import BytesIO

import pytest

from scone_memory.ingestion.text_layer import GARBLED_SHARE, MIN_UNREADABLE, unreadable

pypdf = pytest.importorskip('pypdf')
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject  # noqa: E402


def mixed_pdf(pages):
    """Pages of (text, garbled): a garbled page's font maps every code to a private-use code point."""
    writer = pypdf.PdfWriter()
    for text, garbled in pages:
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica'),
                                 NameObject('/Encoding'): NameObject('/WinAnsiEncoding')})
        if garbled:
            cmap = DecodedStreamObject()
            cmap.set_data(b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /Garbled def "
                          b"1 begincodespacerange <00> <FF> endcodespacerange 1 beginbfrange <20> <7E> <E020> "
                          b"endbfrange endcmap CMapName currentdict /CMap defineresource pop end end")
            font[NameObject('/ToUnicode')] = writer._add_object(cmap)
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject('/Resources')] = DictionaryObject(
            {NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 72 720 Td <{text.encode("cp1252").hex()}> Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    target = BytesIO()
    writer.write(target)
    return target.getvalue()


READABLE = "Juniper ships on Friday from the north yard."
PAGES = ((READABLE, False), ("The crane survey found rust on the jib and slew ring.", True))


def test_private_use_replacement_control_and_cid_runs_are_unreadable():
    assert unreadable("\ue04a\ue075\ue06e\ue069\ue070\ue065\ue072\ue020\ue073\ue068")
    assert unreadable("(cid:12)(cid:15)(cid:44) (cid:7)(cid:9)")
    assert unreadable("\ufffd\ufffd\ufffd\ufffd x \ufffd\ufffd\ufffd\ufffd")
    assert unreadable("\x01\x02\x03\x04\x05\x06\x07\x08 ok")
    assert unreadable("\U000f0001" * 10), "the supplementary private-use planes too"


def test_readable_text_and_a_few_icons_are_not():
    assert not unreadable(READABLE)
    assert not unreadable("港口起重机已检修。发现吊臂生锈！")
    assert not unreadable("\ue001 Home  \ue002 Search  \ue003 Settings " + READABLE * 2), "icon glyphs beside text"
    assert not unreadable("\ue001\ue002"), "two icons on an otherwise blank page are not a garbled text layer"
    assert not unreadable("line one\n\tline two\r\n"), "tabs and line breaks are not control noise"
    assert not unreadable("Total\n\n\n\n\t\t" * 6), "a sparse page is mostly layout whitespace, which is not counted"
    assert not unreadable("")


def test_the_share_and_the_count_both_bite():
    visible = 100
    at_share = int(visible * GARBLED_SHARE)
    # The spaces between words are not characters of either kind.
    assert unreadable("\ue000" * at_share + " a" * (visible - at_share))
    assert not unreadable("\ue000" * (at_share - 1) + " a" * (visible - at_share + 1))
    runs = "".join(f"(cid:{n})" for n in range(1, 6))  # 35 characters, each run counted whole
    assert unreadable(runs + "a" * (visible - len(runs)))
    assert not unreadable(runs + "a" * 90), "the runs are visible characters too: 35 of 125 is under the share"
    assert unreadable("\ue000" * MIN_UNREADABLE)
    assert not unreadable("\ue000" * (MIN_UNREADABLE - 1))


async def test_ocr_in_its_default_mode_recognizes_an_unreadable_page_and_keeps_a_readable_one():
    pytest.importorskip('pypdfium2')
    from scone_memory.ingestion.pdf import PdfLimits
    from scone_memory.ingestion.pdf_ocr import OcrPdfParser

    from .test_pdf_ocr import ObservedOcr

    engine = ObservedOcr()
    parsed = await OcrPdfParser(engine).parse(mixed_pdf(PAGES), PdfLimits())
    assert engine.calls == 1
    assert [page.extraction for page in parsed.pages] == ['text_layer', 'ocr']
    assert parsed.text == f"{READABLE}\n\nCafé uses Polaris"


async def test_without_ocr_the_file_import_keeps_the_text_and_names_the_page():
    from scone_memory.ingestion.formats.registry import BuiltinDocumentParser

    parsed = await BuiltinDocumentParser().parse(mixed_pdf(PAGES), "survey.pdf")
    first, second = parsed.segments
    assert first.text == READABLE and "unreadable" not in first.metadata
    assert second.metadata["unreadable"] == "true" and second.text.startswith("\ue054")
    assert parsed.metadata["unreadable_pages"] == "2"


async def test_a_readable_file_import_says_nothing_new():
    from scone_memory.ingestion.formats.registry import BuiltinDocumentParser

    parsed = await BuiltinDocumentParser().parse(mixed_pdf(((READABLE, False),)), "survey.pdf")
    assert "unreadable_pages" not in parsed.metadata and "unreadable" not in parsed.segments[0].metadata


async def test_an_ingested_pdf_with_an_unreadable_page_is_partial_and_names_it():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.ingestion import ingest_pdf

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        ingested = await ingest_pdf(engine, "default", mixed_pdf(PAGES), filename="survey.pdf")
        episode = await engine.episode("default", ingested.added.episode_id)
        readable = await ingest_pdf(engine, "default", mixed_pdf(((READABLE, False),)), filename="plain.pdf")
        plain = await engine.episode("default", readable.added.episode_id)
    finally:
        await engine.close()
    assert ingested.unreadable_pages == (2,) and ingested.empty_pages == ()
    assert episode.metadata["pdf_coverage"] == "partial" and episode.metadata["pdf_unreadable_pages"] == "2"
    assert readable.unreadable_pages == () and plain.metadata["pdf_coverage"] == "text_layer"
    assert "pdf_unreadable_pages" not in plain.metadata


async def test_the_resumable_ocr_workflow_recognizes_an_unreadable_page_too(tmp_path):
    """The workflow chooses its pages in a loop of its own; it must choose the same pages the parser does."""
    pytest.importorskip('pypdfium2')
    from scone_memory.ingestion.pdf_ocr import OcrPdfOptions

    from .test_pdf_ocr import ObservedOcr
    from .test_pdf_ocr_workflow import open_memory, retain, workflow

    memory = await open_memory(tmp_path)
    original = await retain(memory, mixed_pdf(PAGES))
    engine = ObservedOcr()
    job = workflow(memory, tmp_path, engine, options=OcrPdfOptions())
    try:
        result = await job.run('garbled', space='alpha', attachment_id=original.attachment_id)
        episode = await memory.episode('alpha', result.added.episode_id)
    finally:
        job.close()
        await memory.close()
    assert engine.calls == 1 and result.unreadable_pages == ()
    assert episode.content == f"{READABLE}\n\nCafé uses Polaris"
