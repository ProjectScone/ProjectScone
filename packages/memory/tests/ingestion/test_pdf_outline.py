"""The sections a PDF's own bookmarks give its pages.

A PDF's outline says where each chapter and section begins, and the text
layer does not: a page reading "within 30 days" carried nothing saying it
sits under "Refunds" in "Chapter 2". Here the worker reads the bookmarks,
and each page gets the chain of titles in force on it -- the last bookmark
at each level that begins on or before the page -- as its segment's
``section``, while the text stays exactly as it was. An outline too large
to read whole is read to its bound and said to be capped; one that cannot
be read is said to be unreadable; a PDF without one says nothing new.
"""

from __future__ import annotations

from io import BytesIO

import pytest

pypdf = pytest.importorskip("pypdf")

from scone_memory.ingestion._pdf_worker import extract  # noqa: E402
from scone_memory.ingestion.formats.registry import BuiltinDocumentParser  # noqa: E402
from scone_memory.ingestion.pdf import PdfLimits  # noqa: E402

from .test_pdf_ingestion import pdf_bytes  # noqa: E402

PAGES = ("Introduction to the survey.", "Scope of the crane survey.", "Refunds within 30 days.",
         "Returns by post.", "Appendix tables.")


def with_outline(build) -> bytes:
    reader = pypdf.PdfReader(BytesIO(pdf_bytes(pages=PAGES)))
    writer = pypdf.PdfWriter(clone_from=reader)
    build(writer)
    target = BytesIO()
    writer.write(target)
    return target.getvalue()


def chapters(writer) -> None:
    one = writer.add_outline_item("Chapter 1", 0)
    writer.add_outline_item("Scope", 1, parent=one)
    two = writer.add_outline_item("Chapter 2", 2)
    writer.add_outline_item("Refunds", 2, parent=two)
    writer.add_outline_item("Returns", 3, parent=two)
    writer.add_outline_item("Appendix", 4)


def test_each_page_gets_the_titles_in_force_on_it():
    parsed = extract(with_outline(chapters), PdfLimits())
    assert [page.section for page in parsed.pages] == [
        ("Chapter 1",), ("Chapter 1", "Scope"), ("Chapter 2", "Refunds"), ("Chapter 2", "Returns"), ("Appendix",)]
    assert parsed.outline == "read" and parsed.text == "\n\n".join(PAGES)


def test_a_pdf_without_an_outline_says_nothing_new():
    parsed = extract(pdf_bytes(pages=PAGES), PdfLimits())
    assert all(page.section == () for page in parsed.pages) and parsed.outline == "none"
    assert "section" not in parsed.pages[0].model_dump() and "outline" not in parsed.model_dump()


def test_a_page_before_the_first_bookmark_has_no_section():
    parsed = extract(with_outline(lambda writer: writer.add_outline_item("Chapter 2", 2)), PdfLimits())
    assert [page.section for page in parsed.pages] == [(), (), ("Chapter 2",), ("Chapter 2",), ("Chapter 2",)]


def test_an_outline_past_its_bound_is_read_to_the_bound_and_said_to_be_capped(monkeypatch):
    import scone_memory.ingestion._pdf_worker as worker

    monkeypatch.setattr(worker, "MAX_OUTLINE_ITEMS", 2)
    parsed = extract(with_outline(chapters), PdfLimits())
    assert parsed.outline == "capped"
    assert parsed.pages[1].section == ("Chapter 1", "Scope") and parsed.pages[4].section == ("Chapter 1", "Scope")


def test_a_long_title_is_cut_and_the_file_import_carries_the_section():
    long_title = "Chapter " + "x" * 400

    def build(writer):
        writer.add_outline_item(long_title, 0)

    parsed = extract(with_outline(build), PdfLimits())
    assert len(parsed.pages[0].section[0]) <= 256 and parsed.pages[0].section[0].endswith("…")

    import asyncio

    document = asyncio.run(BuiltinDocumentParser().parse(with_outline(chapters), "survey.pdf"))
    assert document.segments[2].metadata["section"] == "Chapter 2 > Refunds"
    assert document.metadata["outline"] == "read"
    plain = asyncio.run(BuiltinDocumentParser().parse(pdf_bytes(pages=PAGES), "plain.pdf"))
    assert "section" not in plain.segments[0].metadata and "outline" not in plain.metadata


def test_a_deep_outline_of_long_titles_fits_a_metadata_value_keeping_the_innermost():
    """Eight levels of 256-character titles in a script of three-byte characters are
    about six kilobytes; a metadata value holds four, and the import must not fail."""
    import asyncio

    def build(writer):
        parent = None
        for level in range(8):
            parent = writer.add_outline_item(f"{level}" + "章" * 255, 0, parent=parent)

    document = asyncio.run(BuiltinDocumentParser().parse(with_outline(build), "deep.pdf"))
    section = document.segments[0].metadata["section"]
    assert len(section.encode()) <= 4096 and section.startswith("… > ") and section.endswith("7" + "章" * 255)
