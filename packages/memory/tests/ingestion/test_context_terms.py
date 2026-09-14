"""What a passage is under, for the lane that finds what a passage lacks.

A chunk under the heading "Refunds" in a document titled "Billing rules"
need not say either word; a query about billing refunds then misses it
in the lexical lane, which finds the words a passage has and only those.
This derives, with no model, the words a chunk is under -- its headings,
the document's title and the source's name -- and keeps only what the
chunk itself lacks, so an index of it adds what the text lane cannot see
and repeats nothing it can.
"""

from __future__ import annotations

from scone_memory.ingestion import context_terms as module
from scone_memory.ingestion.context_terms import chunk_context, document_title, heading_path, source_words
from scone_memory.ingestion.structure import parse_structure
from scone_memory.retrieval.lexical import tokenize

DOC = ("# Billing rules\n\nWhat the company does with money.\n\n## Refunds\n\n"
       "A customer may ask for money back within 30 days.\n\n## Late fees\n\nAfter the due date 2% is added.\n")


def test_the_heading_path_of_a_chunk_is_outermost_first():
    structure = parse_structure(DOC)
    assert heading_path(structure, DOC.index("A customer")) == ("Billing rules", "Refunds")
    assert heading_path(structure, DOC.index("After the due")) == ("Billing rules", "Late fees")
    assert heading_path(structure, DOC.index("What the company")) == ("Billing rules",)


def test_a_title_is_the_top_heading_or_a_short_first_line():
    assert document_title(DOC, parse_structure(DOC)) == "Billing rules"
    assert document_title("Quarterly plan\n\nWe will ship in March.", None) == "Quarterly plan"
    assert document_title("We will ship in March, come what may.\n\nMore.", None) is None, "a sentence is not a title"
    assert document_title("x" * (module.MAX_TITLE_CHARS + 1) + "\nbody", None) is None
    assert document_title("   \n\n", None) is None


def test_source_words_come_from_the_name_not_the_path():
    assert source_words("docs/billing-rules.md") == ("billing", "rules")
    assert source_words("/tmp/Q3_report.final.PDF") == ("q3", "report", "final")
    assert source_words("session_4a") == ("session", "4a")
    assert source_words(None) == () and source_words("") == ()


def test_a_chunk_keeps_only_the_structure_it_lacks():
    structure = parse_structure(DOC)
    chunk = "A customer may ask for money back within 30 days."
    found = chunk_context(DOC, start=DOC.index(chunk), chunk_text=chunk, source="docs/billing-rules.md",
                          structure=structure)
    assert found.headings == ("Billing rules", "Refunds") and found.title == "Billing rules"
    assert found.source_words == ("billing", "rules")
    text = found.text()
    assert set(tokenize(text)) == {"billing", "rules", "refunds"}, "structure only: no word of the body's own"
    assert text.count("billing") == 1, "each word once, however many places said it"
    assert found.omitted == 0


def test_a_chunk_that_already_says_everything_gets_no_context():
    doc = "Refunds\n\nRefunds are paid in thirty days."
    chunk = "Refunds are paid in thirty days."
    found = chunk_context(doc, start=doc.index(chunk), chunk_text=chunk, source="refunds.txt")
    assert found.text() == "" and found.headings == () and found.title == "Refunds"


def test_context_words_are_capped_with_the_omitted_count_on_the_record(monkeypatch):
    monkeypatch.setattr(module, "MAX_TERMS", 2)
    structure = parse_structure(DOC)
    chunk = "A customer may ask for money back within 30 days."
    found = chunk_context(DOC, start=DOC.index(chunk), chunk_text=chunk, source="docs/billing-and-refund-desk.md",
                          structure=structure)
    assert tokenize(found.text()) == ["billing", "rules"] and found.omitted > 0, "headings come first, they say the most"


def test_headings_come_before_the_source_name_when_the_bound_bites(monkeypatch):
    monkeypatch.setattr(module, "MAX_TERMS", 1)
    doc = "# Escalations\n\nBody sentence about the desk.\n"
    chunk = "Body sentence about the desk."
    found = chunk_context(doc, start=doc.index(chunk), chunk_text=chunk, source="support/handbook.md")
    assert tokenize(found.text()) == ["escalations"], "the heading says the most; the file name waits"
    assert found.omitted == 1, "the title repeats the heading and costs nothing; the file name was left out"


def test_the_structure_is_parsed_when_none_is_given():
    chunk = "After the due date 2% is added."
    found = chunk_context(DOC, start=DOC.index(chunk), chunk_text=chunk, source=None)
    assert found.headings == ("Billing rules", "Late fees")


def test_the_record_is_the_words_and_the_counts():
    doc = "Plan\n\nShip in March."
    found = chunk_context(doc, start=doc.index("Ship"), chunk_text="Ship in March.", source="plan.md")
    assert found.record() == {"headings": [], "title": "Plan", "source_words": ["plan"], "terms": 1, "omitted": 0}, \
        "two places say one word; the index carries it once"
