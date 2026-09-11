"""Names from the ledger shown in Markdown render as the text they are.

A stored subject is text someone wrote, so a report or a note must not let
it become a table column, a heading, a link, an image or raw HTML. Each
name is escaped where it is placed; the JSON forms keep the raw value.
"""
from __future__ import annotations

import re

import pytest

from scone_memory.core.models import Fact
from scone_memory.entities.analysis import analyze_projection
from scone_memory.entities.markdown import literal
from scone_memory.entities.project import project_entities
from scone_memory.entities.report import build_report, render_markdown

PIPE = "Alice | Injected"
HTML = '<img src="https://example.invalid/x.png">'
MARKUP = "**Bold** _claim_ [link](https://example.invalid) #tag ==mark== `code`"
HOSTILE = (PIPE, HTML, MARKUP)


def fact(number: int, subject: str, predicate: str, object_: str) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject, predicate=predicate, object=object_,
                valid_from="2025-01-01T00:00:00Z")


LEDGER = [fact(1, PIPE, "knows", "Bob"), fact(2, HTML, "knows", "Bob"), fact(3, MARKUP, "knows", "Bob"),
          fact(4, "Bob", "works_at", "Acme"), fact(5, "Carol", "knows", "Dan"), fact(6, "Dan", "knows", PIPE)]


@pytest.fixture
def projection():
    return project_entities("alpha", LEDGER, revision=1)


def report_markdown(projection) -> str:
    report = build_report(projection, analyze_projection(projection),
                          meta={"digest": projection.digest, "revision": 1},
                          filters={"status": "current", "as_of": None}, coverage={"facts_read": 6, "facts_counted": 6})
    return render_markdown(report)


def unescaped(text: str) -> str:
    return re.sub(r"\\([!-/:-@\[-`{-~])", r"\1", text)


def bare(text: str, character: str) -> int:
    """How many times ``character`` appears without a backslash escaping it."""
    count, index = 0, 0
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        count += text[index] == character
        index += 1
    return count


def test_a_pipe_in_a_name_does_not_add_a_table_column(projection):
    rows = [line for line in report_markdown(projection).splitlines() if line.startswith("|")]
    assert len(rows) > 2
    assert {bare(row, "|") for row in rows} == {bare(rows[0], "|")}


def test_no_name_becomes_html_a_link_a_tag_or_emphasis(projection):
    markdown = report_markdown(projection)
    assert bare(markdown, "<") == 0 and bare(markdown, "[") == 0
    for markup, escape in (("**Bold**", "\\*"), ("#tag", "\\#"), ("==mark==", "\\="), ("`code`", "\\`")):
        assert markup not in markdown.replace(escape, ""), markup


def test_every_hostile_name_reads_back_verbatim(projection):
    markdown = unescaped(report_markdown(projection))
    for name in HOSTILE:
        assert name in markdown


def test_literal_keeps_one_line_and_escapes_only_what_markdown_reads():
    assert literal("a\nb\r\nc") == "a b c"
    assert literal("works_at") == "works_at"
    assert literal("_claim_") == "\\_claim\\_"
    assert literal("- item") == "\\- item" and literal("1. item") == "1\\. item" and literal("Acme, Inc.") == "Acme, Inc."
    assert unescaped(literal(MARKUP)) == MARKUP
