"""Where the links in an Office document point.

A Word paragraph reading "see the refund policy" with a link on the last
three words kept the words and dropped the address: the target lives in
the part's relationships, marked external, and the reader skipped every
external relationship. A slide's run keeps its link the same way. Here the
text is unchanged, and the segment says which bytes of it link where: the
link's text, its target and its UTF-8 span, an internal bookmark as
``#name``. A link whose target cannot be read is counted, never invented.
"""

from __future__ import annotations

import json

import pytest

from scone_memory.ingestion.formats.office import MAX_LINKS, parse_office
from scone_memory.ingestion.formats.types import DocumentLimits

from .test_office_formats import A, P, R, REL, W, ooxml_archive


def docx(body: str, relationships: str = "") -> bytes:
    parts = {"word/document.xml": f'<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>{body}</w:body></w:document>'}
    if relationships:
        parts["word/_rels/document.xml.rels"] = f'<Relationships xmlns="{REL}">{relationships}</Relationships>'
    return ooxml_archive(parts, main_part="word/document.xml")


def external(identifier: str, target: str) -> str:
    return f'<Relationship Id="{identifier}" Target="{target}" TargetMode="External" Type="{R}/hyperlink"/>'


def run(text: str) -> str:
    return f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r>'


def links(segment) -> list[dict]:
    return json.loads(segment.metadata["links"])


def test_a_word_link_keeps_its_text_and_says_where_it_points():
    body = (f'<w:p>{run("  See the ")}<w:hyperlink r:id="rId7">{run("refund")}{run(" policy")}</w:hyperlink>'
            f'{run(" for returns.")}</w:p>')
    parsed = parse_office(docx(body, external("rId7", "https://example.com/refunds")), "terms.docx", DocumentLimits())
    [segment] = parsed.segments
    assert segment.text == "See the refund policy for returns."
    [link] = links(segment)
    assert link == {"text": "refund policy", "target": "https://example.com/refunds",
                    "start": len("See the "), "end": len("See the refund policy")}
    assert segment.text.encode()[link["start"]:link["end"]].decode() == link["text"]


def test_spans_count_bytes_and_a_bookmark_is_named_with_a_hash():
    body = (f'<w:p>{run("Café ")}<w:hyperlink w:anchor="returns">{run("retours")}</w:hyperlink>{run(" et ")}'
            f'<w:hyperlink r:id="rId2">{run("contact")}</w:hyperlink></w:p>')
    parsed = parse_office(docx(body, external("rId2", "mailto:help@example.com")), "terms.docx", DocumentLimits())
    [segment] = parsed.segments
    found = links(segment)
    assert [(link["text"], link["target"]) for link in found] == [("retours", "#returns"),
                                                                  ("contact", "mailto:help@example.com")]
    for link in found:
        assert segment.text.encode()[link["start"]:link["end"]].decode() == link["text"]


def test_a_paragraph_without_links_says_nothing_new():
    parsed = parse_office(docx(f'<w:p>{run("Plain text.")}</w:p>'), "plain.docx", DocumentLimits())
    assert "links" not in parsed.segments[0].metadata


def test_a_link_whose_target_is_missing_is_counted_not_invented():
    body = f'<w:p>{run("Read ")}<w:hyperlink r:id="rId404">{run("this")}</w:hyperlink></w:p>'
    parsed = parse_office(docx(body), "broken.docx", DocumentLimits())
    [segment] = parsed.segments
    assert segment.text == "Read this" and "links" not in segment.metadata
    assert segment.metadata["links_unresolved"] == "1"


def test_links_past_the_bound_are_counted(monkeypatch):
    import scone_memory.ingestion.formats.office as module

    monkeypatch.setattr(module, "MAX_LINKS", 2)
    body = "<w:p>" + "".join(f'<w:hyperlink r:id="r{n}">{run(f"link{n} ")}</w:hyperlink>' for n in range(4)) + "</w:p>"
    relationships = "".join(external(f"r{n}", f"https://example.com/{n}") for n in range(4))
    [segment] = parse_office(docx(body, relationships), "many.docx", DocumentLimits()).segments
    assert [link["text"] for link in links(segment)] == ["link0", "link1"]
    assert segment.metadata["links_cut"] == "2"
    assert MAX_LINKS >= 2


def test_a_slide_run_keeps_its_link():
    data = ooxml_archive({
        "ppt/presentation.xml": f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst><p:sldId id="200" r:id="r1"/></p:sldIdLst></p:presentation>',
        "ppt/_rels/presentation.xml.rels": f'<Relationships xmlns="{REL}"><Relationship Id="r1" Target="slides/slide1.xml" Type="{R}/slide"/></Relationships>',
        "ppt/slides/slide1.xml": (f'<p:sld xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree><p:sp><p:txBody><a:p>'
                                  f'<a:r><a:t>Book the </a:t></a:r><a:r><a:rPr><a:hlinkClick r:id="h1"/></a:rPr><a:t>survey</a:t></a:r>'
                                  f'</a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>'),
        "ppt/slides/_rels/slide1.xml.rels": f'<Relationships xmlns="{REL}">{external("h1", "https://example.com/survey")}</Relationships>',
    }, main_part="ppt/presentation.xml")
    [segment] = parse_office(data, "deck.pptx", DocumentLimits()).segments
    assert segment.text == "Book the survey"
    assert links(segment) == [{"text": "survey", "target": "https://example.com/survey",
                               "start": len("Book the "), "end": len("Book the survey")}]


def test_a_link_s_own_leading_and_trailing_spaces_are_not_part_of_its_span():
    body = f'<w:p>{run("Read our")}<w:hyperlink r:id="rId1">{run("  returns  ")}</w:hyperlink>{run("page.")}</w:p>'
    [segment] = parse_office(docx(body, external("rId1", "https://example.com/returns")), "t.docx", DocumentLimits()).segments
    [link] = links(segment)
    assert link["text"] == "returns" and segment.text.encode()[link["start"]:link["end"]] == b"returns"


def test_a_target_that_is_not_a_hyperlink_blank_or_too_long_is_unresolved():
    from scone_memory.ingestion.formats.office import MAX_LINK_TARGET

    body = (f'<w:p><w:hyperlink r:id="img">{run("picture")}</w:hyperlink> <w:hyperlink r:id="blank">{run("blank")}</w:hyperlink>'
            f'<w:hyperlink r:id="long">{run("long")}</w:hyperlink><w:hyperlink r:id="ok">{run("fine")}</w:hyperlink></w:p>')
    relationships = (f'<Relationship Id="img" Target="media/image1.png" Type="{R}/image"/>'
                     + external("blank", " ") + external("long", "https://example.com/" + "a" * MAX_LINK_TARGET)
                     + external("ok", "https://example.com/ok"))
    [segment] = parse_office(docx(body, relationships), "t.docx", DocumentLimits()).segments
    assert [link["text"] for link in links(segment)] == ["fine"]
    assert segment.metadata["links_unresolved"] == "3"


def test_long_links_are_recorded_while_they_fit_a_metadata_value_and_the_rest_counted():
    """A metadata value holds 4,096 bytes; two presigned links of about 2,000
    characters each do not fit in one, and the document must still be read."""
    first, second = "https://example.com/a?" + "x" * 1_990, "https://example.com/b?" + "y" * 1_990
    body = (f'<w:p><w:hyperlink r:id="r1">{run("first")}</w:hyperlink> '
            f'<w:hyperlink r:id="r2">{run("second")}</w:hyperlink></w:p>')
    [segment] = parse_office(docx(body, external("r1", first) + external("r2", second)), "long.docx", DocumentLimits()).segments
    assert [link["target"] for link in links(segment)] == [first]
    assert segment.metadata["links_cut"] == "1" and len(segment.metadata["links"].encode()) <= 4096


def test_a_notes_part_with_broken_relationships_is_still_read():
    """Only the links in that part are lost, and they are counted; before links were read,
    a notes part's relationships were never opened, so they must not fail the document now."""
    data = ooxml_archive({
        "ppt/presentation.xml": f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst><p:sldId id="200" r:id="r1"/></p:sldIdLst></p:presentation>',
        "ppt/_rels/presentation.xml.rels": f'<Relationships xmlns="{REL}"><Relationship Id="r1" Target="slides/slide1.xml" Type="{R}/slide"/></Relationships>',
        "ppt/slides/slide1.xml": f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>',
        "ppt/slides/_rels/slide1.xml.rels": f'<Relationships xmlns="{REL}"><Relationship Id="n1" Target="../notesSlides/notesSlide1.xml" Type="{R}/notesSlide"/></Relationships>',
        "ppt/notesSlides/notesSlide1.xml": (f'<p:notes xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree><p:sp><p:txBody><a:p>'
                                            f'<a:r><a:rPr><a:hlinkClick r:id="h1"/></a:rPr><a:t>Speaker link</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>'),
        "ppt/notesSlides/_rels/notesSlide1.xml.rels": (f'<Relationships xmlns="{REL}">{external("h1", "https://example.com/a")}'
                                                       f'{external("h1", "https://example.com/b")}</Relationships>'),
    }, main_part="ppt/presentation.xml")
    slide, notes = parse_office(data, "deck.pptx", DocumentLimits()).segments
    assert notes.text == "Speaker link" and "links" not in notes.metadata and notes.metadata["links_unresolved"] == "1"


def test_the_recorded_links_never_exceed_a_metadata_value_whatever_their_lengths():
    for extra in range(1_280, 1_380, 3):
        body = "<w:p>" + "".join(f'<w:hyperlink r:id="r{n}">{run(f"l{n} ")}</w:hyperlink>' for n in range(3)) + "</w:p>"
        relationships = "".join(external(f"r{n}", f"https://example.com/{n}?" + "z" * extra) for n in range(3))
        [segment] = parse_office(docx(body, relationships), "sweep.docx", DocumentLimits()).segments
        recorded = links(segment)
        assert len(segment.metadata["links"].encode()) <= 4096, extra
        assert len(recorded) + int(segment.metadata.get("links_cut", "0")) == 3
