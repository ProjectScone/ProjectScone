"""The heading level a document gives a paragraph, kept beside its text.

Structure chunking cuts at headings, and it found none in the formats most
people bring: a Word paragraph styled Heading 1, an OpenDocument text:h and
an HTML h2 all became plain paragraphs, because the level lived in a style
or a tag the readers dropped. Here each heading paragraph's segment says
``heading_level`` (1 to 9). A Word level comes from the paragraph's own
outline level, else from its style's -- followed through the styles it is
based on -- or from a built-in style named "heading N", whatever the style's
id is in the document's language. The text is not changed.
"""

from __future__ import annotations

from scone_memory.ingestion.formats.office import parse_office
from scone_memory.ingestion.formats.text import parse_text
from scone_memory.ingestion.formats.types import DocumentLimits

from .test_office_formats import O, R, REL, T, W, archive, ooxml_archive


def levels(parsed) -> list[tuple[str, str | None]]:
    return [(segment.text, segment.metadata.get("heading_level")) for segment in parsed.segments]


STYLES = (f'<w:styles xmlns:w="{W}">'
          '<w:style w:type="paragraph" w:styleId="Titre1"><w:name w:val="heading 1"/></w:style>'
          '<w:style w:type="paragraph" w:styleId="Section"><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>'
          '<w:style w:type="paragraph" w:styleId="Subsection"><w:basedOn w:val="Section"/></w:style>'
          '<w:style w:type="paragraph" w:styleId="Loop"><w:basedOn w:val="Loop"/></w:style>'
          '<w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
          '</w:styles>')


def paragraph(text: str, properties: str = "") -> str:
    return f'<w:p><w:pPr>{properties}</w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>'


def word(body: str) -> bytes:
    return ooxml_archive({
        "word/document.xml": f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>',
        "word/_rels/document.xml.rels": (f'<Relationships xmlns="{REL}"><Relationship Id="s1" Target="styles.xml" '
                                         f'Type="{R}/styles"/></Relationships>'),
        "word/styles.xml": STYLES,
    }, main_part="word/document.xml")


def test_a_word_heading_is_found_by_its_style_name_whatever_the_style_id():
    body = (paragraph("Refunds", '<w:pStyle w:val="Titre1"/>') + paragraph("Within 30 days.", '<w:pStyle w:val="Normal"/>')
            + paragraph("Plain text."))
    assert levels(parse_office(word(body), "terms.docx", DocumentLimits())) == [
        ("Refunds", "1"), ("Within 30 days.", None), ("Plain text.", None)]


def test_an_outline_level_on_the_style_its_base_or_the_paragraph_itself_gives_the_level():
    body = (paragraph("Scope", '<w:pStyle w:val="Section"/>') + paragraph("Cranes", '<w:pStyle w:val="Subsection"/>')
            + paragraph("Direct", '<w:outlineLvl w:val="2"/>') + paragraph("Body level", '<w:outlineLvl w:val="9"/>')
            + paragraph("Looped", '<w:pStyle w:val="Loop"/>'))
    assert levels(parse_office(word(body), "terms.docx", DocumentLimits())) == [
        ("Scope", "2"), ("Cranes", "2"), ("Direct", "3"), ("Body level", None), ("Looped", None)]


def test_a_word_document_without_styles_still_reads():
    styled = paragraph("Refunds", '<w:pStyle w:val="Heading1"/>')
    document = f'<w:document xmlns:w="{W}"><w:body>{styled}</w:body></w:document>'
    data = ooxml_archive({"word/document.xml": document}, main_part="word/document.xml")
    assert levels(parse_office(data, "terms.docx", DocumentLimits())) == [("Refunds", None)]


def test_an_open_document_heading_says_its_outline_level():
    body = ('<office:text><text:h text:outline-level="2">Refunds</text:h><text:h>Untitled level</text:h>'
            '<text:p>Within 30 days.</text:p></office:text>')
    data = archive({"content.xml": f'<office:document-content xmlns:office="{O}" xmlns:text="{T}"><office:body>{body}'
                                   f'</office:body></office:document-content>'})
    assert levels(parse_office(data, "terms.odt", DocumentLimits())) == [
        ("Refunds", "2"), ("Untitled level", "1"), ("Within 30 days.", None)]


def test_an_html_heading_says_its_level_and_nothing_else_changes():
    html = b"<h1>Terms</h1><p>Intro.</p><h3>Refunds <em>and returns</em></h3><div>Within 30 days.</div>"
    assert levels(parse_text(html, "terms.html", DocumentLimits())) == [
        ("Terms", "1"), ("Intro.", None), ("Refunds and returns", "3"), ("Within 30 days.", None)]
