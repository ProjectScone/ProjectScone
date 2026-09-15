"""The heading level a document gives a paragraph, kept beside its text.

Structure chunking cuts at headings, and it found none in the formats most
people bring: a Word paragraph styled Heading 1, an OpenDocument text:h and
an HTML h2 all became plain paragraphs, because the level lived in a style
or a tag the readers dropped. Here each heading paragraph's segment says
``heading_level``: 1 to 9 for Word, 1 to 10 for OpenDocument, 1 to 6 for
HTML. A Word level comes from the paragraph's own
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
          '<w:style w:type="paragraph" w:styleId="Quote"><w:basedOn w:val="Normal"/></w:style>'
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


def test_a_paragraphs_own_outline_level_outranks_its_styles():
    body = paragraph("Both", '<w:pStyle w:val="Section"/><w:outlineLvl w:val="5"/>')
    assert levels(parse_office(word(body), "terms.docx", DocumentLimits())) == [("Both", "6")]


def test_a_heading_inside_a_text_box_keeps_its_level():
    boxed = ('<w:txbxContent>' + paragraph("Sidebar", '<w:pStyle w:val="Titre1"/>')
             + paragraph("Aside", '<w:outlineLvl w:val="1"/>') + '</w:txbxContent>')
    body = f'<w:p><w:r><w:t>Anchor.</w:t><w:pict>{boxed}</w:pict></w:r></w:p>'
    assert levels(parse_office(word(body), "terms.docx", DocumentLimits())) == [
        ("Anchor.", None), ("Sidebar", "1"), ("Aside", "2")]


def test_an_outline_level_that_is_not_a_decimal_number_is_not_a_level():
    # "²" is a digit to str.isdigit and not a number to int.
    body = paragraph("Squared", '<w:outlineLvl w:val="²"/>') + paragraph("Next")
    assert levels(parse_office(word(body), "terms.docx", DocumentLimits())) == [("Squared", None), ("Next", None)]


def test_a_style_chain_past_the_bound_says_its_level_was_not_resolved():
    from scone_memory.ingestion.formats import office

    chain = "".join(f'<w:style w:type="paragraph" w:styleId="S{n}"><w:basedOn w:val="S{n + 1}"/></w:style>'
                    for n in range(office._STYLE_DEPTH))
    styles = (f'<w:styles xmlns:w="{W}">{chain}<w:style w:type="paragraph" w:styleId="S{office._STYLE_DEPTH}">'
              '<w:name w:val="heading 2"/></w:style></w:styles>')
    body = (paragraph("Near", f'<w:pStyle w:val="S1"/>') + paragraph("Far", '<w:pStyle w:val="S0"/>')
            + paragraph("Circular", '<w:pStyle w:val="Loop"/>') + paragraph("Plain", '<w:pStyle w:val="Quote"/>'))
    data = ooxml_archive({
        "word/document.xml": f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>',
        "word/_rels/document.xml.rels": (f'<Relationships xmlns="{REL}"><Relationship Id="s1" Target="styles.xml" '
                                         f'Type="{R}/styles"/></Relationships>'),
        "word/styles.xml": styles.replace('</w:styles>', STYLES.split('>', 1)[1].replace('</w:styles>', '') + '</w:styles>'),
    }, main_part="word/document.xml")
    parsed = parse_office(data, "terms.docx", DocumentLimits())
    assert levels(parsed) == [("Near", "2"), ("Far", None), ("Circular", None), ("Plain", None)]
    unresolved = [segment.metadata.get("heading_level_unresolved") for segment in parsed.segments]
    assert unresolved == [None, "style_chain", "style_chain", None]


def test_a_word_document_without_styles_still_reads():
    styled = paragraph("Refunds", '<w:pStyle w:val="Heading1"/>')
    document = f'<w:document xmlns:w="{W}"><w:body>{styled}</w:body></w:document>'
    data = ooxml_archive({"word/document.xml": document}, main_part="word/document.xml")
    assert levels(parse_office(data, "terms.docx", DocumentLimits())) == [("Refunds", None)]


def test_an_open_document_heading_says_its_outline_level():
    body = ('<office:text><text:h text:outline-level="2">Refunds</text:h><text:h>Untitled level</text:h>'
            '<text:h text:outline-level="²">Squared</text:h><text:p>Within 30 days.</text:p></office:text>')
    data = archive({"content.xml": f'<office:document-content xmlns:office="{O}" xmlns:text="{T}"><office:body>{body}'
                                   f'</office:body></office:document-content>'})
    assert levels(parse_office(data, "terms.odt", DocumentLimits())) == [
        ("Refunds", "2"), ("Untitled level", "1"), ("Squared", "1"), ("Within 30 days.", None)]


def test_an_html_heading_says_its_level_and_nothing_else_changes():
    html = b"<h1>Terms</h1><p>Intro.</p><h3>Refunds <em>and returns</em></h3><div>Within 30 days.</div>"
    assert levels(parse_text(html, "terms.html", DocumentLimits())) == [
        ("Terms", "1"), ("Intro.", None), ("Refunds and returns", "3"), ("Within 30 days.", None)]
