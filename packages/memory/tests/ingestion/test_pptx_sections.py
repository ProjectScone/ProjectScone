"""A deck organised into sections reads like any other.

PowerPoint keeps a deck's sections in an extension list inside
presentation.xml, and each section names its slides again as
``p14:sldId`` elements carrying only the slide's numeric id. The reader
looked for every element named ``sldId`` anywhere in the file, found the
section entries after the real list, took the empty relationship id of
the first one and refused the whole deck as missing a relationship. Six
of seven real decks checked used sections. Only the presentation's own
``sldIdLst`` names the slides.
"""
from __future__ import annotations

from scone_memory.ingestion.formats.office import parse_office
from scone_memory.ingestion.formats.types import DocumentLimits

from .test_office_formats import A, P, R, REL, ooxml_archive

P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"


def slide(text: str) -> str:
    return (f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{text}</a:t>'
            "</a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>")


def test_a_deck_with_sections_reads_its_slides_once_in_order():
    sections = (f'<p:extLst><p:ext uri="{{521415D9-36F7-43E2-AB2F-B90AF26B5E84}}"><p14:sectionLst xmlns:p14="{P14}">'
                '<p14:section name="Opening" id="{A}"><p14:sldIdLst><p14:sldId id="257"/></p14:sldIdLst></p14:section>'
                '<p14:section name="Body" id="{B}"><p14:sldIdLst><p14:sldId id="256"/></p14:sldIdLst></p14:section>'
                "</p14:sectionLst></p:ext></p:extLst>")
    data = ooxml_archive({
        "ppt/presentation.xml": (f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst><p:sldId id="257" r:id="r2"/>'
                                 f'<p:sldId id="256" r:id="r1"/></p:sldIdLst>{sections}</p:presentation>'),
        "ppt/_rels/presentation.xml.rels": (f'<Relationships xmlns="{REL}"><Relationship Id="r1" Target="slides/slide1.xml" '
                                            f'Type="{R}/slide"/><Relationship Id="r2" Target="slides/slide2.xml" '
                                            f'Type="{R}/slide"/></Relationships>'),
        "ppt/slides/slide1.xml": slide("Body slide"),
        "ppt/slides/slide2.xml": slide("Opening slide"),
    }, main_part="ppt/presentation.xml")
    parsed = parse_office(data, "deck.pptx", DocumentLimits())
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ("slide:1/paragraph:1", "Opening slide"), ("slide:2/paragraph:1", "Body slide")]
