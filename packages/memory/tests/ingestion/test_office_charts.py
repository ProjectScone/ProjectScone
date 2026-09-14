"""The numbers a chart in a Word or PowerPoint file was drawn from, as text beside it.

A chart in a deck or a report carries its data: every series' name, its
categories and its values are cached inside the chart part, so the file
opens without the workbook they came from. The reader took the slide's
and the paragraph's text and skipped the chart, so "what was Q2 revenue"
had nothing to find. Here each chart becomes a segment of its own after
the text it sits in: its title and kind, then one line per series. Only
cached values are read -- nothing is recalculated or rendered -- and the
bounds on series and points say when they bit.
"""
from __future__ import annotations

from scone_memory.ingestion.formats import office
from scone_memory.ingestion.formats.office import parse_office
from scone_memory.ingestion.formats.types import DocumentLimits

from .test_docx_parts import document
from .test_office_formats import A, P, R, REL, W, ooxml_archive

C = "http://schemas.openxmlformats.org/drawingml/2006/chart"


def points(tag: str, values: list[str], kind: str = "num") -> str:
    cached = "".join(f'<c:pt idx="{index}"><c:v>{value}</c:v></c:pt>' for index, value in enumerate(values))
    return f"<c:{tag}><c:{kind}Ref><c:f>Sheet1!$A$1</c:f><c:{kind}Cache>{cached}</c:{kind}Cache></c:{kind}Ref></c:{tag}>"


def series(name: str, categories: list[str] | None, values: list[str], *, x: str = "cat", y: str = "val") -> str:
    title = f'<c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>{name}</c:v></c:pt></c:strCache></c:strRef></c:tx>'
    return ("<c:ser>" + title + (points(x, categories, "str") if categories is not None else "")
            + points(y, values) + "</c:ser>")


def chart(plot: str, title: str | None = "Revenue") -> str:
    heading = (f"<c:title><c:tx><c:rich><a:p><a:r><a:t>{title}</a:t></a:r></a:p></c:rich></c:tx></c:title>"
               if title else "")
    return (f'<c:chartSpace xmlns:c="{C}" xmlns:a="{A}"><c:chart>{heading}<c:plotArea>{plot}</c:plotArea>'
            "</c:chart></c:chartSpace>")


BAR = chart("<c:barChart>" + series("2025", ["Q1", "Q2"], ["10", "12.5"]) + series("2024", ["Q1", "Q2"], ["8", "9"])
            + "</c:barChart>")


def deck(chart_xml: str, *, relationship: bool = True) -> bytes:
    frame = (f'<p:graphicFrame><a:graphic><a:graphicData uri="{C}"><c:chart xmlns:c="{C}" xmlns:r="{R}" r:id="c1"/>'
             "</a:graphicData></a:graphic></p:graphicFrame>")
    slide = (f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Results</a:t></a:r>'
             f"</a:p></p:txBody></p:sp>{frame}</p:spTree></p:cSld></p:sld>")
    return ooxml_archive({
        "ppt/presentation.xml": f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst><p:sldId id="200" r:id="r1"/></p:sldIdLst></p:presentation>',
        "ppt/_rels/presentation.xml.rels": f'<Relationships xmlns="{REL}"><Relationship Id="r1" Target="slides/slide1.xml" Type="{R}/slide"/></Relationships>',
        "ppt/slides/slide1.xml": slide,
        "ppt/slides/_rels/slide1.xml.rels": (f'<Relationships xmlns="{REL}"><Relationship Id="c1" Target="../charts/chart1.xml" '
                                            f'Type="{R}/chart"/></Relationships>' if relationship else f'<Relationships xmlns="{REL}"/>'),
        "ppt/charts/chart1.xml": chart_xml,
    }, main_part="ppt/presentation.xml")


def test_a_slides_chart_is_a_segment_of_its_title_kind_and_series_after_the_slides_text():
    parsed = parse_office(deck(BAR), "results.pptx", DocumentLimits())
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ("slide:1/paragraph:1", "Results"),
        ("slide:1/chart:1", "Revenue (bar chart)\n2025: Q1 10; Q2 12.5\n2024: Q1 8; Q2 9")]
    assert parsed.segments[1].metadata == {"member": "ppt/charts/chart1.xml", "content_role": "chart",
                                           "parent_locator": "slide:1", "chart_type": "bar",
                                           "chart_series": "2", "chart_points": "4"}
    assert parsed.parser == "native-xml+charts-v1" and parsed.metadata == {"charts": "1"}


def test_a_word_chart_follows_its_paragraph_and_a_scatter_reads_its_x_and_y():
    scatter = chart("<c:scatterChart>" + series("Load", ["0.5", "4"], ["3.5", "7"], x="xVal", y="yVal")
                    + "</c:scatterChart>", title=None)
    inline = f'<w:drawing><c:chart xmlns:c="{C}" r:id="chart"/></w:drawing>'
    data = document(f"<w:p><w:r><w:t>Before.</w:t>{inline}</w:r></w:p><w:p><w:r><w:t>After.</w:t></w:r></w:p>",
                    relationships=f'<Relationship Id="chart" Target="charts/chart1.xml" Type="{R}/chart"/>',
                    extra={"content/charts/chart1.xml": scatter})
    parsed = parse_office(data, "report.docx", DocumentLimits())
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ("paragraph:1", "Before."), ("paragraph:1/chart:1", "Chart (scatter chart)\nLoad: 0.5 3.5; 4 7"),
        ("paragraph:2", "After.")]


def test_a_series_without_categories_is_read_by_point_number():
    parsed = parse_office(deck(chart("<c:lineChart>" + series("Visits", None, ["5", "6", "7"]) + "</c:lineChart>")),
                          "results.pptx", DocumentLimits())
    assert parsed.segments[1].text == "Revenue (line chart)\nVisits: 1 5; 2 6; 3 7"


def test_a_cached_value_is_one_line_of_text_and_the_first_point_at_an_index_stands():
    doubled = ('<c:cat><c:strRef><c:strCache><c:pt idx="0"><c:v>North\n  East</c:v></c:pt>'
               '<c:pt idx="0"><c:v>Ignored</c:v></c:pt></c:strCache></c:strRef></c:cat>')
    plot = "<c:pieChart>" + series("Share", ["x"], ["40"]).replace(points("cat", ["x"], "str"), doubled) + "</c:pieChart>"
    assert parse_office(deck(chart(plot)), "results.pptx", DocumentLimits()).segments[1].text == (
        "Revenue (pie chart)\nShare: North East 40")


def test_the_series_and_point_bounds_say_when_they_bit(monkeypatch):
    monkeypatch.setattr(office, "MAX_CHART_SERIES", 1)
    monkeypatch.setattr(office, "MAX_CHART_POINTS", 1)
    segment = parse_office(deck(BAR), "results.pptx", DocumentLimits()).segments[1]
    assert segment.text == "Revenue (bar chart)\n2025: Q1 10"
    assert segment.metadata["chart_series_cut"] == "1" and segment.metadata["chart_points_cut"] == "1"
    assert segment.metadata["chart_series"] == "1" and segment.metadata["chart_points"] == "1"


def test_a_chart_that_cannot_be_read_is_counted_and_the_rest_of_the_file_still_reads():
    missing = parse_office(deck(BAR, relationship=False), "results.pptx", DocumentLimits())
    assert [s.text for s in missing.segments] == ["Results"] and missing.metadata == {"charts_unreadable": "1"}
    broken = parse_office(deck("<not-xml"), "results.pptx", DocumentLimits())
    assert [s.text for s in broken.segments] == ["Results"] and broken.metadata == {"charts_unreadable": "1"}
    assert broken.parser == "native-xml"
    foreign = parse_office(deck(BAR.replace(C, "urn:not-a-chart")), "results.pptx", DocumentLimits())
    assert [s.text for s in foreign.segments] == ["Results"] and foreign.metadata == {"charts_unreadable": "1"}


def test_a_file_without_charts_reads_exactly_as_before():
    data = document("<w:p><w:r><w:t>Plain.</w:t></w:r></w:p>")
    parsed = parse_office(data, "report.docx", DocumentLimits())
    assert parsed.parser == "native-xml" and parsed.metadata == {}
    assert "metadata" not in parsed.model_dump(exclude_defaults=True)


def test_a_series_named_by_a_literal_or_not_at_all_is_still_labelled():
    literal = series("ignored", ["Q1"], ["3"]).replace(
        '<c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>ignored</c:v></c:pt></c:strCache></c:strRef></c:tx>',
        "<c:tx><c:v>Typed name</c:v></c:tx>")
    unnamed = "<c:ser>" + points("val", ["4"]) + "</c:ser>"
    parsed = parse_office(deck(chart("<c:areaChart>" + literal + unnamed + "</c:areaChart>")), "results.pptx",
                          DocumentLimits())
    assert parsed.segments[1].text == "Revenue (area chart)\nTyped name: Q1 3\nSeries 2: 1 4"


CX = "http://schemas.microsoft.com/office/drawing/2014/chartex"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def extended(layout: str = "waterfall", levels: int = 1) -> str:
    level = '<cx:lvl ptCount="2"><cx:pt idx="0">Start</cx:pt><cx:pt idx="1">Sales</cx:pt></cx:lvl>'
    outer = '<cx:lvl ptCount="2"><cx:pt idx="0">2025</cx:pt><cx:pt idx="1">2025</cx:pt></cx:lvl>'
    return (f'<cx:chartSpace xmlns:cx="{CX}"><cx:chartData><cx:data id="0">'
            f'<cx:strDim type="cat"><cx:f>Sheet1!$A$2:$A$3</cx:f>{level}{outer * (levels - 1)}</cx:strDim>'
            # A colour dimension is data too, but not the values the series plots.
            '<cx:numDim type="colorVal"><cx:lvl ptCount="2"><cx:pt idx="0">1</cx:pt><cx:pt idx="1">2</cx:pt></cx:lvl></cx:numDim>'
            '<cx:numDim type="val"><cx:f>Sheet1!$B$2:$B$3</cx:f><cx:lvl ptCount="2" formatCode="General">'
            '<cx:pt idx="0">100</cx:pt><cx:pt idx="1">40</cx:pt></cx:lvl></cx:numDim></cx:data></cx:chartData>'
            '<cx:chart><cx:title><cx:tx><cx:txData><cx:v>Quarter</cx:v></cx:txData></cx:tx></cx:title><cx:plotArea>'
            f'<cx:plotAreaRegion><cx:series layoutId="{layout}"><cx:tx><cx:txData><cx:f>Sheet1!$B$1</cx:f>'
            '<cx:v>Cash flow</cx:v></cx:txData></cx:tx><cx:dataId val="0"/></cx:series></cx:plotAreaRegion>'
            "</cx:plotArea></cx:chart></cx:chartSpace>")


def extended_deck(chart_xml: str) -> bytes:
    frame = (f'<mc:AlternateContent xmlns:mc="{MC}"><mc:Choice xmlns:cx1="{CX}" Requires="cx1"><p:graphicFrame>'
             f'<a:graphic><a:graphicData uri="{CX}"><cx:chart xmlns:cx="{CX}" xmlns:r="{R}" r:id="x1"/></a:graphicData>'
             "</a:graphic></p:graphicFrame></mc:Choice><mc:Fallback><p:sp/></mc:Fallback></mc:AlternateContent>")
    slide = (f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Results</a:t></a:r>'
             f"</a:p></p:txBody></p:sp>{frame}</p:spTree></p:cSld></p:sld>")
    return ooxml_archive({
        "ppt/presentation.xml": f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst><p:sldId id="200" r:id="r1"/></p:sldIdLst></p:presentation>',
        "ppt/_rels/presentation.xml.rels": f'<Relationships xmlns="{REL}"><Relationship Id="r1" Target="slides/slide1.xml" Type="{R}/slide"/></Relationships>',
        "ppt/slides/slide1.xml": slide,
        "ppt/slides/_rels/slide1.xml.rels": (f'<Relationships xmlns="{REL}"><Relationship Id="x1" Target="../charts/chartEx1.xml" '
                                            'Type="http://schemas.microsoft.com/office/2014/relationships/chartEx"/></Relationships>'),
        "ppt/charts/chartEx1.xml": chart_xml,
    }, main_part="ppt/presentation.xml")


def test_an_office_2016_chart_reads_its_series_from_the_data_it_names():
    # Waterfall, histogram, treemap and the other newer kinds keep their data apart from their
    # series, which point at it by id.
    parsed = parse_office(extended_deck(extended()), "results.pptx", DocumentLimits())
    assert [(s.locator, s.text) for s in parsed.segments] == [
        ("slide:1/paragraph:1", "Results"), ("slide:1/chart:1", "Quarter (waterfall chart)\nCash flow: Start 100; Sales 40")]
    assert parsed.segments[1].metadata == {"member": "ppt/charts/chartEx1.xml", "content_role": "chart",
                                           "parent_locator": "slide:1", "chart_type": "waterfall",
                                           "chart_series": "1", "chart_points": "2"}


def test_an_office_2016_chart_with_nested_categories_reads_the_first_level_and_says_so():
    segment = parse_office(extended_deck(extended("treemap", levels=2)), "results.pptx", DocumentLimits()).segments[1]
    assert segment.text == "Quarter (treemap chart)\nCash flow: Start 100; Sales 40"
    assert segment.metadata["chart_category_levels_cut"] == "1"
