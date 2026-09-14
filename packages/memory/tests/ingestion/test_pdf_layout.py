"""Reading order from the text layer's own geometry: columns, running lines, and what is left alone."""
from io import BytesIO

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import pdf_layout
from scone_memory.ingestion.documents import ingest_pdf, pdf_provenance
from scone_memory.ingestion.pdf import ParsedPdf, PdfLimits, PdfPage, PdfTextRegion, PypdfParser, validate_pdf
from scone_memory.ingestion.pdf_layout import lay_out_page, running_rows
from scone_memory.ingestion.pdf_ocr import assemble_ocr_pdf
from scone_memory.ocr.layout import ReadingOrderReceipt
from scone_memory.ocr.types import OcrRegion, OcrResult

pypdf = pytest.importorskip('pypdf')
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject  # noqa: E402


def placed(*pages):
    """A PDF whose pages hold text at given points: each page a list of (x, y, text, size)."""
    writer = pypdf.PdfWriter()
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
        NameObject('/BaseFont'): NameObject('/Helvetica'), NameObject('/Encoding'): NameObject('/WinAnsiEncoding')})
    for ops in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(''.join(f'BT /F1 {size} Tf {x} {y} Td <{text.encode("cp1252").hex()}> Tj ET\n'
                                for x, y, text, size in ops).encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    target = BytesIO()
    writer.write(target)
    return target.getvalue()


LEFT = ("The left column begins here with", "a first paragraph of body text that",
        "runs for several lines down the", "page before it ends.")
RIGHT = ("Meanwhile the right column carries", "a different paragraph whose lines",
         "sit beside the left ones on the", "same baselines every time.")
HEADER = "Quarterly Review of Columns"
WORDS = {1: "first", 2: "second", 3: "third"}


def title(number):
    return f"A Two Column Title for the {WORDS[number]} page"


def closing(number):
    return f"A full-width closing line spans both columns of the {WORDS[number]} page."


def two_column_page(number, *, header=HEADER):
    ops = [(72, 740, header, 10), (72, 710, title(number), 16)]
    for i, (left, right) in enumerate(zip(LEFT, RIGHT)):
        ops += [(72, 680 - 14 * i, left, 10), (330, 680 - 14 * i, right, 10)]
    return ops + [(72, 600, closing(number), 10), (300, 40, str(number), 10)]


def rows_of(parsed, number):
    page = parsed.pages[number - 1]
    return parsed.text.encode()[page.start:page.end].decode().split('\n')


async def test_a_two_column_page_reads_column_by_column_with_running_lines_recorded():
    parsed = await PypdfParser().parse(placed(*(two_column_page(n) for n in (1, 2, 3))), PdfLimits())
    assert parsed.parser.endswith(':layout-v1+grid-columns-v1')
    assert rows_of(parsed, 1) == [HEADER, title(1), *LEFT, *RIGHT, closing(1)], "left column whole, then the right"
    assert rows_of(parsed, 2) == [title(2), *LEFT, *RIGHT, closing(2)], "the header stays only where it first appears"
    assert [page.running for page in parsed.pages] == [('1',), (HEADER, '2'), (HEADER, '3')]
    for page in parsed.pages:
        assert page.reading_order == ReadingOrderReceipt(strategy='grid-columns-v1', columns=2, notes=('geometry_inferred',))
        assert page.region_geometry == 'normalized_text_grid' and page.extraction == 'text_layer'
        assert sorted(region.provider_index for region in page.regions) == list(range(len(page.regions)))
        assert {region.reading_column for region in page.regions} == {0, 1, 2}
        assert [region.text for region in page.regions if region.reading_column == 2] == list(RIGHT)
    assert "with                                      Meanwhile" not in parsed.text, "no row carries both columns"


async def test_as_extracted_keeps_the_grid_row_by_row():
    raw = placed(two_column_page(1))
    old = await PypdfParser(layout='as_extracted').parse(raw, PdfLimits())
    assert old.parser.endswith(':layout-v1') and old.pages[0].reading_order is None
    assert old.pages[0].running == () and old.pages[0].regions == ()
    first_body = rows_of(old, 1)[2]
    assert first_body.startswith(LEFT[0]) and first_body.endswith(RIGHT[0]), "the grid interleaves the columns"
    assert rows_of(old, 1)[-1].strip() == '1'
    with pytest.raises(InvalidInput, match="layout"):
        PypdfParser(layout='rows')  # type: ignore[arg-type]


async def test_the_manifest_keeps_regions_and_provenance_resolves_a_chunk():
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        result = await ingest_pdf(memory, 'alpha', placed(two_column_page(1), two_column_page(2), two_column_page(3)))
        manifest = (await memory.attachment('alpha', result.manifest.attachment_id))[1]
        assert b'"schema_version":3' in manifest and b'"reading_column"' in manifest and b'"running":["1"]' in manifest
        recalled = await memory.recall('alpha', 'right column carries a different paragraph', limit=1)
        provenance = await pdf_provenance(memory, 'alpha', result.added.episode_id, chunk_id=recalled.items[0].chunk_id)
        assert provenance.pages and all(page.reading_order.columns == 2 for page in provenance.pages)
        assert all(region.reading_column is not None for page in provenance.pages for region in page.regions)
    finally:
        await memory.close()


def table_page(rows):
    ops = [(72, 740, "Results by quarter, in millions", 10)]
    for i, (label, first, second) in enumerate(rows):
        y = 700 - 14 * i
        ops += [(72, y, label, 10), (300, y, first, 10), (420, y, second, 10)]
    return ops


async def test_a_table_page_stays_whole_and_the_receipt_says_so():
    rows = [("Revenue", "2,903", "6,854"), ("Cost of revenue", "1,650", "2,720"), ("Operations and support", "574", "718"),
            ("Sales and marketing", "1,263", "4,789"), ("Research and development", "587", "2,054"), ("Total costs", "4,074", "10,281")]
    parsed = await PypdfParser().parse(placed(table_page(rows)), PdfLimits())
    page = parsed.pages[0]
    assert page.reading_order.columns == 0 and 'tabular_kept_whole' in page.reading_order.notes
    assert page.regions == () and page.region_geometry == 'normalized_displayed_page_top_left'
    for label, first, second in rows:
        line = next(row for row in rows_of(parsed, 1) if row.startswith(label))
        assert first in line and second in line, "a table row is still one row"


async def test_a_key_value_list_is_not_two_columns():
    rows = [(f"Setting number {i} of the device", str(i * 7), "") for i in range(1, 8)]
    parsed = await PypdfParser().parse(placed([(x, y, t, s) for x, y, t, s in table_page(rows) if t]), PdfLimits())
    page = parsed.pages[0]
    assert page.reading_order.columns == 0 and 'tabular_kept_whole' in page.reading_order.notes
    assert rows_of(parsed, 1)[1].startswith("Setting number 1 of the device") and rows_of(parsed, 1)[1].endswith("7")


def test_running_lines_need_three_pages_and_a_page_is_never_emptied():
    def body(word):
        return [f"Body text of the {word} page that says something.", f"And a second line of the {word} one."]

    pages = [body("first"), body("second"), body("third")]
    assert running_rows([["Header", *pages[0]], ["Header", *pages[1]]]) == [{}, {}], "twice is not running"
    assert running_rows([["Header", *page] for page in pages]) == [{}, {0: 'Header'}, {0: 'Header'}], "kept where it first appears"
    assert running_rows([[*pages[0], "Chapter 3 · 41"], [*pages[1], "Chapter 3 · 42"], [*pages[2], "Chapter 4 · 43"]]) == \
        [{}, {2: 'Chapter 3 · 42'}, {2: 'Chapter 4 · 43'}], "numbers do not tell a footer apart"
    for number in ("7", "Page 7", "7 of 12", "- 7 -", "page 7 / 12"):
        assert running_rows([[*pages[0], number]]) == [{2: number}], number
    assert running_rows([["7"]]) == [{}], "the only row stays"
    assert running_rows([["", "Only line", ""]]) == [{}]
    assert running_rows([["Body text that is long enough.", "2021 annual report", "and more"]]) == [{}], "a year in a title is not a page number"


def grid(*rows):
    return list(rows)


def test_the_region_budget_is_disclosed(monkeypatch):
    rows = grid("Alpha beta gamma delta epsilon      Zeta eta theta iota kappa lambda",
                "mu nu xi omicron pi rho sigma       tau upsilon phi chi psi omega alpha",
                "beta gamma delta epsilon zeta       eta theta iota kappa lambda mu nu",
                "xi omicron pi rho sigma tau         upsilon phi chi psi omega alpha beta")
    laid = lay_out_page(rows, 0)
    assert laid.receipt.columns == 2 and len(laid.regions) == 8
    monkeypatch.setattr(pdf_layout, 'MAX_REGIONS_PER_PAGE', 7)
    bounded = lay_out_page(rows, 0)
    assert bounded.receipt == ReadingOrderReceipt(strategy='grid-columns-v1', columns=0, notes=('no_separating_gutter', 'region_limit'))
    assert bounded.text == "\n".join(rows) and bounded.regions == ()


def test_a_page_narrower_than_a_gutter_stays_as_extracted():
    for rows in (["7"], ["II", "", "OK"], ["ab"]):
        laid = lay_out_page(rows, 0)
        assert laid.receipt == ReadingOrderReceipt(strategy='grid-columns-v1', columns=0, notes=('no_separating_gutter',))
        assert laid.text == "\n".join(rows).rstrip() and laid.regions == ()


async def test_a_document_with_a_nearly_blank_page_is_still_read():
    parsed = await PypdfParser().parse(placed(two_column_page(1), [(300, 400, "II", 14)], two_column_page(3)), PdfLimits())
    assert [page.reading_order.columns for page in parsed.pages] == [2, 0, 2]
    assert rows_of(parsed, 2) == ["II"]


def test_regions_carry_absolute_byte_spans_and_row_gaps():
    rows = grid("Café au lait for the first row        Über the second column's first",
                "naïve second row of the left          façade of the right column here",
                "the third left row with x²   (1)      résumé of the third right row",
                "and a fourth left row to close        ending the right column as well")
    laid = lay_out_page(rows, 10)
    encoded = laid.text.encode()
    for region in laid.regions:
        assert encoded[region.start - 10:region.end - 10] == region.text.encode()
    assert "the third left row with x²   (1)" in laid.text, "runs of one row keep the gap between them"
    assert laid.text.split("\n")[:4] == [rows[0][:30].strip(), rows[1][:30].strip(), "the third left row with x²   (1)", rows[3][:30].strip()]


async def test_ocr_assembly_moves_the_regions_of_pages_it_did_not_recognise():
    parsed = await PypdfParser().parse(placed(two_column_page(1), two_column_page(2), two_column_page(3)), PdfLimits())
    recognized = OcrResult(engine='fixture', width=100, height=100,
                           regions=(OcrRegion(text='recognised words of a scanned page', box=(0.1, 0.1, 0.9, 0.2)),))
    assembled = assemble_ocr_pdf(parsed, {1: recognized}, PdfLimits())
    assert assembled.pages[0].extraction == 'ocr' and assembled.pages[0].running == ()
    later = assembled.pages[1]
    assert later.regions and later.reading_order.columns == 2 and later.running == (HEADER, '2')
    encoded = assembled.text.encode()
    assert all(encoded[region.start:region.end] == region.text.encode() for region in later.regions)


def test_validation_keeps_grid_and_raster_geometry_apart():
    region = PdfTextRegion(text='word', box=(0.1, 0.1, 0.2, 0.2), start=0, end=4, provider_index=0, reading_column=0)
    receipt = ReadingOrderReceipt(strategy='grid-columns-v1', columns=0, notes=('no_separating_gutter',))

    def page(**updates):
        base = dict(number=1, start=0, end=4, width_points=1.0, height_points=1.0, rotation=0, empty=False)
        return ParsedPdf(text='word', parser='p', pages=(PdfPage(**{**base, **updates}),))

    validate_pdf(page(regions=(region,), reading_order=receipt, region_geometry='normalized_text_grid'), PdfLimits())
    with pytest.raises(InvalidInput, match='OCR regions'):
        validate_pdf(page(regions=(region,), reading_order=receipt), PdfLimits())
    with pytest.raises(InvalidInput, match='rectangles'):
        validate_pdf(page(extraction='ocr', ocr_engine='e', regions=(region,), reading_order=receipt,
                          region_geometry='normalized_text_grid'), PdfLimits())
    with pytest.raises(InvalidInput, match='running'):
        validate_pdf(page(extraction='ocr', ocr_engine='e', regions=(region,), reading_order=receipt, running=('7',)), PdfLimits())


async def test_a_small_table_inside_a_column_stays_whole_while_the_columns_are_read():
    ops = [(72, 740, "Findings of the review", 14)]
    prose_left = ("Left column prose that fills the line", "and continues on a second line here",
                  "a third line keeps the paragraph going", "the fourth line is much the same",
                  "a fifth line to make the column tall", "and a sixth line to end the paragraph.")
    prose_right = ("Right column prose that fills its line", "with a second line beside the left",
                   "a third line of the right column here", "a fourth line keeps it prose",
                   "a fifth line of the right column too", "and a sixth line ends it as well.")
    for i, (left, right) in enumerate(zip(prose_left, prose_right)):
        ops += [(72, 700 - 14 * i, left, 10), (330, 700 - 14 * i, right, 10)]
    table = (("Revenue", "2,903"), ("Costs", "1,650"), ("Margin", "1,253"))
    for i, (label, number) in enumerate(table):
        ops += [(72, 600 - 14 * i, label, 10), (200, 600 - 14 * i, number, 10)]
    parsed = await PypdfParser().parse(placed(ops), PdfLimits())
    page = parsed.pages[0]
    assert page.reading_order.columns == 2 and 'tabular_kept_whole' in page.reading_order.notes, \
        "the columns are read; the cut through the table was refused"
    rows = rows_of(parsed, 1)
    assert rows[1:7] == list(prose_left) and rows[-6:] == list(prose_right)
    assert [row for row in rows if row.startswith("Revenue")] == [next(row for row in rows if "2,903" in row)], "a table row is one row"
    assert rows.index("Left column prose that fills the line") < rows.index(next(row for row in rows if row.startswith("Revenue"))) < rows.index("Right column prose that fills its line")
