"""A page's tables as document table cells, the way the office readers give them.

The office readers (DOCX, HTML, XLSX) give a document's tables as
cells with a row, a column, their spans and a byte span of the
segment's text, and the table querying and the evidence that cite a
cell rest on that record. A PDF page had none: the OCR table inference
(`ocr.tables`) proposed grids from geometry, and nothing carried them
to the segment. This does, for the regions of a page the recognizer or
the rules labelled `table` -- a page whose labels name no table carries
none, whatever a grid over its lines would propose, as two columns of
prose align the way a table does -- each proposed cell becomes a
`DocumentTableCell` located on the page,
with the column span the grid inferred, and its text as the byte span
of the page's text that holds exactly the cell's words -- which is only
so when the cell's regions sit together in the reading order. The cells
come in the order the page's text reads them, each keeping its row and
column, so a table a recognizer read column by column (Tesseract reads
a table's columns as blocks) cites its columns in turn; the office
readers' cells, written row by row, come in that same order. A table
with a cell whose words do not sit together is left out rather than
cited wrongly, and the page says how many were.

A table's header is read from its shape: the first row that fills every
column is the header when none of its cells is a number (a bare year is
a label, as over a financial statement's columns) and some column below
it is mostly numbers, since nothing else tells a header from a first
row; a table of words alone gets none. Header cells say so, and every
cell below carries a `column` reference to the header over it, the way
the DOCX reader gives them, so the table query names the columns. No
row span or empty cell is invented.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re

from ..core.errors import InvalidInput
from ..ingestion.formats.table_types import DocumentTableCell, DocumentTableHeader
from .tables import TableCandidate, infer_tables
from .types import OcrRegion

#: Tables a page's segment carries; the rest are counted, not carried.
MAX_TABLES = 64
#: A cell that is a number, by the rule the table query reads one
#: (``retrieval.table_query.NUMERIC_FORM``: a sign, a currency sign,
#: digits with thousands separators and a decimal part) or a percentage.
_NUMBER = re.compile(r'^[+-]?\s*[$\u20ac\u00a3\u00a5]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$')
#: A bare year, which heads a column rather than filling one.
_YEAR = re.compile(r'^(?:1[6-9]|20)\d{2}$')


def _is_number(text: str) -> bool:
    body = text.strip()
    return bool(_NUMBER.match(body)) and not _YEAR.match(body)


def header_row(table: TableCandidate) -> int | None:
    """The row of ``table`` that is its header, by its shape, or None."""
    rows: dict[int, list] = {}
    for cell in table.cells:
        rows.setdefault(cell.row, []).append(cell)
    # Only rows filling every column count: a title across the table, a
    # subtotal or a wrapped word says nothing about the columns.
    full = [row for row, cells in sorted(rows.items())
            if len(cells) == table.columns and all(cell.column_span == 1 for cell in cells)]
    if len(full) < 2 or any(_is_number(cell.text) or not cell.text.strip() for cell in rows[full[0]]):
        return None
    for column in range(table.columns):
        values = [cell.text for row in full[1:] for cell in rows[row] if cell.column == column]
        if values and 2 * sum(_is_number(value) for value in values) > len(values):
            return full[0]
    return None


@dataclass(frozen=True)
class PageTables:
    cells: tuple[DocumentTableCell, ...]
    #: Tables the grid proposed, and those left out because their cells did
    #: not read together in the page's text.
    proposed: int
    unreadable: int
    #: Tables whose header row was read from their shape.
    headed: int = 0


def page_table_cells(regions: Sequence[OcrRegion], spans: Sequence[tuple[int, int]], text: bytes,
                     locator: str) -> PageTables:
    """The document table cells of one page: ``regions`` the page's regions
    in text order, ``spans`` each region's segment-relative byte span,
    ``text`` the segment's text, ``locator`` the page's."""
    if len(regions) != len(spans):
        raise ValueError('every region needs its span')
    chosen = [index for index, region in enumerate(regions) if region.label == 'table']
    if not chosen:
        return PageTables((), 0, 0)
    try:
        layout = infer_tables([regions[index] for index in chosen])
    except InvalidInput:
        return PageTables((), 0, 0)
    cells: list[DocumentTableCell] = []
    unreadable = headed = 0
    for number, table in enumerate(layout.tables[:MAX_TABLES]):
        table_locator = f'{locator}/table:{number + 1}'
        header = header_row(table)
        heads = [(cell.column, cell.column_span, DocumentTableHeader(
                      locator=f'{table_locator}/cell:{cell.row},{cell.column}', text=cell.text, association='column'))
                 for cell in table.cells if header is not None and cell.row == header]
        made: list[DocumentTableCell] = []
        for cell in table.cells:
            indices = sorted(chosen[i] for i in cell.regions)
            start, end = spans[indices[0]][0], spans[indices[-1]][1]
            # The span holds the cell's words and nothing else -- so they
            # sit together in the text, one space apart -- or it would
            # cite something the cell is not.
            if text[start:end].decode('utf-8', 'replace') != cell.text:
                made = []
                break
            over = tuple(head for column, span, head in heads if header is not None and cell.row > header
                         and column < cell.column + cell.column_span and cell.column < column + span)
            made.append(DocumentTableCell(table_locator=table_locator,
                                          locator=f'{table_locator}/cell:{cell.row},{cell.column}',
                                          row=cell.row, column=cell.column, column_span=cell.column_span,
                                          is_header=header is not None and cell.row == header,
                                          text=cell.text, start=start, end=end, headers=over))
        if not made:
            unreadable += 1
            continue
        headed += header is not None
        cells.extend(made)
    # The page's text order, whichever way the page reads: a table's cells
    # hold distinct regions, so no two spans overlap.
    cells.sort(key=lambda c: c.start)
    return PageTables(tuple(cells), len(layout.tables), unreadable, headed)
