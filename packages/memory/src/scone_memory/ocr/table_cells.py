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
cited wrongly, and the page says how many were. Nothing is invented: no
header, no row span, no empty cell.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..core.errors import InvalidInput
from ..ingestion.formats.table_types import DocumentTableCell
from .tables import infer_tables
from .types import OcrRegion

#: Tables a page's segment carries; the rest are counted, not carried.
MAX_TABLES = 64


@dataclass(frozen=True)
class PageTables:
    cells: tuple[DocumentTableCell, ...]
    #: Tables the grid proposed, and those left out because their cells did
    #: not read together in the page's text.
    proposed: int
    unreadable: int


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
    unreadable = 0
    for number, table in enumerate(layout.tables[:MAX_TABLES]):
        table_locator = f'{locator}/table:{number + 1}'
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
            made.append(DocumentTableCell(table_locator=table_locator,
                                          locator=f'{table_locator}/cell:{cell.row},{cell.column}',
                                          row=cell.row, column=cell.column, column_span=cell.column_span,
                                          text=cell.text, start=start, end=end))
        if not made:
            unreadable += 1
            continue
        cells.extend(made)
    # The page's text order, whichever way the page reads: a table's cells
    # hold distinct regions, so no two spans overlap.
    cells.sort(key=lambda c: c.start)
    return PageTables(tuple(cells), len(layout.tables), unreadable)
