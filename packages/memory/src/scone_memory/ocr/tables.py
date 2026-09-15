"""Original, bounded row/gutter analysis of retained OCR rectangles.

A candidate is geometry, not a claim that a page contains a semantic table.
No headers or missing values are invented. A cell that reaches across the
grid's columns -- a title row over three columns, a "Total" beside two
numbers -- is read as spanning them, from its own width against the
column bands the full rows lay down; nothing narrower is widened. Cell
text joins its observations with spaces; individual region references
retain the exact source.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re
from statistics import median
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, ValidationError, model_serializer

from ..core.errors import InvalidInput
from .types import OcrRegion

MAX_REGIONS = 5000
#: A footnote's mark opening a row's first cell.
_NOTE_MARK = re.compile(r'^[*†‡§¹²³⁴⁵⁶⁷⁸⁹]')
#: What a sentence ends with, and a title does not.
_SENTENCE_END = ('.', '!', '?', ';', ':', ',')
#: A citation closing a sentence, as in "history.[28]".
_CITATION = re.compile(r'(?:\s*\[\d{1,4}\])+\s*$')
#: Cells at least this wide, as a share of the page, can be prose.
MIN_PROSE_WIDTH = 0.15
#: A currency sign set on its own, as a statement sets it beside a number.
_CURRENCY = '$\u20ac\u00a3\u00a5'
#: What a number a sign belongs to may open with: a digit, a parenthesis
#: (an accounting negative) or a dash standing for none.
_NUMBER_START = '0123456789(-\u2013\u2014'
#: Rows of fewer cells held above a grid as its possible title, caption
#: or header rows; more than this is prose, not a header.
MAX_HEADER_ROWS = 3


def _is_title(text: str) -> bool:
    """Whether a line above a grid reads as its title rather than a sentence."""
    return not _CITATION.sub('', text).rstrip().endswith(_SENTENCE_END)


def _is_prose(text: str) -> bool:
    """Whether a cell's text reads as a line of prose -- a sentence, or long
    enough to be one -- rather than a label, a number or a wrapped word."""
    return not _is_title(text) or len(text.split()) >= 6 or len(text) >= 40


def _reads_as_prose(boxes: Sequence[Box]) -> bool:
    """Whether a column's cells read as lines of prose: three or more, wide
    enough, and about as wide as each other -- a table's column is narrow or
    ragged."""
    widths = [right - left for left, _, right, _ in boxes]
    return len(widths) >= 3 and max(widths) >= MIN_PROSE_WIDTH and median(widths) >= 0.5 * max(widths)
RegionIndex = Annotated[int, Field(ge=0, lt=MAX_REGIONS)]
Box = tuple[float, float, float, float]


class TableCell(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    row: int = Field(ge=0, lt=1000)
    column: int = Field(ge=0, lt=12)
    #: Columns the cell reaches across, by its width against the column
    #: bands of the grid; 1 for a cell in one column.
    column_span: int = Field(default=1, ge=1, le=12)
    text: str = Field(min_length=1, max_length=2_000_000)
    box: Box
    regions: tuple[RegionIndex, ...] = Field(min_length=1, max_length=MAX_REGIONS)

    @model_serializer(mode='wrap')
    def omit_single_span(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        # A candidate recorded before spans were read serializes as it did.
        value: dict[str, object] = handler(self)
        if self.column_span == 1:
            value.pop('column_span', None)
        return value


class TableCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    rows: int = Field(ge=3, le=1000)
    columns: int = Field(ge=2, le=12)
    cells: tuple[TableCell, ...] = Field(min_length=6, max_length=MAX_REGIONS)


class TableLayout(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    schema_version: Literal[1] = 1
    #: ``aligned-rows-v2`` reads a row of fewer cells as cells spanning the
    #: grid's columns; ``v1`` never did, and is what a stored layout says.
    strategy: Literal['aligned-rows-v1', 'aligned-rows-v2'] = 'aligned-rows-v2'
    origin: Literal['geometry_inferred'] = 'geometry_inferred'
    region_count: int = Field(ge=0, le=MAX_REGIONS)
    tables: tuple[TableCandidate, ...] = Field(max_length=64)
    unassigned: tuple[RegionIndex, ...] = Field(max_length=MAX_REGIONS)
    notes: tuple[Literal['geometry_only', 'unassigned_regions', 'no_aligned_grid'], ...]


@dataclass(frozen=True)
class _Row:
    top: float
    bottom: float
    cells: tuple[tuple[int, ...], ...]
    gutters: tuple[tuple[float, float], ...]


def _box(regions: Sequence[OcrRegion], indices: Sequence[int]) -> Box:
    return (min(regions[i].box[0] for i in indices), min(regions[i].box[1] for i in indices),
            max(regions[i].box[2] for i in indices), max(regions[i].box[3] for i in indices))


def _reaches(left: float, right: float, start: float, end: float, loosely: bool = False) -> bool:
    """Whether a cell over ``left``..``right`` covers the column band
    ``start``..``end``: one inside the other, or the two overlapping by
    half of the narrower and a tenth of the wider -- so a wide cell grazing
    a narrow band does not claim it, and a narrow word inside a wide band
    sits in it. ``loosely``, for a lone cell above the grid -- a title or
    a caption, centred over the columns it names and so short of half of
    the last -- a tenth of the cell's own width is enough."""
    overlap = min(right, end) - max(left, start)
    if (left <= start and right >= end) or (start <= left and right <= end):
        return True
    if loosely:
        return overlap >= 0.1 * (right - left)
    return overlap >= 0.5 * min(right - left, end - start) and overlap >= 0.1 * max(right - left, end - start)


def _rows(regions: Sequence[OcrRegion], gap: float) -> list[_Row]:
    bands: list[list[int]] = []
    top, bottom = 0., 0.
    for index in sorted(range(len(regions)), key=lambda i: (regions[i].box[1], regions[i].box[0], i)):
        box = regions[index].box
        overlap = min(bottom, box[3]) - max(top, box[1])
        if not bands or overlap < .5 * min(bottom - top, box[3] - box[1]):
            bands.append([])
            top, bottom = box[1], box[3]
        else:
            # Intersect the anchor band so a chain of overlapping tall boxes
            # cannot progressively absorb every line on the page.
            top, bottom = max(top, box[1]), min(bottom, box[3])
        bands[-1].append(index)
    rows: list[_Row] = []
    for band in bands:
        cells: list[list[int]] = []
        right = -1.
        gutters: list[tuple[float, float]] = []
        for index in sorted(band, key=lambda i: (regions[i].box[0], i)):
            box = regions[index].box
            # A currency sign on its own, set apart from the number to its
            # right as a statement sets them, is that number's cell.
            sign = bool(cells) and len(cells[-1]) == 1 and len(regions[cells[-1][0]].text.strip()) == 1 and (
                regions[cells[-1][0]].text.strip() in _CURRENCY and regions[index].text[:1] in _NUMBER_START)
            if not cells or (box[0] - right >= gap and not sign):
                if cells:
                    gutters.append((right, box[0]))
                cells.append([])
            elif sign and gutters and len(cells[-1]) == 1:
                # The sign stands at its column's edge, far from the number:
                # the gutter before the cell runs to the number, or a wide
                # number in the column before would close it.
                gutters[-1] = (gutters[-1][0], box[0])
            cells[-1].append(index)
            right = max(right, box[2])
        bounds = _box(regions, band)
        rows.append(_Row(bounds[1], bounds[3], tuple(tuple(cell) for cell in cells), tuple(gutters)))
    return rows


def _near(above: _Row, below: _Row) -> bool:
    """Whether ``above`` sits just over ``below``: no further than twice
    the taller of the two."""
    return above.bottom <= below.top and below.top - above.bottom <= 2 * max(
        above.bottom - above.top, below.bottom - below.top)


def _heads(row: _Row, spans: tuple[tuple[int, int], ...], text: Callable[[Sequence[int]], str]) -> bool:
    """Whether a row of fewer cells placed as ``spans`` above a grid reads
    as its title, caption or header: no cell a sentence, and a lone cell
    across the table from its first column or over two or more columns
    (a caption over the values), not a wrapped word over one column. A
    row set past the first column is a caption, and the comma closing
    "Three Months Ended March 31," -- the years follow below -- is not a
    sentence's, as it is at the margin."""
    for cell in row.cells:
        line = text(cell).rstrip()
        if not _is_title(line.rstrip(',') if spans[0][0] > 0 else line):
            return False
    return len(row.cells) > 1 or spans[0][0] == 0 or spans[0][1] >= 2


def infer_tables(observations: Sequence[OcrRegion]) -> TableLayout:
    """Propose aligned grids, retaining every region as assigned or unassigned.

    Three consecutive separated rows and persistent vertical gutters are needed.
    Pages with more than 5,000 regions or 2 MB of text (including inserted cell
    separators), 64 tables, or 1,000 rows in a table are explicitly refused.
    Multi-line/merged cells, rotations beyond displayed rectangles and semantic
    header recognition are outside this geometric strategy.
    """
    if len(observations) > MAX_REGIONS:
        raise InvalidInput('OCR table analysis exceeds its region limit')
    regions: list[OcrRegion] = []
    text_bytes = 0
    try:
        for value in observations:
            payload = value.model_dump()
            region = OcrRegion.model_validate({key: payload[key] for key in OcrRegion.model_fields if key in payload})
            text_bytes += len(region.text.encode('utf-8'))
            if text_bytes > 2_000_000:
                raise InvalidInput('OCR table analysis exceeds its text limit')
            regions.append(region)
    except (ValidationError, AttributeError, KeyError, UnicodeError):
        raise InvalidInput('OCR table analysis requires valid observations') from None
    widths = [(r.box[2] - r.box[0]) / max(1, len(r.text)) for r in regions]
    gap = min(.06, max(.015, median(widths) * 2.5)) if widths else .015
    tables: list[TableCandidate] = []
    group: list[_Row] = []
    #: Which rows of the group span (their cells placed on the bands), by
    #: row position: the placement of each cell as (first column, span).
    spanning: dict[int, tuple[tuple[int, int], ...]] = {}
    gutters: tuple[tuple[float, float], ...] = ()
    assigned: set[int] = set()

    def _text(indices: Sequence[int]) -> str:
        return ' '.join(regions[i].text for i in indices)

    def bands() -> list[tuple[float, float]]:
        """The grid's column bands: each column's extent across the full
        rows, the widest its cells reach."""
        full = [group[position] for position in range(len(group)) if position not in spanning]
        count = len(full[0].cells)
        columns: list[tuple[float, float]] = []
        for column in range(count):
            boxes = [_box(regions, row.cells[column]) for row in full if len(row.cells) == count]
            columns.append((min(box[0] for box in boxes), max(box[2] for box in boxes)))
        return columns

    def placed(row: _Row, loosely: bool = False) -> tuple[tuple[int, int], ...] | None:
        """A row of fewer cells placed on the grid's columns, each cell over
        the contiguous bands its width covers (``_reaches``), no two cells
        sharing one and none left over; None when the row does not fit."""
        columns = bands()
        taken: list[tuple[int, int]] = []
        used = 0
        for indices in row.cells:
            left, _, right, _ = _box(regions, indices)
            covered = [j for j, (start, end) in enumerate(columns) if _reaches(left, right, start, end, loosely)]
            if not covered or covered != list(range(covered[0], covered[-1] + 1)) or covered[0] < used:
                return None
            taken.append((covered[0], len(covered)))
            used = covered[-1] + 1
        return tuple(taken)

    def emit() -> None:
        nonlocal text_bytes
        full = [position for position in range(len(group)) if position not in spanning]
        if len(full) < 3:
            return
        if len(group) > 1000 or len(tables) >= 64:
            raise InvalidInput('OCR table analysis exceeds its table or row limit')
        text_bytes += sum(len(indices) - 1 for band in group for indices in band.cells)
        if text_bytes > 2_000_000:
            raise InvalidInput('OCR table analysis exceeds its text limit')
        # Two columns of prose side by side align as a grid of two does; a
        # table's first column is narrow or ragged. Neither column is.
        if len(group[full[0]].cells) == 2 and all(
                _reads_as_prose([_box(regions, group[position].cells[column]) for position in full]) for column in (0, 1)):
            return
        cells: list[TableCell] = []
        for row, band in enumerate(group):
            places = spanning.get(row) or tuple((column, 1) for column in range(len(band.cells)))
            for (column, span), indices in zip(places, band.cells):
                cells.append(TableCell(row=row, column=column, column_span=span,
                    text=_text(indices), box=_box(regions, indices), regions=indices))
        tables.append(TableCandidate(rows=len(group), columns=len(group[full[0]].cells), cells=tuple(cells)))
        assigned.update(i for cell in cells for i in cell.regions)

    #: Rows of fewer cells the grid below may own as its title, caption or
    #: header rows -- the rows just above it that joined no grid, and the
    #: rows of a run too short to be one -- the nearest last.
    pending: list[_Row] = []

    def close() -> None:
        nonlocal group, spanning, pending
        if not group:
            return
        # At the foot of the grid, a row of fewer cells that opens with a
        # footnote's mark or holds a line of prose is the table's note or
        # the prose below it, not its last row. A total beside its numbers,
        # a subtotal across them or a wrapped word is.
        while len(group) - 1 in spanning and (_NOTE_MARK.match(_text(group[-1].cells[0]))
                                              or any(_is_prose(_text(cell)) for cell in group[-1].cells)):
            del spanning[len(group) - 1]
            group.pop()
        # The rows of fewer cells just above the first full row -- a title
        # across the table, a caption over its values, a header of years
        # over a blank label column -- are placed now that the grid is
        # known, the nearest first and no further than they sit in a chain
        # (``_near``) and read as a heading (``_heads``). A sentence is not
        # a title, and neither is a wrapped word over a column short of
        # the first.
        full = [position for position in range(len(group)) if position not in spanning]
        if len(full) >= 3:
            columns = len(group[full[0]].cells)
            for row in reversed(pending):
                spans = placed(row) if len(row.cells) < columns else None
                if len(row.cells) == 1 and len(row.cells) < columns:
                    # A caption centred over the columns it names falls
                    # short of half of the last: placed loosely, when that
                    # reaches two or more.
                    loose = placed(row, loosely=True)
                    spans = loose if loose is not None and loose[0][1] >= 2 else spans
                if spans is None or not _near(row, group[0]) or not _heads(row, spans, _text):
                    break
                group.insert(0, row)
                spanning = {position + 1: places for position, places in spanning.items()}
                spanning[0] = spans
            pending = []
        else:
            # A run too short to be a grid may be the header rows of the
            # grid below it: two years over a blank label column.
            pending = (pending + group)[-MAX_HEADER_ROWS:]
        emit()
        group, spanning = [], {}

    for row in _rows(regions, gap):
        width = len(group[-1].cells) if group else 0
        full_rows = [r for position, r in enumerate(group) if position not in spanning]
        expected = len(full_rows[-1].cells) if full_rows else width
        compatible = bool(group) and 2 <= len(row.cells) <= 12 and len(row.cells) == expected
        joined = tuple((max(a, c), min(b, d)) for (a, b), (c, d) in zip(gutters, row.gutters))
        prior = group[-1] if group else None
        near = prior is not None and row.top >= prior.bottom and (
            row.top - prior.bottom <= 2 * max(prior.bottom - prior.top, row.bottom - row.top))
        if compatible:
            compatible = near and all(end - start >= gap for start, end in joined)
        # A row of fewer cells beside full rows may reach across their
        # columns; it joins the group placed on the bands, and does not
        # narrow the gutters the full rows lay down.
        spans = None
        if not compatible and group and near and 1 <= len(row.cells) < expected and expected >= 2:
            spans = placed(row)
        if compatible:
            gutters = joined
            group.append(row)
        elif spans is not None and len(group) >= 2 and len(group) - 1 in spanning and len(group) - 2 in spanning:
            # A third row of fewer cells in a row is prose beside the grid,
            # not wrapped cells of it: the grid ended before them.
            close()
            gutters = row.gutters
        elif spans is not None:
            spanning[len(group)] = spans
            group.append(row)
        else:
            close()
            gutters = row.gutters
            if 2 <= len(row.cells) <= 12:
                group.append(row)
            elif len(row.cells) == 1:
                # A row of one cell may be the title of the grid below.
                pending = (pending + [row])[-MAX_HEADER_ROWS:]
            else:
                pending = []
    close()
    unassigned = tuple(i for i in range(len(regions)) if i not in assigned)
    notes: list[Literal['geometry_only', 'unassigned_regions', 'no_aligned_grid']] = ['geometry_only']
    if unassigned:
        notes.append('unassigned_regions')
    if not tables:
        notes.append('no_aligned_grid')
    return TableLayout(region_count=len(regions), tables=tuple(tables), unassigned=unassigned, notes=tuple(notes))
