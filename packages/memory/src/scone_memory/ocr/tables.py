"""Original, bounded row/gutter analysis of retained OCR rectangles.

A candidate is geometry, not a claim that a page contains a semantic table.
No headers, merged cells or missing values are invented. Cell text joins its
observations with spaces; individual region references retain the exact source.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.errors import InvalidInput
from .types import OcrRegion

MAX_REGIONS = 5000
RegionIndex = Annotated[int, Field(ge=0, lt=MAX_REGIONS)]
Box = tuple[float, float, float, float]


class TableCell(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    row: int = Field(ge=0, lt=1000)
    column: int = Field(ge=0, lt=12)
    text: str = Field(min_length=1, max_length=2_000_000)
    box: Box
    regions: tuple[RegionIndex, ...] = Field(min_length=1, max_length=MAX_REGIONS)


class TableCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    rows: int = Field(ge=3, le=1000)
    columns: int = Field(ge=2, le=12)
    cells: tuple[TableCell, ...] = Field(min_length=6, max_length=MAX_REGIONS)


class TableLayout(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    schema_version: Literal[1] = 1
    strategy: Literal['aligned-rows-v1'] = 'aligned-rows-v1'
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
            if not cells or box[0] - right >= gap:
                if cells:
                    gutters.append((right, box[0]))
                cells.append([])
            cells[-1].append(index)
            right = max(right, box[2])
        bounds = _box(regions, band)
        rows.append(_Row(bounds[1], bounds[3], tuple(tuple(cell) for cell in cells), tuple(gutters)))
    return rows


def infer_tables(observations: Sequence[OcrRegion]) -> TableLayout:
    """Propose aligned grids, retaining every region as assigned or unassigned.

    Three consecutive separated rows and persistent vertical gutters are needed.
    Pages with more than 5,000 regions or 2 MB of text are explicitly refused.
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
            region = OcrRegion.model_validate({key: payload[key] for key in OcrRegion.model_fields})
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
    gutters: tuple[tuple[float, float], ...] = ()
    assigned: set[int] = set()

    def emit() -> None:
        if len(group) < 3:
            return
        if len(group) > 1000 or len(tables) >= 64:
            raise InvalidInput('OCR table analysis exceeds its table or row limit')
        cells = tuple(TableCell(row=row, column=column,
            text=' '.join(regions[i].text for i in indices), box=_box(regions, indices), regions=indices)
            for row, band in enumerate(group) for column, indices in enumerate(band.cells))
        tables.append(TableCandidate(rows=len(group), columns=len(group[0].cells), cells=cells))
        assigned.update(i for cell in cells for i in cell.regions)

    for row in _rows(regions, gap):
        compatible = bool(group) and 2 <= len(row.cells) <= 12 and len(row.cells) == len(group[-1].cells)
        joined = tuple((max(a, c), min(b, d)) for (a, b), (c, d) in zip(gutters, row.gutters))
        if compatible:
            prior = group[-1]
            compatible = (row.top >= prior.bottom
                and row.top - prior.bottom <= 2 * max(prior.bottom - prior.top, row.bottom - row.top)
                and all(end - start >= gap for start, end in joined))
        if not compatible:
            emit()
            group = []
            gutters = row.gutters
        else:
            gutters = joined
        if 2 <= len(row.cells) <= 12:
            group.append(row)
    emit()
    unassigned = tuple(i for i in range(len(regions)) if i not in assigned)
    notes: list[Literal['geometry_only', 'unassigned_regions', 'no_aligned_grid']] = ['geometry_only']
    if unassigned:
        notes.append('unassigned_regions')
    if not tables:
        notes.append('no_aligned_grid')
    return TableLayout(region_count=len(regions), tables=tuple(tables), unassigned=unassigned, notes=tuple(notes))
