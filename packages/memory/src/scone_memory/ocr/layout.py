"""Original bounded whitespace partitioning over observed OCR rectangles.

This estimates column order, not semantic layout or table structure. Words
within a column retain provider order. Crossing body text prevents a split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import OcrRegion, OrderedOcrRegion

ReadingMode = Literal['provider', 'columns_ltr', 'columns_rtl']
LayoutNote = Literal['geometry_inferred', 'no_separating_gutter', 'candidate_limit', 'column_limit']


class ReadingOrderReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    strategy: Literal['whitespace-columns-v1'] = 'whitespace-columns-v1'
    direction: Literal['ltr', 'rtl'] = 'ltr'
    columns: int = Field(ge=0, le=8)
    notes: tuple[LayoutNote, ...] = Field(max_length=4)

    @model_validator(mode='after')
    def consistent_notes(self) -> ReadingOrderReceipt:
        expected = 'geometry_inferred' if self.columns else 'no_separating_gutter'
        if (self.columns == 1 or expected not in self.notes or len(set(self.notes)) != len(self.notes)
                or ('geometry_inferred' in self.notes) == ('no_separating_gutter' in self.notes)):
            raise ValueError('OCR reading order notes do not match its columns')
        return self


def validate_reading_order(regions: Sequence[OrderedOcrRegion], receipt: ReadingOrderReceipt | None) -> None:
    if receipt is None:
        if any(r.provider_index is not None or r.reading_column is not None for r in regions):
            raise ValueError('OCR ordered regions require a reading order receipt')
        return
    receipt = ReadingOrderReceipt.model_validate(receipt.model_dump())
    if (any(type(r.provider_index) is not int or type(r.reading_column) is not int
            or not 0 <= r.reading_column <= receipt.columns for r in regions)
            or sorted(r.provider_index for r in regions if r.provider_index is not None) != list(range(len(regions)))
            or {r.reading_column for r in regions if r.reading_column} != set(range(1, receipt.columns + 1))):
        raise ValueError('OCR reading order does not cover its original regions')


@dataclass(frozen=True)
class OrderedRegions:
    indices: tuple[int, ...]
    columns: tuple[int, ...]
    receipt: ReadingOrderReceipt


@dataclass(frozen=True)
class _Run:
    box: tuple[float, float, float, float]
    indices: tuple[int, ...]


def _line_runs(regions: Sequence[OcrRegion]) -> list[_Run]:
    lines: dict[tuple[int, int], list[int]] = {}
    for i, region in enumerate(regions):
        lines.setdefault((region.block, region.line), []).append(i)
    runs: list[_Run] = []

    def emit(indices: list[int]) -> None:
        runs.append(_Run((min(regions[i].box[0] for i in indices), min(regions[i].box[1] for i in indices),
                          max(regions[i].box[2] for i in indices), max(regions[i].box[3] for i in indices)),
                         tuple(sorted(indices))))

    for indices in lines.values():
        rows: list[list[int]] = []
        bottom = -1.0
        for i in sorted(indices, key=lambda i: regions[i].box[1]):
            if regions[i].box[1] >= bottom:
                rows.append([])
            rows[-1].append(i)
            bottom = max(bottom, regions[i].box[3])
        for row in rows:
            current: list[int] = []
            right = -1.0
            for i in sorted(row, key=lambda i: regions[i].box[0]):
                if current and regions[i].box[0] - right >= .06:
                    emit(current)
                    current = []
                current.append(i)
                right = max(right, regions[i].box[2])
            emit(current)
    return sorted(runs, key=lambda run: run.indices[0])


def _rows(regions: Sequence[_Run], indices: list[int]) -> int:
    rows, bottom = 0, -1.0
    for index in sorted(indices, key=lambda i: regions[i].box[1]):
        box = regions[index].box
        if box[1] >= bottom:
            rows += 1
        bottom = max(bottom, box[3])
    return rows


def _partition(regions: Sequence[_Run], indices: list[int], notes: set[LayoutNote]
               ) -> tuple[list[int], list[int], list[int], list[int]] | None:
    edges = sorted({coordinate for i in indices for coordinate in (regions[i].box[0], regions[i].box[2])})
    gaps = sorted(zip(edges, edges[1:]), key=lambda pair: (-(pair[1] - pair[0]), pair[0]))
    if len(gaps) > 32:
        notes.add('candidate_limit')
    # Every candidate is checked against actual rectangles. The candidate cap
    # bounds work on dense pages, and an omitted candidate is disclosed.
    for lo, hi in gaps[:32]:
        middle = (lo + hi) / 2
        left: list[int] = []
        right: list[int] = []
        crossing: list[int] = []
        for i in indices:
            box = regions[i].box
            (left if box[2] <= middle else right if box[0] >= middle else crossing).append(i)
        if not left or not right:
            continue
        gap = min(regions[i].box[0] for i in right) - max(regions[i].box[2] for i in left)
        if gap < .06 or _rows(regions, left) < 3 or _rows(regions, right) < 3:
            continue
        left_top, left_bottom = min(regions[i].box[1] for i in left), max(regions[i].box[3] for i in left)
        right_top, right_bottom = min(regions[i].box[1] for i in right), max(regions[i].box[3] for i in right)
        overlap = min(left_bottom, right_bottom) - max(left_top, right_top)
        if overlap < .5 * min(left_bottom - left_top, right_bottom - right_top):
            continue
        top, bottom = min(left_top, right_top), max(left_bottom, right_bottom)
        above = [i for i in crossing if regions[i].box[3] <= top]
        below = [i for i in crossing if regions[i].box[1] >= bottom]
        if len(above) + len(below) != len(crossing):
            continue
        return above, left, right, below
    return None


def order_columns(regions: Sequence[OcrRegion], *, direction: Literal['ltr', 'rtl'] = 'ltr') -> OrderedRegions:
    """Keep every observation exactly once, including spanning headings/footers."""
    if direction not in ('ltr', 'rtl') or len(regions) > 50_000:
        raise ValueError('invalid OCR reading order or region budget')
    notes: set[LayoutNote] = set()
    ordered: list[int] = []
    assigned: list[int] = []
    column = 0
    runs = _line_runs(regions)

    def append(indices: list[int], group: int) -> None:
        members = sorted(i for index in indices for i in runs[index].indices)
        ordered.extend(members)
        assigned.extend([group] * len(members))

    def visit(indices: list[int], depth: int) -> None:
        nonlocal column
        split = _partition(runs, indices, notes) if len(indices) >= 6 else None
        if split is None or depth == 3:
            if split is not None:
                notes.add('column_limit')
            if depth:
                column += 1
            append(indices, column if depth else 0)
            return
        above, left, right, below = split
        append(above, 0)
        for child in (left, right) if direction == 'ltr' else (right, left):
            visit(child, depth + 1)
        append(below, 0)

    visit(list(range(len(runs))), 0)
    notes.add('geometry_inferred' if column else 'no_separating_gutter')
    return OrderedRegions(tuple(ordered), tuple(assigned), ReadingOrderReceipt(
        direction=direction, columns=column, notes=tuple(sorted(notes))))
