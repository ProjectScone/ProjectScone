"""Source-declared cell geometry and independently verifiable header references."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ...core.errors import InvalidInput

if TYPE_CHECKING:
    from .types import DocumentSegment

MAX_TABLE_SLOTS = 100_000


class DocumentTableHeader(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    locator: str = Field(min_length=1, max_length=4096)
    text: str = Field(min_length=1, max_length=2_000_000)
    association: Literal['explicit', 'row', 'column', 'rowgroup', 'colgroup']


class DocumentTableCell(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    table_locator: str = Field(min_length=1, max_length=4096)
    locator: str = Field(min_length=1, max_length=4096)
    row: int = Field(ge=0, lt=20_000)
    column: int = Field(ge=0, lt=1000)
    row_span: int = Field(default=1, ge=1, le=20_000)
    column_span: int = Field(default=1, ge=1, le=1000)
    is_header: bool = False
    text: str = Field(max_length=2_000_000)
    start: int = Field(ge=0, le=2_000_000)
    end: int = Field(ge=0, le=2_000_000)
    headers: tuple[DocumentTableHeader, ...] = Field(default=(), max_length=128)


def validate_tables(segments: tuple[DocumentSegment, ...]) -> None:
    cells: dict[str, DocumentTableCell] = {}
    occupied: set[tuple[str, int, int]] = set()
    evidence_bytes = 0
    for segment in segments:
        encoded = segment.text.encode('utf-8')
        previous = 0
        for observed in segment.table_cells:
            try:
                cell = DocumentTableCell.model_validate(observed.model_dump())
                value = cell.text.encode('utf-8')
                evidence_bytes += len(cell.model_dump_json().encode('utf-8'))
            except (ValidationError, AttributeError, UnicodeError):
                raise InvalidInput('document table cell is invalid') from None
            if (not previous <= cell.start <= cell.end <= len(encoded)
                    or encoded[cell.start:cell.end] != value):
                raise InvalidInput('document table cell does not match its text span')
            previous = cell.end
            if cell.locator in cells:
                raise InvalidInput('document table cell locator is duplicated')
            cells[cell.locator] = cell
            area = cell.row_span * cell.column_span
            if (cell.row + cell.row_span > 20_000 or cell.column + cell.column_span > 1000
                    or len(occupied) + area > MAX_TABLE_SLOTS or evidence_bytes > 8_000_000):
                raise InvalidInput('document table evidence exceeds its limit')
            for row in range(cell.row, cell.row + cell.row_span):
                for column in range(cell.column, cell.column + cell.column_span):
                    slot = (cell.table_locator, row, column)
                    if slot in occupied:
                        raise InvalidInput('document table cells overlap')
                    occupied.add(slot)
    for cell in cells.values():
        seen: set[str] = set()
        for header in cell.headers:
            source = cells.get(header.locator)
            if (source is None or not source.is_header or source.text != header.text
                    or source.table_locator != cell.table_locator or source.locator == cell.locator
                    or source.locator in seen):
                raise InvalidInput('document table header does not match its source cell')
            seen.add(source.locator)
            if header.association == 'row' and not (
                    source.column < cell.column and source.row < cell.row + cell.row_span
                    and cell.row < source.row + source.row_span):
                raise InvalidInput('document table row header does not intersect its cell row')
            if header.association == 'column' and not (
                    source.row < cell.row and source.column < cell.column + cell.column_span
                    and cell.column < source.column + source.column_span):
                raise InvalidInput('document table column header does not intersect its cell column')
