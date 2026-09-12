"""Source-declared cell geometry and independently verifiable header references."""
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, SerializerFunctionWrapHandler, model_serializer

from ...core.errors import InvalidInput

if TYPE_CHECKING:
    from .types import DocumentSegment

MAX_TABLE_SLOTS = 100_000
SourceLocator = Annotated[str, Field(min_length=1, max_length=4096)]


class DocumentTableHeader(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    locator: str = Field(min_length=1, max_length=4096)
    text: str = Field(min_length=1, max_length=2_000_000)
    association: Literal['explicit', 'row', 'column', 'rowgroup', 'colgroup']


class DocumentTableContext(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    locator: str = Field(min_length=1, max_length=4096)
    text: str = Field(min_length=1, max_length=2_000_000)
    association: Literal['row_span'] = 'row_span'


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
    context: tuple[DocumentTableContext, ...] = Field(default=(), max_length=128)
    merged_locators: tuple[SourceLocator, ...] = Field(default=(), max_length=20_000)

    @model_serializer(mode='wrap')
    def serialize_optional_sources(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        result: dict[str, object] = handler(self)
        if not self.context:
            result.pop('context', None)
        if not self.merged_locators:
            result.pop('merged_locators', None)
        return result


def validate_tables(segments: tuple[DocumentSegment, ...]) -> None:
    cells: dict[str, DocumentTableCell] = {}
    occupied: set[tuple[str, int, int]] = set()
    evidence_bytes = 0
    source_locators: set[str] = set()
    for segment in segments:
        encoded = segment.text.encode('utf-8')
        previous = 0
        for observed in segment.table_cells:
            try:
                cell = DocumentTableCell.model_validate(observed.model_dump())
                value = cell.text.encode('utf-8')
                raw_size = sum(len(text) for text in (cell.text, cell.locator, cell.table_locator, *cell.merged_locators))
                raw_size += sum(len(h.text) + len(h.locator) for h in cell.headers)
                raw_size += sum(len(c.text) + len(c.locator) for c in cell.context)
                if evidence_bytes + raw_size > 8_000_000:
                    raise InvalidInput('document table evidence exceeds its limit')
                evidence_bytes += len(cell.model_dump_json().encode('utf-8'))
            except (ValidationError, AttributeError, UnicodeError):
                raise InvalidInput('document table cell is invalid') from None
            if (not previous <= cell.start <= cell.end <= len(encoded)
                    or encoded[cell.start:cell.end] != value):
                raise InvalidInput('document table cell does not match its text span')
            previous = cell.end
            for locator in (cell.locator, *cell.merged_locators):
                if not locator or len(locator) > 4096 or locator in source_locators:
                    raise InvalidInput('document table source locator is invalid or duplicated')
                source_locators.add(locator)
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
        contexts: set[str] = set()
        for reference in cell.context:
            source = cells.get(reference.locator)
            if (source is None or source.is_header or source.text != reference.text
                    or source.table_locator != cell.table_locator or source.locator in contexts
                    or not source.row < cell.row < source.row + source.row_span):
                raise InvalidInput('document table row context does not match its spanning source cell')
            contexts.add(source.locator)
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
