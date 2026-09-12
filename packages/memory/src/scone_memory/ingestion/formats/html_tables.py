"""Bounded table evidence from the existing HTML parser's visible event stream."""
from __future__ import annotations

from dataclasses import dataclass, field
from bisect import bisect_left
import re
from typing import Literal, Protocol

from ...core.errors import InvalidInput
from .table_types import DocumentTableCell, DocumentTableHeader, MAX_TABLE_SLOTS
from .types import DocumentLimits, DocumentSegment


class Collector(Protocol):
    segments: list[DocumentSegment]
    size: int
    limits: DocumentLimits

    def check(self, depth: int = 0) -> None: ...
    def add(self, text: str, locator: str, metadata: dict[str, str] | None = None) -> None: ...


@dataclass
class _Cell:
    row: int
    group: int
    attrs: dict[str, str | None]
    header: bool
    hidden: bool
    parts: list[tuple[str, bool]] = field(default_factory=list)
    text: str = ''
    column: int = 0
    width: int = 1
    height: int = 1
    locator: str = ''
    column_header: bool = False
    row_header: bool = False


@dataclass
class _Table:
    locator: str
    segment_start: int
    size_start: int
    cells: list[_Cell] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)
    captions: list[tuple[int, list[tuple[str, bool]]]] = field(default_factory=list)
    column_groups: list[tuple[int, int]] = field(default_factory=list)
    notes: set[str] = field(default_factory=set)
    fallback: str = ''
    group: int = 0
    group_counter: int = 0
    footer_groups: set[int] = field(default_factory=set)
    row_open: bool = False
    active: _Cell | None = None
    caption: list[tuple[str, bool]] | None = None
    colgroup_span: int | None = None
    colgroup_children: int = 0


@dataclass
class _Axis:
    positions: list[int] = field(default_factory=list)
    headers: list[_Cell] = field(default_factory=list)
    data: list[int] = field(default_factory=list)


class _HeaderIndex:
    def __init__(self, grid: dict[tuple[int, int], _Cell], table: _Table) -> None:
        self.rows: dict[int, _Axis] = {}
        self.columns: dict[int, _Axis] = {}
        self.cells = {id(cell) for cell in table.cells}
        self.groups = [cell for cell in table.cells if cell.header
                       and (cell.attrs.get('scope') or '').lower() in {'rowgroup', 'colgroup'}]
        for (row, column), cell in sorted(grid.items()):
            for rays, key, position in ((self.rows, row, column), (self.columns, column, row)):
                axis = rays.setdefault(key, _Axis())
                if cell.header:
                    axis.positions.append(position)
                    axis.headers.append(cell)
                else:
                    axis.data.append(position)


def _span(raw: str | None, maximum: int, *, zero: bool = False) -> int:
    match = re.match(r'[ \t\n\r\f]*\+?([0-9]+)', raw or '')
    if match is None:
        return 1
    digits = match[1].lstrip('0') or '0'
    value = maximum if len(digits) > len(str(maximum)) else min(int(digits), maximum)
    return value if value or zero else 1


def _visible_text(parts: list[tuple[str, bool]]) -> str:
    groups: list[tuple[list[str], bool]] = []
    for text, preformatted in parts:
        if not groups or groups[-1][1] != preformatted:
            groups.append(([], preformatted))
        groups[-1][0].append(text)
    values = [''.join(text) if pre else re.sub(r'[ \t\r\f]+', ' ', ''.join(text)) for text, pre in groups]
    if values and not groups[0][1]:
        values[0] = values[0].lstrip(' \t\r\n\f')
    if values and not groups[-1][1]:
        values[-1] = values[-1].rstrip(' \t\r\n\f')
    return ''.join(values)


class HtmlTables:
    def __init__(self, out: Collector, prefix: str) -> None:
        self.out = out
        self.prefix = prefix
        self.number = 0
        self.depth = 0
        self.table: _Table | None = None
        self.ids: dict[str, _Cell | None] = {}
        self.ids_incomplete = False
        self.work = 0

    def tick(self, amount: int = 1) -> None:
        self.work += amount
        if self.work > 1_000_000:
            raise InvalidInput('document table work exceeds its limit')
        self.out.check()

    def start(self, tag: str, attrs: dict[str, str | None], hidden: bool) -> None:
        if self.table is not None or tag == 'table':
            self.tick()
        cell: _Cell | None = None
        if tag == 'table':
            self.depth += 1
            self.number += 1
            if self.table is None:
                self.table = _Table(f'{self.prefix}table:{self.number}', len(self.out.segments), self.out.size)
            else:
                self.table.fallback = 'nested_table'
        table = self.table
        if table is not None and self.depth == 1:
            if table.colgroup_span is not None and tag != 'col':
                self.end('colgroup')
            if tag in {'thead', 'tbody', 'tfoot'}:
                table.group_counter += 1
                table.group = table.group_counter
                if tag == 'tfoot':
                    table.footer_groups.add(table.group)
            elif tag == 'tr':
                if table.group == 0:
                    table.group_counter += 1
                    table.group = -table.group_counter
                table.rows.append(table.group)
                table.row_open = True
                if len(table.rows) > 20_000:
                    raise InvalidInput('document table rows exceed their limit')
            elif tag in {'td', 'th'}:
                if not table.row_open:
                    table.fallback = table.fallback or 'cell_outside_row'
                cell = _Cell(max(0, len(table.rows) - 1), table.group, attrs, tag == 'th', hidden)
                table.cells.append(cell)
                table.active = cell
                if len(table.cells) > 20_000:
                    raise InvalidInput('document table cells exceed their limit')
            elif tag == 'caption':
                table.caption = []
                table.captions.append((len(table.rows), table.caption))
            elif tag == 'colgroup':
                table.colgroup_span = _span(attrs.get('span'), 1000)
                table.colgroup_children = 0
            elif tag == 'col':
                if table.colgroup_span is None:
                    table.colgroup_span = 1
                    table.colgroup_children = 0
                table.colgroup_children += _span(attrs.get('span'), 1000)
        identifier = attrs.get('id')
        if identifier is not None:
            if len(identifier) > 4096 or len(self.ids) >= 20_000:
                self.ids_incomplete = True
            else:
                self.ids.setdefault(identifier, cell)

    def end(self, tag: str) -> None:
        table = self.table
        if table is None:
            return
        if tag == 'table':
            self.depth -= 1
            if self.depth == 0:
                self.finish(table)
                self.table = None
            return
        if self.depth != 1:
            return
        if tag in {'td', 'th'}:
            table.active = None
        elif tag == 'tr':
            table.row_open = False
        elif tag in {'thead', 'tbody', 'tfoot'}:
            table.group = 0
        elif tag == 'caption':
            table.caption = None
        elif tag == 'colgroup' and table.colgroup_span is not None:
            start = table.column_groups[-1][1] if table.column_groups else 0
            end = start + (table.colgroup_children or table.colgroup_span)
            if end > 1000:
                raise InvalidInput('document table columns exceed their limit')
            table.column_groups.append((start, end))
            table.colgroup_span = None

    def data(self, text: str, preformatted: bool = False) -> None:
        table = self.table
        if table is None or self.depth != 1:
            return
        if table.active is not None:
            table.active.parts.append((text, preformatted))
        elif table.caption is not None:
            table.caption.append((text, preformatted))
        elif text.strip():
            table.fallback = table.fallback or 'text_outside_cells'

    def boundary(self) -> None:
        table = self.table
        if table is not None and table.active is not None:
            self.data('\n')

    def grid(self, table: _Table) -> dict[tuple[int, int], _Cell]:
        occupied: dict[tuple[int, int], _Cell] = {}
        cursors: dict[int, int] = {}
        order = sorted(range(len(table.rows)), key=lambda row: table.rows[row] in table.footer_groups)
        display_rows = {source: displayed for displayed, source in enumerate(order)}
        table.rows = [table.rows[source] for source in order]
        for cell in table.cells:
            cell.row = display_rows.get(cell.row, cell.row)
        group_ends = {group: row + 1 for row, group in enumerate(table.rows)}
        for index, cell in enumerate(table.cells):
            self.tick()
            cell.text = _visible_text(cell.parts)
            cell.locator = f'{table.locator}/cell:{index + 1}'
            cell.width = _span(cell.attrs.get('colspan'), 1000)
            height = _span(cell.attrs.get('rowspan'), 65534, zero=True)
            cell.height = height or group_ends.get(cell.group, cell.row + 1) - cell.row
            if (cell.row + cell.height > 20_000
                    or len(occupied) + cell.width * cell.height > MAX_TABLE_SLOTS):
                raise InvalidInput('document table grid exceeds its limit')
            if cell.row + cell.height > group_ends.get(cell.group, cell.row + 1):
                table.fallback = table.fallback or 'span_outside_row_group'
            column = cursors.get(cell.row, 0)
            while (cell.row, column) in occupied:
                column += 1
            if column + cell.width > 1000:
                raise InvalidInput('document table columns exceed their limit')
            cell.column = column
            cursors[cell.row] = column + cell.width
            for row in range(cell.row, cell.row + cell.height):
                for col in range(column, column + cell.width):
                    if (row, col) in occupied:
                        table.fallback = table.fallback or 'overlapping_cells'
                    occupied[row, col] = cell
        data_rows = {r for (r, _), c in occupied.items() if not c.header}
        data_columns = {c for (_, c), value in occupied.items() if not value.header}
        for cell in table.cells:
            scope = (cell.attrs.get('scope') or '').lower()
            automatic = scope not in {'row', 'col', 'rowgroup', 'colgroup'}
            cell.column_header = scope == 'col' or (automatic and all(
                row not in data_rows for row in range(cell.row, cell.row + cell.height)))
            cell.row_header = scope == 'row' or (automatic and not cell.column_header and all(
                col not in data_columns for col in range(cell.column, cell.column + cell.width)))
        return occupied

    def headers(self, table: _Table, cell: _Cell,
                index: _HeaderIndex) -> tuple[DocumentTableHeader, ...]:
        found: dict[str, DocumentTableHeader] = {}

        def add(header: _Cell, kind: Literal['explicit', 'row', 'column', 'rowgroup', 'colgroup']) -> None:
            if header is cell or header.hidden or not header.header or not header.text.strip():
                return
            found.setdefault(header.locator, DocumentTableHeader(
                locator=header.locator, text=header.text, association=kind))
            if len(found) > 128:
                raise InvalidInput('document table header associations exceed their limit')

        if 'headers' in cell.attrs:
            for identifier in re.findall(r'[^ \t\n\r\f]+', cell.attrs['headers'] or ''):
                self.tick()
                header = self.ids.get(identifier)
                if header is None or id(header) not in index.cells or header.hidden or not header.header:
                    table.notes.add('unresolved_headers')
                    if identifier not in self.ids and self.ids_incomplete:
                        table.notes.add('header_id_limit')
                else:
                    add(header, 'explicit')
            return tuple(found.values())

        def scan(axis: _Axis, position: int, vertical: bool) -> None:
            opaque: set[tuple[int, int]] = set()
            block: set[tuple[int, int]] = set()
            if cell.header:
                block.add((cell.column, cell.width) if vertical else (cell.row, cell.height))
            for cursor in range(bisect_left(axis.positions, position) - 1, -1, -1):
                self.tick()
                coordinate = axis.positions[cursor]
                if bisect_left(axis.data, coordinate) != bisect_left(axis.data, position):
                    opaque.update(block)
                    block.clear()
                position = coordinate
                current = axis.headers[cursor]
                shape = (current.column, current.width) if vertical else (current.row, current.height)
                block.add(shape)
                eligible = current.column_header if vertical else current.row_header
                if shape not in opaque and eligible:
                    add(current, 'column' if vertical else 'row')

        for row in range(cell.row, cell.row + cell.height):
            scan(index.rows[row], cell.column, False)
        for column in range(cell.column, cell.column + cell.width):
            scan(index.columns[column], cell.row, True)
        column_group = next((group for group in table.column_groups
                             if group[0] <= cell.column < group[1]), None)
        for header in index.groups:
            self.tick()
            scope = (header.attrs.get('scope') or '').lower()
            if header.row >= cell.row + cell.height or header.column >= cell.column + cell.width:
                continue
            if scope == 'rowgroup' and cell.group != 0 and header.group == cell.group:
                add(header, 'rowgroup')
            elif scope == 'colgroup' and column_group is not None and column_group[0] <= header.column < column_group[1]:
                add(header, 'colgroup')
        return tuple(found.values())

    def finish(self, table: _Table) -> None:
        grid = self.grid(table)
        if table.fallback:
            for index in range(table.segment_start, len(self.out.segments)):
                segment = self.out.segments[index]
                self.out.segments[index] = segment.model_copy(update={'metadata': {
                    **segment.metadata, 'table_locator': table.locator,
                    'table_status': 'text_fallback', 'table_notes': table.fallback}})
            return
        header_index = _HeaderIndex(grid, table)
        associations = {c.locator: self.headers(table, c, header_index) for c in table.cells if not c.hidden}
        del self.out.segments[table.segment_start:]
        self.out.size = table.size_start
        by_row: dict[int, list[_Cell]] = {}
        for cell in table.cells:
            if not cell.hidden:
                by_row.setdefault(cell.row, []).append(cell)
        for row in range(len(table.rows) + 1):
            for position, caption in table.captions:
                if position == row:
                    self.out.add(_visible_text(caption), f'{table.locator}/caption')
            values = by_row.get(row, [])
            if not any(cell.text.strip() for cell in values):
                continue
            parts: list[str] = []
            evidence: list[DocumentTableCell] = []
            offset = 0
            contextual = any(associations[cell.locator] for cell in values if not cell.header)
            for cell in values:
                headers = associations[cell.locator]
                prefix = (' / '.join(h.text for h in headers) + ': ') if headers and not cell.header else ''
                separator = ('\n' if contextual else ' ') if parts else ''
                parts.append(separator + prefix + cell.text)
                start = offset + len((separator + prefix).encode('utf-8'))
                offset += len(parts[-1].encode('utf-8'))
                if offset > self.out.limits.max_text_bytes:
                    raise InvalidInput('document exceeds its extracted text byte limit')
                evidence.append(DocumentTableCell(table_locator=table.locator, locator=cell.locator,
                    row=row, column=cell.column, row_span=cell.height, column_span=cell.width,
                    is_header=cell.header, text=cell.text, start=start, end=offset, headers=headers))
            metadata = {'table_locator': table.locator, 'table_status': 'structured'}
            if table.notes:
                metadata['table_notes'] = ','.join(sorted(table.notes))
            self.out.add(''.join(parts), f'{table.locator}/row:{row + 1}', metadata)
            self.out.segments[-1] = self.out.segments[-1].model_copy(update={'table_cells': tuple(evidence)})
