"""Bind SpreadsheetML table declarations to retained worksheet cell values."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import re
from xml.etree.ElementTree import Element

from ...core.errors import InvalidInput
from .table_types import DocumentTableCell, DocumentTableHeader, MAX_TABLE_SLOTS
from .types import DocumentSegment

_NAMESPACES = frozenset({'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
                        'http://purl.oclc.org/ooxml/spreadsheetml/main'})
_RELATIONSHIPS = frozenset({'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
                          'http://purl.oclc.org/ooxml/officeDocument/relationships'})


class _Unresolved(ValueError):
    pass


def cell_position(value: str) -> tuple[int, int]:
    match = re.fullmatch(r'\$?([A-Z]{1,3})\$?([1-9][0-9]{0,6})', value)
    if match is None:
        raise _Unresolved('invalid_cell_reference')
    column = 0
    for character in match[1]:
        column = column * 26 + ord(character) - ord('A') + 1
    row = int(match[2])
    if column > 16384 or row > 1048576:
        raise _Unresolved('invalid_cell_reference')
    return row - 1, column - 1


def _range(value: str) -> tuple[int, int, int, int]:
    parts = value.split(':')
    if not 1 <= len(parts) <= 2:
        raise _Unresolved('invalid_table_range')
    first, last = cell_position(parts[0]), cell_position(parts[-1])
    if first[0] > last[0] or first[1] > last[1]:
        raise _Unresolved('invalid_table_range')
    return *first, *last


def _count(value: str, maximum: int) -> int:
    if not re.fullmatch(r'[0-9]{1,10}', value) or int(value) > maximum:
        raise _Unresolved('invalid_table_count')
    return int(value)


@dataclass
class _Table:
    locator: str
    member: str
    name: str
    reference: str
    first_row: int
    first_column: int
    last_row: int
    last_column: int
    header_rows: int
    total_rows: int
    names: tuple[str, ...]
    headers: dict[int, DocumentSegment] = field(default_factory=dict)
    notes: set[str] = field(default_factory=set)


def _text_node_ids(root: Element) -> set[int]:
    return {id(element) for element in root.iter() if element.tag.rsplit('}', 1)[-1]
            in {'t', 'tab', 'ptab', 'br', 'cr', 'noBreakHyphen'}}


def spreadsheet_text_supported(root: Element) -> bool:
    namespace = root.tag.rsplit('}', 1)[0].removeprefix('{')
    ns = '{' + namespace + '}'
    if namespace not in _NAMESPACES or root.tag not in {ns + 'si', ns + 'is'}:
        return False
    expected = {id(element) for element in (*root.findall(ns + 't'), *root.findall(ns + 'r/' + ns + 't'))}
    return expected == _text_node_ids(root)


def _worksheet_cells(worksheet: Element, ns: str, check: Callable[[], None]) -> dict[tuple[int, int], Element]:
    data = worksheet.findall(ns + 'sheetData')
    if len(data) != 1:
        raise _Unresolved('unsupported_worksheet_structure')
    rows = data[0].findall(ns + 'row')
    if {id(element) for element in worksheet.iter() if element.tag.rsplit('}', 1)[-1] == 'row'} != {id(element) for element in rows}:
        raise _Unresolved('unsupported_worksheet_structure')
    result: dict[tuple[int, int], Element] = {}
    previous_row = 0
    for row_element in rows:
        check()
        row = _count(row_element.get('r', str(previous_row + 1)), 1048576)
        if not row:
            raise _Unresolved('invalid_cell_row')
        previous_row, next_column = row, 0
        for cell in row_element:
            check()
            if cell.tag.rsplit('}', 1)[-1] != 'c':
                continue
            if cell.tag != ns + 'c' or any(element.tag.rsplit('}', 1)[-1] in {'v', 'is', 't', 'f'}
                    and not element.tag.startswith(ns) for element in cell.iter()):
                raise _Unresolved('unsupported_worksheet_structure')
            if cell.get('t') == 'inlineStr':
                strings = cell.findall(ns + 'is')
                if (len(strings) != 1 or not spreadsheet_text_supported(strings[0])
                        or _text_node_ids(cell) != _text_node_ids(strings[0])):
                    raise _Unresolved('unsupported_worksheet_structure')
            position = cell_position(cell.attrib['r']) if 'r' in cell.attrib else (row - 1, next_column)
            if position[0] != row - 1 or position[1] >= 16384:
                raise _Unresolved('inconsistent_cell_row')
            next_column = position[1] + 1
            if position in result:
                raise _Unresolved('duplicate_worksheet_cell')
            if len(result) >= MAX_TABLE_SLOTS:
                raise InvalidInput('spreadsheet cell inspection exceeds its limit')
            result[position] = cell
    return result


def _declarations(worksheet: Element, prefix: str, load: Callable[[str], tuple[str, Element]],
                  check: Callable[[], None]) -> tuple[list[_Table], dict[tuple[int, int], _Table]]:
    namespace = worksheet.tag.rsplit('}', 1)[0].removeprefix('{')
    if namespace not in _NAMESPACES:
        raise _Unresolved('unsupported_table_namespace')
    ns = '{' + namespace + '}'
    physical = _worksheet_cells(worksheet, ns, check)
    groups = [child for child in worksheet if child.tag.rsplit('}', 1)[-1] == 'tableParts']
    if len(groups) != 1 or groups[0].tag != ns + 'tableParts':
        raise _Unresolved('invalid_table_parts')
    group = groups[0]
    if len(group) > 1000:
        raise InvalidInput('spreadsheet table declarations exceed their limit')
    if 'count' in group.attrib and _count(group.attrib['count'], 1000) != len(group):
        raise _Unresolved('invalid_table_parts_count')
    tables: list[_Table] = []
    positions: dict[tuple[int, int], _Table] = {}
    members: set[str] = set()
    for ordinal, part in enumerate(group, 1):
        check()
        identifiers = [value for key, value in part.attrib.items()
                       if key in {f'{{{space}}}id' for space in _RELATIONSHIPS}]
        if part.tag != ns + 'tablePart' or len(identifiers) != 1:
            raise _Unresolved('invalid_table_part')
        member, root = load(identifiers[0])
        if member in members or root.tag != ns + 'table':
            raise _Unresolved('invalid_table_definition')
        members.add(member)
        reference = root.get('ref', '')
        top, left, bottom, right = _range(reference)
        height, width = bottom - top + 1, right - left + 1
        if height > 20000 or width > 1000 or len(positions) + height * width > MAX_TABLE_SLOTS:
            raise _Unresolved('table_grid_limit')
        header = _count(root.get('headerRowCount', '1'), 1)
        total = _count(root.get('totalsRowCount', '0'), 1)
        if header + total > height:
            raise _Unresolved('invalid_table_rows')
        columns = root.findall(ns + 'tableColumns')
        if len(columns) != 1 or len(columns[0]) != width:
            raise _Unresolved('invalid_table_columns')
        column_group = columns[0]
        if 'count' in column_group.attrib and _count(column_group.attrib['count'], 1000) != width:
            raise _Unresolved('invalid_table_columns_count')
        names: list[str] = []
        ids: set[int] = set()
        for column in column_group:
            identifier, name = _count(column.get('id', ''), 2**32 - 1), column.get('name', '')
            if (column.tag != ns + 'tableColumn' or not identifier
                    or identifier in ids or not name or len(name) > 4096):
                raise _Unresolved('invalid_table_column')
            ids.add(identifier)
            names.append(name)
        name = root.get('displayName', root.get('name', ''))
        if not name or len(name) > 4096:
            raise _Unresolved('invalid_table_name')
        table = _Table(f'{prefix}/table:{ordinal}', member, name, reference,
                       top, left, bottom, right, header, total, tuple(names))
        if not header:
            table.notes.add('header_row_absent')
        tables.append(table)
        for row in range(top, bottom + 1):
            check()
            for column_number in range(left, right + 1):
                key = row, column_number
                if key in positions:
                    raise _Unresolved('overlapping_table_ranges')
                positions[key] = table
    for position, cell in physical.items():
        matched_table = positions.get(position)
        if matched_table is None or cell.find(ns + 'f') is None:
            continue
        cached = cell.find(ns + 'v')
        if cell.get('t') != 'inlineStr' and (cached is None or not cached.text):
            matched_table.notes.add('missing_cached_formula')
    merge_checks = 0
    for group in worksheet:
        if group.tag.rsplit('}', 1)[-1] != 'mergeCells':
            continue
        if group.tag != ns + 'mergeCells' or len(group) > MAX_TABLE_SLOTS:
            raise _Unresolved('invalid_merge_cells')
        for merge in group:
            check()
            if merge.tag != ns + 'mergeCell':
                raise _Unresolved('unsupported_table_namespace')
            top, left, bottom, right = _range(merge.get('ref', ''))
            merge_checks += len(tables)
            if merge_checks > 1_000_000:
                raise _Unresolved('table_merge_check_limit')
            if any(t.first_row <= bottom and top <= t.last_row and t.first_column <= right
                   and left <= t.last_column for t in tables):
                raise _Unresolved('merged_table_cells')
    return tables, positions


def spreadsheet_tables(segments: list[DocumentSegment], worksheet: Element, *, prefix: str, member: str,
                       load: Callable[[str], tuple[str, Element]], check: Callable[[], None],
                       shared_strings_supported: bool = True) -> Iterator[DocumentSegment]:
    """Yield cell evidence without guessing labels for ordinary worksheet rows.

    Malformed declarations retain the already extracted cells with an explicit
    fallback note. Missing/unsafe package relationships remain parser errors.
    """
    try:
        if not shared_strings_supported:
            raise _Unresolved('unsupported_shared_string_structure')
        tables, positions = _declarations(worksheet, prefix, load, check)
        found: set[tuple[int, int]] = set()
        for segment in segments:
            check()
            row, column = cell_position(segment.metadata['cell'])
            if (row, column) in found:
                raise _Unresolved('duplicate_worksheet_cell')
            found.add((row, column))
            table = positions.get((row, column))
            if table is not None and table.header_rows and row == table.first_row:
                table.headers[column] = segment
        for table in tables:
            if not table.header_rows:
                continue
            for column, name in enumerate(table.names, table.first_column):
                header = table.headers.get(column)
                if header is None:
                    table.notes.add('missing_header_cell')
                elif header.text != name:
                    table.notes.add('column_name_mismatch')
    except _Unresolved as error:
        for segment in segments:
            check()
            yield segment.model_copy(update={'metadata': {**segment.metadata, 'member': member,
                'table_status': 'text_fallback', 'table_notes': str(error)}})
        return
    for segment in segments:
        check()
        row, column = cell_position(segment.metadata['cell'])
        table = positions.get((row, column))
        if table is None:
            yield segment
            continue
        is_header = bool(table.header_rows and row == table.first_row)
        header = table.headers.get(column) if not is_header else None
        headers = () if header is None else (DocumentTableHeader(locator=header.locator, text=header.text,
                                                                 association='column'),)
        label = '' if header is None else header.text + ': '
        text = label + segment.text
        start = len(label.encode('utf-8'))
        cell = DocumentTableCell(table_locator=table.locator, locator=segment.locator,
            row=row - table.first_row, column=column - table.first_column, is_header=is_header,
            text=segment.text, start=start, end=start + len(segment.text.encode('utf-8')), headers=headers)
        metadata = {**segment.metadata, 'member': member, 'table_locator': table.locator,
                    'table_member': table.member, 'table_name': table.name, 'table_range': table.reference,
                    'table_status': 'structured', 'header_basis': 'xlsx_table_declaration',
                    'table_role': 'header' if is_header else 'totals' if table.total_rows and row == table.last_row else 'data'}
        if table.notes:
            metadata['table_notes'] = ','.join(sorted(table.notes))
        yield DocumentSegment(text=text, locator=segment.locator, metadata=metadata, table_cells=(cell,))
