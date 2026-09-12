"""Original WordprocessingML grid assembly with retained merge sources."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import re
from xml.etree.ElementTree import Element

from ...core.errors import InvalidInput
from .table_types import DocumentTableCell, DocumentTableContext, DocumentTableHeader, MAX_TABLE_SLOTS
from .types import DocumentLimits, DocumentSegment

WORD_NAMESPACES = frozenset({
    'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'http://purl.oclc.org/ooxml/wordprocessingml/main',
})


def word_children(root: Element, name: str) -> Iterator[Element]:
    namespace = root.tag.rsplit('}', 1)[0] + '}'
    wrappers = {namespace + tag for tag in ('sdt', 'sdtContent', 'customXml', 'ins', 'moveTo')}
    pending = [iter(root)]
    while pending:
        child = next(pending[-1], None)
        if child is None:
            pending.pop()
        elif child.tag.rsplit('}', 1)[-1] == name:
            yield child
        elif child.tag in wrappers:
            pending.append(iter(child))


def _property(root: Element, container: str, name: str) -> Element | None:
    namespace = root.tag.rsplit('}', 1)[0] + '}'
    return root.find(f'{namespace}{container}/{namespace}{name}')


def _value(element: Element, default: str = '') -> str:
    return element.get(element.tag.rsplit('}', 1)[0] + '}val', default)


class _Unresolved(Exception):
    pass


def _number(element: Element | None, default: int, *, zero: bool = False) -> int:
    if element is None:
        return default
    text = _value(element).strip()
    if re.fullmatch(r'\+?[0-9]+', text) is None:
        raise _Unresolved('invalid_grid_value')
    digits = text.lstrip('+0') or '0'
    if len(digits) > 4 or int(digits) > 1000:
        raise InvalidInput('document table grid exceeds its limit')
    value = int(digits)
    if not value and not zero:
        raise _Unresolved('invalid_grid_span')
    return value


def _merge(element: Element, name: str) -> str:
    prop = _property(element, 'tcPr', name)
    if prop is None:
        return ''
    value = _value(prop, 'continue')
    if value not in {'continue', 'restart'}:
        raise _Unresolved('invalid_merge_value')
    return value


@dataclass
class _Cell:
    row: int
    column: int
    width: int
    header: bool
    text: str
    locator: str
    vertical: str
    height: int = 1
    merged: list[str] = field(default_factory=list)


class _Table:
    def __init__(self, locator: str, limits: DocumentLimits, check: Callable[[], None],
                 text_bytes: int = 0, segments: int = 0) -> None:
        self.locator = locator
        self.limits = limits
        self.check = check
        self.cells: list[_Cell] = []
        self.notes: set[str] = set()
        self.work = 0
        self.slots = 0
        self.text_bytes = text_bytes
        self.segments = segments

    def tick(self, amount: int = 1) -> None:
        self.check()
        self.work += amount
        if self.work > 1_000_000:
            raise InvalidInput('document table work exceeds its limit')

    def join(self, values: list[str], separator: str = '\n') -> str:
        if sum(len(value.encode('utf-8')) for value in values) + max(0, len(values) - 1) * len(separator.encode()) > self.limits.max_text_bytes:
            raise InvalidInput('document exceeds its extracted text byte limit')
        return separator.join(values)

    def absorb(self, target: _Cell, source: _Cell) -> None:
        if source.text:
            target.text = self.join([value for value in (target.text, source.text) if value])
        target.merged.extend((source.locator, *source.merged))

    def assemble(self, rows: list[tuple[Element, list[tuple[Element, str]]]]) -> None:
        active: dict[tuple[int, int], _Cell] = {}
        leading_headers = True
        for row_number, (row, elements) in enumerate(rows):
            marker = _property(row, 'trPr', 'tblHeader')
            declared = marker is not None and _value(marker, 'true') in {'1', 'true', 'on'}
            if declared and not leading_headers:
                self.notes.add('noncontiguous_header')
            leading_headers = leading_headers and declared
            column = _number(_property(row, 'trPr', 'gridBefore'), 0, zero=True)
            horizontal: _Cell | None = None
            current: list[_Cell] = []
            for ordinal, (element, text) in enumerate(elements, 1):
                self.tick()
                width = _number(_property(element, 'tcPr', 'gridSpan'), 1)
                if column + width > 1000:
                    raise InvalidInput('document table columns exceed their limit')
                locator = f'{self.locator}/row:{row_number + 1}/cell:{ordinal}'
                if len(locator) > 4096:
                    raise InvalidInput('document table source locator exceeds its limit')
                cell = _Cell(row_number, column, width, leading_headers, text, locator, _merge(element, 'vMerge'))
                column += width
                merge = _merge(element, 'hMerge')
                if merge == 'continue':
                    if horizontal is None or horizontal.vertical != cell.vertical:
                        raise _Unresolved('unmatched_horizontal_merge')
                    self.absorb(horizontal, cell)
                    horizontal.width += width
                else:
                    current.append(cell)
                    horizontal = cell if merge == 'restart' else None
            after = _number(_property(row, 'trPr', 'gridAfter'), 0, zero=True)
            if column + after > 1000:
                raise InvalidInput('document table columns exceed their limit')
            next_active: dict[tuple[int, int], _Cell] = {}
            for cell in current:
                self.slots += cell.width
                if self.slots > MAX_TABLE_SLOTS:
                    raise InvalidInput('document table grid exceeds its limit')
                key = (cell.column, cell.width)
                if cell.vertical == 'continue':
                    origin = active.get(key)
                    if origin is None:
                        raise _Unresolved('unmatched_vertical_merge')
                    self.absorb(origin, cell)
                    origin.height += 1
                    next_active[key] = origin
                else:
                    self.cells.append(cell)
                    if cell.vertical == 'restart':
                        next_active[key] = cell
            active = next_active

    def render(self, rows: int, metadata: dict[str, str]) -> dict[int, DocumentSegment]:
        by_row: dict[int, list[_Cell]] = {}
        headers: dict[int, list[_Cell]] = {}
        carried: dict[int, list[_Cell]] = {}
        for cell in self.cells:
            by_row.setdefault(cell.row, []).append(cell)
            if cell.header and cell.text.strip():
                for column in range(cell.column, cell.column + cell.width):
                    headers.setdefault(column, []).append(cell)
            elif cell.text.strip():
                for row in range(cell.row + 1, cell.row + cell.height):
                    carried.setdefault(row, []).append(cell)
        result: dict[int, DocumentSegment] = {}
        details = {**metadata, 'table_locator': self.locator, 'table_status': 'structured',
                   'header_basis': 'word_repeating_rows'}
        if self.notes:
            details['table_notes'] = ','.join(sorted(self.notes))
        for row in range(rows):
            self.tick()
            values = by_row.get(row, [])
            if not any(cell.text.strip() for cell in values):
                continue
            parts: list[str] = []
            evidence: list[DocumentTableCell] = []
            offset = 0
            for cell in values:
                references: dict[str, DocumentTableHeader] = {}
                for column in range(cell.column, cell.column + cell.width):
                    for header in headers.get(column, []):
                        self.tick()
                        if header.row < cell.row:
                            references.setdefault(header.locator, DocumentTableHeader(
                                locator=header.locator, text=header.text, association='column'))
                            if len(references) > 128:
                                raise InvalidInput('document table references exceed their limit')
                contexts: list[DocumentTableContext] = []
                for source in carried.get(row, []):
                    self.tick()
                    contexts.append(DocumentTableContext(locator=source.locator, text=source.text))
                    if len(contexts) > 128:
                        raise InvalidInput('document table references exceed their limit')
                prefix = ''
                if not cell.header and (references or contexts):
                    prefix = self.join([c.text for c in contexts] + [h.text for h in references.values()], ' / ') + ': '
                separator = ('\n' if references or contexts else '\t') if parts else ''
                fragment = separator + prefix + cell.text
                start = offset + len((separator + prefix).encode('utf-8'))
                offset += len(fragment.encode('utf-8'))
                if offset > self.limits.max_text_bytes:
                    raise InvalidInput('document exceeds its extracted text byte limit')
                parts.append(fragment)
                evidence.append(DocumentTableCell(table_locator=self.locator, locator=cell.locator,
                    row=cell.row, column=cell.column, row_span=cell.height, column_span=cell.width,
                    is_header=cell.header, text=cell.text, start=start, end=offset,
                    headers=tuple(references.values()), context=tuple(contexts), merged_locators=tuple(cell.merged)))
            self.text_bytes += offset + (2 if self.segments else 0)
            self.segments += 1
            if self.text_bytes > self.limits.max_text_bytes:
                raise InvalidInput('document exceeds its extracted text byte limit')
            if self.segments > self.limits.max_segments:
                raise InvalidInput('document exceeds its segment limit')
            result[row] = DocumentSegment(text=''.join(parts), locator=f'{self.locator}/row:{row + 1}',
                                         metadata=details, table_cells=tuple(evidence))
        return result


def word_table(root: Element, rows: list[Element], locator: str, metadata: dict[str, str],
               extract: Callable[[Element], str], limits: DocumentLimits,
               check: Callable[[], None], *, text_bytes: int = 0, segments: int = 0) -> dict[int, DocumentSegment]:
    table = _Table(locator, limits, check, text_bytes, segments)
    if len(rows) > 20_000:
        raise InvalidInput('document table rows exceed their limit')
    observed: list[tuple[Element, list[tuple[Element, str]]]] = []
    count = 0
    for row in rows:
        values: list[tuple[Element, str]] = []
        for cell in word_children(row, 'tc'):
            table.tick()
            count += 1
            if count > 20_000:
                raise InvalidInput('document table cells exceed their limit')
            values.append((cell, extract(cell)))
        observed.append((row, values))
    try:
        namespace = root.tag.rsplit('}', 1)[0] + '}'
        if (namespace[1:-1] not in WORD_NAMESPACES
                or any(row.tag != namespace + 'tr' or any(cell.tag != namespace + 'tc' for cell, _ in values)
                       for row, values in observed)):
            raise _Unresolved('unsupported_table_namespace')
        if any(_property(row, 'trPr', 'del') is not None for row in rows):
            raise _Unresolved('tracked_row_structure')
        if any(child is not root and child.tag == root.tag for child in root.iter()):
            raise _Unresolved('nested_table')
        table.assemble(observed)
        return table.render(len(rows), metadata)
    except _Unresolved as error:
        details = {**metadata, 'table_locator': locator, 'table_status': 'text_fallback', 'table_notes': str(error)}
        result: dict[int, DocumentSegment] = {}
        for index, (_, cells) in enumerate(observed):
            text = table.join([text for _, text in cells], '\t').strip()
            if text:
                table.text_bytes += len(text.encode('utf-8')) + (2 if table.segments else 0)
                table.segments += 1
                if table.text_bytes > limits.max_text_bytes:
                    raise InvalidInput('document exceeds its extracted text byte limit')
                if table.segments > limits.max_segments:
                    raise InvalidInput('document exceeds its segment limit')
                result[index] = DocumentSegment(text=text, locator=f'{locator}/row:{index + 1}', metadata=details)
        return result
