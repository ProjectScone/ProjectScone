"""A parsed document written back out as Markdown, every line traced to its source.

The readers already keep what a Markdown page needs: segments in reading
order, table cells with their rows, columns and spans, and -- where the
source declares them -- headings with their level, list items with their
depth, captions and preformatted text (``block_role`` and its companions in
segment metadata). This writes those as Markdown and does nothing else: no
model, no layout guess, and no structure the parsed document does not carry.
A PDF's text layer declares no headings, so its pages come out as
paragraphs and the record says ``structure_declared: false``.

Rules, each with a test:

- **Text stays text.** A paragraph that begins ``# `` or ``1. `` is escaped,
  as is inline markup (``*``, ``_`` at a word edge, ``[``, ``<``, a backtick,
  ``|``), so nothing reads as structure the source did not declare.
- **A table is a pipe table only where one can say it.** A pipe table has one
  header row and no spans. A declared header row becomes the header; a table
  that declares none gets an empty header row rather than promoting its
  first row; further header rows are written as body rows. A cell spanning
  rows or columns is written in its first slot with the slots it covers left
  empty, and a note before the table says so. A table the reader could only
  read as text keeps its lines as paragraphs, after a note naming why.
- **Every line traces back.** Each span of Markdown lines names the segment
  locators and the byte ranges it was written from, in the manifest's unit:
  UTF-8 bytes of the extracted text, segments joined by a blank line -- the
  offsets a stored document's chunks use. Lines the writer made up (a
  table's delimiter row, a note) are marked ``generated`` and name the rows
  they describe.
- **The bound cuts between blocks, and says where.** Past ``max_bytes`` no
  further block or table row is written; the record gives the extracted-text
  offset the cut fell at and how many segments were not wholly written.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import json
import re
from typing import Literal

from ...core.errors import InvalidInput
from .table_types import DocumentTableCell
from .types import DocumentLimits, DocumentSegment, ParsedDocument, validate_document

#: Markdown one assembly writes at most. The extracted text is at most 2 MB;
#: escapes, list indents and table pipes add to it, and a sparse grid adds
#: empty cells, so the bound sits well above the text it renders.
MAX_MARKDOWN_BYTES = 8_000_000
#: The deepest heading Markdown has; a Word outline level below it is clamped.
MAX_HEADING_LEVEL = 6

SpanKind = Literal['heading', 'paragraph', 'list_item', 'list_continuation', 'caption', 'code', 'quote',
                   'table_note', 'table_header', 'table_delimiter', 'table_row', 'table_text']

_INLINE = re.compile(r'[\\`*\[\]<|~]|(?<![^\W_])_|_(?![^\W_])|&(?=#?[0-9A-Za-z]+;)')
_LINE_START = re.compile(r'[#>+=-]')
_ORDINAL = re.compile(r'([0-9]{1,9})([.)])(?=\s|$)')
_LEVEL = re.compile(r'[0-9]{1,3}')
#: What a Markdown reader takes as the end of a line.
_BREAK = re.compile(r'\r\n|\r|\n')
_SIDE_IDS = ('note_id', 'comment_id')


@dataclass(frozen=True)
class SourceSpan:
    """Bytes of the extracted text one span was written from."""
    locator: str
    start: int
    end: int

    def record(self) -> dict[str, object]:
        return {'locator': self.locator, 'start': self.start, 'end': self.end}


@dataclass(frozen=True)
class MarkdownSpan:
    """Lines of the Markdown, by byte range and 1-based line numbers, and their sources."""
    kind: SpanKind
    start: int
    end: int
    first_line: int
    last_line: int
    sources: tuple[SourceSpan, ...]
    generated: bool = False

    def record(self) -> dict[str, object]:
        return {'kind': self.kind, 'markdown_start': self.start, 'markdown_end': self.end,
                'first_line': self.first_line, 'last_line': self.last_line, 'generated': self.generated,
                'sources': [source.record() for source in self.sources]}


@dataclass
class TableRendering:
    """How one table was written, and what a pipe table could not hold."""
    locator: str
    rendering: Literal['pipe', 'text']
    rows: int
    columns: int = 0
    rows_written: int = 0
    header_basis: Literal['declared', 'table_columns', 'none_declared'] | None = None
    extra_header_rows: int = 0
    spanning_cells: int = 0
    line_breaks: int = 0
    notes: str = ''
    caption: str | None = None

    def record(self) -> dict[str, object]:
        return {'locator': self.locator, 'rendering': self.rendering, 'rows': self.rows,
                'rows_written': self.rows_written, 'columns': self.columns, 'header_basis': self.header_basis,
                'extra_header_rows': self.extra_header_rows, 'spanning_cells': self.spanning_cells,
                'line_breaks': self.line_breaks, 'notes': self.notes, 'caption': self.caption}


@dataclass(frozen=True)
class MarkdownDocument:
    format: str
    parser: str
    markdown: str
    spans: tuple[MarkdownSpan, ...]
    tables: tuple[TableRendering, ...]
    segments: int
    structure_declared: bool
    headings_clamped: int
    list_levels_clamped: int
    list_kinds_unsaid: int
    roles_unreadable: int
    max_bytes: int
    cut_at: int | None
    segments_omitted: int

    def record(self) -> dict[str, object]:
        blocks: dict[str, int] = {}
        for span in self.spans:
            kind = 'table' if span.kind == 'table_header' else span.kind
            if span.kind not in {'table_note', 'table_delimiter', 'table_row'}:
                blocks[kind] = blocks.get(kind, 0) + 1
        return {'format': self.format, 'parser': self.parser, 'markdown': self.markdown,
                'markdown_bytes': len(self.markdown.encode('utf-8')), 'offset_unit': 'extracted_text_utf8_bytes',
                'segments': self.segments, 'structure_declared': self.structure_declared, 'blocks': blocks,
                'tables': [table.record() for table in self.tables],
                'headings_clamped': self.headings_clamped, 'list_levels_clamped': self.list_levels_clamped,
                'list_kinds_unsaid': self.list_kinds_unsaid, 'roles_unreadable': self.roles_unreadable,
                'bound': {'max_bytes': self.max_bytes, 'cut': self.cut_at is not None, 'cut_at': self.cut_at,
                          'segments_omitted': self.segments_omitted},
                'spans': [span.record() for span in self.spans]}


class _Cut(Exception):
    """The next block does not fit; ``at`` is the first extracted byte left out."""
    def __init__(self, at: int) -> None:
        self.at = at


def _inline(text: str) -> str:
    return _INLINE.sub(lambda match: '\\' + match[0], text)


def _line(text: str) -> str:
    """One line of running text, escaped so it cannot open a block."""
    escaped = _inline(text)
    if _LINE_START.match(escaped):
        return '\\' + escaped
    return _ORDINAL.sub(r'\1\\\2', escaped, count=1) if _ORDINAL.match(escaped) else escaped


def _lines(text: str) -> Iterator[tuple[int, int]]:
    """Character ranges of each line with its edge whitespace trimmed; blank lines as empty ranges."""
    position = 0
    for separator in [*_BREAK.finditer(text), None]:
        end = len(text) if separator is None else separator.start()
        raw = text[position:end]
        stripped = raw.strip()
        start = position + (len(raw) - len(raw.lstrip())) if stripped else position
        yield start, start + len(stripped)
        if separator is not None:
            position = separator.end()


@dataclass
class _List:
    """The open list: its identity at depth zero, and each depth's content column and count."""
    identity: tuple[str | None, str | None]
    depth: int = 0
    columns: list[int] = field(default_factory=list)
    counters: list[int] = field(default_factory=list)
    #: The item open at each depth, where the reader names items (HTML <li>).
    items: list[str | None] = field(default_factory=list)


class _Writer:
    def __init__(self, parsed: ParsedDocument, max_bytes: int) -> None:
        self.parsed = parsed
        self.max_bytes = max_bytes
        self.parts: list[str] = []
        self.size = 0
        self.newlines = 0
        self.spans: list[MarkdownSpan] = []
        self.tables: list[TableRendering] = []
        self.list: _List | None = None
        #: The last count at depth zero of each numbered list, so a list the
        #: document interrupts and resumes (one Word numbering, one <ol>)
        #: carries on counting.
        self.resumed: dict[tuple[str | None, str | None], int] = {}
        self.headings_clamped = self.list_levels_clamped = self.list_kinds_unsaid = self.roles_unreadable = 0
        self.offsets: list[int] = []
        offset = 0
        for segment in parsed.segments:
            self.offsets.append(offset)
            offset += len(segment.text.encode('utf-8')) + 2
        self.captions = {segment.metadata['caption_target']: segment.locator for segment in parsed.segments
                         if segment.metadata.get('block_role') == 'caption' and 'caption_target' in segment.metadata}

    def emit(self, text: str, kind: SpanKind, sources: tuple[SourceSpan, ...], *,
             joined: bool = False, generated: bool = False) -> None:
        lead = ('\n' if joined else '\n\n') if self.parts else ''
        size = len((lead + text).encode('utf-8'))
        if self.size + size > self.max_bytes:
            raise _Cut(min(source.start for source in sources))
        start = self.size + len(lead.encode('utf-8'))
        first = self.newlines + lead.count('\n') + 1
        self.parts.append(lead + text)
        self.size += size
        self.newlines += lead.count('\n') + text.count('\n')
        self.spans.append(MarkdownSpan(kind, start, self.size, first, first + text.count('\n'), sources, generated))

    def require(self, blocks: list[tuple[str, bool]], at: int) -> None:
        """Refuse, before writing any of them, blocks that are only whole together."""
        size = self.size
        for position, (text, joined) in enumerate(blocks):
            lead = ('\n' if joined else '\n\n') if self.parts or position else ''
            size += len((lead + text).encode('utf-8'))
        if size > self.max_bytes:
            raise _Cut(at)

    def source(self, index: int, start: int = 0, end: int | None = None) -> SourceSpan:
        """Characters ``start:end`` of segment ``index`` as extracted-text bytes."""
        text = self.parsed.segments[index].text
        end = len(text) if end is None else end
        base = self.offsets[index] + len(text[:start].encode('utf-8'))
        return SourceSpan(self.parsed.segments[index].locator, base, base + len(text[start:end].encode('utf-8')))

    def trimmed(self, index: int) -> tuple[list[str], SourceSpan]:
        """The segment's non-blank lines, each trimmed, and the bytes from the first to the last."""
        text = self.parsed.segments[index].text
        ranges = [(start, end) for start, end in _lines(text) if end > start]
        return [text[start:end] for start, end in ranges], self.source(index, ranges[0][0], ranges[-1][1])

    def run(self) -> None:
        segments = self.parsed.segments
        index = 0
        while index < len(segments):
            segment = segments[index]
            table = segment.metadata.get('table_locator')
            if table is not None:
                end = index
                while end < len(segments) and segments[end].metadata.get('table_locator') == table:
                    end += 1
                self.list = None
                self.table(table, range(index, end))
                index = end
                continue
            self.block(index)
            index += 1

    def block(self, index: int) -> None:
        metadata = self.parsed.segments[index].metadata
        role = metadata.get('block_role')
        if 'content_role' in metadata:
            self.list = None
            return self.quote(index)
        if role == 'list_item' and _LEVEL.fullmatch(metadata.get('list_level', '0')):
            return self.list_item(index)
        self.list = None
        if role == 'heading' and re.fullmatch(r'[1-9]', metadata.get('heading_level', '')):
            lines, source = self.trimmed(index)
            level = int(metadata['heading_level'])
            if level > MAX_HEADING_LEVEL:
                self.headings_clamped += 1
            text = _inline(' '.join(lines)).replace('#', '\\#')
            return self.emit('#' * min(level, MAX_HEADING_LEVEL) + ' ' + text, 'heading', (source,))
        if role == 'code':
            text = self.parsed.segments[index].text
            longest = max((len(run) for run in re.findall(r'`+', text)), default=0)
            fence = '`' * max(3, longest + 1)
            return self.emit(f'{fence}\n{text}\n{fence}', 'code', (self.source(index),))
        if role == 'caption':
            lines, source = self.trimmed(index)
            return self.emit('*' + '\\\n'.join(_line(line) for line in lines) + '*', 'caption', (source,))
        if role is not None:
            self.roles_unreadable += 1
        self.paragraphs(index, 'paragraph')

    def paragraphs(self, index: int, kind: SpanKind) -> None:
        """Each run of non-blank lines is its own paragraph; a line break inside one is kept."""
        text = self.parsed.segments[index].text
        group: list[tuple[int, int]] = []
        for start, end in [*_lines(text), (0, 0)]:
            if end > start:
                group.append((start, end))
                continue
            if group:
                rendered = '\\\n'.join(_line(text[a:b]) for a, b in group)
                self.emit(rendered, kind, (self.source(index, group[0][0], group[-1][1]),))
                group = []

    def quote(self, index: int) -> None:
        metadata = self.parsed.segments[index].metadata
        lines, source = self.trimmed(index)
        label = metadata['content_role'].replace('_', ' ')
        identifier = next((metadata[key] for key in _SIDE_IDS if key in metadata), None)
        label = _inline(f'{label} {identifier}' if identifier else label)
        body = '\\\n> '.join(_line(line) for line in lines)
        self.emit(f'> {label}: {body}', 'quote', (source,))

    def list_item(self, index: int) -> None:
        metadata = self.parsed.segments[index].metadata
        declared = int(metadata.get('list_level', '0'))
        kind = metadata.get('list_kind')
        identity = (metadata.get('list_id'), kind)
        current = self.list
        item = metadata.get('list_item_id')
        if current is not None and item is not None and item in current.items:
            # More of an item already open -- a second paragraph, or text after
            # its nested list: a paragraph indented under that item's marker.
            depth = current.items.index(item)
            del current.columns[depth + 1:], current.counters[depth + 1:], current.items[depth + 1:]
            lines, source = self.trimmed(index)
            padding = ' ' * current.columns[depth]
            self.emit(padding + ('\\\n' + padding).join(_line(line) for line in lines), 'list_continuation', (source,))
            current.depth = depth
            return
        joined = current is not None and not (declared == 0 and current.identity != identity)
        if current is None or not joined:
            current = self.list = _List(identity, counters=[self.resumed.get(identity, 0)] if identity[0] else [])
            depth = 0
        else:
            depth = min(declared, current.depth + 1)
        if depth != declared:
            self.list_levels_clamped += 1
        if kind not in {'bullet', 'ordered'}:
            self.list_kinds_unsaid += 1
        del current.columns[depth:], current.items[depth:]
        del current.counters[depth + 1:]
        current.counters.extend([0] * (depth + 1 - len(current.counters)))
        current.counters[depth] += 1
        marker = f'{current.counters[depth]}. ' if kind == 'ordered' else '- '
        indent = current.columns[depth - 1] if depth else 0
        column = indent + len(marker)
        lines, source = self.trimmed(index)
        body = ('\\\n' + ' ' * column).join(_line(line) for line in lines)
        self.emit(' ' * indent + marker + body, 'list_item', (source,), joined=joined)
        current.columns.append(column)
        current.items.append(item)
        current.depth = depth
        if depth == 0:
            current.identity = identity
            self.resumed[identity] = current.counters[0]

    def table(self, locator: str, indexes: range) -> None:
        segments = self.parsed.segments
        first = segments[indexes[0]]
        cells = [(index, cell) for index in indexes for cell in segments[index].table_cells]
        everything = tuple(self.source(index) for index in indexes)
        if not cells:
            notes = first.metadata.get('table_notes', '')
            rendering = TableRendering(locator, 'text', len(indexes), notes=notes, caption=self.captions.get(locator))
            self.tables.append(rendering)
            reason = f' ({notes})' if notes else ''
            self.emit('> ' + _inline(f'{locator} was read as text, not as a grid{reason}; its lines follow as '
                                     'paragraphs.'), 'table_note', everything, generated=True)
            for index in indexes:
                self.paragraphs(index, 'table_text')
                rendering.rows_written += 1
            return
        self.grid(locator, first, cells, everything)

    def grid(self, locator: str, first: DocumentSegment, cells: list[tuple[int, DocumentTableCell]],
             everything: tuple[SourceSpan, ...]) -> None:
        origins = {(cell.row, cell.column): cell for _, cell in cells}
        sources: dict[int, dict[int, None]] = {}  # row -> the segments holding its cells, in order
        for index, cell in cells:
            sources.setdefault(cell.row, {})[index] = None
        rows = sorted(sources)
        columns = max(cell.column + cell.column_span for _, cell in cells)
        data_rows = {cell.row for _, cell in cells if not cell.is_header}
        headers = 0
        while headers < len(rows) and rows[headers] not in data_rows:
            headers += 1
        labels = _labels(first.metadata.get('table_columns'))
        basis: Literal['declared', 'table_columns', 'none_declared'] = (
            'declared' if headers else 'table_columns' if labels else 'none_declared')
        if basis == 'table_columns':
            columns = max(columns, len(labels))
        rendering = TableRendering(locator, 'pipe', len(rows), columns, header_basis=basis,
                                   extra_header_rows=max(0, headers - 1), caption=self.captions.get(locator),
                                   spanning_cells=sum(1 for _, c in cells if c.row_span > 1 or c.column_span > 1),
                                   line_breaks=sum(1 for _, c in cells if _BREAK.search(c.text.strip())))
        self.tables.append(rendering)
        note = _note(locator, rendering) if rendering.spanning_cells or rendering.extra_header_rows else None

        def line(values: list[str]) -> str:
            return '| ' + ' | '.join(values) + ' |'

        def cell_text(row: int, column: int) -> str:
            cell = origins.get((row, column))
            if cell is None:
                return ''
            return '<br>'.join(_inline(part.strip()) for part in _BREAK.split(cell.text.strip()))

        def row_sources(row: int) -> tuple[SourceSpan, ...]:
            return tuple(self.source(index) for index in sources[row])

        body = rows
        if basis == 'declared':
            header, body = rows[0], rows[1:]
            header_sources = row_sources(header)
            heading = line([cell_text(header, c) for c in range(columns)])
        else:
            header_sources = everything
            heading = line([_inline(label) for label in labels] + [''] * (columns - len(labels)))
        delimiter = line(['---'] * columns)
        # A note without its table, or a header without its delimiter row,
        # is not a table: the three are written together or not at all.
        self.require([*([(note, False)] if note else []), (heading, False), (delimiter, True)], everything[0].start)
        if note:
            self.emit(note, 'table_note', everything, generated=True)
        self.emit(heading, 'table_header', header_sources, generated=basis != 'declared')
        rendering.rows_written += basis == 'declared'
        self.emit(delimiter, 'table_delimiter', header_sources, joined=True, generated=True)
        for row in body:
            self.emit(line([cell_text(row, c) for c in range(columns)]), 'table_row', row_sources(row), joined=True)
            rendering.rows_written += 1


def _labels(raw: str | None) -> list[str]:
    """Column names a reader wrote for a table with no header cells (CSV, JSON keys)."""
    try:
        labels = json.loads(raw) if raw else []
    except ValueError:
        return []
    return labels if isinstance(labels, list) and all(isinstance(label, str) for label in labels) else []


def _note(locator: str, rendering: TableRendering) -> str:
    parts: list[str] = []
    if rendering.spanning_cells:
        count = rendering.spanning_cells
        parts.append(f'{count} {"cell spans" if count == 1 else "cells span"} more than one row or column, '
                     'which a pipe table cannot show; each is written in its first slot and the slots it '
                     'covers are left empty.')
    if rendering.extra_header_rows:
        count = rendering.extra_header_rows
        parts.append(f'{count} further header {"row is" if count == 1 else "rows are"} written as a body row'
                     f'{"" if count == 1 else "s"}.')
    return f'> {_inline(locator)}: ' + ' '.join(parts)


def assemble_markdown(parsed: ParsedDocument, *, max_bytes: int = MAX_MARKDOWN_BYTES) -> MarkdownDocument:
    """Write ``parsed`` as Markdown with a span map back to its extracted text. Deterministic:
    the same parsed document always gives the same bytes."""
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_MARKDOWN_BYTES:
        raise InvalidInput(f'markdown byte bound must be an integer from 1 to {MAX_MARKDOWN_BYTES}')
    validate_document(parsed, DocumentLimits())
    writer = _Writer(parsed, max_bytes)
    cut_at: int | None = None
    try:
        writer.run()
    except _Cut as cut:
        cut_at = cut.at
    omitted = 0 if cut_at is None else sum(
        1 for offset, segment in zip(writer.offsets, parsed.segments)
        if offset + len(segment.text.encode('utf-8')) > cut_at)
    declared = any('block_role' in s.metadata or s.table_cells for s in parsed.segments)
    return MarkdownDocument(parsed.format, parsed.parser, ''.join(writer.parts), tuple(writer.spans),
                            tuple(writer.tables), len(parsed.segments), declared, writer.headings_clamped,
                            writer.list_levels_clamped, writer.list_kinds_unsaid, writer.roles_unreadable,
                            max_bytes, cut_at, omitted)
