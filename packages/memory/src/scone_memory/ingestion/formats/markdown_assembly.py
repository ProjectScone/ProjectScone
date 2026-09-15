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
  first row; any other header row, at the top or between body rows, is
  written as a body row and counted. A cell spanning rows or columns is
  written in its first slot with the slots it covers left empty, and a note
  before the table says so. A table the reader could only read as text keeps
  its lines as paragraphs, after a note naming why.
- **A table is written whole.** Its segments are gathered where the first one
  is, though the document puts others between them (a spreadsheet cell beside
  the table, a ``<caption>`` after its first row): its caption goes before
  it, the rest after it, and the note says how many.
- **A block stays in its list item.** Code, a heading, a caption or a table
  the source puts inside a list item is indented under that item, or opens
  it; two separate lists that would touch get different markers (``-`` and
  ``*``, ``1.`` and ``1)``), which is how CommonMark keeps lists apart.
- **Every line traces back.** Each span of Markdown lines names the segment
  locators and the byte ranges it was written from, in the manifest's unit:
  UTF-8 bytes of the extracted text, segments joined by a blank line -- the
  offsets a stored document's chunks use. Lines the writer made up (a
  table's delimiter row, a note) are marked ``generated`` and name the rows
  they describe.
- **The bound cuts between blocks, and says where.** Past ``max_bytes`` no
  further block or table row is written; the record gives the first
  extracted-text byte not written and how many segments were not wholly
  written.
"""
from __future__ import annotations

from bisect import bisect_right
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
_START = re.compile(r'[0-9]{1,9}')
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
    #: Segments the document puts between the table's rows, written after it.
    interleaved_segments: int = 0

    def record(self) -> dict[str, object]:
        return {'locator': self.locator, 'rendering': self.rendering, 'rows': self.rows,
                'rows_written': self.rows_written, 'columns': self.columns, 'header_basis': self.header_basis,
                'extra_header_rows': self.extra_header_rows, 'spanning_cells': self.spanning_cells,
                'line_breaks': self.line_breaks, 'notes': self.notes, 'caption': self.caption,
                'interleaved_segments': self.interleaved_segments}


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
    #: List items opened, whether by a list item's text or by a block (code,
    #: a heading, a table) that is the first thing in its item.
    list_items: int = 0

    def record(self) -> dict[str, object]:
        blocks: dict[str, int] = {}
        for span in self.spans:
            kind = 'table' if span.kind == 'table_header' else span.kind
            if span.kind not in {'table_note', 'table_delimiter', 'table_row'}:
                blocks[kind] = blocks.get(kind, 0) + 1
        if self.list_items:
            blocks['list_item'] = self.list_items
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
    """The open list: its identity and markers at depth zero, and each depth's content column, count and kind."""
    identity: tuple[str | None, str | None]
    bullet: str = '-'
    delimiter: str = '.'
    depth: int = 0
    columns: list[int] = field(default_factory=list)
    counters: list[int] = field(default_factory=list)
    #: The kind each depth's count belongs to: a numbered list after a bullet
    #: list under the same item counts from its own start.
    kinds: list[str | None] = field(default_factory=list)
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
        #: Segments wholly written; the bound counts the rest.
        self.written: set[int] = set()
        self.members: dict[str, list[int]] = {}
        self.captions: dict[str, list[int]] = {}
        self.rendered: set[str] = set()
        offset = 0
        for index, segment in enumerate(parsed.segments):
            self.offsets.append(offset)
            offset += len(segment.text.encode('utf-8')) + 2
            if (table := segment.metadata.get('table_locator')) is not None:
                self.members.setdefault(table, []).append(index)
            if segment.metadata.get('block_role') == 'caption' and 'caption_target' in segment.metadata:
                self.captions.setdefault(segment.metadata['caption_target'], []).append(index)
        #: While a block sits in a list item: what its next line starts with
        #: (the item's marker, when the block opens the item) and what every
        #: later line starts with (the item's content column).
        self.opening = self.indent = ''
        #: Whether the next line follows the last without a blank line: an
        #: item joining the list before it.
        self.join = False
        self.list_items = 0

    def shape(self, text: str, first: str) -> str:
        head, *rest = text.split('\n')
        return '\n'.join([first + head, *(self.indent + line if line else line for line in rest)])

    def emit(self, text: str, kind: SpanKind, sources: tuple[SourceSpan, ...], *,
             joined: bool = False, generated: bool = False) -> None:
        text = self.shape(text, self.opening)
        lead = ('\n' if joined or self.join else '\n\n') if self.parts else ''
        size = len((lead + text).encode('utf-8'))
        if self.size + size > self.max_bytes:
            raise _Cut(min(source.start for source in sources))
        start = self.size + len(lead.encode('utf-8'))
        first = self.newlines + lead.count('\n') + 1
        self.parts.append(lead + text)
        self.size += size
        self.newlines += lead.count('\n') + text.count('\n')
        self.spans.append(MarkdownSpan(kind, start, self.size, first, first + text.count('\n'), sources, generated))
        self.list_items += self.opening != self.indent
        self.opening, self.join = self.indent, False

    def require(self, blocks: list[tuple[str, bool]], at: int) -> None:
        """Refuse, before writing any of them, blocks that are only whole together."""
        size = self.size
        for position, (text, joined) in enumerate(blocks):
            text = self.shape(text, self.indent if position else self.opening)
            lead = ('\n' if joined or (self.join and not position) else '\n\n') if self.parts or position else ''
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
        for index, segment in enumerate(self.parsed.segments):
            table = segment.metadata.get('table_locator')
            if table is not None and table not in self.rendered:
                self.table(table)
            if index not in self.written:
                self.block(index)
                self.written.add(index)

    def place(self, index: int) -> bool | None:
        """Put segment ``index`` in the list item the source puts it in. ``True``: the item
        is open, and its lines are indented under it. ``False``: it opens the item, and its
        first line carries the marker. ``None``: it is in no list item, and any open list ends."""
        metadata = self.parsed.segments[index].metadata
        self.opening, self.indent, self.join = '', '', False
        listed = metadata.get('block_role') == 'list_item' or 'list_item_id' in metadata
        if 'content_role' in metadata or not listed or not _LEVEL.fullmatch(metadata.get('list_level', '0')):
            self.list = None
            return None
        current = self.list
        item = metadata.get('list_item_id')
        if current is not None and item is not None and item in current.items:
            # More of an item already open -- a second paragraph, a code block,
            # or text after its nested list: indented under that item's marker.
            depth = current.items.index(item)
            del current.columns[depth + 1:], current.counters[depth + 1:], current.items[depth + 1:]
            current.depth = depth
            self.opening = self.indent = ' ' * current.columns[depth]
            return True
        declared = int(metadata.get('list_level', '0'))
        kind = metadata.get('list_kind')
        identity = (metadata.get('list_id'), kind)
        self.join = current is not None and not (declared == 0 and current.identity != identity)
        if current is None or not self.join:
            previous, current = current, _List(identity)
            if previous is not None and (previous.identity[1] == 'ordered') == (kind == 'ordered'):
                current.bullet = '*' if previous.bullet == '-' else '-'
                current.delimiter = ')' if previous.delimiter == '.' else '.'
            if identity[0] is not None and identity in self.resumed:
                current.counters, current.kinds = [self.resumed[identity]], [kind]
            self.list = current
            depth = 0
        else:
            depth = min(declared, current.depth + 1)
        if depth != declared:
            self.list_levels_clamped += 1
        if kind not in {'bullet', 'ordered'}:
            self.list_kinds_unsaid += 1
        del current.columns[depth:], current.items[depth:], current.counters[depth + 1:], current.kinds[depth + 1:]
        if len(current.counters) == depth or current.kinds[depth] != kind:
            start = metadata.get('list_start', '')
            current.counters[depth:], current.kinds[depth:] = [int(start) - 1 if _START.fullmatch(start) else 0], [kind]
        current.counters[depth] += 1
        marker = f'{current.counters[depth]}{current.delimiter} ' if kind == 'ordered' else f'{current.bullet} '
        indent = current.columns[depth - 1] if depth else 0
        current.columns.append(indent + len(marker))
        current.items.append(item)
        current.depth = depth
        if depth == 0:
            current.identity = identity
            self.resumed[identity] = current.counters[0]
        self.opening, self.indent = ' ' * indent + marker, ' ' * (indent + len(marker))
        return False

    def block(self, index: int) -> None:
        metadata = self.parsed.segments[index].metadata
        role = metadata.get('block_role')
        placed = self.place(index)
        if 'content_role' in metadata:
            return self.quote(index)
        if role == 'list_item' and placed is not None:
            lines, source = self.trimmed(index)
            kind: SpanKind = 'list_continuation' if placed else 'list_item'
            return self.emit('\\\n'.join(_line(line) for line in lines), kind, (source,))
        if role == 'heading' and re.fullmatch(r'[1-9]', metadata.get('heading_level', '')):
            lines, source = self.trimmed(index)
            level = int(metadata['heading_level'])
            if level > MAX_HEADING_LEVEL:
                self.headings_clamped += 1
            text = _inline(' '.join(lines)).replace('#', '\\#')
            return self.emit('#' * min(level, MAX_HEADING_LEVEL) + ' ' + text, 'heading', (source,))
        if role == 'code':
            # Line ends as a Markdown reader counts them, and the fence closes
            # on the line after the code: a final line break is not a blank line.
            text = _BREAK.sub('\n', self.parsed.segments[index].text)
            longest = max((len(run) for run in re.findall(r'`+', text)), default=0)
            fence = '`' * max(3, longest + 1)
            close = '' if text.endswith('\n') else '\n'
            return self.emit(f'{fence}\n{text}{close}{fence}', 'code', (self.source(index),))
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

    def table(self, locator: str) -> None:
        """Write the table where its first segment is. A caption the document puts between
        its rows goes before it; any other segment between them is written after it."""
        segments = self.parsed.segments
        self.rendered.add(locator)
        members = self.members[locator]
        indexes = [index for index in members if segments[index].table_cells] or members
        held = set(indexes)
        for between in self.captions.get(locator, ()):
            if indexes[0] < between < indexes[-1]:
                self.block(between)
                self.written.add(between)
        interleaved = sum(1 for index in range(indexes[0], indexes[-1])
                          if index not in held and index not in self.written)
        captions = self.captions.get(locator)
        caption = segments[captions[0]].locator if captions else None
        self.place(indexes[0])
        first = segments[indexes[0]]
        everything = tuple(self.source(index) for index in indexes)
        if not first.table_cells:
            notes = first.metadata.get('table_notes', '')
            rendering = TableRendering(locator, 'text', len(indexes), notes=notes, caption=caption,
                                       interleaved_segments=interleaved)
            self.tables.append(rendering)
            reason = f' ({notes})' if notes else ''
            self.emit('> ' + _inline(f'{locator} was read as text, not as a grid{reason}; its lines follow as '
                                     'paragraphs.' + _interleaved(interleaved)), 'table_note', everything,
                      generated=True)
            for index in indexes:
                self.paragraphs(index, 'table_text')
                self.written.add(index)
                rendering.rows_written += 1
            return
        cells = [(index, cell) for index in indexes for cell in segments[index].table_cells]
        rendering = TableRendering(locator, 'pipe', 0, caption=caption, interleaved_segments=interleaved)
        self.grid(rendering, first, cells, everything)

    def grid(self, rendering: TableRendering, first: DocumentSegment, cells: list[tuple[int, DocumentTableCell]],
             everything: tuple[SourceSpan, ...]) -> None:
        origins = {(cell.row, cell.column): cell for _, cell in cells}
        sources: dict[int, dict[int, None]] = {}  # row -> the segments holding its cells, in order
        last_rows: dict[int, int] = {}  # segment -> the last row holding one of its cells
        for index, cell in cells:
            sources.setdefault(cell.row, {})[index] = None
            last_rows[index] = max(last_rows.get(index, cell.row), cell.row)
        finished: dict[int, list[int]] = {}  # row -> the segments wholly written once it is
        for index, row in last_rows.items():
            finished.setdefault(row, []).append(index)
        rows = sorted(sources)
        columns = max(cell.column + cell.column_span for _, cell in cells)
        data_rows = {cell.row for _, cell in cells if not cell.is_header}
        header_rows = sum(1 for row in rows if row not in data_rows)
        labels = _labels(first.metadata.get('table_columns'))
        basis: Literal['declared', 'table_columns', 'none_declared'] = (
            'declared' if rows[0] not in data_rows else 'table_columns' if labels else 'none_declared')
        if basis == 'table_columns':
            columns = max(columns, len(labels))
        rendering.rows, rendering.columns, rendering.header_basis = len(rows), columns, basis
        rendering.extra_header_rows = header_rows - (basis == 'declared')
        rendering.spanning_cells = sum(1 for _, c in cells if c.row_span > 1 or c.column_span > 1)
        rendering.line_breaks = sum(1 for _, c in cells if _BREAK.search(c.text.strip()))
        self.tables.append(rendering)
        note = (_note(rendering) if rendering.spanning_cells or rendering.extra_header_rows
                or rendering.interleaved_segments else None)

        def line(values: list[str]) -> str:
            return '| ' + ' | '.join(values) + ' |'

        def cell_text(text: str) -> str:
            return '<br>'.join(_inline(part.strip()) for part in _BREAK.split(text.strip()))

        def row_text(row: int) -> str:
            return line([cell_text(cell.text) if (cell := origins.get((row, c))) else '' for c in range(columns)])

        def row_sources(row: int) -> tuple[SourceSpan, ...]:
            return tuple(self.source(index) for index in sources[row])

        body = rows
        if basis == 'declared':
            header, body = rows[0], rows[1:]
            header_sources = row_sources(header)
            heading = row_text(header)
        else:
            header_sources = everything
            heading = line([cell_text(label) for label in labels] + [''] * (columns - len(labels)))
        delimiter = line(['---'] * columns)
        # A note without its table, or a header without its delimiter row,
        # is not a table: the three are written together or not at all.
        self.require([*([(note, False)] if note else []), (heading, False), (delimiter, True)], everything[0].start)
        if note:
            self.emit(note, 'table_note', everything, generated=True)
        self.emit(heading, 'table_header', header_sources, generated=basis != 'declared')
        self.emit(delimiter, 'table_delimiter', header_sources, joined=True, generated=True)
        if basis == 'declared':
            rendering.rows_written += 1
            self.written.update(finished.get(rows[0], ()))
        for row in body:
            self.emit(row_text(row), 'table_row', row_sources(row), joined=True)
            rendering.rows_written += 1
            self.written.update(finished.get(row, ()))


def _labels(raw: str | None) -> list[str]:
    """Column names a reader wrote for a table with no header cells (CSV, JSON keys)."""
    try:
        labels = json.loads(raw) if raw else []
    except ValueError:
        return []
    return labels if isinstance(labels, list) and all(isinstance(label, str) for label in labels) else []


def _interleaved(count: int) -> str:
    if not count:
        return ''
    return (f' {count} {"segment" if count == 1 else "segments"} the document puts between its rows '
            f'{"is" if count == 1 else "are"} written after it.')


def _note(rendering: TableRendering) -> str:
    parts: list[str] = []
    if rendering.spanning_cells:
        count = rendering.spanning_cells
        parts.append(f'{count} {"cell spans" if count == 1 else "cells span"} more than one row or column, '
                     'which a pipe table cannot show; each is written in its first slot and the slots it '
                     'covers are left empty.')
    if rendering.extra_header_rows:
        count = rendering.extra_header_rows
        further = 'further ' if rendering.header_basis == 'declared' else ''
        parts.append(f'{count} {further}header {"row is" if count == 1 else "rows are"} written as '
                     f'{"a body row" if count == 1 else "body rows"}.')
    return f'> {_inline(rendering.locator)}: ' + (' '.join(parts) + _interleaved(rendering.interleaved_segments)).lstrip()


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
        # A table is written where its first segment is, so a segment before
        # the cut can still be unwritten: the cut names the first byte of any.
        # The segment being written when the bound bit may be partly written.
        partial = bisect_right(writer.offsets, cut.at) - 1
        cut_at = min([cut.at, *(offset for index, offset in enumerate(writer.offsets)
                                if index not in writer.written and index != partial)])
    omitted = 0 if cut_at is None else len(parsed.segments) - len(writer.written)
    declared = any('block_role' in s.metadata or s.table_cells for s in parsed.segments)
    return MarkdownDocument(parsed.format, parsed.parser, ''.join(writer.parts), tuple(writer.spans),
                            tuple(writer.tables), len(parsed.segments), declared, writer.headings_clamped,
                            writer.list_levels_clamped, writer.list_kinds_unsaid, writer.roles_unreadable,
                            max_bytes, cut_at, omitted, writer.list_items)
