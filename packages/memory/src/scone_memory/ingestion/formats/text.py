"""Local, bounded parsers for text, structured text, HTML and MIME messages."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from email import policy
from email.message import Message
from email.parser import BytesParser
from html.parser import HTMLParser
from io import StringIO
from pathlib import PurePath
from time import monotonic

from pydantic import ValidationError
from typing import cast
from xml.etree.ElementTree import Element

from ...core.errors import InvalidInput
from .bounded_xml import parse_xml
from .html_tables import HtmlTables
from .table_types import DocumentTableCell
from .types import DocumentLimits, DocumentSegment, ParsedDocument, validate_document

TEXT_EXTENSIONS = frozenset({
    '.txt', '.text', '.md', '.markdown', '.mdx', '.rst', '.log', '.ini', '.cfg',
    '.conf', '.yaml', '.yml', '.toml', '.py', '.pyi', '.js', '.jsx', '.ts', '.tsx',
    '.java', '.c', '.h', '.cc', '.cpp', '.hpp', '.cs', '.go', '.rs', '.rb', '.php',
    '.swift', '.kt', '.kts', '.scala', '.sh', '.bash', '.zsh', '.sql', '.css',
    '.scss', '.sass', '.less', '.r', '.lua', '.pl', '.ex', '.exs', '.erl', '.hs',
    '.vue', '.svelte', '.graphql', '.gql', '.csv', '.tsv', '.json', '.jsonl',
    '.ndjson', '.xml', '.html', '.htm', '.eml', '.mbox', '.ipynb',
    '.ldjson', '.qmd', '.skill', '.mts', '.cts', '.mjs', '.cjs', '.ejs', '.ets',
    '.groovy', '.gradle', '.cxx', '.cu', '.cuh', '.metal', '.rake', '.luau', '.toc',
    '.zig', '.ps1', '.psm1', '.psd1', '.m', '.mm', '.ml', '.mli', '.jl', '.astro',
    '.dart', '.v', '.sv', '.svh', '.f', '.f90', '.f95', '.f03', '.f08', '.pas',
    '.pp', '.dpr', '.dpk', '.lpr', '.inc', '.dfm', '.lfm', '.lpk', '.tf', '.tfvars',
    '.hcl', '.dm', '.dme', '.dmm', '.dmf', '.sln', '.slnx', '.csproj', '.fsproj',
    '.vbproj', '.xaml', '.razor', '.cshtml', '.cls', '.trigger', '.lisp', '.cl',
    '.lsp', '.asd', '.robot', '.resource', '.svg',
})
_MAX_DEPTH = 100


class _Collector:
    def __init__(self, limits: DocumentLimits) -> None:
        self.limits = limits
        self.segments: list[DocumentSegment] = []
        self.size = 0
        self.deadline = monotonic() + limits.timeout_seconds

    def check(self, depth: int = 0) -> None:
        if depth > _MAX_DEPTH:
            raise InvalidInput('document exceeds its nested depth limit')
        if monotonic() > self.deadline:
            raise InvalidInput('document parsing timed out')

    def add(self, text: str, locator: str, metadata: dict[str, str] | None = None,
            table_cells: tuple[DocumentTableCell, ...] = ()) -> None:
        self.check()
        if not text.strip():
            return
        _valid_unicode(text)
        if len(locator) > 4096:
            raise InvalidInput('document source locator exceeds its limit')
        self.size += len(text.encode('utf-8')) + (2 if self.segments else 0)
        if self.size > self.limits.max_text_bytes:
            raise InvalidInput('document exceeds its extracted text byte limit')
        if len(self.segments) >= self.limits.max_segments:
            raise InvalidInput('document exceeds its segment limit')
        self.segments.append(DocumentSegment(text=text, locator=locator, metadata=metadata or {}, table_cells=table_cells))


def _valid_unicode(text: str) -> None:
    try:
        text.encode('utf-8')
    except UnicodeError:
        raise InvalidInput('document contains invalid Unicode') from None
    if '\x00' in text:
        raise InvalidInput('document contains binary NUL characters')


def _decode(data: bytes) -> str:
    encoding = 'utf-8-sig'
    if data.startswith((b'\xff\xfe\x00\x00', b'\x00\x00\xfe\xff')):
        encoding = 'utf-32'
    elif data.startswith((b'\xff\xfe', b'\xfe\xff')):
        encoding = 'utf-16'
    try:
        text = data.decode(encoding)
    except UnicodeError:
        raise InvalidInput('text must be UTF-8 or BOM-marked UTF-16/UTF-32') from None
    _valid_unicode(text)
    return text


def _lines(text: str, out: _Collector, prefix: str = '', metadata: dict[str, str] | None = None, *,
           first_line: int = 1) -> None:
    for line, value in enumerate(StringIO(text, newline=None), 1):
        if line >= first_line:
            out.add(value.removesuffix('\n'), f'{prefix}line:{line}', metadata)


#: Lines a note's front matter may run to before it is not front matter.
MAX_FRONTMATTER_LINES = 200
#: Keys promoted to document metadata; past this the rest are counted.
MAX_FRONTMATTER_KEYS = 16
MAX_FRONTMATTER_VALUE = 256
_FRONTMATTER_KEY = re.compile(r'^([A-Za-z_][A-Za-z0-9_-]{0,63}):(?:\s+(.*))?$')
_FRONTMATTER_ITEM = re.compile(r'^\s*-\s+(.*)$')
_FRONTMATTER_BLOCK_SCALAR = frozenset({'>', '|', '>-', '|-', '>+', '|+'})
#: Keys a note vault agrees on, kept under their own names; the rest are
#: prefixed so a note's `source:` cannot be read as the engine's, and the
#: reader's own two counters are kept from a note that names them.
_FRONTMATTER_OWN = frozenset({'title', 'tags', 'aliases'})
_FRONTMATTER_RESERVED = frozenset({'frontmatter_keys', 'frontmatter_skipped'})


def _frontmatter_scalar(value: str) -> str:
    """A plain scalar as YAML reads it far enough for a note: quotes
    removed from a quoted one, a trailing ` #` comment cut from a bare one."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
        return value[1:-1].strip()
    comment = value.find(' #')
    if comment >= 0:
        value = value[:comment]
    return value.strip()


def _frontmatter(text: str) -> tuple[dict[str, str], int]:
    """A note's front matter (Obsidian, Jekyll, Hugo, Zettlr: the YAML
    between a first line `---` and the next `---` or `...`), read without a
    YAML parser: `key: scalar`, `key: [a, b]`, and `key:` followed by
    `- item` lines, indented or not. Returns the metadata and the first
    line after the block (1 when there is none), counting lines as the
    line reader does. `title`, `tags` and `aliases` keep their names;
    other keys are `frontmatter_<key>`; lists join with commas. What is
    data and cannot be kept -- a nested mapping, a list of mappings, a
    block scalar, a value over MAX_FRONTMATTER_VALUE, a key past the bound
    or repeated, a key that names one of the reader's own counters -- is
    counted in `frontmatter_skipped`, never guessed at; a blank line, a
    comment and an empty value are not data and are not counted."""
    lines = [value.removesuffix('\n') for value in StringIO(text, newline=None)]
    if not lines or lines[0].strip() != '---':
        return {}, 1
    end = next((index for index in range(1, min(len(lines), MAX_FRONTMATTER_LINES + 1))
                if lines[index].strip() in ('---', '...')), None)
    if end is None:
        return {}, 1
    found: dict[str, str] = {}
    skipped = 0
    index = 1
    while index < end:
        line = lines[index]
        match = _FRONTMATTER_KEY.match(line)
        if match is None:
            if line.strip() and not line.lstrip().startswith('#'):
                skipped += 1  # a line that is neither a key, a comment nor blank: a continuation nobody read
            index += 1
            continue
        key = match.group(1).lower().replace('-', '_')
        raw = (match.group(2) or '').strip()
        index += 1
        value: str | None
        if raw in _FRONTMATTER_BLOCK_SCALAR:
            value = None  # a folded or literal block: its lines are not a scalar the reader takes
            while index < end and (lines[index][:1].isspace() or not lines[index].strip()):
                index += 1
        elif not raw:
            items: list[str] = []
            nested = False
            while index < end:
                item = _FRONTMATTER_ITEM.match(lines[index])
                if item is not None:
                    if _FRONTMATTER_KEY.match(item.group(1).strip()):
                        nested = True  # a list of mappings
                    items.append(_frontmatter_scalar(item.group(1)))
                    index += 1
                elif lines[index][:1].isspace() and lines[index].strip():
                    nested = True  # an indented mapping under the key
                    index += 1
                elif not lines[index].strip():
                    index += 1
                else:
                    break
            if nested:
                value = None
            elif not items:
                continue  # an empty key holds nothing: not data, not counted
            else:
                value = ','.join(item for item in items if item)
        elif raw.startswith('[') and raw.endswith(']'):
            value = ','.join(part for part in (_frontmatter_scalar(p) for p in raw[1:-1].split(',')) if part)
        else:
            value = _frontmatter_scalar(raw)
        name = key if key in _FRONTMATTER_OWN else f'frontmatter_{key}'
        if value is None or name in _FRONTMATTER_RESERVED or name in found or len(value) > MAX_FRONTMATTER_VALUE \
                or len(found) >= MAX_FRONTMATTER_KEYS:
            skipped += 1
            continue
        if value:
            found[name] = value
    if found or skipped:
        found['frontmatter_keys'] = str(len(found))
    if skipped:
        found['frontmatter_skipped'] = str(skipped)
    return found, end + 2


def _table(text: str, delimiter: str, out: _Collector) -> None:
    reader = csv.reader(StringIO(text, newline=''), delimiter=delimiter, strict=True,
                        quoting=csv.QUOTE_NONE if delimiter == '\t' else csv.QUOTE_MINIMAL)
    try:
        header_record = 0
        headers: list[str] = []
        for header_record, headers in enumerate(reader, 1):
            out.check()
            if headers:
                break
        counts = Counter(headers)
        labels = [f'{h} [column {i}]' if h and counts[h] > 1 else h or f'column {i}'
                  for i, h in enumerate(headers, 1)]
        previous_end = reader.line_num
        for row_number, row in enumerate(reader, header_record + 1):
            out.check()
            end = reader.line_num
            # line_num is the last physical source line, including quoted newlines.
            start = previous_end + 1
            previous_end = end
            if not row:
                continue
            if len(row) != len(headers):
                raise InvalidInput('delimited row has a different column count than its header')
            # Each value is a cell with its span in the row's text, so a
            # table query can quote it; the labels are the parser's rendering
            # of the header line, named once per row in table_columns.
            lines = [f'{label}: {value}' for label, value in zip(labels, row)]
            cells = []
            position = 0
            for index, (label, value) in enumerate(zip(labels, row)):
                head = position + len(f'{label}: '.encode('utf-8'))
                cells.append(DocumentTableCell(table_locator='delimited', locator=f'row:{row_number}/column:{index + 1}',
                                               row=row_number - header_record - 1, column=index, text=value,
                                               start=head, end=head + len(value.encode('utf-8'))))
                position += len(lines[index].encode('utf-8')) + 1
            out.add('\n'.join(lines), f'row:{row_number}',
                    {'line_start': str(start), 'line_end': str(end), 'table_locator': 'delimited',
                     'table_columns': json.dumps(labels), 'table_status': 'structured'},
                    table_cells=tuple(cells))
    except csv.Error:
        raise InvalidInput('document contains malformed delimited text') from None
    except ValidationError:
        raise InvalidInput('document exceeds its table row limit') from None


class _Number(str):
    """An exact JSON numeric token; never round through a binary float."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidInput('JSON contains duplicate object keys')
        _valid_unicode(key)
        result[key] = value
    return result


def _bad_constant(value: str) -> object:
    raise InvalidInput(f'JSON contains invalid constant {value}')


def _json(text: str, out: _Collector, prefix: str = '') -> None:
    try:
        value: object = json.loads(text, parse_int=_Number, parse_float=_Number,
                                   parse_constant=_bad_constant, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        raise InvalidInput('document contains malformed or excessively nested JSON') from None
    _json_value(value, '', out, prefix, 0)


def _is_table(value: list[object]) -> bool:
    """An array of flat objects is a table: every element an object whose values are all scalars."""
    if not value or len(value) > 20_000:
        return False
    for element in value:
        if not isinstance(element, dict) or not element:
            return False
        if any(isinstance(item, (dict, list)) for item in element.values()):
            return False
    return True


def _json_value(value: object, pointer: str, out: _Collector, prefix: str, depth: int,
                cell: tuple[str, int, int, tuple[str, ...]] | None = None) -> None:
    out.check(depth)
    if len(prefix) + 1 + len(pointer) > 4096:
        raise InvalidInput('document source locator exceeds its limit')
    if isinstance(value, list) and value and _is_table(cast(list[object], value)):
        # A table's rows: each scalar is a cell with its span in the leaf's
        # text, named by its object key once per row in table_columns, so a
        # JSON document queries like a spreadsheet's declared table.
        columns: list[str] = []
        for element in cast(list[dict[str, object]], value):
            for key in element:
                if key not in columns:
                    columns.append(key)
        if len(columns) > 1000:
            columns = []
        for index, element in enumerate(cast(list[dict[str, object]], value)):
            for key, child in element.items():
                escaped = key.replace('~', '~0').replace('/', '~1')
                at = (f'json:{pointer or "/"}', index, columns.index(key), tuple(columns)) if columns else None
                _json_value(child, f'{pointer}/{index}/{escaped}', out, prefix, depth + 2, at)
        return
    if isinstance(value, dict) and value:
        for key, child in cast(dict[str, object], value).items():
            escaped = key.replace('~', '~0').replace('/', '~1')
            _json_value(child, f'{pointer}/{escaped}', out, prefix, depth + 1)
        return
    if isinstance(value, list) and value:
        for index, child in enumerate(cast(list[object], value)):
            _json_value(child, f'{pointer}/{index}', out, prefix, depth + 1)
        return
    if isinstance(value, str):
        _valid_unicode(value)
    rendered = str(value) if isinstance(value, _Number) else json.dumps(value, ensure_ascii=False)
    text = f'{pointer}: {rendered}' if pointer else rendered
    metadata = {'json_pointer': pointer}
    cells: tuple[DocumentTableCell, ...] = ()
    if cell is not None:
        table, row, column, names = cell
        quoted = isinstance(value, str) and not isinstance(value, _Number)  # a JSON number is a str subclass here
        # The cell quotes the segment as written, so a string's text is its
        # JSON rendering between the quotes, escapes and all: the span then
        # holds exactly those bytes.
        inner: str = rendered[1:-1] if quoted else rendered
        head = len(f'{pointer}: '.encode('utf-8')) + (1 if quoted else 0)
        cells = (DocumentTableCell(table_locator=table, locator=f'{prefix}#{pointer}', row=row, column=column,
                                   text=inner, start=head, end=head + len(inner.encode('utf-8'))),)
        metadata.update({'table_locator': table, 'table_columns': json.dumps(list(names)),
                         'table_status': 'structured', 'header_basis': 'json_object_keys'})
    out.add(text, f'{prefix}#{pointer}', metadata, table_cells=cells)


_BLOCKS = frozenset({'p', 'div', 'section', 'article', 'header', 'footer', 'main', 'aside',
                     'nav', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'ul', 'ol',
                     'table', 'tr', 'thead', 'tbody', 'tfoot', 'blockquote', 'pre',
                     'title', 'body', 'hr', 'address', 'details', 'dialog', 'dd', 'dl',
                     'dt', 'fieldset', 'figcaption', 'figure', 'form', 'hgroup',
                     'menu', 'search'})
_VOID = frozenset({'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link',
                   'meta', 'param', 'source', 'track', 'wbr'})
_HIDDEN = frozenset({'script', 'style', 'template', 'noscript'})
_P_CLOSERS = _BLOCKS - {'li', 'tr', 'thead', 'tbody', 'tfoot', 'title', 'body'}
_SCOPE_BOUNDARIES = frozenset({'applet', 'caption', 'html', 'table', 'td', 'th',
                              'marquee', 'object', 'template'})


class _HTML(HTMLParser):
    def __init__(self, out: _Collector, prefix: str = '', metadata: dict[str, str] | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self.out = out
        self.prefix = prefix
        self.metadata = metadata
        self.stack: list[tuple[str, bool]] = []
        self.parts: list[str] = []
        self.line = 1
        self.has_content = False
        self.title_parts: list[str] = []
        self.tables = HtmlTables(out, prefix, metadata)

    def flush(self) -> None:
        text = ''.join(self.parts)
        if not self.preformatted():
            text = re.sub(r'[ \t\r\f]+', ' ', text).strip(' \t\r\n\f')
        self.parts.clear()
        self.has_content = False
        self.out.add(text, f'{self.prefix}line:{self.line}', self.metadata)

    def preformatted(self) -> bool:
        return any(tag == 'pre' for tag, _ in self.stack)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.close_implied_elements(tag)
        self.out.check(len(self.stack))
        values = dict(attrs)
        style = re.sub(r'\s+', '', values.get('style') or '').lower()
        hidden = (any(item[1] for item in self.stack) or tag in _HIDDEN or 'hidden' in values
                  or 'display:none' in style or 'visibility:hidden' in style)
        if tag in _BLOCKS:
            self.flush()
            if not hidden:
                self.tables.boundary()
        self.tables.start(tag, values, hidden)
        if tag == 'br' and not hidden:
            self.parts.append('\n')
            self.tables.data('\n')
        if tag in {'td', 'th'} and self.parts and not hidden:
            self.parts.append(' ')
        if tag not in _VOID:
            self.stack.append((tag, hidden))

    def close_in_scope(self, targets: frozenset[str], boundaries: frozenset[str]) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            tag = self.stack[index][0]
            if tag in targets:
                self.remove_elements(index)
                return
            if tag in boundaries:
                return

    def close_implied_elements(self, tag: str) -> None:
        if tag != 'col':
            self.close_in_scope(frozenset({'colgroup'}), frozenset({'table', 'template'}))
        if tag in _P_CLOSERS or tag == 'li':
            self.close_in_scope(frozenset({'p'}), _SCOPE_BOUNDARIES | {'button'})
        if tag == 'li':
            self.close_in_scope(frozenset({'li'}), _SCOPE_BOUNDARIES | {'ul', 'ol', 'menu'})
        elif tag in {'dt', 'dd'}:
            self.close_in_scope(frozenset({'dt', 'dd'}), _SCOPE_BOUNDARIES | {'dl'})
        elif tag in {'td', 'th'}:
            self.close_in_scope(frozenset({'td', 'th'}), frozenset({'tr', 'table', 'template'}))
        elif tag == 'tr':
            self.close_in_scope(frozenset({'tr'}), frozenset({'table', 'thead', 'tbody', 'tfoot', 'template'}))
        elif tag in {'thead', 'tbody', 'tfoot'}:
            self.close_in_scope(frozenset({'tr'}), frozenset({'table', 'template'}))
            self.close_in_scope(frozenset({'thead', 'tbody', 'tfoot'}), frozenset({'table', 'template'}))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCKS:
            self.flush()
            if not any(hidden for _, hidden in self.stack):
                self.tables.boundary()
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                self.remove_elements(index)
                break

    def remove_elements(self, index: int) -> None:
        for tag, _ in reversed(self.stack[index:]):
            self.tables.end(tag)
        del self.stack[index:]

    def handle_data(self, data: str) -> None:
        self.out.check()
        if any(item[1] for item in self.stack):
            return
        if any(item[0] == 'title' for item in self.stack):
            self.title_parts.append(data)
        if not self.has_content:
            self.line = self.getpos()[0]
            if not self.preformatted() and (first := re.search(r'[^ \t\r\n\f]', data)):
                self.line += data[:first.start()].count('\n')
        self.has_content = self.has_content or bool(data.strip())
        # PRE retains source whitespace; normal flow collapses only ASCII space.
        normalized = data if self.preformatted() else re.sub(r'[ \t\n\r\f]+', ' ', data)
        self.parts.append(normalized)
        self.tables.data(normalized, self.preformatted())


def _html(text: str, out: _Collector, prefix: str = '', metadata: dict[str, str] | None = None) -> str:
    parser = _HTML(out, prefix, metadata)
    parser.feed(text)
    parser.close()
    parser.flush()
    parser.remove_elements(0)
    return ' '.join(''.join(parser.title_parts).split())


def _xml(text: str, out: _Collector) -> None:
    root = parse_xml(text)
    _xml_element(root, f'/{root.tag}[1]', out, 0)


def _xml_element(element: Element, path: str, out: _Collector, depth: int) -> None:
    out.check(depth)
    for name, value in element.attrib.items():
        out.add(f'@{name}: {value}', f'xml:{path}/@{name}')
    out.add(element.text or '', f'xml:{path}/text()[1]')
    counts: Counter[str] = Counter()
    text_index = 1 if element.text else 0
    for child in element:
        counts[child.tag] += 1
        _xml_element(child, f'{path}/{child.tag}[{counts[child.tag]}]', out, depth + 1)
        if child.tail:
            text_index += 1
        out.add(child.tail or '', f'xml:{path}/text()[{text_index}]')


def _email(data: bytes, out: _Collector) -> dict[str, str]:
    try:
        message = BytesParser(policy=policy.default).parsebytes(data)
    except (ValueError, RecursionError):
        raise InvalidInput('message contains malformed or excessively nested MIME') from None
    for header in ('Subject', 'From', 'To', 'Cc', 'Date', 'Message-ID'):
        for value in message.get_all(header, []):
            out.add(f'{header}: {value}', f'header:{header.lower()}')
    skipped = _mime(message, '1', out, 0)
    return {'attachments_skipped': str(skipped)}


#: Messages a mailbox is read for before the rest are counted, not read.
#: The document's own limits come first: 20,000 segments and 2 MB of
#: text refuse the document whole, as they do an `.eml`, so a mailbox
#: of ordinary mail reaches this bound at about a thousand messages.
MAX_MAILBOX_MESSAGES = 1_000
#: A message's envelope line, the shape every mbox writer makes:
#: `From sender Www Mmm dd hh:mm:ss yyyy`. A body line that merely
#: begins with `From ` (an mboxcl writer leaves it unquoted) is not one.
_MBOX_FROM = re.compile(rb'^From \S+ +[A-Za-z]{3} +[A-Za-z]{3} +\d{1,2} +\d\d:\d\d(?::\d\d)?'
                        rb'(?: +[A-Za-z]{3,5}| +[+-]\d{4})? +\d{4}[^\r\n]*\r?\n', re.MULTILINE)


def _mailbox(data: bytes, out: _Collector) -> dict[str, str]:
    """A Unix mailbox: one message after another, each opened by an
    envelope line. Every message is read as an `.eml` is, its headers and
    text parts under `message:N/`, and each segment carries the message's
    number, date and sender so a passage recalled from a mailbox says
    which mail it came from. A file not opened by an envelope line (an
    `.eml` saved as `.mbox`, or one with text before its first message)
    is read whole as one message, with nothing split and nothing
    unquoted. A failure names the message it was in. Past the bound the
    rest are counted."""
    data = data.removeprefix(b'\xef\xbb\xbf')
    starts = [found.start() for found in _MBOX_FROM.finditer(data)]
    bare = not starts or bool(data[:starts[0]].strip())
    if bare:
        starts = [0]
    read = skipped_attachments = 0
    for number, start in enumerate(starts, 1):
        out.check()
        if number > MAX_MAILBOX_MESSAGES:
            break
        end = starts[number] if number < len(starts) else len(data)
        raw = data[start:end]
        if not bare:
            raw = raw.split(b'\n', 1)[1] if b'\n' in raw else b''
            # mboxrd quoting: a body line that began with `From ` was
            # written with one more `>` than it had. An mboxo writer
            # quotes only the bare line, so a genuine `>From ...` reply
            # line in such a file loses its `>` here; the two cannot be
            # told apart from the bytes.
            raw = re.sub(rb'^>(?=>*From )', b'', raw, flags=re.MULTILINE)
        try:
            try:
                message = BytesParser(policy=policy.default).parsebytes(raw)
            except (ValueError, RecursionError):
                raise InvalidInput('message contains malformed or excessively nested MIME') from None
            about = {'message': str(number)}
            for header in ('Date', 'From', 'Subject'):
                value = message.get(header)
                if value:
                    about[header.lower()] = str(value)[:200]
            for header in ('Subject', 'From', 'To', 'Cc', 'Date', 'Message-ID'):
                for value in message.get_all(header, []):
                    out.add(f'{header}: {value}', f'message:{number}/header:{header.lower()}', about)
            skipped_attachments += _mime(message, '1', out, 0, f'message:{number}/', about)
        except InvalidInput as refused:
            raise InvalidInput(f'mailbox message {number}: {refused}') from None
        read += 1
    return {'messages': str(read), 'messages_unread': str(max(0, len(starts) - read)),
            'attachments_skipped': str(skipped_attachments)}


def _mime(message: Message, path: str, out: _Collector, depth: int, prefix: str = '',
          about: dict[str, str] | None = None) -> int:
    out.check(depth)
    if message.get_content_disposition() == 'attachment' or message.get_filename():
        return 1
    if message.defects:
        raise InvalidInput('message contains malformed MIME content')
    if message.is_multipart():
        payload = message.get_payload()
        if not isinstance(payload, list):
            raise InvalidInput('message contains malformed multipart content')
        return sum(_mime(part, f'{path}.{i}', out, depth + 1, prefix, about)
                   for i, part in enumerate(cast(list[Message], payload), 1))
    if message.get_content_type() not in {'text/plain', 'text/html'}:
        return 0
    raw = message.get_payload(decode=True)
    if message.defects:
        raise InvalidInput('message contains malformed MIME text encoding')
    if not isinstance(raw, bytes):
        raise InvalidInput('message contains malformed text content')
    try:
        text = raw.decode(message.get_content_charset() or 'ascii')
    except (UnicodeError, LookupError):
        raise InvalidInput('message text has an invalid charset or encoding') from None
    _valid_unicode(text)
    if message.get_content_type() == 'text/html':
        _html(text, out, f'{prefix}mime:{path}/', about)
    else:
        _lines(text, out, f'{prefix}mime:{path}/', about)
    return 0


def parse_text(data: bytes, filename: str, limits: DocumentLimits) -> ParsedDocument:
    """Extract supported text formats without network access or silent truncation."""
    suffix = PurePath(filename).suffix.lower()
    if suffix not in TEXT_EXTENSIONS:
        raise InvalidInput('unsupported text document extension')
    if len(data) > limits.max_input_bytes:
        raise InvalidInput('document exceeds its input byte limit')
    if suffix == '.ipynb':
        from .notebook import parse_notebook
        return parse_notebook(data, limits)
    out = _Collector(limits)
    metadata: dict[str, str] = {}
    if suffix == '.eml':
        metadata = _email(data, out)
    elif suffix == '.mbox':
        metadata = _mailbox(data, out)
    else:
        text = _decode(data)
        if suffix in {'.csv', '.tsv'}:
            _table(text, ',' if suffix == '.csv' else '\t', out)
        elif suffix == '.json':
            _json(text, out)
        elif suffix in {'.jsonl', '.ndjson', '.ldjson'}:
            for line, value in enumerate(StringIO(text, newline='\n'), 1):
                out.check()
                if value.strip(' \t\r\n'):
                    _json(value, out, f'line:{line}')
        elif suffix in {'.html', '.htm'}:
            metadata['title'] = _html(text, out)
        elif suffix == '.xml':
            _xml(text, out)
        elif suffix in {'.md', '.markdown', '.mdx'}:
            metadata, first_line = _frontmatter(text)
            _lines(text, out, first_line=first_line)
            if not out.segments:
                # A note that is only front matter: the block is the document.
                _lines(text, out)
        else:
            _lines(text, out)
    if not out.segments:
        raise InvalidInput('document contains no extractable text')
    version = 'scone-text-tables-v1' if any(s.table_cells or 'table_status' in s.metadata for s in out.segments) else 'scone-text-v1'
    parsed = ParsedDocument(format=suffix[1:], parser=version,
                            segments=tuple(out.segments), metadata=metadata)
    validate_document(parsed, limits)
    return parsed
