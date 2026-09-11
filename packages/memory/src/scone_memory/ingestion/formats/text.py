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
from typing import cast
from xml.etree.ElementTree import Element

from ...core.errors import InvalidInput
from .types import DocumentLimits, DocumentSegment, ParsedDocument, validate_document

TEXT_EXTENSIONS = frozenset({
    '.txt', '.text', '.md', '.markdown', '.mdx', '.rst', '.log', '.ini', '.cfg',
    '.conf', '.yaml', '.yml', '.toml', '.py', '.pyi', '.js', '.jsx', '.ts', '.tsx',
    '.java', '.c', '.h', '.cc', '.cpp', '.hpp', '.cs', '.go', '.rs', '.rb', '.php',
    '.swift', '.kt', '.kts', '.scala', '.sh', '.bash', '.zsh', '.sql', '.css',
    '.scss', '.sass', '.less', '.r', '.lua', '.pl', '.ex', '.exs', '.erl', '.hs',
    '.vue', '.svelte', '.graphql', '.gql', '.csv', '.tsv', '.json', '.jsonl',
    '.ndjson', '.xml', '.html', '.htm', '.eml',
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

    def add(self, text: str, locator: str, metadata: dict[str, str] | None = None) -> None:
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
        self.segments.append(DocumentSegment(text=text, locator=locator, metadata=metadata or {}))


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


def _lines(text: str, out: _Collector, prefix: str = '') -> None:
    for line, value in enumerate(text.splitlines(), 1):
        out.add(value, f'{prefix}line:{line}')


def _table(text: str, delimiter: str, out: _Collector) -> None:
    reader = csv.reader(StringIO(text, newline=''), delimiter=delimiter, strict=True)
    try:
        headers = next(reader, [])
        counts = Counter(headers)
        labels = [f'{h} [column {i}]' if h and counts[h] > 1 else h or f'column {i}'
                  for i, h in enumerate(headers, 1)]
        previous_end = reader.line_num
        for row_number, row in enumerate(reader, 2):
            out.check()
            end = reader.line_num
            # line_num is the last physical source line, including quoted newlines.
            start = previous_end + 1
            previous_end = end
            if len(row) != len(headers):
                raise InvalidInput('delimited row has a different column count than its header')
            out.add('\n'.join(f'{label}: {value}' for label, value in zip(labels, row)),
                    f'row:{row_number}', {'line_start': str(start), 'line_end': str(end)})
    except csv.Error:
        raise InvalidInput('document contains malformed delimited text') from None


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


def _json_value(value: object, pointer: str, out: _Collector, prefix: str, depth: int) -> None:
    out.check(depth)
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
    out.add(f'{pointer}: {rendered}' if pointer else rendered,
            f'{prefix}#{pointer}', {'json_pointer': pointer})


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
    def __init__(self, out: _Collector, prefix: str = '') -> None:
        super().__init__(convert_charrefs=True)
        self.out = out
        self.prefix = prefix
        self.stack: list[tuple[str, bool]] = []
        self.parts: list[str] = []
        self.line = 1
        self.has_content = False
        self.title_parts: list[str] = []

    def flush(self) -> None:
        text = re.sub(r'[^\S\n]+', ' ', ''.join(self.parts)).strip()
        self.parts.clear()
        self.has_content = False
        self.out.add(text, f'{self.prefix}line:{self.line}')

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.close_implied_elements(tag)
        self.out.check(len(self.stack))
        values = dict(attrs)
        style = re.sub(r'\s+', '', values.get('style') or '').lower()
        hidden = (any(item[1] for item in self.stack) or tag in _HIDDEN or 'hidden' in values
                  or 'display:none' in style or 'visibility:hidden' in style)
        if tag in _BLOCKS:
            self.flush()
        if tag == 'br' and not hidden:
            self.parts.append('\n')
        if tag in {'td', 'th'} and self.parts and not hidden:
            self.parts.append(' ')
        if tag not in _VOID:
            self.stack.append((tag, hidden))

    def close_in_scope(self, targets: frozenset[str], boundaries: frozenset[str]) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            tag = self.stack[index][0]
            if tag in targets:
                del self.stack[index:]
                return
            if tag in boundaries:
                return

    def close_implied_elements(self, tag: str) -> None:
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
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        self.out.check()
        if any(item[1] for item in self.stack):
            return
        if any(item[0] == 'title' for item in self.stack):
            self.title_parts.append(data)
        if not self.has_content:
            self.line = self.getpos()[0]
        self.has_content = self.has_content or bool(data.strip())
        # HTML collapses source whitespace; only explicit BR adds a line break.
        self.parts.append(re.sub(r'\s+', ' ', data))


def _html(text: str, out: _Collector, prefix: str = '') -> str:
    parser = _HTML(out, prefix)
    parser.feed(text)
    parser.close()
    parser.flush()
    return ' '.join(''.join(parser.title_parts).split())


def _xml(text: str, out: _Collector) -> None:
    try:
        from defusedxml.ElementTree import fromstring
    except ImportError:
        raise InvalidInput('XML documents require scone-memory[documents]') from None
    try:
        root: Element = fromstring(text, forbid_dtd=True)
    except Exception:
        raise InvalidInput('document contains malformed or unsafe XML') from None
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


def _mime(message: Message, path: str, out: _Collector, depth: int) -> int:
    out.check(depth)
    if message.get_content_disposition() == 'attachment' or message.get_filename():
        return 1
    if message.defects:
        raise InvalidInput('message contains malformed MIME content')
    if message.is_multipart():
        payload = message.get_payload()
        if not isinstance(payload, list):
            raise InvalidInput('message contains malformed multipart content')
        return sum(_mime(part, f'{path}.{i}', out, depth + 1)
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
        _html(text, out, f'mime:{path}/')
    else:
        _lines(text, out, f'mime:{path}/')
    return 0


def parse_text(data: bytes, filename: str, limits: DocumentLimits) -> ParsedDocument:
    """Extract supported text formats without network access or silent truncation."""
    suffix = PurePath(filename).suffix.lower()
    if suffix not in TEXT_EXTENSIONS:
        raise InvalidInput('unsupported text document extension')
    if len(data) > limits.max_input_bytes:
        raise InvalidInput('document exceeds its input byte limit')
    out = _Collector(limits)
    metadata: dict[str, str] = {}
    if suffix == '.eml':
        metadata = _email(data, out)
    else:
        text = _decode(data)
        if suffix in {'.csv', '.tsv'}:
            _table(text, ',' if suffix == '.csv' else '\t', out)
        elif suffix == '.json':
            _json(text, out)
        elif suffix in {'.jsonl', '.ndjson', '.ldjson'}:
            for line, value in enumerate(text.splitlines(), 1):
                out.check()
                if value.strip():
                    _json(value, out, f'line:{line}')
        elif suffix in {'.html', '.htm'}:
            metadata['title'] = _html(text, out)
        elif suffix == '.xml':
            _xml(text, out)
        else:
            _lines(text, out)
    if not out.segments:
        raise InvalidInput('document contains no extractable text')
    parsed = ParsedDocument(format=suffix[1:], parser='scone-text-v1',
                            segments=tuple(out.segments), metadata=metadata)
    validate_document(parsed, limits)
    return parsed
