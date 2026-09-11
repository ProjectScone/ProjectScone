"""Native text extraction from ZIP-based Office, OpenDocument and EPUB files.

No macros, formulas, external relationships, embedded objects or scripts execute.
Spreadsheet values are stored values; layout, images and chart rendering are not
interpreted. Repeated ODS cells/rows carry repeat metadata rather than expanding.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import PurePosixPath
import posixpath
import re
from time import monotonic
from urllib.parse import unquote, urlsplit
from xml.etree.ElementTree import Element

from pydantic import ValidationError

from ...core.errors import InvalidInput
from .archive import SafeArchive
from .types import DocumentLimits, DocumentSegment, ParsedDocument, validate_document

OFFICE_EXTENSIONS = frozenset({'docx', 'xlsx', 'pptx', 'odt', 'ods', 'odp', 'epub'})
_ODF_REVISION_METADATA = frozenset({
    '{urn:oasis:names:tc:opendocument:xmlns:text:1.0}tracked-changes',
    '{urn:oasis:names:tc:opendocument:xmlns:office:1.0}change-info',
})


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _attr(element: Element, name: str, default: str = '') -> str:
    return next((value for key, value in element.attrib.items() if _local(key) == name), default)


def _elements(element: Element, name: str) -> Iterator[Element]:
    return (child for child in element.iter() if _local(child.tag) == name)


def _child(element: Element, name: str) -> Element | None:
    return next((child for child in element if _local(child.tag) == name), None)


class _Output:
    def __init__(self, limits: DocumentLimits) -> None:
        self.limits = limits
        self.segments: list[DocumentSegment] = []
        self.text_bytes = 0
        self.deadline = monotonic() + limits.timeout_seconds

    def check(self) -> None:
        if monotonic() > self.deadline:
            raise InvalidInput('document parsing exceeded its timeout')

    def join(self, parts: Iterable[str], separator: str = '') -> str:
        pieces: list[str] = []
        size = 0
        for part in parts:
            self.check()
            size += len(part.encode('utf-8')) + (len(separator.encode('utf-8')) if pieces else 0)
            if size > self.limits.max_text_bytes:
                raise InvalidInput('document exceeds its extracted text byte limit')
            pieces.append(part)
        return separator.join(pieces)

    def add(self, text: str, locator: str, metadata: dict[str, str] | None = None) -> None:
        self.check()
        text = text.strip()
        if not text:
            return
        self.text_bytes += len(text.encode('utf-8')) + (2 if self.segments else 0)
        if self.text_bytes > self.limits.max_text_bytes:
            raise InvalidInput('document exceeds its extracted text byte limit')
        if len(self.segments) >= self.limits.max_segments:
            raise InvalidInput('document exceeds its segment limit')
        self.segments.append(DocumentSegment(text=text, locator=locator, metadata=metadata or {}))


def _resolve(source: str, target: str) -> str:
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or parsed.query or '\\' in target or not parsed.path:
        raise InvalidInput('document relationship is not a local archive member')
    path = unquote(parsed.path)
    if '\\' in path or '\x00' in path:
        raise InvalidInput('document relationship has an unsafe path')
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path) if not path.startswith('/') else path[1:])
    if resolved == '..' or resolved.startswith('../') or resolved in ('', '.'):
        raise InvalidInput('document relationship escapes its archive')
    return resolved


def _relationships(bundle: SafeArchive, source: str) -> dict[str, tuple[str, str]]:
    path = PurePosixPath(source)
    relpath = str(path.parent / '_rels' / (path.name + '.rels'))
    if relpath not in bundle.names:
        return {}
    result: dict[str, tuple[str, str]] = {}
    for entry in _elements(bundle.xml(relpath), 'Relationship'):
        identifier = entry.get('Id', '')
        if not identifier or identifier in result:
            raise InvalidInput('document contains invalid relationship identifiers')
        if entry.get('TargetMode') == 'External':
            continue
        result[identifier] = (entry.get('Target', ''), entry.get('Type', '').rsplit('/', 1)[-1])
    return result


def _related(source: str, relations: dict[str, tuple[str, str]], identifier: str, kind: str) -> str:
    relation = relations.get(identifier)
    if relation is None or relation[1] != kind:
        raise InvalidInput('document is missing a required internal relationship')
    return _resolve(source, relation[0])


def _blocks(root: Element, names: set[str]) -> Iterator[Element]:
    stack = [iter(root)]
    while stack:
        child = next(stack[-1], None)
        if child is None:
            stack.pop()
        elif _local(child.tag) in names:
            yield child
        else:
            stack.append(iter(child))


def _run_text(root: Element, output: _Output) -> str:
    return output.join(
        (element.text or '') if _local(element.tag) == 't' else '\t' if _local(element.tag) == 'tab' else '\n'
        for element in root.iter() if _local(element.tag) in {'t', 'tab', 'br', 'cr'}
    )


def _table(root: Element, output: _Output, locator: str) -> None:
    for row_number, row in enumerate((child for child in root if _local(child.tag) == 'tr'), 1):
        text = output.join((output.join((_run_text(paragraph, output) for paragraph in _elements(cell, 'p')), '\n') for cell in row if _local(cell.tag) == 'tc'), '\t')
        output.add(text, f'{locator}/row:{row_number}')


def _docx(bundle: SafeArchive, output: _Output) -> None:
    root = bundle.xml('word/document.xml')
    body = _child(root, 'body')
    if _local(root.tag) != 'document' or body is None:
        raise InvalidInput('DOCX is missing its document body')
    paragraphs = tables = 0
    for block in _blocks(body, {'p', 'tbl'}):
        if _local(block.tag) == 'tbl':
            tables += 1
            _table(block, output, f'table:{tables}')
        else:
            paragraphs += 1
            output.add(_run_text(block, output), f'paragraph:{paragraphs}')


def _xlsx(bundle: SafeArchive, output: _Output) -> None:
    source = 'xl/workbook.xml'
    root = bundle.xml(source)
    if _local(root.tag) != 'workbook':
        raise InvalidInput('XLSX is missing its workbook')
    relations = _relationships(bundle, source)
    shared: list[str] = []
    shared_size = 0
    for identifier, (_, kind) in relations.items():
        if kind != 'sharedStrings':
            continue
        for item in _elements(bundle.xml(_related(source, relations, identifier, kind)), 'si'):
            text = _run_text(item, output)
            shared_size += len(text.encode('utf-8'))
            if shared_size > output.limits.max_archive_bytes:
                raise InvalidInput('spreadsheet shared strings exceed their byte limit')
            shared.append(text)
    for sheet_number, sheet in enumerate(_elements(root, 'sheet'), 1):
        output.check()
        name = sheet.get('name', f'Sheet{sheet_number}')
        path = _related(source, relations, _attr(sheet, 'id'), 'worksheet')
        worksheet = bundle.xml(path)
        if _local(worksheet.tag) != 'worksheet':
            raise InvalidInput('XLSX relationship is not a worksheet')
        for row_number, row in enumerate(_elements(worksheet, 'row'), 1):
            actual_row = _positive(row.get('r', str(row_number)), 1_048_576)
            for column, cell in enumerate((child for child in row if _local(child.tag) == 'c'), 1):
                output.check()
                reference = cell.get('r', f'{_column(column)}{actual_row}')
                value = _child(cell, 'v')
                text = '' if value is None else value.text or ''
                cell_type = cell.get('t', '')
                if cell_type == 's':
                    try:
                        index = int(text)
                    except ValueError:
                        raise InvalidInput('XLSX has an invalid shared string index') from None
                    if index < 0 or index >= len(shared):
                        raise InvalidInput('XLSX shared string index is out of range')
                    text = shared[index]
                elif cell_type == 'inlineStr':
                    text = _run_text(cell, output)
                metadata = {'sheet': name, 'cell': reference}
                if _child(cell, 'f') is not None:
                    metadata['formula'] = 'cached-value'
                output.add(text, f'sheet:{name}/cell:{reference}', metadata)


def _drawing(root: Element, output: _Output, prefix: str) -> None:
    paragraphs = tables = 0
    for block in _blocks(root, {'p', 'tbl'}):
        if _local(block.tag) == 'tbl':
            tables += 1
            _table(block, output, f'{prefix}/table:{tables}')
        else:
            paragraphs += 1
            output.add(_run_text(block, output), f'{prefix}/paragraph:{paragraphs}')


def _pptx(bundle: SafeArchive, output: _Output) -> None:
    source = 'ppt/presentation.xml'
    root = bundle.xml(source)
    if _local(root.tag) != 'presentation':
        raise InvalidInput('PPTX is missing its presentation')
    relations = _relationships(bundle, source)
    for number, slide in enumerate(_elements(root, 'sldId'), 1):
        output.check()
        # A slide has both an unqualified numeric id and a namespaced relationship id.
        identifier = next((value for key, value in slide.attrib.items() if key.endswith('}id')), '')
        path = _related(source, relations, identifier, 'slide')
        slide_root = bundle.xml(path)
        if _local(slide_root.tag) != 'sld':
            raise InvalidInput('PPTX relationship is not a slide')
        _drawing(slide_root, output, f'slide:{number}')
        slide_relations = _relationships(bundle, path)
        for relation_id, (_, kind) in slide_relations.items():
            if kind == 'notesSlide':
                notes_path = _related(path, slide_relations, relation_id, kind)
                _drawing(bundle.xml(notes_path), output, f'slide:{number}/notes')


def _positive(value: str, maximum: int = 2_147_483_647) -> int:
    try:
        number = int(value)
    except ValueError:
        raise InvalidInput('document contains an invalid positive integer') from None
    if number < 1 or number > maximum:
        raise InvalidInput('document count exceeds its supported range')
    return number


def _column(number: int) -> str:
    characters: list[str] = []
    while number:
        number, remainder = divmod(number - 1, 26)
        characters.append(chr(65 + remainder))
    return ''.join(reversed(characters))


def _visible_text(root: Element, output: _Output, *, odf: bool = False) -> str:
    def parts() -> Iterator[str]:
        stack: list[Element | str] = [root]
        while stack:
            output.check()
            current = stack.pop()
            if isinstance(current, str):
                yield current
                continue
            name = _local(current.tag)
            if current is not root and current.tail:
                stack.append(current.tail)
            if name in {'script', 'style', 'head'}:
                continue
            if odf and name == 's':
                count = _positive(_attr(current, 'c', '1'), output.limits.max_text_bytes)
                yield ' ' * count
            elif name in {'tab'}:
                yield '\t'
            elif name in {'line-break', 'br'}:
                yield '\n'
            else:
                boundary = not odf and name in {'p', 'div', 'section', 'article', 'main', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'pre', 'blockquote', 'tr'}
                if boundary:
                    yield '\n'
                    stack.append('\n')
                stack.extend(reversed(current))
                if current.text:
                    yield current.text
    text = output.join(parts())
    return text if odf else re.sub(r'\n{2,}', '\n', text)


def _odf_paragraphs(root: Element, output: _Output) -> Iterator[str]:
    for paragraph in _blocks(root, {'p', 'h'}):
        yield _visible_text(paragraph, output, odf=True)


def _ods(root: Element, output: _Output) -> None:
    for sheet_number, sheet in enumerate(_elements(root, 'table'), 1):
        name = _attr(sheet, 'name', f'Sheet{sheet_number}')
        row_number = 1
        for row in _blocks(sheet, {'table-row'}):
            output.check()
            row_repeat = _positive(_attr(row, 'number-rows-repeated', '1'))
            column_number = 1
            for cell in row:
                if _local(cell.tag) not in {'table-cell', 'covered-table-cell'}:
                    continue
                column_repeat = _positive(_attr(cell, 'number-columns-repeated', '1'))
                text = output.join(_odf_paragraphs(cell, output), '\n')
                if not text:
                    text = next((_attr(cell, attribute) for attribute in ('string-value', 'value', 'date-value', 'time-value', 'boolean-value') if _attr(cell, attribute)), '')
                reference = f'{_column(column_number)}{row_number}'
                metadata = {'sheet': name, 'cell': reference, 'row_repeat': str(row_repeat), 'column_repeat': str(column_repeat)}
                if _attr(cell, 'formula'):
                    metadata['formula'] = 'cached-value'
                output.add(text, f'sheet:{name}/cell:{reference}', metadata)
                column_number += column_repeat
                if column_number > 2_147_483_648:
                    raise InvalidInput('ODS column range exceeds its supported range')
            row_number += row_repeat
            if row_number > 2_147_483_648:
                raise InvalidInput('ODS row range exceeds its supported range')


def _odf(bundle: SafeArchive, output: _Output, extension: str) -> None:
    if 'META-INF/manifest.xml' in bundle.names:
        if next(_elements(bundle.xml('META-INF/manifest.xml'), 'encryption-data'), None) is not None:
            raise InvalidInput('encrypted OpenDocument files are unsupported')
    root = bundle.xml('content.xml')
    body = _child(root, 'body')
    if body is None:
        raise InvalidInput('OpenDocument is missing its body')
    expected = {'odt': 'text', 'ods': 'spreadsheet', 'odp': 'presentation'}[extension]
    content = _child(body, expected)
    if content is None:
        raise InvalidInput('OpenDocument content does not match its extension')
    for element in content.iter():
        output.check()
        if element.tag in _ODF_REVISION_METADATA:
            # A tail belongs to the surrounding live content, not this subtree.
            tail = element.tail
            element.clear()
            element.tail = tail
    if extension == 'ods':
        _ods(content, output)
        return
    if extension == 'odp':
        for number, page in enumerate(_elements(content, 'page'), 1):
            _odf_content(page, output, f'slide:{number}/', {'slide_name': _attr(page, 'name')})
        return
    _odf_content(content, output, '', {})


def _odf_content(root: Element, output: _Output, prefix: str, metadata: dict[str, str]) -> None:
    paragraphs = tables = 0
    for block in _blocks(root, {'p', 'h', 'table'}):
        if _local(block.tag) != 'table':
            paragraphs += 1
            output.add(_visible_text(block, output, odf=True), f'{prefix}paragraph:{paragraphs}', metadata)
            continue
        tables += 1
        for number, row in enumerate(_blocks(block, {'table-row'}), 1):
            text = output.join((output.join(_odf_paragraphs(cell, output), '\n') for cell in row if _local(cell.tag) in {'table-cell', 'covered-table-cell'}), '\t')
            output.add(text, f'{prefix}table:{tables}/row:{number}', metadata)


def _epub(bundle: SafeArchive, output: _Output) -> None:
    if 'META-INF/encryption.xml' in bundle.names:
        raise InvalidInput('encrypted EPUB resources are unsupported')
    container = bundle.xml('META-INF/container.xml')
    rootfile = next(_elements(container, 'rootfile'), None)
    if rootfile is None:
        raise InvalidInput('EPUB is missing its package rootfile')
    package_path = _resolve('', rootfile.get('full-path', ''))
    package = bundle.xml(package_path)
    manifest: dict[str, tuple[str, str]] = {}
    for item in _elements(package, 'item'):
        identifier = item.get('id', '')
        if not identifier or identifier in manifest:
            raise InvalidInput('EPUB has invalid manifest identifiers')
        manifest[identifier] = (item.get('href', ''), item.get('media-type', ''))
    for chapter, itemref in enumerate(_elements(package, 'itemref'), 1):
        output.check()
        resource = manifest.get(itemref.get('idref', ''))
        if resource is None:
            raise InvalidInput('EPUB spine refers to a missing manifest item')
        if resource[1] not in {'application/xhtml+xml', 'text/html'}:
            continue
        path = _resolve(package_path, resource[0])
        document = bundle.xml(path, allow_doctype=True)
        body = next(_elements(document, 'body'), None)
        if body is None:
            raise InvalidInput('EPUB chapter has no XHTML body')
        output.add(_visible_text(body, output), f'chapter:{chapter}/{path}', {'member': path})


def parse_office(data: bytes, filename: str, limits: DocumentLimits) -> ParsedDocument:
    """Extract ordered native text without filesystem extraction or execution."""
    extension = PurePosixPath(filename).suffix.lower().lstrip('.')
    if extension not in OFFICE_EXTENSIONS:
        raise InvalidInput('unsupported native Office or EPUB format')
    output = _Output(limits)
    try:
        with SafeArchive(data, limits) as bundle:
            if extension == 'docx':
                _docx(bundle, output)
            elif extension == 'xlsx':
                _xlsx(bundle, output)
            elif extension == 'pptx':
                _pptx(bundle, output)
            elif extension == 'epub':
                _epub(bundle, output)
            else:
                _odf(bundle, output, extension)
        output.check()
        if not output.segments:
            raise InvalidInput('document contains no extractable text')
        parsed = ParsedDocument(format=extension, parser='native-xml', segments=tuple(output.segments))
        validate_document(parsed, limits)
        return parsed
    except (ValidationError, ValueError, OverflowError, RecursionError):
        raise InvalidInput('document contains invalid or oversized content') from None
