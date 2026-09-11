"""Native text extraction from ZIP-based Office, OpenDocument and EPUB files.

No macros, formulas, external relationships, embedded objects or scripts execute.
Spreadsheet values are stored values; layout, images and chart rendering are not
interpreted. Repeated ODS cells/rows carry repeat metadata rather than expanding.
"""
from __future__ import annotations

from collections import deque
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
from .table_types import DocumentTableCell
from .word_tables import word_children, word_table

OFFICE_EXTENSIONS = frozenset({'docx', 'xlsx', 'pptx', 'odt', 'ods', 'odp', 'epub'})
_ODF_REVISION_METADATA = frozenset({
    '{urn:oasis:names:tc:opendocument:xmlns:text:1.0}tracked-changes',
    '{urn:oasis:names:tc:opendocument:xmlns:office:1.0}change-info',
})
_ODF_TEXT = '{urn:oasis:names:tc:opendocument:xmlns:text:1.0}'
_ODF_OFFICE = '{urn:oasis:names:tc:opendocument:xmlns:office:1.0}'
_ODF_SIDE_CONTENT = frozenset({_ODF_OFFICE + 'annotation', _ODF_TEXT + 'note'})
_ODF_SPEAKER_NOTES = frozenset({'{urn:oasis:names:tc:opendocument:xmlns:presentation:1.0}notes'})
_DC = '{http://purl.org/dc/elements/1.1/}'
_WORD_NAMESPACES = (
    'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'http://purl.oclc.org/ooxml/wordprocessingml/main',
)
_RUN_METADATA = frozenset(
    f'{{{namespace}}}{name}'
    for namespace in _WORD_NAMESPACES
    for name in ('del', 'moveFrom', 'pPr', 'rPr', 'rt')
) | frozenset({
    '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}rPh',
    '{http://purl.oclc.org/ooxml/spreadsheetml/main}rPh',
})
_WORD_RUNS = frozenset(f'{{{namespace}}}r' for namespace in _WORD_NAMESPACES)
_WORD_TEXTBOXES = frozenset(f'{{{namespace}}}txbxContent' for namespace in _WORD_NAMESPACES)
_WORD_TEXT_NAMESPACES = frozenset(_WORD_NAMESPACES) | frozenset({
    'http://schemas.microsoft.com/office/word/2010/wordprocessingShape',
    'urn:schemas-microsoft-com:vml',
})
_WORD_REFERENCES = {
    f'{{{namespace}}}{tag}': kind
    for namespace in _WORD_NAMESPACES
    for tag, kind in (('footnoteReference', 'footnote'), ('endnoteReference', 'endnote'),
                      ('commentRangeStart', 'comment'), ('commentReference', 'comment'))
}
_RUN_CHARACTERS = {'tab': '\t', 'ptab': '\t', 'br': '\n', 'cr': '\n', 'noBreakHyphen': '\u2011'}
_PACKAGE_RELATIONSHIPS = '{http://schemas.openxmlformats.org/package/2006/relationships}'
_OFFICE_RELATIONSHIP_NAMESPACES = frozenset({
    'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'http://purl.oclc.org/ooxml/officeDocument/relationships',
})
_MAIN_PART_TYPES = frozenset(namespace + '/officeDocument' for namespace in _OFFICE_RELATIONSHIP_NAMESPACES)


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

    def add(self, text: str, locator: str, metadata: dict[str, str] | None = None, *,
            table_cells: tuple[DocumentTableCell, ...] = ()) -> None:
        self.check()
        if not table_cells:
            text = text.strip()
        if not text.strip():
            return
        self.text_bytes += len(text.encode('utf-8')) + (2 if self.segments else 0)
        if self.text_bytes > self.limits.max_text_bytes:
            raise InvalidInput('document exceeds its extracted text byte limit')
        if len(self.segments) >= self.limits.max_segments:
            raise InvalidInput('document exceeds its segment limit')
        self.segments.append(DocumentSegment(text=text, locator=locator, metadata=metadata or {}, table_cells=table_cells))


def _resolve(source: str, target: str) -> str:
    if '?' in target or '#' in target:
        raise InvalidInput('document relationship cannot have a query or fragment')
    if (any(ord(character) <= 32 or ord(character) == 127 for character in target)
            or any(ord(character) < 32 or ord(character) == 127 for character in unquote(target))):
        raise InvalidInput('document relationship contains invalid whitespace or controls')
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


def _relationship_entries(bundle: SafeArchive, path: str) -> Iterator[Element]:
    root = bundle.xml(path)
    if root.tag != _PACKAGE_RELATIONSHIPS + 'Relationships':
        raise InvalidInput('OOXML has invalid package relationships')
    identifiers: set[str] = set()
    for entry in root:
        if entry.tag != _PACKAGE_RELATIONSHIPS + 'Relationship':
            continue
        identifier = entry.get('Id', '')
        if not identifier or identifier in identifiers:
            raise InvalidInput('OOXML has invalid package relationship identifiers')
        identifiers.add(identifier)
        yield entry


def _relationships(bundle: SafeArchive, source: str) -> dict[str, tuple[str, str]]:
    path = PurePosixPath(source)
    relpath = str(path.parent / '_rels' / (path.name + '.rels'))
    if relpath not in bundle.names:
        return {}
    result: dict[str, tuple[str, str]] = {}
    for entry in _relationship_entries(bundle, relpath):
        if entry.get('TargetMode', 'Internal') != 'Internal':
            continue
        namespace, _, kind = entry.get('Type', '').rpartition('/')
        result[entry.get('Id', '')] = (entry.get('Target', ''), kind if namespace in _OFFICE_RELATIONSHIP_NAMESPACES else '')
    return result


def _related(source: str, relations: dict[str, tuple[str, str]], identifier: str, kind: str) -> str:
    relation = relations.get(identifier)
    if relation is None or relation[1] != kind:
        raise InvalidInput('document is missing a required internal relationship')
    return _resolve(source, relation[0])


def _main_part(bundle: SafeArchive) -> str:
    if '_rels/.rels' not in bundle.names:
        raise InvalidInput('OOXML is missing its package relationships')
    candidates = [relation for relation in _relationship_entries(bundle, '_rels/.rels')
                  if relation.get('Type') in _MAIN_PART_TYPES]
    if len(candidates) != 1:
        raise InvalidInput('OOXML requires exactly one main document relationship')
    selected = candidates[0]
    if selected.get('TargetMode', 'Internal') != 'Internal':
        raise InvalidInput('OOXML main document relationship must be internal')
    target = selected.get('Target', '')
    return _resolve('', target)


def _blocks(
    root: Element, names: set[str], *,
    include_tags: frozenset[str] = frozenset(), skip_tags: frozenset[str] = frozenset(),
) -> Iterator[Element]:
    stack = [iter(root)]
    while stack:
        child = next(stack[-1], None)
        if child is None:
            stack.pop()
        elif child.tag in skip_tags:
            continue
        elif _local(child.tag) in names or child.tag in include_tags:
            yield child
        else:
            stack.append(iter(child))


def _hidden_run(element: Element) -> bool:
    if element.tag not in _WORD_RUNS:
        return False
    namespace = element.tag.rsplit('}', 1)[0] + '}'
    properties = element.find(namespace + 'rPr')
    hidden = None if properties is None else properties.find(namespace + 'vanish')
    return hidden is not None and hidden.get(namespace + 'val', 'true') not in {'0', 'false', 'off'}


def _prune_run_metadata(root: Element, output: _Output) -> None:
    stack = [root]
    while stack:
        output.check()
        parent = stack.pop()
        children = []
        for child in parent:
            output.check()
            namespace = child.tag.rsplit('}', 1)[0] + '}'
            word = namespace[1:-1] in _WORD_NAMESPACES
            deleted_cell = word and child.tag == namespace + 'tc' and child.find(
                f'{namespace}tcPr/{namespace}cellDel') is not None
            row_marker = word and child.tag == namespace + 'del' and parent.tag == namespace + 'trPr'
            if (child.tag not in _RUN_METADATA or row_marker) and not _hidden_run(child) and not deleted_cell:
                children.append(child)
        parent[:] = children
        stack.extend(children)


def _run_text(root: Element, output: _Output) -> str:
    _prune_run_metadata(root, output)
    return output.join(
        (element.text or '') if _local(element.tag) == 't' else _RUN_CHARACTERS[_local(element.tag)]
        for element in root.iter() if _local(element.tag) == 't' or _local(element.tag) in _RUN_CHARACTERS
    )


def _table(root: Element, output: _Output, locator: str) -> None:
    for row_number, row in enumerate((child for child in root if _local(child.tag) == 'tr'), 1):
        output.add(_row_text(row, output), f'{locator}/row:{row_number}')


def _row_text(row: Element, output: _Output) -> str:
    return output.join((output.join((_run_text(paragraph, output) for paragraph in _elements(cell, 'p')), '\n')
                        for cell in row if _local(cell.tag) == 'tc'), '\t')


def _word_content(
    root: Element, output: _Output, prefix: str, metadata: dict[str, str],
) -> Iterator[tuple[Element, str, list[Element]]]:
    _prune_run_metadata(root, output)
    paragraphs = tables = 0
    for block in _blocks(root, {'p', 'tbl'}, include_tags=frozenset(_WORD_REFERENCES)):
        if block.tag in _WORD_REFERENCES:
            yield block, prefix.rstrip('/') or 'body', []
            continue
        if _local(block.tag) == 'tbl':
            tables += 1
            rows = list(word_children(block, 'tr'))
            detached = [_detach_textboxes(row, output) for row in rows]
            structured = word_table(block, rows, f'{prefix}table:{tables}', metadata,
                lambda cell: output.join((_run_text(p, output) for p in _elements(cell, 'p')), '\n'),
                output.limits, output.check, text_bytes=output.text_bytes, segments=len(output.segments))
            for number, (row, boxes) in enumerate(zip(rows, detached), 1):
                locator = f'{prefix}table:{tables}/row:{number}'
                segment = structured.get(number - 1)
                if segment is not None:
                    output.add(segment.text, locator, segment.metadata, table_cells=segment.table_cells)
                yield row, locator, boxes
        else:
            paragraphs += 1
            locator = f'{prefix}paragraph:{paragraphs}'
            boxes = _detach_textboxes(block, output)
            output.add(_run_text(block, output), locator, metadata)
            yield block, locator, boxes


def _detach_textboxes(root: Element, output: _Output) -> list[Element]:
    boxes: list[Element] = []
    stack: list[tuple[Element, Iterator[Element], list[Element]]] = [(root, iter(root), [])]
    while stack:
        output.check()
        parent, children, retained = stack[-1]
        child = next(children, None)
        if child is None:
            parent[:] = retained
            stack.pop()
        elif child.tag in _WORD_TEXTBOXES:
            boxes.append(child)
        else:
            retained.append(child)
            stack.append((child, iter(child), []))
    return boxes


def _word_prefix(locator: str, suffix: str) -> str:
    if len(locator) + len(suffix) > 4096:
        raise InvalidInput('document source locator exceeds its limit')
    return locator + suffix


def _word_attribute(element: Element, name: str, default: str = '') -> str:
    namespace = element.tag.rsplit('}', 1)[0] + '}'
    return element.get(namespace + name, default)


def _word_note_id(element: Element) -> str:
    value = _word_attribute(element, 'id')
    if re.fullmatch(r'[+-]?[0-9]{1,32}', value) is None:
        raise InvalidInput('DOCX contains an invalid note or comment identifier')
    return str(int(value))


def _word_note_part(
    bundle: SafeArchive, source: str, relations: dict[str, tuple[str, str]], kind: str,
) -> tuple[str, dict[str, Element]]:
    identifiers = [key for key, (_, relationship_kind) in relations.items() if relationship_kind == kind + 's']
    if len(identifiers) != 1:
        raise InvalidInput('DOCX reference requires exactly one note or comment relationship')
    path = _related(source, relations, identifiers[0], kind + 's')
    root = bundle.xml(path, supported_namespaces=_WORD_TEXT_NAMESPACES)
    if root.tag not in {f'{{{namespace}}}{kind}s' for namespace in _WORD_NAMESPACES}:
        raise InvalidInput('DOCX note or comment relationship has the wrong part type')
    entry_tag = root.tag[:-1]
    entries: dict[str, Element] = {}
    for entry in root:
        identifier = _word_note_id(entry) if entry.tag == entry_tag else None
        if identifier is None:
            continue
        if identifier in entries:
            raise InvalidInput('DOCX contains duplicate note or comment identifiers')
        entries[identifier] = entry
    return path, entries


def _docx(bundle: SafeArchive, output: _Output) -> None:
    source = _main_part(bundle)
    root = bundle.xml(source, supported_namespaces=_WORD_TEXT_NAMESPACES)
    body = _child(root, 'body')
    if _local(root.tag) != 'document' or body is None:
        raise InvalidInput('DOCX is missing its document body')
    relations = _relationships(bundle, source)
    parts: dict[str, tuple[str, dict[str, Element]]] = {}
    emitted: set[tuple[str, str]] = set()
    pending = deque([(body, '', {'member': source})])
    while pending:
        content, prefix, metadata = pending.popleft()
        for block, locator, boxes in _word_content(content, output, prefix, metadata):
            for number, box in enumerate(boxes, 1):
                details = {'member': metadata['member'], 'content_role': 'textbox', 'parent_locator': locator}
                pending.append((box, _word_prefix(locator, f'/textbox:{number}/'), details))
            for reference in block.iter():
                output.check()
                kind = _WORD_REFERENCES.get(reference.tag)
                if kind is None:
                    continue
                identifier = _word_note_id(reference)
                if (kind, identifier) in emitted:
                    continue
                if kind not in parts:
                    parts[kind] = _word_note_part(bundle, source, relations, kind)
                path, entries = parts[kind]
                note = entries.get(identifier)
                if note is None or _word_attribute(note, 'type', 'normal') != 'normal':
                    raise InvalidInput('DOCX reference has no ordinary note or comment target')
                emitted.add((kind, identifier))
                details = {'member': path, 'content_role': kind, 'parent_locator': locator,
                           'comment_id' if kind == 'comment' else 'note_id': identifier}
                if kind == 'comment':
                    for key in ('author', 'date', 'initials'):
                        if value := _word_attribute(note, key):
                            details[key] = value
                suffix = f'/{kind}:{identifier}/'
                pending.append((note, _word_prefix(locator, suffix), details))


def _xlsx(bundle: SafeArchive, output: _Output) -> None:
    source = _main_part(bundle)
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
    source = _main_part(bundle)
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
            if odf and current.tag in _ODF_SIDE_CONTENT:
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
    for paragraph in _blocks(root, {'p', 'h'}, skip_tags=_ODF_SIDE_CONTENT):
        yield _visible_text(paragraph, output, odf=True)


def _odf_side_content(
    element: Element, output: _Output, parent_locator: str,
    metadata: dict[str, str], counts: dict[str, int],
) -> None:
    details = {key: value for key, value in metadata.items() if key not in {
        'formula', 'content_role', 'parent_locator', 'note_id', 'citation', 'comment_name', 'author', 'date',
    }}
    body = element
    if element.tag == _ODF_TEXT + 'note':
        kind = element.get(_ODF_TEXT + 'note-class', 'footnote')
        if kind not in {'footnote', 'endnote'}:
            raise InvalidInput('OpenDocument contains an invalid note class')
        note_body = element.find(_ODF_TEXT + 'note-body')
        if note_body is None:
            raise InvalidInput('OpenDocument note is missing its body')
        body = note_body
        if identifier := element.get(_ODF_TEXT + 'id'):
            details['note_id'] = identifier
        citation = element.find(_ODF_TEXT + 'note-citation')
        if citation is not None:
            details['citation'] = output.join(citation.itertext())
    else:
        kind = 'comment'
        if name := element.get(_ODF_OFFICE + 'name'):
            details['comment_name'] = name
        for key, tag in (('author', 'creator'), ('date', 'date')):
            value = element.find(_DC + tag)
            if value is not None:
                details[key] = output.join(value.itertext())
    counts[kind] = counts.get(kind, 0) + 1
    locator = f'{parent_locator}/{kind}:{counts[kind]}'
    details.update(content_role=kind, parent_locator=parent_locator)
    output.add(output.join(_odf_paragraphs(body, output), '\n'), locator, details)
    _odf_annotations(body, output, locator, details)


def _odf_annotations(root: Element, output: _Output, locator: str, metadata: dict[str, str]) -> None:
    counts: dict[str, int] = {}
    for element in _blocks(root, set(), include_tags=_ODF_SIDE_CONTENT):
        _odf_side_content(element, output, locator, metadata, counts)


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
                locator = f'sheet:{name}/cell:{reference}'
                output.add(text, locator, metadata)
                _odf_annotations(cell, output, locator, metadata)
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
            locator = f'slide:{number}'
            metadata = {'slide_name': _attr(page, 'name')}
            _odf_content(page, output, locator + '/', metadata, skip_tags=_ODF_SPEAKER_NOTES)
            for note_number, notes in enumerate(_blocks(page, set(), include_tags=_ODF_SPEAKER_NOTES), 1):
                note_suffix = 'notes' if note_number == 1 else f'notes:{note_number}'
                _odf_content(notes, output, f'{locator}/{note_suffix}/', {
                    **metadata, 'content_role': 'speaker_notes', 'parent_locator': locator,
                })
        return
    _odf_content(content, output, '', {})


def _odf_content(
    root: Element, output: _Output, prefix: str, metadata: dict[str, str], *,
    skip_tags: frozenset[str] = frozenset(),
) -> None:
    paragraphs = tables = 0
    counts: dict[str, int] = {}
    for block in _blocks(root, {'p', 'h', 'table'}, include_tags=_ODF_SIDE_CONTENT, skip_tags=skip_tags):
        if block.tag in _ODF_SIDE_CONTENT:
            _odf_side_content(block, output, prefix + 'body', metadata, counts)
            continue
        if _local(block.tag) != 'table':
            paragraphs += 1
            locator = f'{prefix}paragraph:{paragraphs}'
            output.add(_visible_text(block, output, odf=True), locator, metadata)
            _odf_annotations(block, output, locator, metadata)
            continue
        tables += 1
        for number, row in enumerate(_blocks(block, {'table-row'}), 1):
            text = output.join((output.join(_odf_paragraphs(cell, output), '\n') for cell in row if _local(cell.tag) in {'table-cell', 'covered-table-cell'}), '\t')
            locator = f'{prefix}table:{tables}/row:{number}'
            output.add(text, locator, metadata)
            _odf_annotations(row, output, locator, metadata)


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
        parser = 'native-xml-word-tables-v1' if extension == 'docx' and any(
            'table_status' in segment.metadata for segment in output.segments) else 'native-xml'
        parsed = ParsedDocument(format=extension, parser=parser, segments=tuple(output.segments))
        validate_document(parsed, limits)
        return parsed
    except (ValidationError, ValueError, OverflowError, RecursionError):
        raise InvalidInput('document contains invalid or oversized content') from None
