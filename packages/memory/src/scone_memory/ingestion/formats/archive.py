"""Read Office/EPUB containers without extraction, expansion bombs or XML entities."""
from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
import re
from xml.etree.ElementTree import Element, TreeBuilder
from xml.parsers import expat
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

from ...core.errors import InvalidInput
from .types import DocumentLimits

XML_MAX_BYTES = 16 * 1024 * 1024
XML_MAX_NODES = 200_000
XML_MAX_DEPTH = 128
XML_MAX_EXPANDED_BYTES = 32 * 1024 * 1024


def _epub_doctype(name: str, system_id: str | None, public_id: str | None, internal: int) -> None:
    if internal:
        raise InvalidInput('document contains unsafe XML: internal DTD subsets are unsupported')


def _preflight_xml(data: bytes | str, allow_doctype: bool) -> str:
    """Check raw names/attributes before a namespace-aware parser expands them."""
    parser = expat.ParserCreate()
    depth = nodes = 0
    encoding = ('utf-8' if isinstance(data, str) else
                'utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else
                'utf-16-le' if data.startswith(b'<\x00') else
                'utf-16-be' if data.startswith(b'\x00<') else 'utf-8-sig')

    def declaration(version: str, declared_encoding: str | None, standalone: int) -> None:
        nonlocal encoding
        if declared_encoding and declared_encoding.lower().replace('_', '-') != 'utf-16':
            encoding = declared_encoding

    def start(name: str, attributes: dict[str, str]) -> None:
        nonlocal depth, nodes
        depth += 1
        nodes += 1
        if nodes > XML_MAX_NODES or depth > XML_MAX_DEPTH:
            raise InvalidInput('XML node or depth limit exceeded')
        if len(name.encode()) > 1024 or len(attributes) > 256:
            raise InvalidInput('XML name or attribute count limit exceeded')
        for key, value in attributes.items():
            if len(key.encode()) > 1024 or (key == 'xmlns' or key.startswith('xmlns:')) and len(value.encode()) > 1024:
                raise InvalidInput('XML name or namespace limit exceeded')

    def end(name: str) -> None:
        nonlocal depth
        depth -= 1

    def doctype(name: str, system_id: str | None, public_id: str | None, internal: int) -> None:
        if not allow_doctype:
            raise InvalidInput('document contains unsafe XML: DTDs are unsupported')
        _epub_doctype(name, system_id, public_id, internal)

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.StartDoctypeDeclHandler = doctype
    parser.XmlDeclHandler = declaration
    if allow_doctype:
        # Missing HTML entity definitions are deferred to the safe local entity map.
        # No external entity handler or parameter-entity loading is configured.
        parser.UseForeignDTD(True)
    for offset in range(0, len(data), 65536):
        parser.Parse(data[offset:offset + 65536], False)
    parser.Parse(b'', True)
    return encoding


_XML_ENTITIES = re.compile(
    r'<!\[CDATA\[.*?\]\]>|<!--.*?-->|<\?.*?\?>|<!DOCTYPE(?:[^>"\']|"[^"]*"|\'[^\']*\')*>|&([^;\s<&]+);',
    re.DOTALL,
)


def _epub_entities(data: bytes, encoding: str) -> str:
    """Resolve only XML entity tokens; comments, CDATA and declarations stay literal."""
    from html.entities import html5

    def resolve(match: re.Match[str]) -> str:
        name = match.group(1)
        if name is None or name.startswith('#'):
            return match.group()
        value = html5.get(name + ';')
        if value is None:
            raise InvalidInput('document contains unsafe XML: unknown entity name')
        return ''.join(f'&#{ord(character)};' for character in value)

    normalized = _XML_ENTITIES.sub(resolve, data.decode(encoding))
    if len(normalized.encode()) > XML_MAX_EXPANDED_BYTES:
        raise InvalidInput('XML expanded content limit exceeded')
    return normalized


class _XmlTreeBuilder(TreeBuilder):
    def __init__(self) -> None:
        super().__init__()
        self._nodes = 0
        self._depth = 0
        self._expanded = 0

    def _charge(self, value: str) -> None:
        self._expanded += len(value.encode())
        if self._expanded > XML_MAX_EXPANDED_BYTES:
            raise InvalidInput('XML expanded content limit exceeded')

    def start(self, tag: str, attrs: dict[str, str]) -> Element:
        self._nodes += 1
        self._depth += 1
        if self._nodes > XML_MAX_NODES or self._depth > XML_MAX_DEPTH:
            raise InvalidInput('XML node or depth limit exceeded')
        self._charge(tag)
        for name, value in attrs.items():
            self._charge(name)
            self._charge(value)
        return super().start(tag, attrs)

    def data(self, data: str) -> None:
        self._charge(data)
        super().data(data)

    def end(self, tag: str) -> Element:
        result = super().end(tag)
        self._depth -= 1
        return result


class SafeArchive:
    def __init__(self, data: bytes, limits: DocumentLimits):
        if len(data) > limits.max_input_bytes:
            raise InvalidInput('document exceeds its input byte limit')
        self._limit = limits.max_archive_bytes
        try:
            self._zip = ZipFile(BytesIO(data))
            entries = self._zip.infolist()
            self.names = tuple(item.filename for item in entries)
            if len(entries) > limits.max_archive_entries or len(set(self.names)) != len(entries):
                raise InvalidInput('archive has too many or duplicate entries')
            if sum(item.file_size for item in entries) > self._limit:
                raise InvalidInput('archive exceeds its expanded byte limit')
            for item in entries:
                if item.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
                    raise InvalidInput('archive compression must be stored or deflated')
                path = PurePosixPath(item.filename)
                if (path.is_absolute() or '..' in path.parts or '\\' in item.filename
                        or item.flag_bits & 1 or (item.external_attr >> 16) & 0o170000 == 0o120000):
                    raise InvalidInput('archive contains an unsafe or encrypted entry')
        except (BadZipFile, OSError, ValueError):
            if hasattr(self, '_zip'):
                self._zip.close()
            raise InvalidInput('document is not a valid ZIP container') from None
        except InvalidInput:
            self._zip.close()
            raise

    def __enter__(self) -> SafeArchive:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._zip.close()

    def read(self, name: str) -> bytes:
        return self._read(name, self._limit, 'archive member')

    def _read(self, name: str, maximum: int, label: str) -> bytes:
        try:
            if self._zip.getinfo(name).file_size > maximum:
                raise InvalidInput(f'{label} exceeds its byte limit')
            with self._zip.open(name) as stream:
                data = stream.read(maximum + 1)
            if len(data) > maximum:
                raise InvalidInput(f'{label} exceeds its byte limit')
            return data
        except (KeyError, BadZipFile, RuntimeError, OSError, NotImplementedError):
            raise InvalidInput('archive member is missing or unreadable') from None

    def xml(self, name: str, *, allow_doctype: bool = False) -> Element:
        try:
            from defusedxml.ElementTree import DefusedXMLParser
        except ImportError:
            raise InvalidInput('XML documents require scone-memory[documents]') from None
        try:
            data = self._read(name, min(self._limit, XML_MAX_BYTES), 'XML member')
            encoding = _preflight_xml(data, allow_doctype)
            parser = DefusedXMLParser(target=_XmlTreeBuilder(), forbid_dtd=not allow_doctype,
                                      forbid_entities=True, forbid_external=True)
            content: bytes | str = data
            if allow_doctype:
                content = _epub_entities(data, encoding)
                _preflight_xml(content, True)
                parser.parser.StartDoctypeDeclHandler = _epub_doctype
            for offset in range(0, len(content), 65536):
                parser.feed(content[offset:offset + 65536])
            return parser.close()
        except InvalidInput:
            raise
        except Exception:
            raise InvalidInput('document contains malformed or unsafe XML') from None
