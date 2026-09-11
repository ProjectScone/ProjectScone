"""Bound XML names and tree construction for standalone and container readers."""
from __future__ import annotations

import re
from xml.etree.ElementTree import Element, TreeBuilder
from xml.parsers import expat

from ...core.errors import InvalidInput

XML_MAX_BYTES = 16 * 1024 * 1024
XML_MAX_NODES = 200_000
XML_MAX_DEPTH = 128
XML_MAX_EXPANDED_BYTES = 32 * 1024 * 1024
_MC = '{http://schemas.openxmlformats.org/markup-compatibility/2006}'


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


def _epub_entities(data: bytes | str, encoding: str) -> str:
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

    normalized = _XML_ENTITIES.sub(resolve, data if isinstance(data, str) else data.decode(encoding))
    if len(normalized.encode()) > XML_MAX_EXPANDED_BYTES:
        raise InvalidInput('XML expanded content limit exceeded')
    return normalized


class _XmlTreeBuilder(TreeBuilder):
    def __init__(self, supported_namespaces: frozenset[str] | None = None) -> None:
        super().__init__()
        self._nodes = 0
        self._depth = 0
        self._expanded = 0
        self._supported = supported_namespaces
        self._namespaces: dict[str, list[str]] = {}
        self._choices: dict[Element, bool | None] = {}

    def start_ns(self, prefix: str, uri: str) -> None:
        self._charge(prefix)
        self._charge(uri)
        if self._supported is not None:
            self._namespaces.setdefault(prefix, []).append(uri)

    def end_ns(self, prefix: str) -> None:
        if self._supported is not None:
            held = self._namespaces[prefix]
            held.pop()
            if not held:
                del self._namespaces[prefix]

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
        element = super().start(tag, attrs)
        if self._supported is not None and tag == _MC + 'Choice':
            required = attrs.get('Requires', '').split()
            if not required or any(prefix not in self._namespaces for prefix in required):
                self._choices[element] = None
            else:
                self._choices[element] = all(self._namespaces[prefix][-1] in self._supported for prefix in required)
        return element

    def data(self, data: str) -> None:
        self._charge(data)
        super().data(data)

    def end(self, tag: str) -> Element:
        result = super().end(tag)
        self._depth -= 1
        return result

    def close(self) -> Element:
        root = super().close()
        if self._supported is None:
            return root
        stack = [root]
        while stack:
            result = stack.pop()
            if result.tag != _MC + 'AlternateContent':
                stack.extend(reversed(result))
                continue
            selected = fallback = None
            choices = 0
            for branch in result:
                if branch.tag == _MC + 'Choice' and fallback is None:
                    choices += 1
                    eligible = self._choices.get(branch)
                    if eligible is None:
                        raise InvalidInput('XML alternate content has invalid namespace requirements')
                    if selected is None and eligible:
                        selected = branch
                elif branch.tag == _MC + 'Fallback' and fallback is None and choices:
                    fallback = branch
                else:
                    raise InvalidInput('XML alternate content has invalid branches')
            if selected is None:
                selected = fallback
            if not choices or selected is None:
                raise InvalidInput('XML alternate content has no supported branch or fallback')
            result[:] = [selected]
            stack.append(selected)
        return root


def parse_xml(data: bytes | str, *, allow_doctype: bool = False,
              supported_namespaces: frozenset[str] | None = None) -> Element:
    try:
        from defusedxml.ElementTree import DefusedXMLParser
    except ImportError:
        raise InvalidInput('XML documents require scone-memory[documents]') from None
    try:
        if len(data if isinstance(data, bytes) else data.encode('utf-8')) > XML_MAX_BYTES:
            raise InvalidInput('XML exceeds its byte limit')
        encoding = _preflight_xml(data, allow_doctype)
        parser = DefusedXMLParser(target=_XmlTreeBuilder(supported_namespaces), forbid_dtd=not allow_doctype,
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
