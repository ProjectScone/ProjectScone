"""What a Word paragraph declares about its role: heading, list item or caption.

A paragraph is a heading because its own outline level, or the outline level
or name of the style it is based on, says so -- never because its text is
short or its font large. A list item is one because numbering is attached
to it (or to its style) and that numbering is not ``numId 0``, which Word
writes to take a list off. The list's kind is read from the numbering part;
where that part is missing or does not define the level, the item is still a
list item and its kind is left unsaid.

Style IDs are what paragraphs reference, and they are localized
(``Berschrift1`` is a German Heading 1), so the style's name decides. Only a
style ID the styles part does not define is read by its spelling
(``Heading2``, ``Title``, ``Caption``), and the metadata says which basis
was used. A paragraph that names no style is not given the document's
default paragraph style: that style is body text in every document Word
writes, and consulting it would only let a malformed one relabel all text.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from xml.etree.ElementTree import Element

#: Styles followed through ``basedOn`` for one paragraph. A chain longer
#: than this is read no further, and the document's ``structure_notes`` say
#: ``style_chain_cut``: a heading declared further up would have been missed.
MAX_STYLE_DEPTH = 32

_HEADING_NAME = re.compile(r'heading ([1-9])', re.IGNORECASE)
_HEADING_ID = re.compile(r'heading([1-9])', re.IGNORECASE)
_DIGITS = re.compile(r'[0-9]{1,4}')


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _child(element: Element, name: str) -> Element | None:
    return next((child for child in element if _local(child.tag) == name), None)


def _value(element: Element | None, name: str = 'val') -> str | None:
    if element is None:
        return None
    namespace = element.tag.rsplit('}', 1)[0] + '}' if element.tag.startswith('{') else ''
    return element.get(namespace + name)


def _property(element: Element, *path: str) -> Element | None:
    current: Element | None = element
    for name in path:
        current = None if current is None else _child(current, name)
    return current


def _outline(properties: Element | None) -> int | None:
    """Outline level 0-8 is a heading at level + 1; 9 is body text, said explicitly."""
    raw = _value(_property(properties, 'outlineLvl') if properties is not None else None)
    if raw is None or _DIGITS.fullmatch(raw) is None or int(raw) > 9:
        return None
    return int(raw)


def _numbering(properties: Element | None) -> tuple[str | None, str | None]:
    numbered = None if properties is None else _property(properties, 'numPr')
    if numbered is None:
        return None, None
    return _value(_property(numbered, 'numId')), _value(_property(numbered, 'ilvl'))


@dataclass(frozen=True)
class _Style:
    name: str
    based_on: str | None
    outline: int | None
    number: str | None
    level: str | None


class WordStructure:
    """Resolves a paragraph's declared role from its properties, styles and numbering."""

    def __init__(self, styles: Element | None, numbering: Element | None) -> None:
        self.styles: dict[str, _Style] = {}
        self.kinds: dict[tuple[str, str], str] = {}
        #: What stopped a resolution short, for the document's structure_notes.
        self.notes: set[str] = set()
        for style in styles if styles is not None else ():
            if (identifier := _value(style, 'styleId')) is not None:
                properties = _child(style, 'pPr')
                number, level = _numbering(properties)
                self.styles[identifier] = _Style(_value(_child(style, 'name')) or '',
                                                 _value(_child(style, 'basedOn')), _outline(properties), number, level)
        if numbering is not None:
            self._read_numbering(numbering)

    def _read_numbering(self, numbering: Element) -> None:
        # An abstract definition carries its id as an attribute; a numbering
        # instance names one in a child element and may override its levels.
        abstract: dict[str, list[Element]] = {}
        for definition in numbering:
            if (identifier := _value(definition, 'abstractNumId')) is not None:
                abstract[identifier] = list(definition)
        for instance in numbering:
            if (identifier := _value(instance, 'numId')) is None:
                continue
            formats = {_value(level, 'ilvl') or '0': _value(_child(level, 'numFmt'))
                       for level in abstract.get(_value(_child(instance, 'abstractNumId')) or '', [])}
            for override in instance:
                if (level := _child(override, 'lvl')) is not None:
                    formats[_value(override, 'ilvl') or '0'] = _value(_child(level, 'numFmt'))
            for level_id, fmt in formats.items():
                if fmt is not None and fmt != 'none':
                    self.kinds[(identifier, level_id)] = 'bullet' if fmt == 'bullet' else 'ordered'

    def _chain(self, identifier: str | None) -> list[_Style]:
        """The style and those it is based on, nearest first; a cycle ends it."""
        chain: list[_Style] = []
        seen: set[str] = set()
        while identifier in self.styles and identifier not in seen:
            if len(chain) == MAX_STYLE_DEPTH:
                self.notes.add('style_chain_cut')
                break
            seen.add(identifier)
            chain.append(self.styles[identifier])
            identifier = chain[-1].based_on
        return chain

    def role(self, paragraph: Element, check: Callable[[], None]) -> dict[str, str]:
        check()
        properties = _child(paragraph, 'pPr')
        style_id = _value(_property(properties, 'pStyle')) if properties is not None else None
        chain = self._chain(style_id)
        heading = self._heading(properties, style_id, chain)
        if heading is not None:
            return heading
        if any(style.name.lower() == 'caption' for style in chain) or (
                style_id and style_id not in self.styles and style_id.lower() == 'caption'):
            return {'block_role': 'caption'}
        return self._list_item(properties, chain)

    def _heading(self, properties: Element | None, style_id: str | None,
                 chain: list[_Style]) -> dict[str, str] | None:
        outline = _outline(properties)
        basis = 'outline_level'
        if outline is None:
            basis = 'style'
            for style in chain:
                named = _HEADING_NAME.fullmatch(style.name)
                if style.outline is not None:
                    outline = style.outline
                elif named is not None:
                    outline = int(named[1]) - 1
                elif style.name.lower() == 'title':
                    outline = 0
                if outline is not None:
                    break
        if outline is None and style_id and style_id not in self.styles:
            basis = 'style_id'
            by_id = _HEADING_ID.fullmatch(style_id)
            outline = int(by_id[1]) - 1 if by_id else 0 if style_id.lower() == 'title' else None
        if outline is None or outline == 9:
            return None
        return {'block_role': 'heading', 'heading_level': str(outline + 1), 'heading_basis': basis}

    def _list_item(self, properties: Element | None, chain: list[_Style]) -> dict[str, str]:
        number, level = _numbering(properties)
        for style in chain:
            number = number if number is not None else style.number
            level = level if level is not None else style.level
        if number is None or number == '0':
            return {}
        level = level or '0'
        role = {'block_role': 'list_item', 'list_id': number}
        if re.fullmatch(r'[0-8]', level):
            role['list_level'] = level
        if (kind := self.kinds.get((number, level))) is not None:
            role['list_kind'] = kind
        return role
