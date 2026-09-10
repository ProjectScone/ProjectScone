"""Extract attributed image occurrences from supplied HTML without fetching URLs."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from html.parser import HTMLParser
from typing import Literal

from ..core.errors import InvalidInput
from .image_context import ImageAttribute, ImageContext

_VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
_HIDDEN = {'script', 'style', 'template', 'noscript'}


@dataclass(frozen=True)
class HtmlImageContext:
    src: str
    context: ImageContext


@dataclass
class _Node:
    tag: str
    attributes: dict[str, str]
    parent: int | None
    content: list[int | str] = field(default_factory=list)


class _Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.nodes = [_Node('root', {}, None)]
        self.stack = [0]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if len(self.nodes) >= 10_000 or len(self.stack) >= 256:
            raise InvalidInput('image context HTML exceeds its node limit')
        index = len(self.nodes)
        self.nodes.append(_Node(tag, {key: value or '' for key, value in attrs}, self.stack[-1]))
        self.nodes[self.stack[-1]].content.append(index)
        if tag not in _VOID:
            self.stack.append(index)

    def handle_endtag(self, tag: str) -> None:
        for position in range(len(self.stack) - 1, 0, -1):
            if self.nodes[self.stack[position]].tag == tag:
                del self.stack[position:]
                break

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        self.nodes[self.stack[-1]].content.append(data)

    def hidden(self, index: int) -> bool:
        cursor: int | None = index
        while cursor is not None:
            if self.nodes[cursor].tag in _HIDDEN or 'hidden' in self.nodes[cursor].attributes:
                return True
            cursor = self.nodes[cursor].parent
        return False

    def text(self, index: int) -> str:
        pending: list[int | str] = [index]
        parts: list[str] = []
        while pending:
            value = pending.pop()
            if isinstance(value, str):
                parts.append(value)
            elif self.nodes[value].tag not in _HIDDEN:
                pending.extend(reversed(self.nodes[value].content))
        return ' '.join(' '.join(parts).split())


def image_contexts_from_html(html: str, *, source: str) -> tuple[HtmlImageContext, ...]:
    """Read alt/title, explicit ARIA descriptions, figure captions and data-*.

    Source + deterministic node locator identifies the occurrence. Shared figure
    captions apply to that figure's images; entities are never inferred from text.
    The caller supplies the corresponding image bytes separately.
    """
    if not isinstance(html, str) or len(html.encode('utf-8')) > 1_000_000:
        raise InvalidInput('image context HTML exceeds its byte limit')
    source_sha256 = hashlib.sha256(html.encode()).hexdigest()
    output_bytes = 0
    document = _Document()
    document.feed(html)
    document.close()
    ids: dict[str, list[int]] = {}
    for index, node in enumerate(document.nodes):
        if node.attributes.get('id'):
            ids.setdefault(node.attributes['id'], []).append(index)
    images: list[HtmlImageContext] = []
    for index, node in enumerate(document.nodes):
        if node.tag != 'img' or document.hidden(index) or not node.attributes.get('src'):
            continue
        attrs: list[ImageAttribute] = []
        locator = f'html-node:{index}'
        for name, value in node.attributes.items():
            if not value.strip():
                continue
            if name in ('alt', 'title', 'aria-label') or name.startswith('data-'):
                kind: Literal['alt', 'title', 'description', 'metadata'] = 'alt' if name == 'alt' else 'title' if name == 'title' else 'description' if name == 'aria-label' else 'metadata'
                attrs.append(ImageAttribute(kind=kind, value=value, origin='html', name=name, locator=f'{locator}@{name}'))
        for identifier in node.attributes.get('aria-describedby', '').split():
            referenced = ids.get(identifier, [])
            if len(referenced) == 1:
                text = document.text(referenced[0])
                if text:
                    attrs.append(ImageAttribute(kind='description', value=text, origin='html',
                        name='aria-describedby', locator=f'html-node:{referenced[0]}'))
        parent = node.parent
        while parent is not None and document.nodes[parent].tag != 'figure':
            parent = document.nodes[parent].parent
        if parent is not None:
            for child in document.nodes[parent].content:
                if isinstance(child, int) and document.nodes[child].tag == 'figcaption':
                    text = document.text(child)
                    if text:
                        attrs.append(ImageAttribute(kind='caption', value=text, origin='html', locator=f'html-node:{child}'))
        if not attrs:
            continue
        if len(node.attributes['src']) > 4096:
            raise InvalidInput('image source reference exceeds its character limit')
        context = ImageContext(source=source, locator=locator, source_sha256=source_sha256, attributes=tuple(attrs))
        output_bytes += len(context.model_dump_json().encode())
        if output_bytes > 4_000_000:
            raise InvalidInput('image context HTML exceeds its output byte limit')
        images.append(HtmlImageContext(node.attributes['src'], context))
        if len(images) > 1000:
            raise InvalidInput('image context HTML exceeds its image limit')
    return tuple(images)
