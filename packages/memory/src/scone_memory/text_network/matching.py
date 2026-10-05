"""Untransformed prose spans and exact Unicode literal matches."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import re
import unicodedata

_FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')
_LIST = re.compile(r'(?:[-+*]|[0-9]{1,9}[.)])([ \t]+)')
_HEADING = re.compile(r'^ {0,3}#{1,6}(?:\s|$)')
_RULE = re.compile(r'^\s*\|?[\s:|-]+\|?\s*$')
_WORD = re.compile(r"[^\W\d_][\w’'-]*", re.UNICODE)
_STOP = frozenset('A An The I It He She They We You This That These Those In On At And But Or Chapter Part Section When Where Why How His Her Their Our My Your Of To From As For With By'.casefold().split())


def _indent(line: str, cursor: int, maximum: int | None = None) -> tuple[int, int]:
    width = 0
    while cursor < len(line) and line[cursor] in ' \t' and (maximum is None or width < maximum):
        width += 4-width % 4 if line[cursor] == '\t' else 1
        cursor += 1
    return cursor, width


@dataclass(frozen=True)
class _FenceScope:
    marker: str
    containers: tuple[tuple[str, int], ...]


def _opening(line: str) -> tuple[str, tuple[tuple[str, int], ...]]:
    """Read only structural prefixes; all cited offsets remain in the source."""
    cursor = 0
    containers: list[tuple[str, int]] = []
    while cursor < len(line):
        after_indent, width = _indent(line, cursor)
        if width > 3:
            break
        if line[after_indent:after_indent+1] == '>':
            cursor = after_indent+1
            if cursor < len(line) and line[cursor] in ' \t':
                cursor += 1
            containers.append(('quote', 0))
            continue
        marker = _LIST.match(line, after_indent)
        if marker is None:
            break
        # A list marker followed by >4 spaces supplies one padding space;
        # the remainder is indentation belonging to its content.
        gap = marker.group(1)
        padding = len(gap.expandtabs(4))
        used = len(gap) if padding <= 4 else 1
        cursor = marker.start(1)+used
        containers.append(('list', width+marker.start(1)-after_indent+(padding if padding <= 4 else 1)))
    return line[cursor:], tuple(containers)


def _inside(line: str, containers: tuple[tuple[str, int], ...]) -> str | None:
    cursor = 0
    for kind, width in containers:
        if kind == 'quote':
            cursor, indent = _indent(line, cursor)
            if indent > 3 or line[cursor:cursor+1] != '>':
                return None
            cursor += 1
            if cursor < len(line) and line[cursor] in ' \t':
                cursor += 1
        else:
            cursor, indent = _indent(line, cursor, width)
            if indent < width:
                return '' if not line[cursor:].strip() else None
    return line[cursor:]


def prose_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield original paragraphs/table rows; never flatten fenced code."""
    cursor = 0
    start: int | None = None
    end = 0
    fence: _FenceScope | None = None
    for line in text.splitlines(keepends=True):
        bare = line.rstrip('\r\n')
        if fence is not None:
            inside = _inside(bare, fence.containers)
            if inside is not None:
                marker = _FENCE.fullmatch(inside)
                if marker and marker[1][0] == fence.marker[0] and len(marker[1]) >= len(fence.marker) and not marker[2].strip():
                    fence = None
                cursor += len(line)
                continue
            else:
                # An unclosed quote/list fence ends with its container; it
                # must not swallow following unindented prose.
                fence = None
        content, containers = _opening(bare)
        marker = _FENCE.fullmatch(content)
        if marker and not (marker[1][0] == '`' and '`' in marker[2]):
            if start is not None:
                yield start, end
                start = None
            fence = _FenceScope(marker[1], containers)
        elif start is None and _indent(content, 0)[1] >= 4:
            # Indented code is separate from fences. Backticks within it do
            # not start a new fence that continues outside the code block.
            pass
        elif not bare.strip() or _HEADING.match(bare) or '|' in bare:
            if start is not None:
                yield start, end
                start = None
            if bare.strip() and not _RULE.fullmatch(bare):
                yield cursor, cursor + len(bare)
        else:
            if start is None:
                start = cursor
            end = cursor + len(bare)
        cursor += len(line)
    if start is not None:
        yield start, end


def _boundary(char: str) -> bool:
    return not char or not (unicodedata.category(char)[0] in 'LMN' or char == '_')


def first_match(text: str, folded: str, phrase: str) -> tuple[int, int] | None:
    """One whole folded match, rejecting half-character expansions (ß→ss)."""
    needle = phrase.casefold()
    cursor = 0
    characters = enumerate(text)
    index = -1
    folded_start = folded_end = 0
    while True:
        found = folded.find(needle, cursor)
        if found < 0:
            return None
        cursor = found + len(needle)
        while folded_end <= found:
            index, char = next(characters)
            folded_start = folded_end
            folded_end += len(char.casefold())
        first, whole_start = index, folded_start == found
        while folded_end < cursor:
            index, char = next(characters)
            folded_start = folded_end
            folded_end += len(char.casefold())
        last = index + 1
        if (whole_start and folded_end == cursor and text[first:last].casefold() == needle
                and _boundary(text[first-1] if first else '') and _boundary(text[last] if last < len(text) else '')):
            return first, last


def candidate_names(text: str) -> set[str]:
    words = list(_WORD.finditer(text))
    candidates: set[str] = set()
    index = 0
    while index < len(words):
        word = words[index]
        value = word.group()
        index += 1
        if not value[0].isupper() or value.casefold() in _STOP or len(value) < 2:
            continue
        end = word.end()
        count = 1
        while index < len(words) and count < 4:
            following = words[index]
            next_value = following.group()
            if (not next_value[0].isupper() or next_value.casefold() in _STOP
                    or not text[end:following.start()].isspace() or '\n' in text[end:following.start()]):
                break
            end = following.end()
            count += 1
            index += 1
        candidate = text[word.start():end]
        if len(candidate) <= 80:
            candidates.add(candidate)
    return candidates
