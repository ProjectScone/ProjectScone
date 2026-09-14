"""Parameter descriptions a function's docstring gives, in the three common styles.

Google (``Args:`` with indented ``name (type): text``), NumPy (a ``Parameters``
heading underlined with dashes, ``name : type`` then indented text) and Sphinx
(``:param name:`` or ``:param type name:``). A description continued on further
lines is joined into one line. Nothing is evaluated; a docstring is only read.
"""
from __future__ import annotations

import inspect
import re

_GOOGLE = re.compile(r'^(\s*)(Args|Arguments|Parameters|Params|Keyword Args|Keyword Arguments|Other Parameters):\s*$')
_GOOGLE_ENTRY = re.compile(r'^\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$')
_NUMPY = re.compile(r'^(\s*)(Parameters|Other Parameters|Keyword Arguments)\s*$')
_UNDERLINE = re.compile(r'^\s*-{3,}\s*$')
_NUMPY_ENTRY = re.compile(r'^((?:\*{0,2}\w+\s*,\s*)*\*{0,2}\w+)\s*(?::.*)?$')
#: Sphinx reads these field names alike.
_SPHINX = re.compile(r'^\s*:(?:param|parameter|arg|argument|key|keyword)\s+(?:[^:]*\s)?\*{0,2}(\w+)\s*:\s*(.*)$')
_SPHINX_TYPE = re.compile(r'^\s*:type\s+\w+\s*:')


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _read(docstring: str | None) -> tuple[dict[str, str], list[str], set[int]]:
    lines = inspect.cleandoc(docstring or '').splitlines()
    found: dict[str, list[str]] = {}
    section: set[int] = set()
    index = 0
    while index < len(lines):
        line = lines[index]
        google = _GOOGLE.match(line)
        numpy = _NUMPY.match(line) if index + 1 < len(lines) and _UNDERLINE.match(lines[index + 1]) else None
        sphinx = _SPHINX.match(line)
        if sphinx:
            said = found.setdefault(sphinx[1], [sphinx[2]])
            section.add(index)
            index += 1
            while index < len(lines):
                ahead = index
                while ahead < len(lines) and not lines[ahead].strip():
                    ahead += 1
                # Blank lines belong to the field only when its text carries on, indented, after them.
                if ahead >= len(lines) or _indent(lines[ahead]) <= _indent(line):
                    break
                said.extend(lines[ahead:ahead + 1])
                section.update(range(index, ahead + 1))
                index = ahead + 1
            continue
        if _SPHINX_TYPE.match(line):
            section.add(index)
            index += 1
            continue
        if google or numpy:
            # A heading on a docstring's first line lost its indent to cleandoc while the entries under it
            # lost theirs, so there the entries sit at the heading's column and still belong to it.
            header = -1 if google and index == 0 else _indent(line)
            section.update({index} if google else {index, index + 1})
            index += 1 if google else 2
            entry: int | None = None
            current: list[str] | None = None
            while index < len(lines):
                body = lines[index]
                if not body.strip():
                    if index + 1 < len(lines) and lines[index + 1].strip() and _indent(lines[index + 1]) <= header \
                            and (google or index + 2 < len(lines) and _UNDERLINE.match(lines[index + 2])):
                        break
                    index += 1
                    continue
                depth = _indent(body)
                if google and depth <= header:
                    break
                if numpy and depth <= header and index + 1 < len(lines) and _UNDERLINE.match(lines[index + 1]):
                    break
                if entry is None:
                    entry = depth
                matched = (_GOOGLE_ENTRY if google else _NUMPY_ENTRY).match(body.strip()) if depth == entry else None
                if matched and google:
                    current = found.setdefault(matched[1], [matched[2]])
                elif matched:
                    # ``x, y : int`` describes both names with the lines that follow.
                    current = []
                    for name in matched[1].split(','):
                        found.setdefault(name.strip().lstrip('*'), current)
                elif current is not None:
                    current.append(body)
                section.add(index)
                index += 1
            continue
        index += 1
    descriptions = {name: ' '.join(' '.join(parts).split()) for name, parts in found.items()}
    return {name: text for name, text in descriptions.items() if text}, lines, section


def parameter_descriptions(docstring: str | None) -> dict[str, str]:
    """Each documented parameter's description, by name."""
    return _read(docstring)[0]


def without_parameters(docstring: str | None) -> str:
    """The docstring, cleaned, with its parameter section taken out."""
    _, lines, section = _read(docstring)
    kept = '\n'.join(line for index, line in enumerate(lines) if index not in section)
    return re.sub(r'\n{3,}', '\n\n', kept).strip()


__all__ = ['parameter_descriptions', 'without_parameters']
