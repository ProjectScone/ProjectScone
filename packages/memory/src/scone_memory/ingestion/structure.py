"""Original, offline Markdown structure over unchanged UTF-8 source bytes.

This intentionally recognizes textual headings, fences and pipe tables only.
It neither renders Markdown nor infers layout, OCR, claims or cell semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..core.models import MAX_CONTENT_BYTES

_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_SEPARATOR = re.compile(r"^:?-{3,}:?$")
MAX_STRUCTURE_NODES = 16_384


class ByteSpan(BaseModel):
    model_config = ConfigDict(frozen=True)
    start: int
    end: int


class StructureBlock(ByteSpan):
    block_id: str
    section_id: str
    kind: Literal["heading", "text", "fenced_code", "table"]
    header: ByteSpan | None = None
    separator: ByteSpan | None = None
    rows: tuple[ByteSpan, ...] = ()


class StructureSection(ByteSpan):
    section_id: str
    parent_section_id: str | None
    title: str
    level: int


class DocumentStructure(BaseModel):
    model_config = ConfigDict(frozen=True)
    content_hash: str
    source_bytes: int
    sections: tuple[StructureSection, ...]
    blocks: tuple[StructureBlock, ...]


@dataclass
class _Section:
    section_id: str
    parent_section_id: str | None
    title: str
    level: int
    start: int
    end: int


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_header(first: str, second: str) -> bool:
    cells = _cells(second)
    return ("|" in first and "|" in second and len(_cells(first)) == len(cells)
            and all(_SEPARATOR.fullmatch(cell) is not None for cell in cells))


def parse_structure(content: str) -> DocumentStructure:
    """Parse at most MAX_CONTENT_BYTES; all offsets are source-relative bytes.

    Section zero is the whole document. Each heading's section includes its
    descendants through the next heading of the same or lower level. A fence
    suppresses heading/table recognition until a compatible closing fence.
    IDs bind to the complete source hash and cannot survive edited bytes.
    """
    raw = content.encode("utf-8")
    if len(raw) > MAX_CONTENT_BYTES:
        raise ValueError(f"structure source exceeds {MAX_CONTENT_BYTES} UTF-8 bytes")
    digest = hashlib.sha256(raw).hexdigest()
    lines = raw.splitlines(keepends=True)
    texts = [line.decode("utf-8").rstrip("\r\n") for line in lines]
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    root_id = f"{digest}:section:root"
    sections = [_Section(root_id, None, "", 0, 0, len(raw))]
    stack = [0]
    blocks: list[StructureBlock] = []
    node_count = 1
    index = 0
    while index < len(lines):
        start_index = index
        heading = _HEADING.fullmatch(texts[index])
        fence = _FENCE.fullmatch(texts[index])
        kind: Literal["heading", "text", "fenced_code", "table"] = "text"
        header = separator = None
        rows: tuple[ByteSpan, ...] = ()
        if heading is not None:
            kind = "heading"
            level = len(heading[1])
            while sections[stack[-1]].level >= level:
                sections[stack.pop()].end = offsets[index]
            section_id = f"{digest}:section:{offsets[index]}"
            title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2] or "").strip()
            sections.append(_Section(section_id, sections[stack[-1]].section_id,
                                     title, level, offsets[index], len(raw)))
            node_count += 1
            stack.append(len(sections) - 1)
            index += 1
        elif fence is not None and not (fence[1][0] == "`" and "`" in fence[2]):
            kind = "fenced_code"
            marker = fence[1]
            index += 1
            while index < len(lines):
                closing = _FENCE.fullmatch(texts[index])
                index += 1
                if (closing is not None and closing[1][0] == marker[0]
                        and len(closing[1]) >= len(marker) and not closing[2].strip()):
                    break
        elif index + 1 < len(lines) and _table_header(texts[index], texts[index + 1]):
            kind = "table"
            header = ByteSpan(start=offsets[index], end=offsets[index + 1])
            separator = ByteSpan(start=offsets[index + 1], end=offsets[index + 2])
            index += 2
            row_spans: list[ByteSpan] = []
            while index < len(lines) and "|" in texts[index] and texts[index].strip():
                if _HEADING.fullmatch(texts[index]) or _FENCE.fullmatch(texts[index]):
                    break
                row_spans.append(ByteSpan(start=offsets[index], end=offsets[index + 1]))
                if node_count + len(row_spans) + 3 > MAX_STRUCTURE_NODES:
                    raise ValueError("structure node limit exceeded")
                index += 1
            rows = tuple(row_spans)
        else:
            index += 1
        blocks.append(StructureBlock(
            block_id=f"{digest}:block:{offsets[start_index]}", section_id=sections[stack[-1]].section_id,
            kind=kind, start=offsets[start_index], end=offsets[index], header=header,
            separator=separator, rows=rows))
        node_count += 1 + len(rows) + (2 if header is not None else 0)
        if node_count > MAX_STRUCTURE_NODES:
            raise ValueError("structure node limit exceeded")
    return DocumentStructure(content_hash=digest, source_bytes=len(raw), blocks=tuple(blocks),
                             sections=tuple(StructureSection(**section.__dict__) for section in sections))
