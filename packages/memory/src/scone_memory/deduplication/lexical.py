"""Unicode-normalized copied spans with explicit bounded matching work."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .types import CopiedSpan, DocumentRevision


@dataclass(frozen=True)
class NormalizedText:
    value: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]


def _ascii(text: str) -> NormalizedText:
    lowered = text.lower()
    if not re.search(r"[^\S ]| {2,}", text):
        return NormalizedText(lowered, tuple(range(len(text))), tuple(range(1, len(text) + 1)))
    values: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    previous = 0
    for whitespace in re.finditer(r"\s+", text):
        start, end = whitespace.span()
        values.extend((lowered[previous:start], " "))
        starts.extend(range(previous, start))
        ends.extend(range(previous + 1, start + 1))
        starts.append(start)
        ends.append(end)
        previous = end
    values.append(lowered[previous:])
    starts.extend(range(previous, len(text)))
    ends.extend(range(previous + 1, len(text) + 1))
    return NormalizedText("".join(values), tuple(starts), tuple(ends))


def normalize(text: str) -> NormalizedText:
    if text.isascii():
        return _ascii(text)
    values: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    clusters: list[tuple[list[str], int, int]] = []
    offset = 0
    for char in text:
        end = offset + len(char.encode("utf-8"))
        if unicodedata.combining(char) and clusters:
            previous, start, _ = clusters[-1]
            previous.append(char)
            clusters[-1] = (previous, start, end)
        else:
            clusters.append(([char], offset, end))
        offset = end
    for cluster, start, end in clusters:
        for char in unicodedata.normalize("NFKC", "".join(cluster)).casefold():
            char = " " if char.isspace() else char
            if char == " " and values and values[-1] == " ":
                ends[-1] = end
                continue
            values.append(char)
            starts.append(start)
            ends.append(end)
    return NormalizedText("".join(values), tuple(starts), tuple(ends))


class CopiedSpanMatcher:
    def __init__(self, document: DocumentRevision, min_chars: int, operations: int) -> None:
        self.document = document
        self.query = normalize(document.text)
        self.min_chars = min_chars
        self.remaining = operations
        self.truncated = False
        self.index: dict[str, list[int]] = {}
        for position in range(len(self.query.value) - min_chars + 1):
            key = self.query.value[position:position + min_chars]
            self.index.setdefault(key, []).append(position)

    def _spend(self) -> bool:
        self.remaining -= 1
        if self.remaining < 0:
            self.truncated = True
            return False
        return True

    def compare(self, source: DocumentRevision) -> list[CopiedSpan]:
        if not self.document.text or not source.text:
            return []
        if self.document.text == source.text:
            return [self._whole(source, "exact")]
        target = normalize(source.text)
        if self.query.value == target.value:
            return [self._whole(source, "normalized")]
        return self._scan(source, target)

    def _whole(self, source: DocumentRevision, kind: str) -> CopiedSpan:
        # Callers restrict kind to these two literals.
        return CopiedSpan(
            self.document.byte_offset, self.document.byte_offset + len(self.document.text.encode()),
            source.source_id, source.revision, source.byte_offset,
            source.byte_offset + len(source.text.encode()), source.chunk_id,
            "exact" if kind == "exact" else "normalized",
        )

    def _scan(self, source: DocumentRevision, target: NormalizedText) -> list[CopiedSpan]:
        result: list[CopiedSpan] = []
        seen: set[tuple[int, int]] = set()
        diagonal_ends: dict[int, int] = {}
        position = 0
        query = self.query.value
        while position <= len(target.value) - self.min_chars and self._spend():
            key = target.value[position:position + self.min_chars]
            for query_position in self.index.get(key, ()):
                if not self._spend():
                    break
                diagonal = query_position - position
                if position + self.min_chars <= diagonal_ends.get(diagonal, -1):
                    continue
                left, source_left = query_position, position
                right, source_right = query_position + self.min_chars, position + self.min_chars
                while left and source_left and query[left - 1] == target.value[source_left - 1]:
                    if not self._spend():
                        break
                    left -= 1
                    source_left -= 1
                while right < len(query) and source_right < len(target.value) and query[right] == target.value[source_right]:
                    if not self._spend():
                        break
                    right += 1
                    source_right += 1
                diagonal_ends[diagonal] = max(diagonal_ends.get(diagonal, -1), source_right)
                # Normalization can expand one source code point (ß -> ss).
                # Trim partial clusters rather than overstate copied bytes.
                while left < right and (
                    (left > 0 and self.query.starts[left] == self.query.starts[left - 1])
                    or (source_left > 0 and target.starts[source_left] == target.starts[source_left - 1])
                ):
                    left += 1
                    source_left += 1
                while right > left and (
                    (right < len(query) and self.query.ends[right - 1] == self.query.ends[right])
                    or (source_right < len(target.value) and target.ends[source_right - 1] == target.ends[source_right])
                ):
                    right -= 1
                    source_right -= 1
                while left < right and query[left].isspace():
                    left += 1
                    source_left += 1
                while right > left and query[right - 1].isspace():
                    right -= 1
                    source_right -= 1
                if right - left < self.min_chars or (left, right) in seen:
                    continue
                seen.add((left, right))
                result.append(CopiedSpan(
                    self.document.byte_offset + self.query.starts[left],
                    self.document.byte_offset + self.query.ends[right - 1],
                    source.source_id, source.revision,
                    source.byte_offset + target.starts[source_left],
                    source.byte_offset + target.ends[source_right - 1],
                    source.chunk_id, "copied",
                ))
            position += 1
        return result


def covered_bytes(spans: list[CopiedSpan]) -> int:
    total, previous_end = 0, -1
    for start, end in sorted((span.start, span.end) for span in spans):
        total += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total
