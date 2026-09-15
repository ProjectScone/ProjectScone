"""One chunk per unit the file's reader named, a long unit split by length."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from .chunker import Span, chunk_spans
from .structure import SourceUnit


@dataclass(frozen=True)
class UnitCut:
    spans: tuple[Span, ...] = ()
    #: Units read.
    units: int = 0
    #: Units longer than the target, cut by length inside themselves.
    split_units: int = 0
    #: Chunks those cuts added beyond one per unit.
    by_size: int = 0
    #: Units by kind: page, slide, row, line (a JSON Lines record), segment, frame, text.
    kinds: dict[str, int] = field(default_factory=dict)

    def record(self) -> dict[str, object]:
        return {"chunks": len(self.spans), "units": self.units, "split_units": self.split_units,
                "by_size": self.by_size, "kinds": dict(self.kinds)}


def unit_spans(content: str, units: Sequence[SourceUnit], target: int) -> UnitCut:
    """Each unit as one chunk, in code points; a unit longer than ``target`` code points, the
    length cut's own measure, is split exactly as the length cut would split it. The blank
    lines between units belong to none."""
    if target <= 0:
        raise ValueError("target must be positive")
    encoded = content.encode()
    spans: list[Span] = []
    split = extra = 0
    byte = point = 0
    for unit in sorted(units, key=lambda one: one.start):
        if not 0 <= unit.start <= unit.end <= len(encoded) or unit.start < byte:
            raise ValueError("units must lie in the content, in order, without overlapping")
        point += len(encoded[byte:unit.start].decode())
        text = encoded[unit.start:unit.end].decode()
        if len(text) > target:
            inside = chunk_spans(text, target)
            spans.extend(Span(point + part.start, point + part.end) for part in inside)
            split += 1
            extra += len(inside) - 1
        elif text:
            spans.append(Span(point, point + len(text)))
        point += len(text)
        byte = unit.end
    return UnitCut(tuple(spans), len(units), split, extra, dict(Counter(unit.kind for unit in units)))
