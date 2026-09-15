"""The order passages are handed to a model in.

Recall ranks passages best first, and that is the order a person reads
a list in. A model reading a long context attends least to its middle
("lost in the middle", Liu et al. 2023): what sits first and last is
used, what sits between is skimmed. Arranging the best passages at both
ends and the weakest in the middle is LlamaIndex's `LongContextReorder`;
here it is a choice a context assembler makes, off by default. The
receipt names the arrangement, and `ranked` undoes it: the passages
carry no rank of their own, so the ranked order is recovered by
inverting the named arrangement, not read off each passage.

The arrangement is fixed and reversible: rank 1 first, rank 2 last,
rank 3 second, rank 4 second to last, and so on inward. Nothing is
dropped and nothing is scored; only the order changes.
"""
from __future__ import annotations

from typing import Literal, Sequence, TypeVar

from ..core.errors import InvalidInput

T = TypeVar("T")

ReadingOrder = Literal["ranked", "ends"]
READING_ORDERS: tuple[ReadingOrder, ...] = ("ranked", "ends")


def validate_reading_order(value: object) -> ReadingOrder:
    if value not in READING_ORDERS:
        raise InvalidInput(f"reading_order must be one of {', '.join(READING_ORDERS)}")
    return value  # type: ignore[return-value]


def ends_first(ranked: Sequence[T]) -> list[T]:
    """``ranked`` best first, rearranged so the best sit at both ends and
    the weakest in the middle: odd ranks from the front, even ranks from
    the back, inward."""
    front = list(ranked[0::2])
    back = list(ranked[1::2])
    return front + back[::-1]


def arranged(ranked: Sequence[T], order: ReadingOrder) -> list[T]:
    """``ranked`` in the requested reading order."""
    return ends_first(ranked) if order == "ends" else list(ranked)


def ranked(items: Sequence[T], order: ReadingOrder) -> list[T]:
    """The ranked order back from a block in the named reading order:
    the inverse of ``arranged``."""
    if order != "ends":
        return list(items)
    front_count = (len(items) + 1) // 2
    front, back = list(items[:front_count]), list(items[front_count:])[::-1]
    restored: list[T] = []
    for index in range(len(items)):
        restored.append(front[index // 2] if index % 2 == 0 else back[index // 2])
    return restored
