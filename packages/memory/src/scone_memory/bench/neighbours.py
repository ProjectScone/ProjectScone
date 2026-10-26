"""Exact nearest neighbours, as the ground truth for an approximate index.

An approximate index (HNSW, IVF, quantised vectors) is judged by how
many of the true nearest neighbours it returns. That needs the true
neighbours, computed by brute force over every stored vector with the
same similarity and the same filter the index was asked to apply: a
ground truth computed without the filter would mark an index wrong for
correctly leaving out a document the caller was not allowed to see.

Cosine similarity is undefined for a zero vector. Such documents are
skipped and counted, never scored as zero, because a zero would place
them above every document with a negative similarity.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ExactNeighbours:
    """The true top ``k`` by cosine similarity, best first."""

    ranked: tuple[tuple[str, float], ...]
    #: Documents scored: stored, allowed by the filter, and not zero.
    considered: int
    #: Documents left out because their vector has no direction.
    skipped_zero: int
    #: Documents left out by the filter.
    filtered_out: int


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))


def exact_top_k(query: Sequence[float], vectors: Mapping[str, Sequence[float]], k: int, *,
                allowed: Optional[Callable[[str], bool]] = None) -> ExactNeighbours:
    """The ``k`` stored vectors most similar to ``query`` by cosine.

    Ties are broken by document id, so the ground truth is the same on
    every run and on every machine. ``allowed`` applies the same filter
    the index under test was given.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    query_norm = _norm(query)
    if query_norm == 0 or not math.isfinite(query_norm):
        raise ValueError("the query vector must be finite and not zero")
    scored: list[tuple[str, float]] = []
    skipped_zero = filtered_out = 0
    for document, vector in vectors.items():
        if len(vector) != len(query):
            raise ValueError(f"{document!r} has {len(vector)} dimensions; the query has {len(query)}")
        if allowed is not None and not allowed(document):
            filtered_out += 1
            continue
        norm = _norm(vector)
        if norm == 0:
            skipped_zero += 1
            continue
        dot = math.fsum(a * b for a, b in zip(query, vector))
        scored.append((document, dot / (query_norm * norm)))
    best = heapq.nsmallest(k, scored, key=lambda pair: (-pair[1], pair[0]))
    return ExactNeighbours(ranked=tuple(best), considered=len(scored),
                           skipped_zero=skipped_zero, filtered_out=filtered_out)
