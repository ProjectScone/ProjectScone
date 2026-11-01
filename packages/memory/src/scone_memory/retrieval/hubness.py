"""Hubs: passages that are near everything, and so near every query.

In a high-dimensional space a few points sit close to many others
(Radovanović et al., 2010). In retrieval they turn up in the results of
query after query whatever was asked, because they are central, not
because they answer. The standard measure is the skew of the
k-occurrence distribution: how often each passage appears in some
query's top ``k``. Evenly used passages give a skew near zero; a long
tail of hubs gives a large positive one.

The remedy borrowed here is CSLS (Lample et al., 2018): rank by
similarity less the passage's neighbourhood density, the mean similarity
to its ``k`` nearest neighbours. A hub is dense by definition, so it
pays for being near everything; an ordinary passage pays little. CSLS
writes it as ``2 * cos - r_query - r_passage``; within one query
``r_query`` is the same for every candidate, so the order is that of
``cos - r_passage / 2`` and ``weight`` scales the penalty from there.

The density can come from the whole space, which is what CSLS defines,
or from the candidates of the query itself (``pool_density``), which
costs no extra search and is what a recall can afford. Which is good
enough is a measurement, not something this module decides.

Measured (BEIR DBpedia-Entity, 999,000 abstracts, exact search, two
embedders, the weight chosen on the other half of the queries): there
were no hubs to correct. Over 1,000 further queries no abstract was in
more than 3 top-10 lists; the skew of about 10 comes from the roughly
989,000 that appeared in none, which is why ``Hubness`` reports the
maximum and the count beside it. The smallest weight tried (0.1) was
chosen every time and still did not help: at 768-d nDCG@10 fell 0.0004
to 0.0013 (significant with the in-pool density, p=0.001 and 0.02), at
3072-d it moved +0.0005 to -0.0019 (none significant). Larger weights
only did more harm. Recall does not use it.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Optional, Sequence, TypeVar

Key = TypeVar("Key", bound=Hashable)


@dataclass(frozen=True)
class Hubness:
    """How unevenly passages appear across many queries' results."""

    #: Skew of the k-occurrence distribution over the whole population.
    skewness: Optional[float]
    #: Most appearances by any one passage.
    max_occurrence: int
    #: Passages that appeared in at least one result.
    appeared: int
    population: int
    lists: int


def k_occurrence(results: Iterable[Sequence[Key]]) -> Counter[Key]:
    """How many results each passage appears in; once per result."""
    counts: Counter[Key] = Counter()
    for result in results:
        counts.update(set(result))
    return counts


def hubness(results: Sequence[Sequence[Key]], population: int) -> Hubness:
    """Skew of k-occurrence across ``population`` passages.

    Passages that never appeared count as zero occurrences: leaving them
    out would describe only the passages that were found, and the skew
    is about how unevenly the whole collection is found.
    """
    counts = k_occurrence(results)
    if type(population) is not int or population < len(counts):
        raise ValueError(f"population must be an integer of at least the {len(counts)} passages that appeared")
    if population == 0:
        return Hubness(None, 0, 0, 0, len(results))
    absent = population - len(counts)
    mean = sum(counts.values()) / population
    second = (sum((count - mean) ** 2 for count in counts.values()) + absent * mean ** 2) / population
    third = (sum((count - mean) ** 3 for count in counts.values()) + absent * (-mean) ** 3) / population
    skewness = third / second ** 1.5 if second > 0 else None
    return Hubness(skewness=skewness, max_occurrence=max(counts.values(), default=0), appeared=len(counts),
                   population=population, lists=len(results))


def neighbourhood_density(similarities: Iterable[float], k: int) -> Optional[float]:
    """Mean of the ``k`` highest similarities: CSLS's ``r``. ``None`` with
    no neighbour to average."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    nearest = heapq.nlargest(k, similarities)
    return math.fsum(nearest) / len(nearest) if nearest else None


def pool_density(vectors: Mapping[Key, Sequence[float]], k: int) -> dict[Key, Optional[float]]:
    """Each candidate's density among the other candidates of one query.

    It costs ``n * (n - 1) / 2`` dot products and no search, so a recall
    can afford it over its candidates. A zero vector has no density.
    """
    keys = list(vectors)
    units: dict[Key, list[float]] = {}
    for key in keys:
        norm = math.sqrt(math.fsum(value * value for value in vectors[key]))
        if norm:
            units[key] = [value / norm for value in vectors[key]]
    likeness: dict[Key, list[float]] = {key: [] for key in units}
    directed = list(units)
    for index, left in enumerate(directed):
        for right in directed[index + 1:]:
            similarity = math.fsum(a * b for a, b in zip(units[left], units[right]))
            likeness[left].append(similarity)
            likeness[right].append(similarity)
    return {key: neighbourhood_density(likeness[key], k) if key in likeness else None for key in keys}


def hub_penalised(similarity: Mapping[Key, float], density: Mapping[Key, Optional[float]], *,
                  weight: float = 0.5) -> list[tuple[Key, float]]:
    """Candidates ordered by ``similarity - weight * density``, best first.

    ``weight`` 0.5 is CSLS's order. A candidate with no density keeps
    its similarity less the mean penalty of the others, so missing a
    density neither rewards nor punishes it; equal scores keep the order
    given.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be a finite, non-negative number")
    known = [value for key in similarity if (value := density.get(key)) is not None]
    typical = math.fsum(known) / len(known) if known else 0.0
    scored = []
    for position, (key, score) in enumerate(similarity.items()):
        value = density.get(key)
        scored.append((key, score - weight * (typical if value is None else value), position))
    scored.sort(key=lambda entry: (-entry[1], entry[2]))
    return [(key, score) for key, score, _ in scored]
