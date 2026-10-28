"""Where a space's vectors sit, and cosine measured from their centre.

Embedding models do not spread text evenly over the sphere: their
vectors share a common direction, so two unrelated passages already
score well above zero and the useful differences between candidates
are squeezed into a narrow band at the top. Subtracting the corpus mean
before taking the cosine (Su et al., 2021, call the full version
whitening; centring is its first step) removes that shared direction.

Whether it helps depends on the model, which is why ``anisotropy`` is
here too: the mean cosine of random pairs says how much shared direction
there is to remove. A model whose random pairs already average near zero
has nothing for centring to take away.

The centre is the mean of **unit** vectors, because cosine compares
directions; a long vector would otherwise pull the centre towards
itself. ``Centre`` values merge, so one can be kept up to date as
passages are written, or built per shard and combined.

Measured (BEIR DBpedia-Entity, 381 judged queries over 999,000 entity
abstracts, exact search, two embedders): random pairs averaged 0.082
(embeddinggemma, 768-d) and 0.078 (text-embedding-3-large, 3072-d), and
centring cost nDCG@10 0.0029 with both (81/111 queries, p=0.04; 93/99,
p=0.7). Re-ordering the raw top 200 with ``recentred`` gave the same
answer as a full centred search. Recall does not centre.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, TypeVar

Key = TypeVar("Key")


@dataclass(frozen=True)
class Centre:
    """The mean direction of a set of vectors."""

    mean: tuple[float, ...]
    #: Vectors averaged.
    count: int
    #: Zero vectors left out: they have no direction to average.
    skipped_zero: int = 0


def _unit(vector: Sequence[float]) -> Optional[list[float]]:
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not math.isfinite(norm):
        raise ValueError("a vector holds a non-finite value")
    return [value / norm for value in vector] if norm else None


def centre_of(vectors: Iterable[Sequence[float]]) -> Optional[Centre]:
    """The mean of the unit vectors; ``None`` when no vector had a direction."""
    total: list[float] = []
    count = skipped = 0
    for vector in vectors:
        unit = _unit(vector)
        if unit is None:
            skipped += 1
            continue
        if not total:
            total = [0.0] * len(unit)
        elif len(unit) != len(total):
            raise ValueError(f"a vector has {len(unit)} dimensions; the first had {len(total)}")
        for index, value in enumerate(unit):
            total[index] += value
        count += 1
    if not count:
        return None
    return Centre(mean=tuple(value / count for value in total), count=count, skipped_zero=skipped)


def merge_centres(first: Centre, second: Centre) -> Centre:
    """The centre of two disjoint sets, from their centres alone."""
    if len(first.mean) != len(second.mean):
        raise ValueError("the centres have different dimensions")
    count = first.count + second.count
    mean = tuple((a * first.count + b * second.count) / count for a, b in zip(first.mean, second.mean))
    return Centre(mean=mean, count=count, skipped_zero=first.skipped_zero + second.skipped_zero)


def centred_cosine(query: Sequence[float], vector: Sequence[float], centre: Centre) -> Optional[float]:
    """Cosine between ``query`` and ``vector`` once the centre is taken
    from both. ``None`` when either lies at the centre, where it has no
    direction left to compare."""
    if not len(query) == len(vector) == len(centre.mean):
        raise ValueError("the query, the vector and the centre must have the same dimensions")
    unit_query, unit_vector = _unit(query), _unit(vector)
    if unit_query is None or unit_vector is None:
        return None
    shifted_query = [value - mean for value, mean in zip(unit_query, centre.mean)]
    shifted_vector = [value - mean for value, mean in zip(unit_vector, centre.mean)]
    norms = math.sqrt(math.fsum(v * v for v in shifted_query)) * math.sqrt(math.fsum(v * v for v in shifted_vector))
    if norms < 1e-12:
        return None
    return math.fsum(a * b for a, b in zip(shifted_query, shifted_vector)) / norms


def recentred(query: Sequence[float], candidates: Mapping[Key, Sequence[float]],
              centre: Centre) -> list[tuple[Key, float]]:
    """Candidates ordered by centred cosine, best first. A candidate at the
    centre has no centred direction and goes last, in the order given."""
    scored: list[tuple[Key, float]] = []
    undirected: list[Key] = []
    for key, vector in candidates.items():
        score = centred_cosine(query, vector, centre)
        if score is None:
            undirected.append(key)
        else:
            scored.append((key, score))
    scored.sort(key=lambda pair: -pair[1])
    return scored + [(key, -math.inf) for key in undirected]


@dataclass(frozen=True)
class Anisotropy:
    """How much direction a set of vectors shares."""

    #: Mean cosine over the sampled pairs: near zero is evenly spread.
    mean_cosine: Optional[float]
    pairs: int
    seed: int


def anisotropy(vectors: Sequence[Sequence[float]], *, pairs: int = 2000, seed: int = 0) -> Anisotropy:
    """Mean cosine of random distinct pairs, drawn with a fixed seed.

    Zero vectors are left out before drawing. Fewer than two directed
    vectors give no pair and a ``None`` mean.
    """
    if type(pairs) is not int or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    units = [unit for unit in (_unit(vector) for vector in vectors) if unit is not None]
    if len(units) < 2:
        return Anisotropy(mean_cosine=None, pairs=0, seed=seed)
    rng = random.Random(seed)
    total = 0.0
    for _ in range(pairs):
        left, right = rng.sample(range(len(units)), 2)
        total += math.fsum(a * b for a, b in zip(units[left], units[right]))
    return Anisotropy(mean_cosine=total / pairs, pairs=pairs, seed=seed)
