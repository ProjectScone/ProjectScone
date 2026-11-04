# Measured retrieval modules that recall leaves off

Some mechanisms from the retrieval literature live in
[`retrieval/`](../src/scone_memory/retrieval/) as plain functions over
scores and vectors, with no store, model or network behind them. Each was
measured before anything used it, and none earned a place in the default
recall path, so `recall` does not call them. They are kept so that an
application can measure them on its own collection, where the result may
differ, and build on them without reimplementing the method. Each module's
docstring records what was measured, on what, and what came of it.

## Hubness and the CSLS penalty

In a high-dimensional space a few points sit close to many others. In
retrieval they turn up in the results of query after query whatever was
asked, because they are central, not because they answer.
[`retrieval/hubness.py`](../src/scone_memory/retrieval/hubness.py) measures
that and offers the standard remedy.

```python
from scone_memory.retrieval.hubness import hubness, pool_density, hub_penalised

# Five queries' top-3 lists over a collection of 8 passages; p1 is in every one.
results = [["p1", "p2", "p3"], ["p1", "p4", "p2"], ["p1", "p5", "p6"],
           ["p1", "p3", "p7"], ["p1", "p2", "p4"]]
report = hubness(results, population=8)
# Hubness(skewness=0.95..., max_occurrence=5, appeared=7, population=8, lists=5)

# One query's candidates, re-ordered by similarity less neighbourhood density.
similarity = {"p1": 0.82, "p2": 0.80, "p3": 0.74}
vectors = {"p1": [0.6, 0.6, 0.5], "p2": [0.9, 0.1, 0.2], "p3": [0.2, 0.9, 0.1]}
density = pool_density(vectors, k=2)
hub_penalised(similarity, density, weight=0.5)
# [('p2', 0.52...), ('p3', 0.46...), ('p1', 0.43...)]
```

`hubness` reads the skew of the k-occurrence distribution: how many of the
given result lists each passage appears in. Passages that never appeared
count as zero, because the question is how unevenly the whole collection
is found, so `population` must be the size of the collection. A skew near
zero means passages are found evenly; a large positive skew means a long
tail of hubs. A large skew can also come from most passages never being
found at all, which is why `max_occurrence` and `appeared` are reported
beside it: read them before concluding there are hubs.

`hub_penalised` orders candidates by `similarity - weight * density`, the
order of CSLS (cross-domain similarity local scaling) at `weight` 0.5. A
hub is dense by definition, so it pays for being near everything; an
ordinary passage pays little. The density can come from the whole
collection, which is what CSLS defines, or from the query's own candidates
with `pool_density`, which costs `n * (n - 1) / 2` dot products and no
extra search. A candidate with no density (a zero vector, or one missing
from the map) is charged the mean penalty of the others, so a missing
density neither rewards nor punishes it. Equal scores keep the order
given.

On the collection it was measured on there were no hubs to correct, and
every weight tried lowered ranking quality slightly, so recall does not
use it.
