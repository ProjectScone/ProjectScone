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

## Predicting a result list's quality

A recall cannot see whether what it found is relevant, but it can see the
shape of what it found. Query performance prediction reads that shape.
[`retrieval/qpp.py`](../src/scone_memory/retrieval/qpp.py) computes the
usual predictors for one result list.

```python
from scone_memory.retrieval.geometry import centre_of
from scone_memory.retrieval.qpp import predict

scores = [0.82, 0.80, 0.61, 0.58, 0.57, 0.55, 0.54, 0.52, 0.51, 0.50, 0.49, 0.47]
top = [[0.6, 0.6, 0.5], [0.9, 0.1, 0.2], [0.2, 0.9, 0.1]]   # the top results' vectors
prediction = predict(scores, k=10, vectors=top)
# Prediction(dispersion=0.109..., nqc=0.189..., wig=0.02..., gap=0.32..., coherence=0.626...,
#            k=10, depth=12, coherence_over=3)

# Coherence measured from the collection's centre rather than the origin.
centre = centre_of(sample_of_stored_vectors)
predict(scores, k=10, vectors=top, centre=centre).coherence
```

`scores` are best first and as deep as the recall read them; the reference
mean is taken over all of them. From the top `k`:

- `dispersion` is the spread (standard deviation) of the top scores.
- `nqc` is that spread over the reference mean. NQC divides by a corpus
  score; a dense index has none to give, so the mean of the list read
  stands in for it.
- `wig` is how far the top scores stand above the reference mean, without
  the query-length term, which belongs to term-matching scores.
- `gap` is the top score less the `k`-th (or the last, with fewer).
- `coherence` is the mean cosine between the top results' own vectors:
  results that agree with one another are about one thing, results
  scattered over the space found nothing in particular. In an anisotropic
  space any two vectors have a high cosine, so a `Centre`
  ([`retrieval/geometry.py`](../src/scone_memory/retrieval/geometry.py))
  can be given to measure agreement from the collection's centre.

A signal is `None` where there was too little to read: fewer than two
scores for the score signals, fewer than two vectors with a direction for
`coherence`, and a zero reference mean for `nqc`. `k` must be at least 2,
and scores that are not finite or not best first are refused.
`coherence_over` says how many vectors the coherence was measured over.

Classical predictors transfer poorly to dense retrieval, so none of these
is trusted here. On the collection measured, the score signals ordered
queries by quality about equally well, and better than coherence, centred
or not; but the agreement was moderate. That is enough to report beside
an answer, not enough to act on alone: gating query feedback on them did
not help ([`retrieval/feedback.py`](../src/scone_memory/retrieval/feedback.py)).
