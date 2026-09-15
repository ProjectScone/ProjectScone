# Recall and ingestion hot paths v1

On a fixed corpus of 504 files cut into 10,635 chunks, with hash embeddings and
200 fixed recalls, the median CPU time of one recall fell from **561 ms to
168 ms** on the in-memory store and from **524 ms to 378 ms** on SQLite.
Ingestion CPU time fell from **6.50 s to 4.13 s** (in-memory) and from **7.26 s
to 5.00 s** (SQLite). All 400 recalls returned the same chunk ids, scores,
similarities, lanes, spans and declarations, to the last bit, before and after.
The in-memory text index costs more memory for it: **31.1 MB became 57.0 MB**
at this corpus size.

This measures Scone's own Python on the built-in stores with the hash embedder.
It says nothing about answer quality, about a model-backed embedder's cost, or
about Qdrant, Postgres or any other adapter.

## How it was measured

`benchmarks/hot_paths.py` reads `packages/memory/docs` and `packages/memory/src`
as text from one pinned revision (`acc6492e`), never from the working tree, so
the code under test cannot change what it is measured on. It ingests every file
in batches of 32 through `MemoryEngine.remember_many`, with one fixed record
timestamp and one fixed engine clock, into a fresh `InMemoryDocumentStore` +
`InMemoryVectorIndex` and then into a fresh SQLite database. It runs five
warm-up recalls, then 200 recalls of two to five words drawn from the corpus by
a seeded generator, at `limit=10`. It reports wall and CPU seconds for the
ingestion, chunks per second, and nearest-rank p50/p95 per recall; `--dump`
keeps every recall's output and `--profile` writes cProfile stats.

```sh
cd packages/memory
PYTHONPATH=src python benchmarks/hot_paths.py --json metrics.json --dump recalls.json
```

"Before" (A) is `acc6492e` exactly, extracted to a separate directory and
checked file for file against `git archive`. "After" (B) is the same revision
with this change. Both ran the same copy of the script with the same corpus and
queries, one process at a time, on an Apple M3 Max (14 cores, 36 GB) under
Python 3.14.7, interleaved A1 B1 B2 A2 A3 B3 between 00:23 and 01:10 CDT on
15 September 2026. Each run measured both stores.

The machine was shared with other jobs throughout, so wall-clock times moved by
up to 2.4x between runs of the same tree (A2's in-memory ingestion took 21.9 s,
A1's 15.8 s, at the same CPU time). CPU time (`time.process_time`, this process
only) is the steadier comparison, and the tables lead with it. Every run
reported zero degraded recalls, so no latency below is a lane that did not run.

## Results: medians of three interleaved runs

In-memory store:

| Metric | Before runs (A1, A2, A3) | Before | After runs (B1, B2, B3) | After | After / before |
|---|---|---:|---|---:|---:|
| Recall CPU p50, ms | 558.8, 574.6, 561.0 | 561.0 | 168.3, 175.4, 158.9 | 168.3 | 0.30 |
| Recall CPU p95, ms | 649.1, 680.1, 664.2 | 664.2 | 248.2, 275.0, 245.0 | 248.2 | 0.37 |
| Ingestion CPU, s | 6.50, 6.50, 6.37 | 6.50 | 4.13, 4.28, 3.95 | 4.13 | 0.64 |
| Chunks per CPU second | 1,636.9, 1,636.2, 1,670.1 | 1,636.9 | 2,572.1, 2,485.5, 2,692.2 | 2,572.1 | 1.57 |
| Recall wall p50, ms | 873.0, 1,732.7, 1,295.1 | 1,295.1 | 305.9, 458.6, 295.3 | 305.9 | 0.24 |
| Recall wall p95, ms | 1,865.9, 3,481.7, 2,163.5 | 2,163.5 | 898.9, 938.4, 826.2 | 898.9 | 0.42 |
| Ingestion wall, s | 15.77, 21.91, 12.76 | 15.77 | 9.59, 11.48, 13.66 | 11.48 | 0.73 |
| Chunks per wall second | 674.5, 485.4, 833.3 | 674.5 | 1,109.4, 926.6, 778.8 | 926.6 | 1.37 |

SQLite store:

| Metric | Before runs (A1, A2, A3) | Before | After runs (B1, B2, B3) | After | After / before |
|---|---|---:|---|---:|---:|
| Recall CPU p50, ms | 524.4, 537.9, 506.8 | 524.4 | 367.2, 383.9, 377.5 | 377.5 | 0.72 |
| Recall CPU p95, ms | 602.3, 614.7, 576.6 | 602.3 | 420.9, 438.0, 446.6 | 438.0 | 0.73 |
| Ingestion CPU, s | 7.26, 7.73, 7.20 | 7.26 | 5.00, 5.31, 4.79 | 5.00 | 0.69 |
| Chunks per CPU second | 1,464.2, 1,376.3, 1,476.6 | 1,464.2 | 2,126.9, 2,004.1, 2,221.4 | 2,126.9 | 1.45 |
| Recall wall p50, ms | 929.9, 1,390.7, 871.8 | 929.9 | 730.1, 1,422.0, 911.1 | 911.1 | 0.98 |
| Recall wall p95, ms | 1,936.7, 2,450.8, 1,936.6 | 1,936.7 | 1,327.1, 2,537.6, 2,769.4 | 2,537.6 | **1.31** |
| Ingestion wall, s | 12.59, 29.68, 14.52 | 14.52 | 12.03, 16.68, 10.80 | 12.03 | 0.83 |
| Chunks per wall second | 844.9, 358.3, 732.4 | 732.4 | 883.7, 637.6, 984.7 | 883.7 | 1.21 |

SQLite's wall-clock recall did not improve in these runs: its p50 median is
level and its p95 median is worse, while its CPU time per recall fell by more
than a quarter in every run. B2 and B3 ran while other jobs held the machine,
and wall time on a loaded machine includes time waiting for a core; this run
cannot separate that from a real regression, so it is reported, not explained
away. A quiet machine is needed to settle it.

## The same answers

`--dump` wrote, for each of the 200 recalls on each store, every returned
item's chunk id, episode id, `repr` of its score and similarity, lane ranks,
character span, line range, declaration and superseded flag, plus the recall's
fact ids, degraded lanes, top similarity and space size. 4,000 items in all.
The six dumps (A1-A3, B1-B3) are byte-identical: MD5
`830c17dcb29416d7884ddd20ca6be316` for every one.

## Where the time went

cProfile of the before tree, one run of each phase (the profiler inflates
Python-level calls, so these seconds are shares, not latencies):

| Phase | Profiled total | Largest Scone hot spot |
|---|---:|---|
| In-memory recall, 200 queries | 326.8 s | `InMemoryVectorIndex._cosine`: 285.5 s (87%), both norms recomputed for every point on every query |
| SQLite recall, 200 queries | 314.5 s | `SqliteVectorIndex._scored`: 274.5 s (87%), both norms recomputed per row through a generator |
| In-memory ingestion | 11.1 s | `code._python` declaration walk 3.6 s (32%); value checks on vectors 2.1 s (19%); `HashEmbedder._one` 2.1 s (19%); `byte_spans` 1.0 s (9%) |
| SQLite ingestion | 12.6 s | the same order; `sqlite3.Connection.execute` (C) 1.2 s |

The declaration walk also ran at recall: naming a recalled code chunk's
declaration parses its whole file, and only the last eight files' declarations
are kept (`MAX_PARSED`), so 1,388 of 1,780 lookups parsed a file again. That was
23.9 s (7%) of the in-memory recall profile.
The in-memory BM25 search, which scored every document, was 12.7 s (4%).

The three hot spots, and what was done to each:

1. **Cosine scoring in both built-in vector indexes.** The in-memory index now
   keeps each point's norm from when it was written and computes one dot
   product per point per query. SQLite unpacks each stored vector to floats once
   and takes its norm and dot product with `sum(map(operator.mul, ...))`.
   `sum` adds the same products in the same order as the generators did, so
   every score is identical; `math.sumprod` was avoided because it rounds
   differently and would reorder near-ties.
2. **The Python declaration walk.** A definition is a statement, and statements
   only stand in the bodies of statements, exception handlers and match cases,
   so the walk no longer descends into expressions, which are most of a tree.
   Line starts come from one regular expression instead of a character loop.
3. **Per-value checks on every embedding.** When every value in a vector is a
   plain `float` or `int`, finiteness is checked in one `map(math.isfinite)`
   pass; anything else (a bool, a subclass, a numpy scalar) still goes value by
   value under the old rule. `validate_vector` uses the same `map` form.

Three more, smaller: the hash embedder hashes each distinct token once per text
and remembers the bucket and sign of the last 65,536 distinct tokens (a cache,
not a bound on output: an evicted token is hashed again to the same place);
`byte_spans` encodes only the stretches between span ends rather than every
character alone; and the in-memory BM25 index keeps, for every term, the
documents holding it, and scores only documents holding a query term or family
member, since every other document scores zero and was never returned.

Each change has a test holding it to the old formula written out the slow way:
`tests/backends/test_vector_scoring.py`, `tests/ingestion/test_declaration_walk.py`,
`tests/ingestion/test_vector_value_checks.py`,
`tests/ingestion/test_byte_span_conversion.py` and
`tests/retrieval/test_lexical_candidates.py`, several of them over this
package's own files. `tests/benchmarks/test_hot_paths.py` holds the benchmark
to asking the same queries and recalling the same way twice.

## What it cost

Postings are sets of document ids. `tracemalloc` over a `Bm25` holding the
10,635 chunk texts of this corpus measured 31,117,626 bytes before and
56,964,226 bytes after: 83% more. Measured in place instead, by summing
`sys.getsizeof` over the in-memory store's text indexes after the benchmark's
own ingestion, the store's text lane went from 30,776,080 to 56,622,730 bytes.
The store keeps a second `Bm25` per space for chunks' context text; it held no
documents on this corpus, and where context text is written it carries postings
too. A deployment holding many in-memory spaces pays the growth per space. Lists
would be smaller than sets but make removing a document cost the length of every
posting it appears in; that trade was not measured.

The in-memory vector index's kept norms took 848,012 bytes for 10,635 points.

## Where the time goes now

A cProfile of the after tree (a seventh run, whose dump has the same MD5 as the
six above) ran while the machine's load average was about 60, and cProfile times
by the wall clock, so only its shares are worth reading, and loosely:

| Phase | Largest shares after the change |
|---|---|
| In-memory recall | vector search 56%; naming declarations 32%; BM25 search 8% |
| SQLite recall | `_scored` 76%; naming declarations 16% |
| In-memory ingestion | text index `add` (mostly tokenizing) 34%; declaration walk 29%; hash embedding 21% |
| SQLite ingestion | declaration walk 33%; hash embedding 25% |

## Not done, and why

- **Recall still re-parses source files to name declarations.** The walk is
  cheaper, but `MAX_PARSED` still keeps only eight files. A larger cache holds
  whole file contents as its keys, so raising it trades memory for time, and
  that trade needs its own measurement across corpus sizes rather than a number
  picked to suit this corpus.
- **SQLite still scores every row in Python.** Scoring inside SQLite or with an
  approximate index would change which chunks come back or how ties break;
  vectorising with numpy would change the rounding of every score. Either breaks
  the rule that results stay identical.
- **A word family scans the whole vocabulary** (`term.startswith(prefix)` over
  every term). A sorted vocabulary with bisection would remove it; it was not a
  top hot spot here and was left alone.
