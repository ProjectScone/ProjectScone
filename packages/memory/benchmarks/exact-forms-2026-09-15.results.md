# Exact word forms in the text lane (2026-09-15)

**Question.** The northstar sweep
([northstar-defaults-2026-09-14](northstar-defaults-2026-09-14.results.md))
traced an MRR drop to `85157c19`, "Count a query word once beside its
family": 0.8386 on its parent and 0.8074 on it, at the old vector weight.
That commit was a correctness fix. "billing" asked with the stem prefix
`bill` had been scored twice, once as the word and once inside the family.
Once the double count was gone, a passage holding the query's own word
form earned no more than a passage holding only a relative ("bills").
Can that credit come back without the double count, and should it be a
default?

**Answer.** Yes, on the samples measured. `SCONE_LEXICAL_EXACT_FORMS`
(`MemoryEngine(..., lexical_exact_forms=)`) keeps the family as one
saturating term. In a passage that holds one of the query's own words, the
family's count is weighed at that word's idf, not the family's. On the
frozen sample and the second sample it raised MRR and kept or raised R@5.
On a third sample, chosen before it was run, MRR also rose. It is now on
by default, and `SCONE_LEXICAL_EXACT_FORMS=0` turns it off.

| engine defaults, frozen n=50 (seed 42) | R@5 | all@5 | R@15 | MRR |
|---|---|---|---|---|
| before: exact forms off on this branch (main's ranking) | 0.90 | 0.76 | 1.00 | 0.8379 |
| **after: exact forms on** | **0.90** | **0.76** | **1.00** | **0.8444** |
| LlamaIndex 0.14.24 BM25 + vector, reciprocal rank | 0.88 | 0.74 | 0.98 | 0.8307 |

| engine defaults, 100 other items (seed 7, outside the frozen 50) | R@5 | all@5 | R@15 | MRR |
|---|---|---|---|---|
| before: exact forms off | 0.96 | 0.84 | 0.99 | 0.8854 |
| **after: exact forms on** | **0.97** | **0.87** | **0.99** | **0.9042** |
| LlamaIndex BM25 + vector | 0.91 | 0.71 | 0.99 | 0.8099 |

The gains are small on the frozen sample, where MRR rises 0.0065 and R@10
goes from 0.92 to 0.94. On the second sample MRR rises 0.019, R@5 by one
item and all-sessions@5 by three items. Not every item moved up. On the
frozen sample 4 items rose in reciprocal rank and 1 fell (from 1/11 to
1/12). On the second sample 6 rose and 4 fell: one from rank 1 to rank 2,
and three by one place further down.

## The signal

A family counts once: 85157c19 removed the word from beside its own
family, and that stays. What changes is the weight of that one term.

- A passage holding only "billing" scores what "billing" alone would
  score (the word's idf, its own count).
- A passage holding "billing" and "bills" scores their joint count at
  "billing"'s idf. That is the family's count at the word's weight, not
  the word followed by the family again.
- A passage holding only relatives scores exactly as it did.
- With two query words in one family, a passage takes the rarest one it
  holds.

The credit a passage gains is bounded by the gap between the word's idf
and the family's. It can never exceed the saturated score of one term at
the word's idf.

The in-memory scorer applies this directly. SQLite's `bm25()` cannot
weigh one phrase differently for different rows. So for each query word,
the store reads the family phrase's part of `bm25()` in rows that hold the
word: the `bm25()` of `"bill"* AND "billing"`, less that of `"billing"`.
Each phrase keeps its idf over the whole index, so the difference is that
part exactly. It then moves the part to the word's idf, computed as FTS5
computes it, floor included. Tests pin that both stores score an
exact-form passage as the word alone, and leave a relative's score
unchanged.

## What ran

- **Data.** `bench-data/longmemeval_s.json`. The frozen sample is the
  stratified `n=50, seed=42` sample. The second sample is `n=100, seed=7`,
  drawn from the 450 items outside it (`--holdout-of 42:50`). The third
  sample is `n=100, seed=11`, drawn from the 350 items outside both.
- **Both sides.** The hashed-token embedder, no model, no reranker. Chunks
  are 700 characters, fusion is by rank, and the hashed vector weight is
  0.01, which is today's defaults. Every session is one document. Both
  rankings are folded to distinct sessions and scored by
  `bench.comparative`'s rule, as in the northstar sweep.
- **Commands** (from `packages/memory`, `PYTHONPATH=$PWD/src`, Python 3.14):

```
python benchmarks/northstar_defaults.py --data ../../bench-data/longmemeval_s.json --n 50 --seed 42 --chunks 700 --fusions rank --weights 0.01 --diversities none --exact-forms off,on --check --out frozen.json
python benchmarks/northstar_defaults.py --data ../../bench-data/longmemeval_s.json --n 100 --seed 7 --holdout-of 42:50 --chunks 700 --fusions rank --weights 0.01 --diversities none --exact-forms off,on --check --out holdout.json
```

  `--exact-forms off,on` asks each item's engine both ways; the setting is
  an engine attribute read at every recall. `--check` ran `compare()` at
  engine defaults, which were still off at the time. Its numbers matched
  the off rows exactly: frozen 0.90 / 0.76 / 1.00 / 0.8379, second sample
  0.96 / 0.84 / 0.99 / 0.8854. These are the northstar results file's
  numbers at 0.01 on main, so the off rows are main's ranking; main itself
  was not re-run. After the default was turned on, `--exact-forms default
  --check` ran again on both samples. `compare()` gave 0.90 / 0.76 / 1.00 / 0.8444
  on the frozen sample and 0.97 / 0.87 / 0.99 / 0.9042 on the second, the
  on rows exactly. LlamaIndex gave 0.8307 and 0.8099 (0.8098 in the first
  `compare()` of the second sample, 0.8099 in its sweep row).
- **Lanes.** At weight 0.01 the fused rows equal the text lane alone on both
  samples, with the setting off and on, as the northstar file found. The
  vector lane alone is unchanged by the setting (0.6196 and 0.6775 MRR).

## Order of work, disclosed

The two named samples are not blind to the design. Before any code was
written, a scratch copy of the scorer was patched with candidate designs
and run at engine defaults on both samples. That copy reproduced the
engine's own numbers when set to today's behaviour.

| design (scratch scorer) | frozen R@5 / all@5 / R@15 / MRR | second sample R@5 / all@5 / R@15 / MRR |
|---|---|---|
| today (family once) | 0.90 / 0.76 / 1.00 / 0.8379 | 0.96 / 0.84 / 0.99 / 0.8854 |
| the old double count | 0.92 / 0.76 / 1.00 / 0.8467 | 0.95 / 0.83 / 0.99 / 0.8828 |
| half the double count | 0.90 / 0.76 / 1.00 / 0.8523 | 0.96 / 0.86 / 0.99 / 0.8967 |
| a quarter of the double count | 0.90 / 0.76 / 1.00 / 0.8406 | 0.96 / 0.87 / 0.99 / 0.8931 |
| relatives weighed at half inside the family | 0.88 / 0.74 / 1.00 / 0.8382 | 0.95 / 0.84 / 0.99 / 0.8806 |
| the family at twice its weight (a control) | 0.90 / 0.76 / 1.00 / 0.8200 | 0.97 / 0.80 / 0.99 / 0.8948 |
| the larger of word and family | 0.90 / 0.76 / 1.00 / 0.8394 | 0.96 / 0.86 / 0.99 / 0.9037 |
| **the family at the word's idf (shipped)** | **0.90 / 0.76 / 1.00 / 0.8444** | **0.97 / 0.87 / 0.99 / 0.9042** |

These results rule out a plain double count: it lost MRR on the second
sample. They also rule out weighing relatives down, which lost R@5 on
both. The control, the family at twice its weight, cost 0.018 MRR on
the frozen sample. On the second sample it raised MRR but lowered
all-sessions@5 by four items, so a heavier family is not the same lever.

A fraction of the double count also passed both samples, and at a half
it was ahead on the frozen sample. It was not taken, because it is part
of the double count 85157c19 removed: at that fraction a covered word
counts one and a half times. The shipped design counts it once.

The shipped design's scratch numbers equal the implementation's on both
samples, to four places. Before anything was committed, it was run once
on the third sample (100 items outside both named samples, seed 11). That
sample is the blind check:

| third sample, n=100 seed 11 (scratch scorer) | R@5 | all@5 | R@15 | MRR |
|---|---|---|---|---|
| today | 0.95 | 0.80 | 0.99 | 0.8546 |
| **the family at the word's idf** | **0.96** | **0.80** | **0.99** | **0.8702** |

## SQLite

The benches above run the in-memory stores. The frozen-sample check was
also run with `SqliteDocumentStore` (fused, engine defaults otherwise).
Off, it gave R@5 0.90, all@5 0.78, R@15 0.98 and MRR 0.8392. On, it gave
0.90, 0.78, 1.00 and 0.8444. SQLite's off numbers differ from the
in-memory store's. FTS5's `bm25()` uses its own idf formula and counts over
the whole index, which is one known difference; the cause was not
isolated.

A text-lane read costs more on SQLite with the setting on. The test store
was one space holding the frozen 50 items' sessions: 45,169 chunks, with
the lexical index fully synchronised. Each of the 50 questions called
`search_terms` with its stem prefixes and a limit of 60. The runs were
three interleaved rounds on a loaded machine (load average about 18), and
the figures are medians of the three rounds' medians, means and maxima:

| store | exact forms off | exact forms on |
|---|---|---|
| SQLite, as committed: median (mean, slowest) | 22.3 ms (24.7, 62.2) | 27.1 ms (34.8, 129.5) |
| in-memory `Bm25` over the same chunks: median | 25.5 ms | 26.3 ms |

The first SQLite form joined each family's `bm25()` as a subquery.
SQLite looked the prefix up again for every row it matched: the median
was 2,268 ms a query for ten questions, against 21 ms off. That form was
replaced before the default was turned on. The replacement returned the
same 60 chunks in the same order for all 50 questions, with scores within
2.4e-16.

## What is still open

- **The mechanism is not isolated.** The design table is consistent with
  the credit belonging to the word's own idf rather than to a heavier
  family, but no item was examined to say why this corpus rewards it.
- **Only English suffix rules make families**, so this moves nothing for
  a language the stem rules do not know.
- **Postgres, Elasticsearch and the other stores** take no prefixes, so
  they have no families and nothing here applies to them.
