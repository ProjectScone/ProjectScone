# North star, first row: retrieval defaults against LlamaIndex's best (2026-09-14)

**Question.** The scoreboard's first row had the engine's defaults behind
LlamaIndex's BM25+vector fusion on LongMemEval-S. Which of the levers the
engine already has -- fusion mode, the vector lane's weight, chunk size,
diversity -- beats it on R@5 **and** MRR, and does that hold on items it
was not chosen on?

**Answer.** Lowering the hashed-token embedder's vector weight from 0.25 to
0.01 (rank fusion, 700-character chunks, no diversity) beats the reference
on all four numbers on the frozen sample. On 100 items outside it, it beats
the reference on three and ties on R@15. It
is now the default for hashed embedders; `SCONE_VECTOR_WEIGHT=0.25`
restores the old one, and any other embedder keeps 1.0.

| engine defaults, frozen n=50 | R@5 | all-sessions@5 | R@15 | MRR |
|---|---|---|---|---|
| before: main `acc6492e`, hashed vector weight 0.25 | 0.88 | 0.72 | 0.98 | 0.807 |
| **after: hashed vector weight 0.01** | **0.90** | **0.76** | **1.00** | **0.838** |
| LlamaIndex 0.14.24 BM25Retriever + VectorIndexRetriever, reciprocal rank | 0.88 | 0.74 | 0.98 | 0.831 |

| engine defaults, 100 other items | R@5 | all-sessions@5 | R@15 | MRR |
|---|---|---|---|---|
| before (0.25) | 0.95 | 0.79 | 0.99 | 0.852 |
| **after (0.01)** | **0.96** | **0.84** | **0.99** | **0.885** |
| LlamaIndex BM25 + vector | 0.91 | 0.71 | 0.99 | 0.810 |

The margins on the frozen sample are small: R@5 is one item of 50 and
all-sessions@5 one item. The second sample is where the change is
visible: +0.05 all-sessions@5 and +0.033 MRR over the old default, and
+0.075 MRR over the reference.

## What ran

- **Data.** `bench-data/longmemeval_s.json`. Frozen sample: the Rust
  harness's stratified sample, `n=50`, `seed=42` (every item has evidence).
  Second sample: `n=100`, `seed=7`, drawn from the 450 items outside the
  frozen 50 (`--holdout-of 42:50`).
- **Both sides.** The hashed-token embedder `hash-256-t3-u16.0.0`, no
  model, no reranker, no query rewriting. Every session is one document.
  The reference is `compare(hybrid=True)`: LlamaIndex's `SentenceSplitter`
  (512 tokens, no overlap), `VectorStoreIndex`, and `QueryFusionRetriever`
  over its vector and BM25 retrievers by reciprocal rank, every node ranked.
  Ours is `MemoryEngine` over in-memory stores, asked for 30 passages
  (15 x the per-episode cap). Both rankings are folded to distinct
  sessions and scored by `bench.comparative`'s rule.
- **Sweep.** `benchmarks/northstar_defaults.py`. It ingests each item once
  per chunk size and asks once per setting. Fusion and diversity are recall
  arguments. The vector weight is the engine attribute every recall reads,
  so the script sets it between recalls. `--check` also runs `compare()` at
  engine defaults. On both samples its numbers matched the sweep's 0.25
  row exactly: frozen 0.88 / 0.72 / 0.98 / 0.8074, second sample 0.95 /
  0.79 / 0.99 / 0.852.
- **Commands** (from `packages/memory`, `PYTHONPATH=$PWD/src`, Python 3.14.7):

```
python benchmarks/northstar_defaults.py --data ../../bench-data/longmemeval_s.json --n 50 --seed 42 --diversities none --check --out sweep-a.json
python benchmarks/northstar_defaults.py --data ../../bench-data/longmemeval_s.json --n 50 --seed 42 --chunks 700 --weights 0.01,0.02 --diversities none --out sweep-c.json
python benchmarks/northstar_defaults.py --data ../../bench-data/longmemeval_s.json --n 100 --seed 7 --holdout-of 42:50 --diversities none --check --out holdout.json
python benchmarks/northstar_defaults.py --data ../../bench-data/longmemeval_s.json --n 100 --seed 7 --holdout-of 42:50 --chunks 700 --weights 0.01,0.02 --diversities none --out holdout-c.json
python benchmarks/northstar_defaults.py --data ../../bench-data/longmemeval_s.json --n 50 --seed 42 --chunks 700 --fusions rank,score --weights 0.01,0.25 --diversities 0.3 --out sweep-d.json
```

Defaults `--chunks 700,2000 --fusions rank,score,distribution --weights 0.05,0.1,0.25,0.5,1.0`.

**Order of work.** This is disclosed because it matters for the second sample. The
first sweep covered weights 0.05 to 1.0. At 700 characters, only one row
beat the reference on the frozen sample's R@5 and MRR, and it used
relative-score fusion. The second sample then showed the text lane alone
with a higher MRR than every fused row at 700 characters. The weights 0.01 and 0.02 were added after that, and run
on both samples. The default was chosen by the rule set in advance: beat
the reference on the frozen sample's R@5 and MRR. The second sample is
therefore a check on a direction it helped suggest, not a blind holdout.

## Every row

Frozen n=50 on the left, the 100 other items on the right. Rows without
a chunk size are the reference. `lanes=` rows run one lane alone.

| configuration | R@5 | all@5 | R@15 | MRR | holdout R@5 | all@5 | R@15 | MRR |
|---|---|---|---|---|---|---|---|---|
| llamaindex default (vector, chunk 512 tokens) | 0.66 | 0.46 | 0.92 | 0.573 | 0.82 | 0.49 | 0.94 | 0.639 |
| llamaindex hybrid (BM25+vector, RRF, chunk 512 tokens) | 0.88 | 0.74 | 0.98 | 0.831 | 0.91 | 0.71 | 0.99 | 0.810 |
| chunk=700 fusion=rank vector_weight=0.01 | 0.90 | 0.76 | 1.00 | 0.838 | 0.96 | 0.84 | 0.99 | 0.885 |
| chunk=700 fusion=rank vector_weight=0.02 | 0.90 | 0.74 | 1.00 | 0.838 | 0.96 | 0.84 | 0.99 | 0.875 |
| chunk=700 fusion=rank vector_weight=0.05 | 0.88 | 0.72 | 0.98 | 0.833 | 0.95 | 0.84 | 0.99 | 0.864 |
| chunk=700 fusion=rank vector_weight=0.1 | 0.88 | 0.72 | 0.98 | 0.823 | 0.95 | 0.84 | 0.99 | 0.858 |
| chunk=700 fusion=rank vector_weight=0.25 | 0.88 | 0.72 | 0.98 | 0.807 | 0.95 | 0.79 | 0.99 | 0.852 |
| chunk=700 fusion=rank vector_weight=0.5 | 0.86 | 0.74 | 0.96 | 0.788 | 0.94 | 0.78 | 0.99 | 0.825 |
| chunk=700 fusion=rank vector_weight=1.0 | 0.84 | 0.72 | 0.92 | 0.759 | 0.93 | 0.74 | 0.98 | 0.789 |
| chunk=700 fusion=score vector_weight=0.01 | 0.90 | 0.74 | 1.00 | 0.838 | 0.96 | 0.84 | 0.99 | 0.885 |
| chunk=700 fusion=score vector_weight=0.02 | 0.90 | 0.74 | 1.00 | 0.838 | 0.96 | 0.84 | 0.99 | 0.885 |
| chunk=700 fusion=score vector_weight=0.05 | 0.88 | 0.72 | 1.00 | 0.837 | 0.96 | 0.84 | 0.99 | 0.884 |
| chunk=700 fusion=score vector_weight=0.1 | 0.88 | 0.72 | 0.98 | 0.836 | 0.95 | 0.85 | 0.99 | 0.878 |
| chunk=700 fusion=score vector_weight=0.25 | 0.90 | 0.74 | 0.96 | 0.834 | 0.95 | 0.82 | 0.99 | 0.868 |
| chunk=700 fusion=score vector_weight=0.5 | 0.88 | 0.72 | 0.94 | 0.823 | 0.95 | 0.82 | 0.99 | 0.875 |
| chunk=700 fusion=score vector_weight=1.0 | 0.86 | 0.72 | 0.92 | 0.817 | 0.95 | 0.79 | 0.98 | 0.843 |
| chunk=700 fusion=distribution vector_weight=0.01 | 0.90 | 0.74 | 1.00 | 0.838 | 0.96 | 0.84 | 0.99 | 0.854 |
| chunk=700 fusion=distribution vector_weight=0.02 | 0.90 | 0.74 | 1.00 | 0.838 | 0.96 | 0.84 | 0.99 | 0.854 |
| chunk=700 fusion=distribution vector_weight=0.05 | 0.88 | 0.72 | 0.98 | 0.836 | 0.96 | 0.84 | 0.99 | 0.854 |
| chunk=700 fusion=distribution vector_weight=0.1 | 0.88 | 0.72 | 0.98 | 0.826 | 0.95 | 0.84 | 0.99 | 0.853 |
| chunk=700 fusion=distribution vector_weight=0.25 | 0.90 | 0.74 | 0.96 | 0.817 | 0.95 | 0.83 | 0.99 | 0.857 |
| chunk=700 fusion=distribution vector_weight=0.5 | 0.90 | 0.76 | 0.92 | 0.803 | 0.95 | 0.80 | 0.99 | 0.851 |
| chunk=700 fusion=distribution vector_weight=1.0 | 0.86 | 0.74 | 0.92 | 0.772 | 0.93 | 0.76 | 0.98 | 0.827 |
| chunk=2000 fusion=rank vector_weight=0.05 | 0.92 | 0.76 | 1.00 | 0.839 | 0.96 | 0.82 | 0.99 | 0.874 |
| chunk=2000 fusion=rank vector_weight=0.1 | 0.92 | 0.78 | 1.00 | 0.839 | 0.96 | 0.83 | 0.99 | 0.878 |
| chunk=2000 fusion=rank vector_weight=0.25 | 0.90 | 0.78 | 0.98 | 0.837 | 0.96 | 0.79 | 0.99 | 0.868 |
| chunk=2000 fusion=rank vector_weight=0.5 | 0.88 | 0.76 | 0.98 | 0.798 | 0.96 | 0.77 | 0.99 | 0.859 |
| chunk=2000 fusion=rank vector_weight=1.0 | 0.86 | 0.74 | 0.98 | 0.753 | 0.94 | 0.73 | 0.98 | 0.846 |
| chunk=2000 fusion=score vector_weight=0.05 | 0.92 | 0.78 | 0.98 | 0.839 | 0.95 | 0.82 | 0.99 | 0.882 |
| chunk=2000 fusion=score vector_weight=0.1 | 0.92 | 0.78 | 0.98 | 0.838 | 0.96 | 0.83 | 0.99 | 0.887 |
| chunk=2000 fusion=score vector_weight=0.25 | 0.92 | 0.80 | 0.96 | 0.845 | 0.96 | 0.84 | 0.99 | 0.890 |
| chunk=2000 fusion=score vector_weight=0.5 | 0.90 | 0.78 | 0.96 | 0.848 | 0.95 | 0.81 | 0.99 | 0.888 |
| chunk=2000 fusion=score vector_weight=1.0 | 0.90 | 0.76 | 0.96 | 0.810 | 0.94 | 0.76 | 1.00 | 0.860 |
| chunk=2000 fusion=distribution vector_weight=0.05 | 0.92 | 0.78 | 1.00 | 0.810 | 0.95 | 0.83 | 0.99 | 0.858 |
| chunk=2000 fusion=distribution vector_weight=0.1 | 0.92 | 0.80 | 1.00 | 0.809 | 0.96 | 0.83 | 0.99 | 0.869 |
| chunk=2000 fusion=distribution vector_weight=0.25 | 0.92 | 0.80 | 0.98 | 0.819 | 0.97 | 0.84 | 0.99 | 0.866 |
| chunk=2000 fusion=distribution vector_weight=0.5 | 0.88 | 0.76 | 0.98 | 0.799 | 0.96 | 0.81 | 0.99 | 0.857 |
| chunk=2000 fusion=distribution vector_weight=1.0 | 0.88 | 0.76 | 0.98 | 0.754 | 0.94 | 0.75 | 0.98 | 0.839 |
| chunk=700 lanes=text | 0.90 | 0.76 | 1.00 | 0.838 | 0.96 | 0.84 | 0.99 | 0.885 |
| chunk=700 lanes=vector | 0.72 | 0.50 | 0.88 | 0.620 | 0.80 | 0.52 | 0.95 | 0.677 |
| chunk=2000 lanes=text | 0.92 | 0.76 | 1.00 | 0.832 | 0.95 | 0.82 | 0.99 | 0.882 |
| chunk=2000 lanes=vector | 0.70 | 0.52 | 0.88 | 0.577 | 0.84 | 0.51 | 0.95 | 0.658 |

### Diversity (maximal marginal relevance at 0.3), frozen n=50, 700 characters

| configuration | R@5 | all@5 | R@15 | MRR | same without diversity |
|---|---|---|---|---|---|
| rank, vector weight 0.01, diversity 0.3 | 0.90 | 0.68 | 1.00 | 0.836 | 0.90 / 0.76 / 1.00 / 0.838 |
| rank, vector weight 0.25, diversity 0.3 | 0.90 | 0.70 | 0.98 | 0.802 | 0.88 / 0.72 / 0.98 / 0.807 |
| score, vector weight 0.01, diversity 0.3 | 0.90 | 0.72 | 1.00 | 0.838 | 0.90 / 0.74 / 1.00 / 0.838 |
| score, vector weight 0.25, diversity 0.3 | 0.90 | 0.72 | 0.98 | 0.838 | 0.90 / 0.74 / 0.96 / 0.834 |

Diversity at 0.3 never raised all-sessions@5, which is the measure it is
meant to help. It lowered it in every row, by up to eight points (0.76 to
0.68 at the new default). R@5 rose by one item in one row, and MRR moved by
at most 0.005 either way. It is also the costly setting. Asked for 30
passages, one recall took about 2.7 s with it and 0.02 s without, on a
single probe item. That is why
this grid is smaller than the others: the full grid's diversity half was
started, ran at about 150 s an item on the loaded machine, and was stopped.
It is not a default.

## Before and after, repeated

`compare(hybrid=True)` at engine defaults on the frozen sample. Before is a
clean checkout of main (`acc6492e`) and after is this change. The runs
were interleaved (before, after) three times on the same machine while
it was loaded. The hashed embedder and in-memory stores are
deterministic, so the three runs should agree exactly. The median is
reported.

| round | before (0.25): R@5 / all@5 / R@15 / MRR | after (0.01): R@5 / all@5 / R@15 / MRR | reference | wall before / after |
|---|---|---|---|---|
| 1 | 0.88 / 0.72 / 0.98 / 0.8074 | 0.90 / 0.76 / 1.00 / 0.8379 | 0.88 / 0.74 / 0.98 / 0.8307 | 208 s / 176 s |
| 2 | 0.88 / 0.72 / 0.98 / 0.8074 | 0.90 / 0.76 / 1.00 / 0.8379 | 0.88 / 0.74 / 0.98 / 0.8307 | 186 s / 159 s |
| 3 | 0.88 / 0.72 / 0.98 / 0.8074 | 0.90 / 0.76 / 1.00 / 0.8379 | 0.88 / 0.74 / 0.98 / 0.8307 | 155 s / 135 s |
| **median** | **0.88 / 0.72 / 0.98 / 0.8074** | **0.90 / 0.76 / 1.00 / 0.8379** | **0.88 / 0.74 / 0.98 / 0.8307** | 186 s / 159 s |

All three rounds were identical. Wall time covers both sides and a loaded
machine, and this change makes no claim about it.

## Why the other winners are not defaults

- **Relative-score fusion at 0.25** (0.90 / 0.74 / 0.96 / 0.834 frozen;
  0.95 / 0.82 / 0.99 / 0.868 on the second sample) also meets the rule on
  the frozen sample. It loses to rank fusion at 0.01 on both samples'
  MRR, and it costs one item of R@15. It would also change the default
  fusion mode for every hashed-embedder recall. The recency term
  (`SCONE_RECENCY_WEIGHT`, 0.005 at age zero) is sized against rank
  fusion's scores, about 0.016 at the top, and under score fusion
  (about 1.0 at the top) it would shrink to a rounding error. These benches
  cannot see that: every session is years older than the clock.
- **2,000-character chunks** gave the best rows overall. Relative-score
  fusion at 0.25 reached 0.92 / 0.80 / 0.96 / 0.845 on the frozen sample and
  0.96 / 0.84 / 0.99 / 0.890 on the second. Chunk size decides which chunks
  exist, and so what a stored space holds. Passages would be about three
  times longer in every answer's context, and near the 512-token window of
  small real embedders. No setting names it (`chunk_target` is a
  constructor argument), and nothing here measures a real embedder.
  It is a lever for a measured change of its own, not a side effect of
  this one.
- **Distribution fusion** was never ahead of relative-score fusion on
  MRR at the same weight and chunk size, on either sample.

## What is still open

- **At 0.01 the hashed vector lane barely touches the order on these
  samples.** The fused rows score exactly as the text lane alone on both.
  Their top five sessions match the text lane's in 149 of 150 items, and
  their top fifteen in 126 (42 of 50 and 84 of 100). The lane still runs,
  for the confidence signal and a failed text lane. Beating the reference with a
  vector lane that knows something is the next row's work: a real embedder
  on both sides.
- **Where the old measurement went.** The commit that set 0.25
  (`a3e8fa57`) measured 0.88 / 0.70 / 1.00 / 0.833. Re-run on its own tree
  with this sample, it gives 0.88 / 0.74 / 1.00 / 0.8386. Main gave
  0.88 / 0.72 / 0.98 / 0.8074. Measuring the merges between them, and
  then the commits of the branch that differed, points at one commit: `85157c19` "Count a query word once beside its family"
  (0.8386 on its parent `8a2ed272`, 0.8074 on it). It is a correctness
  fix, because a word its stem prefix covers had been scored twice. On this
  bench, though, the double count had been helping. The in-memory text
  lane's change is the only part of that commit this bench runs; its
  SQLite changes do not reach it. Why the double count helped is not
  isolated. One reading is that an exact word form is better evidence than
  a family member. An exact-form signal beside the family, weighed on its
  own, is a candidate lever for the text lane. It is not undone here.
