# North star with a real embedder on both sides (2026-09-15)

**Question.** Every north star number so far ran the hashed-token
embedder. With `bge-small-en-v1.5` on both sides -- the engine through
`LocalEmbedder`, LlamaIndex through its embedding interface over the same
model -- does the engine at its defaults for a real embedder beat
LlamaIndex's BM25+vector fusion on LongMemEval-S, and does any vector
weight, fusion mode or chunk size do so on both samples?

**The rule, written before any real-embedder number was read.** A row
beats the reference when its R@5 and its MRR are each strictly higher
than LlamaIndex hybrid's, on the frozen 50 and on the 100 other items.
If a row does, it becomes the default for non-hashed embedders. If more
than one does, the winner that changes the least is taken (a vector
weight alone before a fusion mode, a fusion mode before a chunk size),
and among those the higher MRR averaged over the two samples. If no row
does, nothing changes and this file is the record.

**Status: not measured. No leg finished, so no default changed.** The
runner and the shared vector cache are built, tested and proven (see
below). The frozen n=50 leg at 700 characters was started twice and
stopped at item 5 of 50, because it was projected to run past the
3-hour limit set for a leg. The second sample and the 2,000-character
and 512-token chunkings were not started. No R@5, all@5, R@15 or MRR
with a real embedder exists yet for either side, and none is claimed.

## Why it stopped, with the costs measured

All on one Apple M3 Max (10 performance and 4 efficiency cores, 36 GB),
shared with other sessions' builds, test runs and a local model server.

| what | load average | measured |
|---|---|---|
| probe: 256 texts of 700 characters through `LocalEmbedder` | 16-23 | 11.3-14.6 s (17.5-22.6 texts/s) |
| probe: 256 texts of 2,000 characters | 16-23 | 31.2 s (8.2 texts/s) |
| smoke: two items outside the frozen 50 (`--n 2 --seed 42`: `gpt4_2ba83207`, `gpt4_213fd887`), 700 characters, `--check`, niced (15) | about 16 | 185 s wall. Ours embedded 1,841 texts in 103.5 s, LlamaIndex 481 texts (479 nodes and the 2 questions) in 74.9 s, and the cache dropped 0 |
| the same smoke's `compare()` at defaults, straight after | about 16 | embedded 0 texts: ours read 1,852 of 1,852 from the cache, LlamaIndex 481 of 481 |
| leg A, first start, niced (15) | 35-60 | items 1-2 in 915 s; stopped |
| leg A, second start, not niced (5) | 44-54 | items 1-2 (already cached) in 30 s, model load included; items 3-5 in 1,012 s; stopped at 5/50 |

Each frozen item holds about 50 sessions, between 479,852 and 513,111
characters of text. Projected from the stopped leg by characters
still to embed, the 45 remaining items would have taken 4.2 hours at
that load. Embedding is the whole cost: a cached item took 3 s,
fusion grid included. The cache file
(`~/.scone-memory/northstar-bge-small-en-v1.5.sqlite`, 8,567 vectors when
stopped) keeps everything embedded so far, so a resumed leg starts from
there.

At the smoke run's load, about 90 s an item, the frozen leg would take
about 75 minutes and the second sample about 2.5 hours. That is a
projection, not a measurement, and it rests on the smoke run's two items,
which are not in the frozen 50 (`stratified_sample(n=2, seed=42)` does not
draw the first two of `n=50`). Their sizes, 496,576 and 486,478
characters, are inside the frozen items' range. The 4.2-hour projection
above comes from frozen items 3-5.

Every cost above was measured with the reference's splitter counting
tiktoken's tokens, before the change in the next section. Under it the
smoke run's two items make 492 reference nodes instead of 479. The
embedding times were not measured again: the machine's load average was
41-49, and a time taken at that load would not compare with one taken at 16.

## What the model's window cut

BGE reads 512 tokens of its own and fastembed drops the rest of an input
without a warning. LlamaIndex's `SentenceSplitter(chunk_size=512)` counts
tiktoken's tokens by default, and BGE counts more for the same
conversation text. Counted with BGE's own tokenizer, without truncation,
over the embedded text of each node:

| items | reference nodes, tiktoken count (before) | past 512 | largest | tokens cut | reference nodes, BGE count (after) | past 512 | largest |
|---|---|---|---|---|---|---|---|
| frozen 1-3 (`6aeb4375_abs`, `06db6396`, `07741c44`) | 729 | 297 (41%) | 576 | 6,114 of 331,928 | 743 | 0 | 512 |
| smoke (`gpt4_2ba83207`, `gpt4_213fd887`) | 479 | 233 (49%) | 771 | 5,602 of 223,033 | 492 | 0 | 512 |

Our 700-character chunks never came near the window: on frozen items
1-2, 0 of 1,871 embedded texts, the largest 225 tokens. At 2,000
characters, 11 of 635 ran past it, the largest 578. So before the change
the cut fell almost only on the reference, and nothing recorded it.

Two changes, in `bench.comparative`:

- **The reference splits in the model's tokens.** With an embedder that
  counts its tokens (`count_tokens`, as `LocalEmbedder` does), the
  reference's `SentenceSplitter` counts with that tokenizer, and a chunk's
  two markers are paid once, as the engine's `chunk_tokens` counts them.
  So the reference's 512 and our `512t` are the same measure, what the
  model reads. The hashed embedder has no tokenizer and keeps LlamaIndex's
  default. On the frozen sample at `n=10` with the hashed embedder, this
  branch before and after the change gave identical rows, per-item
  rankings and `compare()` scores. `compare()`'s config and the sweep's
  result name the reference's tokenizer (`llamaindex_tokenizer`).
- **Each side's record counts what the window cut.** `over_window` is the
  number of texts asked for that ran past the model's window, and
  `tokens_past_window` is the tokens cut. Texts read back from the cache
  count too, since their vectors were cut when they were made. Both are
  `null` for a model with no window or no tokenizer.

## Commands

From `packages/memory`, `PYTHONPATH=$PWD/src`, Python 3.14.7. These are the
legs as queued. The first was stopped at item 5 and the others never
started. Legs 1 and 2 of the plan, the defaults row (rank fusion, weight
1.0) and the weight 0.5 and distribution-fusion rows, come out of the
first command in one embedding pass: the fusion and weight grid is asked
per item after ingestion.

```
C=(--data ../../bench-data/longmemeval_s.json --fusions rank,score,distribution --weights 1.0,0.5,0.25 --diversities none --embedder bge-small-en-v1.5 --model-cache ~/.scone-memory/fastembed --embedding-cache ~/.scone-memory/northstar-bge-small-en-v1.5.sqlite)
python benchmarks/northstar_defaults.py $C --n 50 --seed 42 --chunks 700 --check --out A-frozen-700.json
python benchmarks/northstar_defaults.py $C --n 100 --seed 7 --holdout-of 42:50 --chunks 700 --check --out B-second-700.json
python benchmarks/northstar_defaults.py $C --n 50 --seed 42 --chunks 2000,512t --out C-frozen-2000-512t.json
python benchmarks/northstar_defaults.py $C --n 100 --seed 7 --holdout-of 42:50 --chunks 2000,512t --out D-second-2000-512t.json
```

The smoke run:

```
python benchmarks/northstar_defaults.py --data ../../bench-data/longmemeval_s.json --n 2 --seed 42 --chunks 700 --fusions rank --weights 1.0 --diversities none --check --embedder bge-small-en-v1.5 --model-cache ~/.scone-memory/fastembed --embedding-cache ~/.scone-memory/northstar-bge-small-en-v1.5.sqlite --out smoke.json
```

Its two items scored 1.0 on every measure for every row, both sides.
Two items show that the pipeline runs end to end, and say nothing about
which side is better.

The hashed-path check, run on main `3eed1cff` and on this branch:

```
python benchmarks/northstar_defaults.py --data ../../bench-data/longmemeval_s.json --n 10 --seed 42 --chunks 700,2000 --fusions rank,score --weights 0.01,0.25 --diversities none --check --out hashed.json
```

## How it is set up to run

- **Data.** `bench-data/longmemeval_s.json`. Frozen sample: `n=50`,
  `seed=42`, the Rust harness's stratified sample (every item has
  evidence). Second sample: `n=100`, `seed=7`, drawn from the 450 items
  outside the frozen 50 (`--holdout-of 42:50`).
- **Embedder, both sides.** `bge-small-en-v1.5`, the quantised ONNX build
  fastembed 0.8.0 downloads (`qdrant/bge-small-en-v1.5-onnx-q`), onnxruntime
  1.30.0 on the CPU, 384 dimensions. Ours is `LocalEmbedder`, and the
  reference gets the same object through `SconeEmbedding`, LlamaIndex's
  `BaseEmbedding` interface over our `Embedder` port. Neither side adds
  BGE's query instruction ("Represent this sentence for searching relevant
  passages: "). Both embed the question as plain text, so neither is at
  BGE's recommended use. Both sides' chunks are counted in the tokens BGE
  reads (see "What the model's window cut"). A 2,000-character chunk of
  ours can still run past the window, and the record counts it.
- **Reference.** `compare(hybrid=True)`: LlamaIndex 0.14.24's
  `SentenceSplitter` (512 of BGE's tokens, markers included, no overlap;
  tiktoken's tokens with the hashed embedder), `VectorStoreIndex`, and
  `QueryFusionRetriever` fusing its vector and BM25 retrievers by
  reciprocal rank, with every node ranked. Its plain vector retriever is
  listed for scale.
- **Ours.** `MemoryEngine` over in-memory stores, one fresh engine per
  item and chunking, asked for 30 passages (15 x the per-episode cap).
  With a non-hashed embedder the engine's default vector weight is 1.0,
  and its fusion is rank. Both rankings are folded to distinct sessions
  and scored by `bench.comparative`'s rule. No model, reranker or query
  rewriting runs on either side.
- **Embedding once.** Both sides embed through a `CachedEmbedder` over one
  SQLite file (the engine's `SqliteEmbeddingCache`, bound 1,000,000 vectors),
  keyed by model id, width and exact text. A text either side has embedded
  is read back, never embedded again, and a rerun embeds nothing. The run's
  JSON records, per side, the texts asked for, read back and embedded, the
  seconds spent in the model, the texts the model's window cut and the
  tokens it cut, and the cache's evictions.
- **The hashed path is unchanged.** `--embedder` defaults to `hash`. On the
  frozen sample at `n=10` (700 and 2,000 characters, rank and score fusion,
  weights 0.01 and 0.25, `--check`), main `3eed1cff` and this branch gave
  identical rows and identical per-item rankings, in three runs of each (interleaved; wall seconds, median of three, 20 on main and 19 here, on a loaded machine).
