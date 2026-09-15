# Semantic double merge v1: does the second pass change retrieval?

First run on 15 September 2026 at commit dd2ee903 (branch
`feature/semantic-double-merge-chunking`, parent e8ff7965); the merge
rows and `--pairs` were run again the same day on the review fixes that
follow 055f95a5, which stop the second pass joining or comparing a piece
of a sentence longer than the target (see "What the review changed"). The
numbers below are that second run. From `packages/memory`, with

```
PYTHONPATH=$PWD/src python benchmarks/semantic_double_merge.py \
    --data ../../bench-data/longmemeval_s.json --n 50 --seed 42 --thresholds 0.2,0.4 --out merge.json
PYTHONPATH=$PWD/src python benchmarks/semantic_double_merge.py \
    --data ../../bench-data/longmemeval_s.json --n 50 --seed 42 --pairs --out pairs.json
```

The second run passed `--no-length`: the length row does not read the
threshold. Its `semantic` row matched the first run's in every score,
receipt and per-item session ranking. The first command also ran once
before the first commit on the same source
(the commit only removed an unreachable guard, moved two keyword
arguments and added comments); both runs gave the same scores, the same
receipts and the same session ranking for every item.

## What ran

- **Data.** `bench-data/longmemeval_s.json`, the Rust harness's stratified
  sample, `n=50`, `seed=42`; all 50 items have evidence. 2,355 sessions,
  one record each, as `bench.runner.run` stores them.
- **Both sides.** The hashed-token embedder `hash-256-t3-u16.0.0`, no
  model, no reranker, no query rewriting. Every row is
  `bench.comparative.compare(ks=(5, 10, 15), hybrid=True)`: a fresh
  in-memory engine per item at its defaults (vector weight 0.01 for a
  hashed embedder, 700-character target), recall folded to distinct
  sessions, against LlamaIndex 0.14.24 `QueryFusionRetriever(VectorIndexRetriever
  + BM25Retriever)` over `SentenceSplitter(512)`. The reference is the
  same in every row.
- **Rows.** `length` is the engine's default cut. `semantic` is
  `semantic_aware=True`. `semantic+merge@T` adds
  `semantic_merge_threshold=T`. Receipt counts are summed over every
  record the rows stored.
- **Deterministic.** One run is the number; no timing is claimed. Wall
  time on a shared, heavily loaded machine was 317 s and 694 s for the
  whole run.

## Output

| our cut | R@5 | all-sessions@5 | R@10 | R@15 | MRR | NDCG@5 | chunks | first-pass chunks | merges | stopped by similarity | stopped by size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| length (default) | 0.90 | 0.76 | 0.92 | 1.00 | 0.838 | 0.807 | 46,637 | | | | |
| semantic | 0.90 | 0.74 | 0.92 | 0.96 | 0.857 | 0.824 | 51,248 | | | | |
| semantic + merge at 0.4 | 0.90 | 0.74 | 0.92 | 0.96 | 0.857 | 0.824 | 50,996 | 51,248 | 252 | 38,122 | 10,520 |
| semantic + merge at 0.2 | 0.90 | 0.74 | 0.92 | 0.96 | 0.857 | 0.825 | 49,997 | 51,248 | 1,251 | 20,336 | 27,307 |
| *LlamaIndex BM25 + vector* | 0.88 | 0.74 | 0.90 | 0.98 | 0.831 | 0.784 | | | | | |

Unrounded MRR: 0.8379 (length), 0.8572 (all three semantic rows), 0.8307
(reference); NDCG@5 0.8236 (semantic, merge at 0.4) and 0.8246 (merge
at 0.2). Against the reference, per item at k=5: one won, none lost in
every row; at k=15 the semantic rows win one and lose two, the length
row wins one and loses none.

Against the `semantic` row, the top-15 session ranking changed for 32
items at 0.2 (the top 5 for 7) and for 6 at 0.4 (the top 5 for 2).

`merges + stopped by similarity + stopped by size` is the first-pass
chunks less the records that were cut (51,248 - 2,354 = 48,894; one of
the 2,355 sessions repeats another in its item and is stored as a
duplicate, with no receipt).

### The scale a threshold is chosen on (`--pairs`)

The first pass made 51,250 chunks (the duplicate session's 2 counted).
Of the 48,895 neighbouring boundaries, 2,190 are beside or inside a
sentence longer than the target and are never compared. Of the 46,705
pairs of whole-sentence chunks, 6,790 would fit the 700-character target
together -- only those can ever merge -- and 39,915 would not.

| cosine of mean sentence vectors | 10th | 25th | median | 75th | 90th | 95th |
| --- | --- | --- | --- | --- | --- | --- |
| pairs that would fit | -0.022 | 0.000 | 0.036 | 0.162 | 0.300 | 0.377 |
| pairs that would not | 0.057 | 0.143 | 0.252 | 0.373 | 0.487 | 0.557 |

Pairs that would fit, at or above a threshold: 2,402 at 0.1, 1,386 at
0.2, 679 at 0.3, 276 at 0.4, 94 at 0.5. The thresholds 0.2 and 0.4 were
chosen before the first retrieval run, so that the pass would actually
fire, from the first version of this table, which also counted pairs
touching a piece (7,060 fitting, 2,603 / 1,527 / 769 / 330 / 117 at the
same thresholds; 0.2 and 0.4 sat at about its 78th and 95th
percentiles). On this one they sit at about the 80th and 96th. Merges in the
run are fewer than these counts because a merge changes the next pair.

## What it shows

- **The second pass does what it says and is measured doing it.** At 0.2
  it removed 1,251 chunks (2.4% of the semantic cut's), at 0.4 252 (0.5%),
  and the receipt says why every other join did not happen.
- **It moves no retrieval score on this sample.** R@5, all-sessions@5,
  R@10, R@15 and MRR are identical across the three semantic rows; NDCG@5
  moves by 0.001 at 0.2. Rankings do change -- the top 5 of 7 items at
  0.2 -- but no item's first answer session moved and no item gained or
  lost a hit at 5, 10 or 15.
- **Most stops at 0.2 are the size bound.** 27,307 of the 47,643 joins
  not made were counted under size: 2,190 of them are the boundaries
  beside or inside a sentence longer than the target, never compared, and
  the other 25,117 were whole-sentence pairs alike enough and too long
  together. A chunk the first pass ended because the next sentence did
  not fit can never fit the next whole chunk, so every such cut between
  alike chunks is counted there; how many of the 25,117 were size cuts
  rather than valleys was not counted.
- **The semantic cut itself** matches the length cut at k=5 and 10, is
  higher on MRR (0.857 vs 0.838) and NDCG@5 (0.824 vs 0.807), and one
  item lower at k=15 (0.96 vs 1.00), with 9.9% more chunks.

## What the review changed

The first run's second pass compared a piece of a long sentence by the
whole sentence's vector, and could join it to the sentence before it.
Stopping that took 132 merges away at 0.2 (1,383 to 1,251) and 54 at 0.4
(306 to 252): those joins, and what they changed in the pairs after them.
Chunk counts rose to match (49,865 to 49,997; 50,942 to 50,996). At 0.2
stopped-by-similarity fell by 398 (20,734 to 20,336) and stopped-by-size
rose by 530 (26,777 to 27,307), which is 398 plus the 132 merges no
longer made; at 0.4, 852 and 906, which is 852 plus 54. No retrieval score moved, to four places; seven items' session
rankings differ from the first run at 0.2 and one at 0.4. The first
pass (`semantic` row, `groups`) is unchanged: no session in this sample
has a sentence whose size cut reaches past the ceiling, so moving that
bound earlier, which the review also asked for, changed no count here.

## What it does not show

- 50 items: one item is 0.02 of a recall column.
- With the hashed-token embedder the vector lane carries 0.01 of the fused
  ranking, so a cut changes the ranking mostly through the text lane. A
  neural embedder's cosines sit on another scale entirely; neither
  threshold here means anything for it, and nothing was measured with one.
- Nothing about answer quality, and no timing.
