# Complete SQuAD and HotpotQA dev sets: Scone vs LlamaIndex retrieval

## Run

- **Dates:** 29 September 2026.
- **Questions:** every question of SQuAD v1.1 dev (10,570) and HotpotQA distractor dev (7,405).
- **Corpora:** each dataset searches one pooled corpus. For SQuAD, all 2,067 paragraphs. For HotpotQA, every gold and
  distractor paragraph the dev questions offer, deduplicated: 66,635.
- **Documents:** both systems see each document as its title, a newline, then its text.
- **Embedder:** local `bge-base-en-v1.5` for both, through one cache.
- **Scone:** engine defaults, with each system's top 10 documents taken after folding passages to documents.
  - The in-memory index used [`fast_index.py`](fast_index.py): a numpy shortlist rescored with the default's own formula.
  - Its results equal the default's bit for bit (tested, including ties, and with a mutant that the test caught).
- **LlamaIndex:** `VectorStoreIndex` (`SentenceSplitter`, 512 model tokens) fused with its `BM25Retriever` by
  reciprocal rank. Each retriever returns 64 nodes.
- **Code and raw files:** runner [`run.py`](run.py); raw files in `bench-runs/public-matched-2026-09-29/`.
- **Gold:** read only for scoring. SQuAD has 1 gold paragraph per question and HotpotQA has 2.

## SQuAD (10,570 questions, 2,067 paragraphs)

| Metric | Scone | LlamaIndex | Scone only | LlamaIndex only | Sign test |
| --- | ---: | ---: | ---: | ---: | --- |
| hit@1 | **79.3%** | 77.5% | | | |
| all@2 | **89.5%** | 88.5% | 422 | 318 | p = 0.00015 |
| all@5 | **95.4%** | 94.9% | 199 | 141 | p = 0.002 |
| all@10 | **97.7%** | 97.3% | 104 | 64 | p = 0.003 |
| MRR@10 | **86.4** | 85.2 | | | |

Search latency: Scone 14 ms median (19 ms p95); LlamaIndex 35 ms (38 ms p95).

A 50-question smoke test, drawn from the first articles, showed Scone 20 points behind at hit@1. The full set
reverses it, which is why samples are not reported as results.

## HotpotQA (7,405 questions, 66,635 paragraphs)

**All questions** (all@k means both gold paragraphs are in the top k):

| Metric | Scone | LlamaIndex | Scone only | LlamaIndex only | Sign test |
| --- | ---: | ---: | ---: | ---: | --- |
| hit@1 | 87.1% | 87.4% | | | |
| all@2 | 37.9% | 37.7% | 308 | 296 | p = 0.65 |
| all@5 | 65.5% | 65.0% | 259 | 222 | p = 0.10 |
| all@10 | 76.9% | 76.6% | 189 | 167 | p = 0.27 |
| MRR@10 | 92.1 | 92.3 | | | |

**By question type:**

| Type | n | Scone all@5 | LlamaIndex all@5 | Scone all@10 | LlamaIndex all@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| comparison | 1,487 | 91.7% | 91.3% | 98.5% | 98.0% |
| bridge | 5,918 | 58.9% | 58.4% | 71.5% | 71.2% |

Search latency: Scone 113 ms median (179 ms p95); LlamaIndex 1,051 ms median (1,068 ms p95).

Index build: LlamaIndex took 2,356 s over 66,745 nodes. Scone ingested from the same vector cache.

## What it shows

- **SQuAD:** Scone's defaults retrieve better than LlamaIndex's best configuration. The margin is small (+1 to +2
  points) but significant at every depth, and search is 2.4x faster.
- **HotpotQA:** the two systems tie on retrieval, and Scone searches the 66k-paragraph corpus 9x faster.
- **The weakness both share is the second hop of bridge questions.** Both find a first gold paragraph for 86% of
  bridge questions at rank 1. But both gold paragraphs reach the top 5 for only about 59%. The second paragraph is
  about an entity named in the first and shares little with the question. This is where multi-hop retrieval
  (HippoRAG 2's personalized PageRank; iterative retrieval) reports its gains, and where Scone has nothing today.

## Not yet measured

- **Answer accuracy (exact match and F1).** The answer stage is ready and smoke-tested. It needs a reader for about
  36,000 answers: hosted Gemma 4 31B waits on the API spending cap, and local Gemma 4 E4B would take about a day.
- **Other competitors.** Systems other than LlamaIndex have not been run in this harness.
