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

## Experiment: a second hop without a model ([`twohop.py`](twohop.py)) — not adopted

- **What it does:** after Scone's recall, it recalls again with the question plus each of the first two documents'
  text, up to the engine's 1,000-character query limit.
- **How results combine:** the three lists are fused by reciprocal rank with equal weights.
- **Discipline:** parameters were fixed before any result was seen. All 7,405 questions ran, and the first pass
  matched the main run on every question.

| Metric | Scone | Scone + two-hop | LlamaIndex |
| --- | ---: | ---: | ---: |
| hit@1 | **87.1%** | 81.0% | 87.4% |
| all@2 | **37.9%** | 36.1% | 37.7% |
| all@5 | **65.5%** | 64.1% | 65.0% |
| all@10 | 76.9% | **82.3%** | 76.6% |
| bridge all@10 | 71.5% | **79.1%** (642–190 vs Scone, p < 1e-50) | 71.2% |
| comparison all@5 | **91.7%** | 75.8% | 91.3% |
| median latency | **113 ms** | 868 ms | 1,051 ms |

### Verdict

**Not adopted.** Deep recall rises, but the top ranks and comparison questions suffer.

- **The second hop is reachable without a model.** Bridge all@10 rises 7.6 points.
- **Equal-weight fusion is the wrong combination.** Documents found only by a hop displace correct first-pass
  documents from the top ranks.
- **Next:**
  - Let hops fill positions below the first pass's leading documents, and only when the first pass looks incomplete.
  - Tune that on a fixed half of the questions and report on the other half. These results have now been seen, so
    they cannot also be a clean test.

## Second hop, rule chosen on one half and tested on the other ([`hop_rule.py`](hop_rule.py))

**Rule:** keep the first pass's top 3 whole, then fill the remaining places round-robin from one hop search and
the rest of the first pass. The hop searches with the question plus the top document's text. The rule was chosen on
the development half (3,729 questions). The test half (3,676) was then scored once:

| Test half | Scone | **Scone + hop** | LlamaIndex |
| --- | ---: | ---: | ---: |
| hit@1 | 87.2% | **87.2%** | 87.4% |
| all@2 | 37.3% | **37.3%** | 36.8% |
| all@5 | 64.8% | **69.8%** | 64.2% |
| all@10 | 76.4% | **83.1%** | 75.9% |
| bridge all@5 / all@10 | 58.5% / 71.2% | **65.4% / 79.9%** | 57.8% / 70.8% |
| comparison all@5 / all@10 | 91.3% / 98.3% | 88.8% / 96.7% | 91.2% / 97.6% |

Paired counts on all@k (Scone + hop wins vs the other side's wins):

| Against | all@5 | all@10 |
| --- | --- | --- |
| Scone | 281–96 (p = 4e-22) | 346–98 (p = 2e-33) |
| LlamaIndex | 346–139 (p = 2e-21) | 372–107 (p = 2e-35) |

- **Cost:** one extra recall per question, plus a small loss on comparison questions (all@5 2–20 against Scone).
- **Next:** the rule is benchmark code so far. The next step is to make it an engine option and measure it there
  end to end.

### The engine's implementation, end to end on the held-out half ([`enginehop.py`](enginehop.py))

The rule above was measured over saved document lists. `retrieval.second_hop.recall_with_hop` implements it in the
engine, over passages:
- it seeds the hop with the leading passage, not the leading document;
- it keeps the leading three episodes whole;
- it holds to the per-episode cap.

That code ran on all 3,676 test-half questions:

| Test half | Scone | **Scone + engine hop** | LlamaIndex |
| --- | ---: | ---: | ---: |
| hit@1 | 87.2% | **87.2%** | 87.4% |
| all@2 | 37.3% | **37.3%** | 36.8% |
| all@5 | 64.8% | **69.8%** | 64.2% |
| all@10 | 76.4% | **83.2%** | 75.9% |
| bridge all@5 / all@10 | 58.5% / 71.2% | **65.4% / 80.0%** | 57.8% / 70.8% |
| comparison all@5 / all@10 | 91.3% / 98.3% | 88.6% / 96.6% | 91.2% / 97.6% |
| latency p50 / p95 | 113 ms / 179 ms | 424 ms / 659 ms | 1,051 ms / 1,068 ms |

Paired counts on all@k (engine hop wins vs the other side's wins):

| Against | all@5 | all@10 |
| --- | --- | --- |
| Scone | 283–99 (p = 1e-21) | 344–93 (p = 6e-35) |
| LlamaIndex | 348–142 (p = 5e-21) | 369–101 (p = 7e-37) |

- **Agreement with the estimate:** the engine reproduces the estimate from the saved lists within 0.2 points. It
  keeps hit@1 and all@2 unchanged and still searches 2.5x faster than LlamaIndex.
- **What it costs:** on comparison questions (19% of HotpotQA), it trails plain Scone (all@5 2–21). It is still level
  with LlamaIndex at all@10 (3–10, p = 0.09).
- **Next:** gate the hop on bridge-like questions. That gate must be chosen on the development half.
