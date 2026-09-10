# Public QA baseline — 8 September 2026

The [preregistered protocol](public-qa-v1.protocol.md) produced **600 completed
responses**: three self-managed models answered the same 200 original questions,
100 from HotpotQA and 100 from SQuAD. There were **zero runtime failures**.
Model weights were not fine-tuned for this experiment. Questions, settings,
retrieved requests, and scoring were fixed before inference; no answers were
rewritten or replaced with retries.

## Generation

| Model | Overall exact match | Token F1 | Hotpot exact match | SQuAD exact match | Inference p50 / p95 | Observed loaded-model size |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 E4B | 64.0% | 72.9% | 46.0% | 82.0% | 2.29 / 5.88 s | 9.64 GB |
| Llama 3.2 3B | 59.0% | 68.7% | 44.0% | 74.0% | 1.46 / 3.83 s | 3.09 GB |
| Llama 3.1 8B | 61.0% | 72.5% | 44.0% | 78.0% | 3.36 / 11.88 s | 5.92 GB |

Each overall score uses 200 responses; each dataset-specific score uses 100.
Gemma abstained 20 times, Llama 3.2 once, and Llama 3.1 12 times. Abstentions
score zero on these answerable questions and remain in the denominator.
Exact match and token F1 use the published answer references and standard
normalization, with no semantic judge or answer repair. Equivalent wording can
lose exact-match credit; these numbers are not full semantic-accuracy measures.

After the run, the unchanged [official HotpotQA evaluator](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
was applied to the 300 original Hotpot responses. Every per-answer exact-match
and F1 score agrees with this harness (floating-point tolerance `1e-12`).
Evaluator SHA256: `d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0`.
Supporting-fact and joint metrics were not measured: the frozen format requested
answers only, without predicted `[document title, sentence index]` pairs.
The official script's zero placeholders for missing support predictions must
not be interpreted as measured support quality. Retrieval annotation coverage
below is a separate metric.

Gemma had the highest exact-match score in this run. Llama 3.2 had the lowest
median latency and loaded-model size. These observations identify a measured
tradeoff, not a universal model ranking or a demonstrated optimum.

An exploratory paired exact-match comparison found 18 questions correct only
for Gemma and 12 correct only for Llama 3.1 (two-sided exact McNemar p = 0.362).
This small observed gap does not establish a reliable ranking. The comparison
was post hoc; multiple comparisons and shared source topics limit inference.

## Retrieval and context

| Dataset | Supporting-document recall@5 | Recall@10 | All support documents@5 / @10 | Prepared annotation coverage | All annotations in context |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 87.5% | 97.0% | 75.0% / 94.0% | 86.8% | 72.0% |
| SQuAD | 100.0% | 100.0% | 100.0% / 100.0% | 99.0% | 99.0% |

Recall@k counts supporting document IDs among the first k ranked native-recall
chunks. Duplicate document IDs do not refill the ranking. These are standalone
recall measurements; the actual model requests came from MemoryContext at
limit 5 with an 8,000-byte context budget.

Prepared annotation coverage measures original supporting-sentence presence
for HotpotQA and reference-answer text presence within its source for SQuAD.
Hotpot's 86.8% is a per-question average; 196 of 225 annotated sentences were
present overall. Annotation presence does not establish semantic sufficiency.
All 1,000 retained source entries were independently checked against their
source documents, IDs, URLs, and verbatim text.

MemoryContext preparation p50/p95 was **24.6/32.5 ms**, measured separately
before generation. Inference timings include prompt processing and model
loading where incurred. Model blocks rotated every 20 questions, with no
simultaneous inference. Timing was not isolated from other applications.
Loaded-model sizes are Ollama `/api/ps` observations, not process RSS or peak
system RAM. p95 uses nearest rank.

## What this exposes

- The full set of annotated evidence reached the model for only 72 of 100
  Hotpot questions. At native recall@10, 94 questions had all supporting source
  documents. Candidate selection and context retention need investigation.
- Complete annotated evidence did not ensure a correct answer: Gemma's Hotpot
  exact match was 40/72 (55.6%) when all annotated sentences were present.
  Retrieval improvements alone will not resolve every generation error.
- One SQuAD question retrieved its correct source document, but the retained
  chunk ended before the answer. Passage continuity needs explicit handling.

A bounded inspection also found both answer-selection mistakes and wording
penalties. For example, a correct explanatory “Yes…” response receives zero
under Hotpot's categorical yes/no scoring rule; “International Watch Company”
also loses exact-match credit against “International Watch Co.” These examples
are illustrative, not estimated failure rates. They do not change any score.

These are diagnostic observations, not changes to the scored run. The 200
additional reserved questions remain unrun. Any subsequent configuration must
be frozen before they are used for evaluation.

## Reproduction and scope

Follow the [runner instructions](README.md). The shared corpus contains 2,176
original source paragraphs, including all supplied distractors and the reserved
questions' contexts. Inference had no gold-source filters or manually seeded
claim graph. It used Python 3.14.7, SQLite, Qdrant 1.19.1, cached
`bge-small-en-v1.5`, temperature 0, `reasoning_effort=none`, 256 output tokens,
an 8192-token model context, and a 120-second generation deadline.

- [HotpotQA development distractor file, pinned community mirror](https://huggingface.co/datasets/namlh2004/hotpotqa/blob/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json).
  SHA256: `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`.
  The official CMU endpoint failed TLS, so canonical-byte identity was not
  independently established. See the [official project](https://hotpotqa.github.io/).
- [SQuAD 1.1 development file](https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json).
  SHA256: `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`.

Downloaded corpora, gold labels, exact requests, raw responses, CSV scores,
memory samples, and the full report are retained under the ignored
`bench-runs/public-qa-2026-09-08/` directory. The final observations SHA256 is
`3a1c12d8b250d0bccf71f0c9679fb1277925421e820be711de404d09574085c5`;
the generation manifest SHA256 is
`e609d582f2d86f9ad20391af997297d6a3bac688c59d6376c8a1106f08a6bf67`.
Package/protocol source and input hashes stayed unchanged; installed model
digests were checked before every block and at completion.

This is a sampled development-set baseline over a pooled paragraph corpus,
not an official leaderboard submission or full-Wikipedia scalability test.
Public questions may overlap model pretraining. Reserved question IDs are
separate, but articles/topics need not be disjoint. One response per question
does not measure repeat variance. Citation quality, unanswerable-question
handling, and every optional agentic retrieval feature are outside this run.
Earlier synthetic development probes used different protocols and are not
pooled with these scores.
