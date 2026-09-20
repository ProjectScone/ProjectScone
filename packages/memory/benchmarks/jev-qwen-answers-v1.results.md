# Qwen, Qdrant and Jev answer results — 2026-09-20

The corrected run improved overall exact match from **65.5% to 70.0%** and
token F1 from **74.77% to 80.37%** when adding Jev. These are reference-answer
metrics on 200 reused development questions, not semantic accuracy, an untouched
holdout, or an official leaderboard submission. There is substantial room to improve.

## Reproducible comparison

Run `jev-qwen-answers-20260919/run-2` used immutable source snapshot `85fe3cdb`
and the [frozen protocol](jev-qwen-answers-v1.protocol.md). All 2,176 corpus
documents were validated before inference; 2,830 SQLite chunks matched 2,830
points in local Docker Qdrant, with zero pending lexical rows. The prior index
was reused after validation; every answer was regenerated. Run 1 remains aborted
and unscored because its first baseline query encountered an incomplete text index.

Both arms used Scone MemoryEngine and MemoryContext, Qwen3 Embedding 8B (4,096
dimensions), paid Gemma 4 31B, 64 retrieval candidates, five delivered sources,
and 8,000 context bytes. Jev reranked at most 32 candidates and resolved to
`typesafe/jev-1.13-20260917` on all 200 treatment cases. No gold labels were read
during inference. The source, input, protocol, request and output integrity checks
passed before scoring; `code_and_inputs_unchanged` is true.

All 400 retrieval preparations were healthy, with no degraded lanes. Generation
completed on 399/400 requests. One baseline HotpotQA generation timed out after
60.16 seconds (`hotpotqa:5ae64b015542992ae0d162bc`, ChatError); it remains a zero
in the original denominator. Its Jev answer was correct. This availability event
therefore accounts for 0.5 percentage points of the overall exact-match difference;
do not attribute the entire difference to retrieval quality.

## Scores

| Dataset | Qwen hybrid EM | Qwen + Jev EM | Qwen hybrid F1 | Qwen + Jev F1 |
| --- | ---: | ---: | ---: | ---: |
| All, 200 | 65.50% | 70.00% | 74.77% | 80.37% |
| HotpotQA, 100 | 50.00% | 59.00% | 60.19% | 71.23% |
| SQuAD 1.1, 100 | 81.00% | 81.00% | 89.35% | 89.50% |

Exact match: 10 wins, 1 loss, 189 ties. F1: 16 wins, 4 losses, 180 ties.
Abstentions: 24 -> 11 (HotpotQA 21 -> 8, SQuAD 3 -> 3).

| Evidence coverage | Qwen hybrid | Qwen + Jev |
| --- | ---: | ---: |
| Mean supporting-document recall@5 | 92.75% | 98.25% |
| All supporting documents@5 | 86.00% | 97.00% |
| All annotated support sentences / a reference answer span present | 84.50% | 95.50% |

The last row is a post-run exact-substring diagnostic, not semantic entailment:
HotpotQA 71/100 -> 93/100; SQuAD 98/100 in both arms. A correct document does
not guarantee that the necessary sentence survives chunk selection.

Compared with the qualified historical HashEmbedder + Jev result, Qwen + Jev
increased overall EM 68.5% -> 70.0% and F1 79.01% -> 80.37%. HotpotQA EM
57% -> 59%; SQuAD EM 80% -> 81%. Embeddings, fusion weights, vector backend,
index readiness and hosted generation all differ across runs, so this comparison
does not isolate the embedding model's effect.

## Timing and cache audit

| Timing, median / p95 | Qwen hybrid | Qwen + Jev |
| --- | ---: | ---: |
| Context preparation | 35 / 174 ms | 561 / 1,204 ms |
| Generation | 740 / 9,291 ms | 798 / 4,044 ms |
| Preparation + generation | 805 / 9,317 ms | 1,397 / 10,029 ms |

Queries were pre-embedded in 20.744 seconds. Of 400 subsequent query lookups,
392 hit the cache. Eight original questions had outer whitespace stripped by
production recall and required one additional vector each; their second arm
reused that vector. Total vectors embedded: 208, cumulative embedding time
36.0 seconds. Both arms share vectors, but timings are predominantly cache-warm
with eight first-arm misses, not fully warm or uncached production latency.

## Error audit and next experiments

All 60 Jev answers failing exact match were reviewed as question/answer/reference
records. Examples include legitimate aliases (`Steve Carell` vs `Steven John
Carell`), extra units (`781 plates` vs `781`), wrong answer selection (Kanye West's
album), missing bridge evidence (the company behind the Park Avenue South
recording), and underspecified questions (`How was this possible`). No aliases,
gold labels, answers or scores were rewritten.

The four F1 regressions are retained:

- SQuAD `5726f90b708984140094d75f`: `781` -> `781 plates` (the only EM loss).
- SQuAD `5727cd0f4b864d1900163d74`: a shorter transit-benefits paraphrase lost overlap.
- HotpotQA `5a77671255429966f1a36d21`: strategic negotiation -> optimal strategy.
- HotpotQA `5a7261635542997f8278398a`: `North-east Lithuania` -> `North-east`.

Next evaluate Scone's native bounded evidence-tool loop: follow-up searches for
bridge facts and source-window reads. Freeze its settings before generation and
keep this single-pass baseline. Then separate generation improvements from
retrieval improvements using identical evidence. New held-out questions are
required to assess generalization after development on this set.

## External targets and comparison limits

The official [HotpotQA board](https://hotpotqa.github.io/) lists distractor answer
EM/F1 72.69/85.04 for Beam Retrieval and fullwiki 67.46/80.52 for AISO. The official
[SQuAD 1.1 board](https://rajpurkar.github.io/SQuAD-explorer/) lists ANNA at
90.622/95.719. Checked 2026-09-20; these are the leaders displayed by those boards,
not an exhaustive claim about all newer research.

Our 100-question subsets and pooled 2,176-document corpus match neither official
HotpotQA setting nor SQuAD's supplied-passage test. These scores are directional
targets only. Beating them numerically here would not establish a leaderboard
win; that needs matching splits, corpus, task conditions and official evaluation.
Aim for 100%, while preserving failures and reporting what the experiment proves.

## Artifact integrity

Raw prompts, source text, responses, scores, receipts and databases remain in
`bench-runs/jev-qwen-answers-20260919/run-2/` outside git. This reviewed results
document records the scores without committing generated corpora or databases.

- Manifest SHA256: `c5fc2240365a80347307eef874420a37440778a20438a9429a2629df1628f4fb`
- Prepared requests SHA256: `346fba52aee50fbda599ae621bd9b6330ab36ca6226539ebf680db1cae8b07c8`
- Answers SHA256: `179709a9b3b96ea5b8eda77f80478b4da1df81dbb510a5d1b5b405f30214e84f`

Score with the snapshot's `python -m scone_memory.testing.jev_answers score`,
passing the original dataset directory and this run's output directory.
