# Jev answer evaluation v1 — 2026-09-19

The complete paired run improved reference-answer scores through Scone's native
RAG path. Exact match rose from 62.5% to 68.5%, and token F1 from 70.78% to
79.01%. **Exact match is not semantic answer accuracy**: correct aliases and
paraphrases can fail it. These results are neither a quality ceiling nor proof
that every answer failing exact match is wrong.

**Later readiness audit:** the first baseline preparation reported a partial
SQLite text index; the paired Jev preparation followed the next backfill pass.
The remaining 399 preparations reported no degraded lane. Both first-question
answers were `Ofcom` and scored EM/F1 1.0, so that pair contributes zero to the
reported answer-score gain. Nevertheless, index readiness was not held equal
on that pair: treat this as a qualified historical result, not a fully controlled
comparison. The Qwen run exposed the same setup flaw and was stopped unscored.
New ingestion and explicit pre-run readiness checks correct it for subsequent
experiments; recorded observations and headline metrics here are unchanged.

## Fixed comparison

The [protocol](jev-answers-v1.protocol.md) and harness were committed/pushed as
`d0aead10` before inference. Scone ingested all 2,176 downloaded paragraphs into
fresh SQLite stores using HashEmbedder. All 200 reserved questions ran in both
arms, with alternating order. Both used native `MemoryContext`, five sources,
8,000 context bytes, the existing short-answer prompt, and paid
`google/gemma-4-31b-it` at temperature zero with thinking disabled. The treatment
added the shipped Jev ranker, pinned to `typesafe/jev-1.13-20260917`.

All 400 preparations and answers completed. Jev applied on all 200 treatment
cases. Code, protocol and input hashes remained unchanged. The inference
command did not read gold labels; scoring followed terminal completion and
artifact/request integrity checks. No answer was retried or replaced.

Artifacts remain outside git at `bench-runs/jev-answers-20260919/run-1/` in the
primary checkout: exact requests, delivered sources, answers, context receipts,
timings, per-question scores, source index and integrity manifests.

## Results

| Dataset | Scone arm | Exact match | Token F1 | Abstentions | Failures |
| --- | --- | ---: | ---: | ---: | ---: |
| All, 200 | Hybrid | 62.5% | 70.78% | 35 | 0 |
| All, 200 | Hybrid + Jev | 68.5% | 79.01% | 14 | 0 |
| HotpotQA, 100 | Hybrid | 44.0% | 52.80% | 32 | 0 |
| HotpotQA, 100 | Hybrid + Jev | 57.0% | 69.33% | 11 | 0 |
| SQuAD, 100 | Hybrid | 81.0% | 88.77% | 3 | 0 |
| SQuAD, 100 | Hybrid + Jev | 80.0% | 88.68% | 3 | 0 |

Exact match improved on 15 questions, regressed on 3 and tied on 182. Token F1
improved on 21, regressed on 4 and tied on 175. Fourteen of the exact-match wins
replaced a baseline abstention. Of 22 total abstention-to-answer changes, 14
matched a reference exactly and 3 had zero token F1; fewer abstentions alone
are not proof of better answers. One new abstention occurred.

Exploratory paired bootstrap intervals, calculated after scoring using 10,000
question resamples with `Random(20260919)` and sorted IDs: exact-match change
+6.0 percentage points, 95% percentile interval +2.0 to +10.0; F1 change +8.225
points, interval +4.333 to +12.425. These describe uncertainty across these
questions, not repeated-generation variability or generalization.

| Timing | Hybrid p50 / p95 | Hybrid + Jev p50 / p95 |
| --- | ---: | ---: |
| Context preparation | 49 / 88 ms | 566 / 784 ms |
| Answer generation | 816 / 2,003 ms | 792 / 2,281 ms |
| Preparation + generation | 869 / 2,111 ms | 1,396 / 2,821 ms |

The end-to-end median increased by approximately 527 ms. Timings are from one
sequential run on a shared host, not an isolated performance benchmark.

## Score audit and regressions

All 63 treatment answers that failed exact match were inspected as question,
answer, reference and delivered-annotation coverage records. No scores were
manually changed. Examples of why exact match must not be called semantic
accuracy:

- “Denver Broncos” versus reference “Broncos”: exact match 0, F1 0.667.
- “Steve Carell” versus “Steven John Carell”: exact match 0, F1 0.4.
- “It had a resonant frequency” versus “the earth had a resonant frequency”:
  exact match 0, F1 0.75.
- “Teach” versus “teaching”: both exact match and token F1 are 0.

This is a qualitative audit, not an independently adjudicated semantic-accuracy
score. Some questions/reference spans are ambiguous or underspecified; they
remain in the original denominator.

The three exact-match regressions:

1. `hotpotqa:5a7281075542994cef4bc2e9`: “The Moscow Kremlin” became “Kremlin
   Arsenal.” Jev put the Arsenal passage first; it explicitly mentions 1736,
   while the question's “fortified complex” points to Moscow Kremlin. Both
   passages were delivered. This is a real answer-selection ambiguity/failure
   against the unchanged reference, not missing source retrieval.
2. `hotpotqa:5a7f341655429930675136a0`: the Linda McCartney publication answer
   became an abstention. The Alison Castle passage was present, but the second
   annotated source connecting Linda McCartney to the Beatles was absent.
   The baseline's exact match does not establish fully grounded reasoning.
3. `squad:56e0fde0cd28a01900c673ec`: the Tesla answer changed to the pronoun
   paraphrase above. The correct source ranked first in both arms.

The additional F1-only regression, `squad:57335ddbd058e614000b592f`, expanded
“the plain moraine plateau” to “the plain moraine plateau of Warsaw”; the
reference is “moraine.” Neither arm matched exactly.

## Remaining gaps and next evaluation

Delivered supporting-document recall matched the retrieval comparison:
85.75% -> 97%. A descriptive audit found every annotated HotpotQA support
sentence (or a SQuAD answer span) present in the delivered context for
73% -> 91.5% of questions. Exact substring coverage is not semantic entailment.

There are both retrieval and answer-selection gaps. One retrieve-then-answer
pass cannot reliably recover bridge entities absent from its initial candidates.
Scone's existing agent loop can perform follow-up `search_memory` and
`read_memory` calls with scope enforcement and source revalidation; that path
needs a separate measured comparison. It was not enabled in this experiment.

Preserve this baseline when testing the agent loop. The same 200 questions are
now development/evaluation data, not an untouched holdout. Do not tune aliases,
questions or scoring to raise these numbers. A strong permitted neural embedder,
independent semantic adjudication, new held-out data, repeated runs and real UI
acceptance remain separate requirements. Jev is still opt-in.
