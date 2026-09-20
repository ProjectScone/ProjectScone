# Jev follow-up experiments — 2026-09-20

Two completed experiments show improvements and regressions. Neither setting is
ready to replace the baseline globally, and neither establishes official dataset
or held-out accuracy. Every failed attempt remains in the denominator.

## Native agent workflow: 200 questions

Immutable snapshot `ba8d35fb`, [protocol](jev-agent-answers-v1.protocol.md), native
EvidenceToolLoop, scoped memory search/read tools, Qwen 8B, local Docker Qdrant,
Jev 1.13 and Gemma 4 31B. All documents, text-index readiness and vector/chunk
counts were validated. All 200 turns completed, with retained sources and no
tool failure. All 203 Jev reranks applied with the pinned model. Input/code and
artifact integrity checks passed before scoring.

| Dataset | Previous single-pass EM | Agent EM | Previous F1 | Agent F1 |
| --- | ---: | ---: | ---: | ---: |
| All, 200 | 70.00% | 70.50% | 80.37% | 81.87% |
| HotpotQA, 100 | 59.00% | 64.00% | 71.23% | 78.10% |
| SQuAD, 100 | 81.00% | 77.00% | 89.50% | 85.65% |

Exact match: 10 wins, 9 losses, 181 ties. F1: 16 wins, 11 losses, 173 ties.
Abstentions fell 11 -> 8 overall, but rose 3 -> 5 on SQuAD. Agent full-turn
latency: median 1,590 ms, p95 5,244 ms. The previous single-pass preparation plus
generation median was 1,397 ms. Runs were sequential at different times; this
is a workflow comparison, not an isolated causal estimate of tool access.

197 questions used only the host initial search. Three used one additional
model-requested search; none used read_memory:

- HotpotQA `5abd6db755429933744ab7d0`: author-nationality bridge recovered;
  abstention -> `Irish`, EM 0 -> 1.
- HotpotQA `5ac1b3a75542994ab5c67dd2`: company founding-year bridge recovered;
  abstention -> `1971`, EM 0 -> 1.
- SQuAD `57268da7f1498d1400e8e39f`: correct `Three` -> abstention despite the
  needed sentence being present. More tool use did not guarantee a better answer.

The SQuAD losses include changed answer selection (`Lorentz force` instead of
`unified electromagnetic force`), unnecessary abstention, expanded answers, and
a name split across reversed passages (`Michael P.` / `Millardi`). The latter
exposed a real ingestion defect: initials were treated as preferred sentence
boundaries. A separate synthetic regression reproduces it; its fix is not part
of this frozen run. Other EM losses include aliases and formatting (`Amazon`
vs `Amazon.com`, `5` vs `five`), which remain losses under the unchanged metric.

## Identical-evidence reasoning pilot: 40 questions, 80 answers

Immutable snapshot `3fab4b2b`, [protocol](jev-reasoning-answers-v1.protocol.md).
Twenty questions per dataset were selected by ID hash, without selecting errors
or reading gold. Both arms regenerated answers from the identical saved Jev
evidence and prompt, with equal 2,048-token completion limits and alternating
order. Only Gemma's reasoning setting changed. This is a small development pilot.

| Dataset | Direct EM | Reasoning EM | Direct F1 | Reasoning F1 |
| --- | ---: | ---: | ---: | ---: |
| All, 40 | 62.50% | 65.00% | 77.38% | 82.54% |
| HotpotQA, 20 | 40.00% | 40.00% | 60.67% | 68.00% |
| SQuAD, 20 | 85.00% | 90.00% | 94.08% | 97.08% |

EM: 2 wins, 1 loss, 37 ties. F1: 3 wins, 1 loss, 36 ties. Reasoning improved
the response to the Ben McNiece date question, `Ghanaian` -> `Ghana`, and the
embargo answer's span. It shortened `December 1993` to `December`, losing EM.
Median generation latency increased 779 -> 5,537 ms; p95 5,693 -> 14,873 ms.
Direct completed 40/40; reasoning completed 39/40, with one empty ChatError at
47.37 seconds retained as failure (`hotpotqa:5abf8ae85542990832d3a14b`). Existing
receipts do not identify its precise provider/finish cause, so none is asserted.

The SQuAD pilot's 90% EM and 97.08% F1 cannot be compared directly with official
leaderboard scores or the full 100-question subset. A full-set confirmation is
needed before claiming general improvement. The pilot favors further testing
of evidence use, while its latency and failure are real costs.

## Artifacts and integrity

Artifacts stay outside git in the main checkout's `bench-runs` directory. Full
prompts, tool transcripts, retained sources, answers and per-question scores are
preserved. No gold labels, answer aliases or output strings were changed.

| Run | Manifest SHA256 | Answers SHA256 |
| --- | --- | --- |
| `jev-agent-answers-20260920/run-1` | `13fd3bf83a2c9f56304f5bc66852eb5cd4b4c8a38ab3a0c9b2ccda0b09087dc0` | `ce267ae34aaa92e55c21f13e475625205ec014551d11b7578aa68aceb6c972e9` |
| `jev-reasoning-answers-20260920/run-1` | `7b9380adcb762f218abb4c88ded21846228c87cc6dba8a3d2d302500869294f0` | `9aea38244a970ab54ea206edf6c87f5f8f9b66f0c7c18a798daece057b2e9649` |

Remaining work: full-set reasoning confirmation, ingestion-boundary reevaluation,
agent evidence-use improvements, new held-out questions and UI acceptance. These
experiments do not close the framework's capability goal.
