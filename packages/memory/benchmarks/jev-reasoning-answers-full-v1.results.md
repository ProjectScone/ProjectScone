# Fixed-evidence reasoning confirmation — 2026-09-20

On the reused 200-question development set, Gemma reasoning improves exact
match from 69.5% to 73.5% and answer-token F1 from 80.37% to 83.72%. Both
datasets improve, including the 160 questions outside the pilot. Median
generation time increases from 857 ms to 5,882 ms, and one reasoning attempt
times out. This supports further testing, not a global default change or a
claim of official leaderboard performance.

## Frozen comparison

Run `jev-reasoning-answers-20260920/run-2` completed from immutable commit
`978e6844`, following the [full-set protocol](jev-reasoning-answers-full-v1.protocol.md).
All 400 scheduled attempts are present; code, protocol, source, request and
artifact integrity checks passed before scoring. No attempts were retried or
removed, and no references or answer-normalization rules were changed.

The upstream evidence was prepared by Scone's ingestion and retrieval path:
Qwen3 Embedding 8B, local Docker Qdrant, hybrid retrieval and Jev 1.13 reranking.
This experiment reuses those exact saved messages; it does not rerun retrieval.
Both arms use Scone's native generation adapter and QA scorer, paid
`google/gemma-4-31b-it`, temperature 0, 2,048 completion tokens, and identical
90-second provider / 95-second capture limits. Arm order alternates. The only
request difference is reasoning effort: none versus medium.

The source evidence predates the name-initial chunking fix. This run cannot
measure that fix, nor the later finish-reason diagnostic fix. It also does not
combine reasoning with the separately evaluated native agent workflow.

## All 200 questions

| Dataset | Direct EM | Reasoning EM | Direct F1 | Reasoning F1 |
| --- | ---: | ---: | ---: | ---: |
| All, 200 | 69.50% | 73.50% | 80.37% | 83.72% |
| HotpotQA, 100 | 60.00% | 62.00% | 72.50% | 76.93% |
| SQuAD, 100 | 79.00% | 85.00% | 88.25% | 90.50% |

Paired exact match: 10 wins, 2 losses, 188 ties. Paired F1: 16 wins,
5 losses, 179 ties. Abstentions fall from 12 to 8; the separate timeout is
not counted as an abstention and remains scored zero.

| Dataset | Direct completed | Reasoning completed | Direct abstentions | Reasoning abstentions |
| --- | ---: | ---: | ---: | ---: |
| All | 200/200 | 199/200 | 12 | 8 |
| HotpotQA | 100/100 | 99/100 | 8 | 6 |
| SQuAD | 100/100 | 100/100 | 4 | 2 |

Generation latency, including failed attempts: median 857 -> 5,882 ms;
p95 7,804 -> 25,068 ms. These are generation timings, not full ingestion,
retrieval or UI response times. Cloud serving variability remains possible even
with alternating order and temperature zero.

## The 160 questions outside the pilot

Exclude the exact 40 IDs in the completed pilot's manifest. These remaining
questions were already used in previous RAG experiments, so they are not an
untouched holdout.

| Dataset | Direct EM | Reasoning EM | Direct F1 | Reasoning F1 |
| --- | ---: | ---: | ---: | ---: |
| All, 160 | 71.25% | 75.00% | 81.12% | 83.32% |
| HotpotQA, 80 | 65.00% | 66.25% | 75.46% | 77.92% |
| SQuAD, 80 | 77.50% | 83.75% | 86.79% | 88.73% |

Paired exact match: 7 wins, 1 loss, 152 ties. Paired F1: 11 wins,
4 losses, 145 ties. Thus the aggregate direction persists outside the pilot,
but this is still development-set evidence, not independent held-out proof.

## Answer changes and every regression

The 10 EM gains include three former abstentions (`Russian Empire`,
`Leonard Cohen`, `Percy Shelley`), a country/demonym change (`Ghanaian` ->
`Ghana`), and six more precise answer spans. These are score improvements;
the experiment does not separately establish citation faithfulness or whether
the model relied only on supplied evidence.

All five F1 regressions are retained below. The two EM regressions are the
date and full-name shortening errors.

| Question ID | Direct answer | Reasoning answer | F1 before -> after |
| --- | --- | --- | ---: |
| `hotpotqa:5ae0ba9055429924de1b715c` | December 1993 | December | 1.000 -> 0.667 |
| `squad:56dfb6d17aa994140058e057` | simple self-starting design | INSUFFICIENT_EVIDENCE | 0.500 -> 0.000 |
| `squad:572774cf5951b619008f8a54` | Michael P. Millardi | Millardi | 1.000 -> 0.500 |
| `hotpotqa:5add84a15542997545bbbd5e` | Comedy adventure | timeout, empty answer | 0.667 -> 0.000 |
| `squad:5728e8212ca10214002daa6d` | Talking to criminal investigators. | Use the arrest to make an impression on the officers. | 0.444 -> 0.308 |

The timeout was captured as `TimeoutError` after 95,006 ms with no public
output. Its underlying provider cause is unknown. The full-name regression
uses the old evidence with a split name; the separate chunking fix has not yet
been evaluated by regenerating these answers.

There are still 53 EM misses, including 38 HotpotQA and 15 SQuAD questions.
Some misses are answer-span/alias differences, others are wrong facts,
abstentions, incomplete answers or the timeout. Partial lexical overlap can
also reward an incorrect answer: the inertial-frame question changes from
abstention to an incorrect sentence and receives F1 0.125. F1 is token overlap,
not a semantic-correctness or faithfulness judgment.

Next useful comparisons are reingestion with the boundary fix, better agent
use of source-reading tools, and confirmation on new questions. A larger model
or more reasoning cannot substitute for checking those retrieval and evidence
use defects. No production defaults were changed by these measurements.

## Artifacts and reproduction

Raw prompts, answers, per-question scores and completion receipts remain in
`bench-runs/jev-reasoning-answers-20260920/run-2/` outside git. The committed
report is the reviewed score record; generated artifacts are not committed.

| Artifact | SHA256 |
| --- | --- |
| Manifest | `d7fae0414bc7935305def0625713983a69f16037fe8afb3718f706f672660d18` |
| Answers | `2e9f0847a607629c0e7f1763bad015e89f82a4a9149cbc88737cff60ecc0efdd` |
| Source prepared requests | `346fba52aee50fbda599ae621bd9b6330ab36ca6226539ebf680db1cae8b07c8` |

From the frozen snapshot's `packages/memory` directory, run:

```sh
PYTHONPATH=src python benchmarks/jev_reasoning_answers.py score \
  --dataset /Users/msturman00/ProjectScone/bench-runs/public-qa-2026-09-08 \
  --output /Users/msturman00/ProjectScone/bench-runs/jev-reasoning-answers-20260920/run-2
```

The source corpus pools 2,176 documents around sampled questions. This matches
neither official HotpotQA distractor/full-wiki evaluation nor official SQuAD
test evaluation. No leaderboard win or 100% accuracy is established.

Separate current-branch validation at `6fedac14`: full framework suite
13,284 passed, 3,280 optional tests skipped, 48 warnings; all seven affected
source modules pass mypy. This verifies the implementation regressions,
not the validity of every generated answer or full UI acceptance.
