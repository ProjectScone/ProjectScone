# Repeated search-result compaction v1 results

**Decision: keep compaction disabled by default.** Across 200 paired development
questions, exact match remains **56.5%**. Offered tool-result bytes fall **28.11%**,
serialized provider-request bytes fall **5.07%**, and reported prompt tokens fall
**6.65%**. This is a measured presentation saving, with mixed answer changes;
it does not establish an overall accuracy improvement.

## Frozen comparison

The [protocol](public-search-compaction-v1.protocol.md) and production change
were committed and pushed as `94854a9` before inference. All 400 planned turns
reached terminal states: 395 completed and five failed. Frozen code, input,
embedding, model and logical-store checks passed before scoring. No failed turn
was retried or replaced. The original questions and raw answers were unchanged.

Both arms use all 100 HotpotQA and 100 SQuAD 1.1 development questions, the same
2,176 original source paragraphs including distractors, 2,835 cached BGE vectors,
a copied SQLite ledger and a separate self-managed Qdrant 1.19.1 instance.
The 200 reserved questions remain unrun. This pooled corpus experiment is not
the official per-question Hotpot distractor protocol or a held-out evaluation.

The sole varied option is `EvidenceToolLoop(compact_search_results=...)`.
Initial search is enabled and computation disabled. Even question indices run
off/on; odd indices run on/off. Every requested search actually executes in both
arms. The on arm can replace an exactly repeated, nonempty full payload with a
smaller reference to its first result. The reference also gives generic
next-step guidance; byte reduction and that guidance are not isolated effects.
Full fresh evidence receipts remain retained and validated before publication.

Inference uses the same `gemma4-e4b-ctx8k:latest`, 8,192-token context,
temperature zero, thinking disabled and 2,048-token per-call ceiling. The
original shortest-complete-answer instruction is preserved. There is no
fine-tuning, question-specific instruction, output-schema option, answer
reviewer, forced follow-up search or answer repair. Default budgets are four tool calls,
four rounds and 120 seconds per turn, including initial retrieval. Python is
3.14.7; embedding initialization uses cached BGE-small-en-v1.5 offline.

The unchanged upstream Hotpot evaluator agrees with all 100 off and 100 on
answer scores. Supporting-fact and joint metrics are not measured. Its first
invocation stopped before evaluation because this runtime lacked `ujson`;
providing the already cached dependency allowed the unchanged script to run.
No model call, scoring rule, answer or frozen artifact changed for that fix.

## Answer quality

| Dataset | Metric | Off | On |
|---|---|---:|---:|
| All 200 | Exact match | 56.5% | 56.5% |
| All 200 | Token F1 | 0.677965 | 0.681265 |
| Hotpot, 100 | Exact match | 42% | 44% |
| Hotpot, 100 | Token F1 | 0.549918 | 0.566267 |
| SQuAD, 100 | Exact match | 71% | 69% |
| SQuAD, 100 | Token F1 | 0.806012 | 0.796263 |

Exact match has **three gains, three losses and 194 ties**. F1 has eight gains,
seven losses and 185 ties. Hotpot contributes two EM gains and no EM losses;
SQuAD contributes one gain and three losses. All failures receive zero credit
in the full 200-question denominator for each arm. Completed exact-string
`INSUFFICIENT_EVIDENCE` answers number 21 off and 22 on. Other strings such as
`None` are scored verbatim, not silently reclassified or repaired.

There are 23 pairs with different raw answer strings. These are all six EM
changes; other wording changes and partial-credit differences remain in the
preserved raw records.

| Dataset / question ID | Off answer | On answer | EM effect |
|---|---|---|---|
| Hotpot `5ae692e35542996d980e7c0e` | Victor Emmanuel III | 11 November 1869 | 0 → 1 |
| Hotpot `5ae0c7e755429945ae95944c` | Bay Lake | Orange County | 0 → 1 |
| SQuAD `57107a3ea58dae1900cd69de` | Initial retrieval failed | Charleston | 0 → 1 |
| SQuAD `56e1b169cd28a01900c67a73` | computational power | None | 1 → 0 |
| SQuAD `57263ea0271a42140099d7c5` | seven | seven-layer | 1 → 0 |
| SQuAD `56f8b4d79b226e1400dd0e79` | The most radically anti-Semitic tract ever published | The newspaper described it as the most radically anti-Semitic tract ever published. | 1 → 0 |

The SQuAD gain is a failed-versus-completed pair, not demonstrated reasoning
improvement. The last loss also shows why exact match and token F1 must both be
reported: the longer answer has F1 0.75. No claim of statistical significance
or generalization follows from this single development run.

The saved actions illustrate different effects. For the attraction/county
question, off repeats the original search four times and returns the city;
on follows the repeated-result reference with a county-specific search and
returns the expected county. For the birthdate question, both arms issue the
same sequence of searches, but off names the king and on supplies his date of
birth. Conversely, the computational-power question uses the same searches
and neighboring-chunk read in both arms, yet on replaces the correct answer
with `None`. These traces motivate further testing of requested answer roles
and synthesis; they do not justify rewriting these benchmark questions or
adding their expected answers to retrieval.

## Retrieval and presentation

| Observation | Off | On |
|---|---:|---:|
| Fresh search preparations | 708 | 677 |
| Fresh tool preparations, all types | 770 | 769 |
| Identical nonempty search preparations | 287 | 256 |
| Offered search references | 0 | 256 |
| Offered read references | 0 | 4 |
| Full prepared payload bytes / observations | 1,737,668 / 770 | 1,658,495 / 769 |
| Offered tool payload bytes / observations | 1,737,390 / 768 | 1,249,030 / 771 |
| Provider request bytes / observations | 6,904,183 / 768 | 6,554,405 / 771 |
| Reported prompt tokens / usage observations | 1,512,995 / 767 | 1,412,360 / 771 |
| Reported completion tokens / usage observations | 19,027 / 767 | 19,066 / 771 |

Prepared payload totals include unavailable results. Offered totals count each
distinct tool-result message once, when first submitted to a model call;
provider-request totals include the actual serialized requests and repeated
conversation history. These denominators describe different observations and
must not be treated as interchangeable. Usage is absent for one cancelled
provider call. Token totals are provider-reported and may include cached input.
The between-arm savings include changed tool choices and the different failure
mix; they are not a compression ratio on an identical sequence of requests.

Search references explicitly record `searched_again: true`. The difference in
fresh-search counts comes from later model actions, not skipped searches.
Model-selected attempts change from 508 searches and 62 reads to 477 searches
and 96 reads. Four on-arm read attempts use the existing read-reuse path.
Full receipt retention means offered-byte savings are not storage savings.

When tools are disabled for final synthesis, both arms already use
`synthesis_history`, which deduplicates quoted sources and removes ranking
scores and tool-navigation messages. The compaction guidance is not included
in that final request. The experiment changes intermediate navigation payloads
and their resulting model actions, while final synthesis retains this existing
source projection. Offered-payload reduction therefore should not be read as
an equal reduction in every provider request.

Both initial searches succeed in 197 pairs: 143 parsed packets match exactly
and 54 differ. Three pairs contain at least one initial-search failure. The
54 successful differences (18 Hotpot, 36 SQuAD) affect only 81 `items[].score`
fields, with a maximum absolute difference of 0.000003. Source IDs, ordering,
text, metadata and coverage match in every
successful pair. These score differences are preserved in the raw requests;
the pairs are not described as byte-identical initial contexts.

Literal annotation coverage averages 94.32% off and 94.98% on. All annotated
snippets appear among offered sources for 179/200 off and 182/200 on questions:
Hotpot 82/100 → 85/100, SQuAD 97/100 → 97/100. Support-document recall averages
95.25% → 96%. These are literal source-coverage diagnostics; they do not prove
that evidence is sufficient, correctly interpreted or factually complete.

## Failures, calls and timing

All five failed turns occur on SQuAD and remain scored:

- `0012-off`: whole-turn `TimeoutError`, 120,018.8 ms. Three model calls
  completed; the fourth was cancelled without an HTTP response. The exception
  message is empty, so classification uses the recorded exception type and
  provider-call state as well as the message.
- `0013-on`, `0014-on`, `0015-off`, `0016-off`: initial retrieval unavailable
  with a timeout packet, 1,507.839–1,766.94 ms, and no provider calls.
- A later retrieval timeout is offered to the model in `0017-on`, which still
  completes. It is recorded as a tool error rather than a failed turn.

| Observation | Off | On |
|---|---:|---:|
| Completed / failed turns | 197 / 3 | 198 / 2 |
| Provider calls, including failures | 768 | 771 |
| Completed provider responses | 767 | 771 |
| Turns with 0 / 1 / 4 provider calls | 2 / 8 / 190 | 2 / 7 / 191 |
| Turn p50 / p95, ms (n=200 each) | 17,178.084 / 41,039.369 | 15,472.968 / 36,903.652 |
| Provider p50 / p95, ms | 4,538.155 / 11,004.146 | 4,053.584 / 10,285.809 |
| Maximum reported prompt / total tokens | 4,635 / 4,665 | 3,929 / 3,957 |

Provider timing uses all 768 off and 771 on attempts. All 1,538 completed
provider responses report `finish_reason=stop`; none reports an 8,192-token
prompt. These observations do not independently prove absence of provider-side
context trimming. Compaction did not reduce model-call count, and most turns
still use all four calls.

This was a shared-workstation run. The user's memory server, another resident
model and unrelated containers remained running. An observation soon after
the early failures showed about 4.8 GB of swap in use and substantial CPU use
by unrelated containers. That establishes contention was present, not that it
caused a particular timeout. Turn and provider latencies include that workload,
caching and recording overhead; they are not isolated throughput comparisons.

## Integrity and provenance

An independent audit reproduced the answer totals and byte/token counters,
checked all 400 original plans, question/request hashes and saved rows, and
verified 1,539 provider requests and 1,538 completed responses and parsed
decisions. It matched all search references to freshly prepared results and
verified that full fresh receipts remained retained. It also checked frozen
inputs, original ledger equality with only the two permitted additive indexes,
embedding and Qdrant-copy provenance, and 22 model-residency records. The report
retains all failures and does not promote compaction to the production default.

Original inputs, raw HTTP requests and decoded responses, prepared/offered
payload strings, receipts, decisions, answers, model-residency records and
host observations remain in ignored
`bench-runs/public-search-compaction-dev-2026-09-09/`. No generated artifacts,
credentials or private project corpus are added to this report.

- Model digest: `858470a535014f5c6847e388069aa64cb4f892e99309c851b59be288eec784cf`
- `manifest.json` SHA-256: `8f6ab16d8f2a0e1f91906d9d0283533b55594df8ff263755afed464bba2f081a`
- `observations.json` SHA-256: `e537eac84626f0bfc2adf7911ccb6befe80ca9ca46b67c324f1213d55282c828`
- `scores.json` SHA-256: `63ba6b1b2032fb86940f9ecfd96ba2605d8ea3030c63c0ac90e102f86d797cd6`
- `score.py` SHA-256: `96658e6953839609cf413fe2a07928283f3652db8f348936842efcfb8d78a82a`
- `check_official.py` SHA-256: `380fae6668e2168a4f4b48a84a305ba12f240d6be7fefd5fa1341cc4c2c3617d`
- `official-parity.json` SHA-256: `a9e4aeaf79b1e7d445a16dddac9e6ef6dbe8f305c7b94fe7eace7dc0b0b2a2a7`
