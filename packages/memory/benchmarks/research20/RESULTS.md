# Research portfolio results — 2026-09-23

All twenty diagnostics completed: fifteen deterministic simulations and five
direct Jev API experiments. This is a screening portfolio, not twenty validated
inventions or an end-to-end RAG benchmark. Methods and baselines are described in
[the index](README.md) and the four track notes.

The integrated run used the same live fixtures, prompts and scoring rules as the
earlier pilot. Neither run changed them. Repeating the same cases measures some
provider variation; it does not create independent validation examples.

## Twenty comparisons

Arrows show the observed baseline → method, not always improvement. IDs 1–10
have eight constructed cases each, 11–15 have ten each, and 16–20 have twelve
evaluation cases each. These cases are deliberately small and correlated.

| ID | Experiment | Metric and direction | Integrated result | Main qualification |
| --- | --- | --- | --- | --- |
| 1 | Dependency invalidation | Mean invalidation error, lower | 1.625 → .500 | Hidden dependencies leave four conclusions stale across the cases. |
| 2 | Correlated provenance | Brier loss, lower | .2867 → .1797 | Origin families and probabilities are supplied; grouping can suppress independent support. |
| 3 | Bitemporal memory | Exact state accuracy, higher | 62.5% → 87.5% | Incorrect event timestamps break the method. |
| 4 | Conditional decisions | Activation accuracy, higher | 25% → 87.5% | Conditions are supplied; omitted conditions still fail. |
| 5 | Conflicting alternatives | Mean set F1, higher | .8542 → .8833 | Mean retained alternatives grow .875 → 1.875. |
| 6 | Missing premises | Complete coverage, higher | 50% → 75% | Requires supplied premise hints; wrong hints lose. |
| 7 | Complementary evidence | Realized group reward, higher | .70 → .75 | Exact subset search costs more compute than independent ranking. |
| 8 | Counterevidence | Decision accuracy, higher | 50% → 75% | Confirmation-only baseline is weak; misleading opposition hurts. |
| 9 | Information gain | Realized log loss in bits, lower | .7820 → .9174 | Negative result: wrong likelihood forecasts produce confident errors. |
| 10 | Sufficiency stopping | Mean passages read, lower | 4 → 1.5 | Completeness drops 100% → 75%; this is not a free efficiency gain. |
| 11 | Sparse contracts | Total simulated calls, lower | 168 → 96 | Includes compilation and exact fallback; disabling fallback causes 43 abstentions. |
| 12 | Forecast compilation | Total simulated calls, lower | 76 → 33 | Stronger lazy-cache control uses only 21; forecast wastes 12 initial calls. |
| 13 | Risk-weighted refresh | Remaining stale impact, lower | 74 → 45 | Equal budgets; wrong estimates make two scenarios worse. |
| 14 | Shared premises | Total simulated calls, lower | 120 → 29 | Requires complete immutable identities; claim-only caching makes 42 errors. |
| 15 | Adaptive audits | Cumulative stale impact, lower | 50 → 57 | Negative result despite equal 136-audit budgets. |
| 16 | Absence/closure gate | Correct eligibility cases, higher | 12/12 → 10/12 | Two valid absence claims rejected; two method questions versus one. |
| 17 | Self-citation filter | Correct eligibility cases, higher | 9/12 → 11/12 | One recovery uses changed evidence; one is variation on identical input. |
| 18 | Paraphrase stability | Correct eligibility cases, higher | 12/12 → 12/12 | Three correlated questions bring no measured gain here. |
| 19 | Conflict decomposition | Correct status cases, higher | 9/12 → 12/12 | Equal questions; three current equal-authority conflicts recovered. |
| 20 | Empirical calibration | Correct eligibility cases, higher | 12/12 → 12/12 | Twelve separate calibration cases select .5; no risk guarantee. |

Simulated calls and impact units are not measured latency, money or product
accuracy. Policies consume supplied metadata; scoring labels stay outside
selection. Extraction quality, annotation cost and stronger practical baselines
remain open. Exact caching and several other mechanisms are established ideas,
not claims of new research discoveries.

## Live measurements and repeatability

Both runs resolved `jev-latest` to **jev-1.13.0**, used five direct provider
requests, and reported **24,472 input / 3,108 output tokens per run**. Each run
sent 168 Noul questions. Experiment 18 shares its first question with its
baseline, and experiment 20 shares evaluation probabilities across thresholds;
adding the per-arm question counts would therefore double-count shared work.

| ID | Pilot baseline → method | Integrated baseline → method | Integrated batch time |
| --- | --- | --- | --- |
| 16 | 12/12 → 10/12 | 12/12 → 10/12 | 602 ms |
| 17 | 9/12 → 10/12 | 9/12 → 11/12 | 359 ms |
| 18 | 12/12 → 12/12 | 12/12 → 12/12 | 460 ms |
| 19 | 9/12 → 12/12 | 9/12 → 12/12 | 559 ms |
| 20 | 12/12 → 12/12 | 12/12 → 12/12 | 355 ms |

Times include a complete batch containing both arms. They are not per-question
times, per-arm speedups, time to first answer, or webapp latency. Five batches
are insufficient for a tail-latency estimate. Across both runs no method falsely
approved an unsupported binary fixture, but missed valid support and the small
authored case set prevent any general safety or accuracy conclusion.

In experiment 17, filtering removes the `assistant-bridge` false approval in
both runs. The integrated run's additional `user-fact` recovery uses identical
evidence and question text in both arms: probabilities .69 and .80 cross the
threshold despite no assistant entry being removed. That case reflects provider
variation, not a causal benefit from filtering. Integrated false approvals fall
1 → 0 and missed support 2 → 1, but only the former follows changed evidence.

## What merits further work

Conflict decomposition is the strongest live lead: equal question budgets and
the same three recovered conflicts in both runs. Next evaluation should include
untouched external cases with supersession, unequal authority and irrelevant
opposing statements before any production integration. The self-citation result
supports provenance-aware evidence filtering, but does not evaluate Scone's
existing production grounding policy or justify discarding all summaries.

Keep the absence gate, adaptive audit variant and current information-gain
policy out of production on this evidence. Sufficiency stopping needs a
reliable completeness estimator. Forecast compilation must justify extra total
work against lazy memoization, not merely beat compile-all.

## Verification and retained evidence

- 48 experimental tests passed, including the underlying memory contracts.
- Strict mypy passed for all nine portfolio Python modules.
- The complete report has all twenty unique IDs, no error, and matching source
  SHA-256 hashes; every provider request and response is retained without keys.
- Raw generated reports stay outside Git under the local ignored directory
  `bench-runs/research-portfolio-20-2026-09-23/`: `live-pilot.json` and
  `full-1.json`. Re-run instructions are in the README. These local artifacts
  are not downloadable from the MR; the committed tables preserve observations.

No live server configuration, dependencies or production retrieval behavior
changed. The research branch builds on the experimental memory-contract branch.
