# Twenty Scone research diagnostics

Goal: implement and execute twenty distinct, bounded native experiments with
explicit hypotheses, baselines, retained observations and honest limits. These
are research prototypes, not twenty production features or claims of world-first
novelty. Models use configured APIs; infrastructure and artifacts remain local.

Four independent tracks share only `common.ExperimentResult`. No live service
configuration changes. Existing deferred-ingestion changes stay isolated.

| ID | Experiment | Intervention | Primary comparison |
| --- | --- | --- | --- |
| 1 | Dependency-aware invalidation | Trace affected conclusions | Global invalidation |
| 2 | Origin-correlated support | Group copied evidence | Independent noisy-OR |
| 3 | Bitemporal memory | Separate event and observation time | Latest ingestion |
| 4 | Conditional decisions | Retain activation conditions | Unconditional stored fact |
| 5 | Conflict-preserving memory | Keep incompatible alternatives | Highest-score winner |
| 6 | Missing-premise retrieval | Select a missing logical link | Relevance top-k |
| 7 | Evidence complementarity | Select useful combinations | Independent passage utility |
| 8 | Counterevidence search | Reserve budget for refutation | Confirmatory retrieval |
| 9 | Decision information gain | Ask what separates hypotheses | Highest relevance |
| 10 | Sufficiency stopping | Stop after complete support | Fixed evidence count |
| 11 | Sparse counterfactual contracts | Query a subset of worlds | Exhaustive compilation |
| 12 | Workload-directed compilation | Compile likely future states | Compile all states |
| 13 | Risk-weighted revalidation | Schedule by expected loss | FIFO refresh |
| 14 | Shared-premise reuse | Reuse common judgments | Recheck each decision |
| 15 | Drift audits | Sample stale confidence adaptively | Uniform audit budget |
| 16 | Negative evidence | Require closure before inferring absence | Ungated Jev judgment |
| 17 | Self-citation quarantine | Exclude derived assistant assertions | All text as evidence |
| 18 | Paraphrase stability | Require agreement across question forms | One judgment |
| 19 | Conflict decomposition | Separate existence of supporting/refuting facts | Joint support judgment |
| 20 | Empirical abstention calibration | Choose threshold on separate calibration split | Fixed threshold |

Implementation: `validity.py` owns 1–5, `selection.py` owns 6–10,
`efficiency.py` owns 11–15, and `reliability.py` owns 16–20. Each track provides
`run() -> list[ExperimentResult]`; reliability is asynchronous and accepts the
shared direct Jev client. Each track owns a matching test file and notes.
The root runner writes all observations incrementally, checks exactly 20 unique
IDs on complete runs, and renders a comparable index without conflating metrics.

Review focus: fabricated oracle signals, label leakage, unequal evidence/call
budgets, uncertainty mistaken for truth, stale-snapshot reuse. Simulations must
name supplied oracle metadata and include controls/counterexamples. Live prompts
and labels must be separate; provider failures remain failures, not successes.
Thresholds are frozen before evaluation; no held-out claim for synthetic fixtures.

Execution checklist:
- [x] Track 1: pure mechanisms, diagnostics, tests and limits.
- [x] Track 2: distinct evidence-selection mechanisms and diagnostics.
- [x] Track 3: cost/accuracy simulations with unused work included.
- [x] Track 4: direct Jev diagnostics and split calibration.
- [x] Run twenty, preserve reports, inspect failures and type-check.
- [x] Independent review, commit, push and draft MR.

Delivered in [draft MR 161](https://github.com/ProjectScone/ProjectScone/pull/161),
stacked on the experimental memory-contract branch. See [results](RESULTS.md)
for completed measurements, negative findings and remaining research gates.
