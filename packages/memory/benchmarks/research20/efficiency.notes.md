# Efficiency diagnostics 11–15

These are native deterministic algorithms evaluated on 50 synthetic scenarios
(ten per experiment), not provider, wall-clock, dollar, or semantic-accuracy
benchmarks. The unit-cost judgment callback is exact for its synthetic truth
function. Reports retain every workload, selection, cost, and adverse result.
Prior-work references establish related ideas, not novelty or empirical support
for the particular scheduler variants here.

| ID | Primary observation | Interpretation |
| --- | --- | --- |
| 11 | Exhaustive 168 calls; sparse plus fallback 96 | Full coverage removes savings. Turning fallback off causes 43 abstentions; coverage is 65.3%. |
| 12 | Compile-all 76; forecast 33; lazy 21 calls | Forecast shifts online work from 21 to 14 calls but adds 12 unused calls. Lazy wins total cost under these assumptions. |
| 13 | FIFO 74; weighted 45 impact units left stale | Equal per-case budgets. Wrong estimates produce two adverse cases. |
| 14 | Per occurrence 120; exact-identity reuse 29 calls | Claim-only caching produces 42 wrong answers across context/version changes. |
| 15 | Uniform 50; adaptive 57 stale impact-round units | Both spend 136 audits. Adaptive loses overall, including three adverse cases. |

## Mechanisms and controls

11 selects empty, full, and single-family deletion worlds before seeing requests.
Unknown worlds are never inferred using monotonicity: they either abstain or call
the synthetic judge and memoize that exact state. Cost includes all initial work,
unused initial worlds, and fallback. The synthetic judge is nonmonotonic.

12 receives a separate estimated demand distribution. Actual request streams
include shifted, reversed, absent, uniform, and concentrated demand. The selector
never sees future requests or their labels. All-state and initially empty lazy
memoization controls share the same truth callback. Precomputation cannot beat
lazy total calls under unit costs; its online miss advantage is not a latency
claim. No workload fitting or held-out calibration is claimed.

13 schedules supplied drift probability times supplied decision impact. FIFO uses
age. Both receive the same budget and entries. Actual changed entries appear only
in the evaluator after selection. Repair is perfect and unit cost. Scenarios
include no changes, all changes, wrong risk, zero budget, and full budget. Initial
cache creation is common to both refresh policies and outside this refresh epoch.

14 memoizes immutable premise keys containing claim, evidence version, context,
model, and policy. Context fixtures explicitly vary workspace, time, and access.
The unsafe claim-only ablation demonstrates invalidation failure. Complete host
identity and deterministic judges are assumptions; this does not prove cached
judgments true. Alternating versions can reuse each exact historical snapshot;
a version string must never be silently reused for different contents.

15 maintains estimated risk and age, reserving one cyclic exploration slot per
three audits. The rest rank risk times age times impact. Only observed audit
outcomes update estimates. Uniform cyclic audit has identical per-round budget.
Changes create stale entries until repair; repeated changes do not toggle truth.
Residual loss is measured after each round and accumulated. Surprise low-prior
changes are explicit failures. Budgets under three reserve no exploration slot.
As in 13, both begin with the same already-built cache; the experiment measures
incremental audit cost, not original ingestion.

## Verification

Seven behavioral tests pass in `test_portfolio_efficiency.py`, including exact
unknown-state abstention, fallback memoization and unused work, bad forecasts,
risk/FIFO disagreement, each premise identity dimension, equal audit budget,
surprise drift, and all five retained reports. Strict mypy passes for the module
and test file. The parent runner owns integrated portfolio and repository checks.
No external API, model installation, credentials, or service changes were used.
