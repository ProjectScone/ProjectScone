# Validity diagnostics (1–5)

These are five small research diagnostics with reusable pure mechanisms, not
claims of invention or production features. Each has eight hand-authored cases,
including controls and counterexamples. All use the same evidence within each
comparison. No model, API, service, retrieval index, or credentials are used.
There is no held-out evaluation claim, learned routing, threshold selection, or
inferred provenance. The fixtures deliberately expose representation boundaries.

| ID | Mechanism | Main observed metric: baseline → method | Boundary exposed |
| --- | --- | --- | --- |
| 1 | Transitive necessary-premise invalidation | Error 1.625 → 0.5 conclusions/update | An omitted shared dependency leaves four conclusions stale |
| 2 | Group copied evidence by origin before noisy-OR | Brier loss 0.286704 → 0.179712 | Lower confidence worsens some true cases; incorrect grouping suppresses independent evidence |
| 3 | Separate event and observation cutoffs | Exact accuracy 0.625 → 0.875 | Incorrect event metadata reverses the improvement |
| 4 | Activate a decision only when all known conditions match | Exact accuracy 0.25 → 0.875 | An omitted safety precondition makes both methods fail |
| 5 | Preserve supplied conflicting alternatives | Plausible-set F1 0.854167 → 0.883333 | Spurious and resolved alternatives make retention worse |

The invalidation metric equally weights unnecessary and missed invalidations.
The method invalidates 1.375 conclusions/update versus 3.5 for global invalidation,
but misses four stale conclusions across the diagnostic. These are counts of
selected work, not measured runtime savings. Dependencies mean necessary
premises; alternative sufficient proofs need a different representation.

Origin grouping assumes complete perfect correlation within an origin and
independence across origins. Noisy-OR is justified only for independent sufficient
causes, not arbitrary fact confidence. Hand-authored binary outcomes and support
probabilities make the Brier values descriptive, not evidence of calibration.
A true copied claim explicitly has worse Brier loss after grouping.

Temporal records represent state transitions persisting until the next event.
The observation cutoff applies to both methods. Later observations correct an
identical event timestamp; exact timestamp ties preserve input order. Expiry,
retractions, simultaneous conflicts, and timestamp extraction are untested.
The labels represent state as known at the cutoff, not future hindsight.

Conditional activation uses supplied equality conjunctions. Missing conditions
withhold action; the fixture explicitly labels withholding as correct in those
contexts. A false stored premise, missing condition, stale context, disjunction,
or conflicting policy needs further work. This tests activation, not truth.

Conflict preservation retains 1.875 distinct alternatives/case versus 0.875 for
winner selection. Plausibility sets are independently authored evaluation labels,
not provided to the selection algorithms. F1 describes recall/precision of these
sets, not single-answer accuracy. Scores select the baseline winner only; the
method does not know which alternatives are spurious or obsolete.

Related primary papers supplied and verified by the coordinating agent:

- Dependency rollback: <https://arxiv.org/abs/2608.10502>
- BeliefMem: <https://arxiv.org/abs/2605.05583>
- Temporal mutable memory: <https://arxiv.org/abs/2609.16073>
- TANGLE: <https://arxiv.org/abs/2608.13921>

These references are related work, not assertions that these prototypes reproduce
their methods or establish new results beyond them. `run()` returns five reports
with per-case inputs, outputs, labels, controls, and failures. The common runner
owns artifact persistence and cross-track reporting.

Verification: nine focused pytest checks cover dependency cycles, transitive
closure, untouched sources, copied support, invalid probabilities, both temporal
cutoffs, event correction, missing conditions, duplicate alternatives, empty
inputs, and report serialization/counterexamples. Strict mypy covers the module
and tests. Full portfolio and repository checks are coordinated by the root agent.
