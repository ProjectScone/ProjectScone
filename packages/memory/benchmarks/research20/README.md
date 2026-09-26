# Twenty native Scone research experiments

This portfolio turns twenty hypotheses into runnable diagnostics. It does not
claim twenty inventions, twenty production features, or superiority to other
RAG frameworks. The aim is to identify which mechanisms merit expensive
integration and independent evaluation, including the circumstances where they
fail. [Implementation plan](PLAN.md), [observed results for all twenty](RESULTS.md).

**Evidence types are deliberately separate:** IDs 1–15 are deterministic
simulations with supplied metadata/signals. IDs 16–20 call the configured direct
Jev API on authored synthetic text. None is an external benchmark or a complete
retrieval-to-answer evaluation. No external RAG framework is imported. Storage
and artifacts remain local; the live webapp is unchanged.

## The twenty experiments

| ID | Question tested | Code / interpretation |
| --- | --- | --- |
| 1 | Which conclusions should a source change invalidate? | [Validity](validity.py), [notes](validity.notes.md) |
| 2 | How much confidence comes from copies rather than independent evidence? | Validity |
| 3 | Can memory distinguish when a fact held from when we learned it? | Validity |
| 4 | Can stored decisions deactivate when their conditions stop holding? | Validity |
| 5 | When should memory preserve conflicting alternatives? | Validity |
| 6 | Does retrieving a missing premise beat another relevant passage? | [Selection](selection.py), [notes](selection.notes.md) |
| 7 | Can two individually weak passages be valuable together? | Selection |
| 8 | When is a deliberate search for counterevidence useful? | Selection |
| 9 | Which evidence request best separates competing answers? | Selection |
| 10 | When should a retrieval process stop collecting evidence? | Selection |
| 11 | How much counterfactual compilation can be skipped safely? | [Efficiency](efficiency.py), [notes](efficiency.notes.md) |
| 12 | When does anticipating future evidence states pay for itself? | Efficiency |
| 13 | Which stale decision should be rechecked first? | Efficiency |
| 14 | Can decisions reuse shared premise judgments without stale reuse? | Efficiency |
| 15 | Can limited audits detect drift before it affects many decisions? | Efficiency |
| 16 | Can absence claims require evidence that the search domain is complete? | [Reliability](reliability.py), [fixtures](reliability_cases.py) |
| 17 | Does excluding assistant self-citations prevent circular support? | Reliability |
| 18 | Does agreement across paraphrases identify fragile judgments? | Reliability |
| 19 | Does judging the existence of opposing arguments expose conflict? | Reliability |
| 20 | Can a separate calibration split improve the approval threshold? | Reliability |

All algorithms use their own inputs, not evaluation labels. Simulation inputs
such as dependencies, provenance, premise coverage, change probabilities and
costs are supplied. Their extraction and calibration are unsolved by these
experiments. Comparing a richer representation against a simpler baseline
demonstrates an assumption's consequences, not that real data will supply it.

## Run and inspect

Use the existing Scone Python environment with pytest and httpx installed:

```sh
# Fifteen offline diagnostics. The report explicitly remains incomplete for 20.
PYTHONPATH=packages/memory/src:packages/memory/benchmarks python -m research20.run \
  --output /tmp/scone-research20-offline.json

# All twenty, including five direct Jev batches. No local language model.
PYTHONPATH=packages/memory/src:packages/memory/benchmarks python scripts/local_env.py \
  --env-file /absolute/path/to/.env.local -- python -m research20.run \
  --live --output /tmp/scone-research20-live.json

PYTHONPATH=packages/memory/src:packages/memory/benchmarks python -m pytest -q \
  packages/memory/tests/experimental
```

Choose a new output file for every run. Results are saved after each experiment,
with raw observations, limits, primary references and a source-file hash manifest.
Completed live batches retain exact synthetic requests, probabilities, resolved
model, tokens and timing. Authentication headers are never recorded. Initialization
and provider failures make the report incomplete; they cannot create a positive
result. Cancellation may leave the latest completed experiment on disk.

Use `--track validity`, `selection`, `efficiency`, or `reliability` to run only
five diagnostics; `reliability` also requires `--live`. Partial reports retain
`complete: false` because that flag means all twenty completed.

Each experiment's primary metric has its own units and direction. Do not average
F1, Brier loss, model-call counts and eligibility accuracy into a single “RAG
score.” Extra metrics expose costs or failures that the primary score can hide.
Simulated call counts are not measured latency or billing savings.

## Live protocol, fixed before provider calls

Twelve cases per experiment. Experiment 20 additionally uses twelve disjoint
calibration fixtures, with evaluation cases shared with experiment 18. There are
therefore correlations both within and across experiments. No training or
public held-out dataset is used.

The direct client sends independent Noul questions with each case in that
question's instructions. Common state contains no evidence or labels. Labels
are passed only to metrics; experiment 20 uses calibration labels solely after
obtaining probabilities, and never uses evaluation labels to choose its threshold.
Model probabilities are not guaranteed calibrated for any of these tasks.

Support approvals use .8. Experiment 16 additionally requires closure at .8 and
counterexample presence at most .2. Experiment 18 requires all three paraphrases
to approve. Experiment 19 uses the existing contract's .8/.2 support/contradiction
policy for both arms. Experiment 20 selects the lowest of .5, .6, .7, .8, .9, .95
or abstention-above-1 that makes zero false approvals on the calibration split.
That is an empirical heuristic, not conformal coverage or a statistical guarantee.

Arms share a provider batch. Experiments 17 and 19 use equal question counts;
16 and 18 spend more method questions; 20 spends additional calibration questions.
Batch timings cannot be attributed causally to either arm. No questions or
thresholds are revised after inspecting the reported live run.

## Research context and next gate

The relevant ideas have substantial prior art. These experiments explore their
combination in Scone; none is a faithful reproduction of the cited systems:

* [Dependency-guided memory repair](https://arxiv.org/abs/2608.10502),
  [BeliefMem](https://arxiv.org/abs/2605.05583), and
  [mutable-memory temporal consistency](https://arxiv.org/abs/2609.16073)
  motivate testing how conclusions remain valid as memory changes.
* [FLARE](https://arxiv.org/abs/2305.06983),
  [information gain pruning](https://arxiv.org/abs/2601.17532), and
  [sufficient context](https://arxiv.org/abs/2411.06037)
  motivate selecting evidence by its effect on decisions.
* [Semantic compilation](https://arxiv.org/abs/2608.20845) and
  [RAGONITE](https://arxiv.org/abs/2412.10571) motivate paying for reusable
  evidence analysis, while measuring when that work is wasted.
* [TANGLE](https://arxiv.org/abs/2608.13921) studies unresolved memory conflicts;
  [TypeSafe's consistency cookbook](https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook)
  motivates checking stability rather than treating one probability as truth.

Before promotion: replace supplied oracle signals with measured extraction,
evaluate an untouched public temporal-memory/conflict dataset, compare stronger
baselines under matched total budgets, and run native source-retirement and
permission consistency checks. A successful simulation alone does not authorize
serving answers or actions from one of these policies.
