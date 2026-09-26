# Direct Jev diagnostics (16–20)

These five experiments isolate semantic judgment on authored text. Each has
twelve evaluation cases. Experiment 20 additionally uses twelve distinct
calibration cases; its evaluation cases are shared with experiment 18. Request
payloads contain evidence and judgments to perform, never expected answers.

The first live run resolved to `jev-1.13.0`. It made five provider requests,
containing 168 Noul questions total, and reported 24,472 input tokens and 3,108
output tokens. Both arms share each batch; there is no per-arm latency claim.
Results were not used to change prompts, thresholds or fixture labels.

The integrated confirmation run retained all fixtures and prompts. Experiment 17
improved to 11/12 for the method (one missed support case); all other accuracy
counts below repeated. That extra recovery has identical input in both arms;
it reflects provider variation, not filtering. See [both runs and batch timings](RESULTS.md). The details
below describe the pilot specifically, not a pooled independent sample.

| ID | Baseline → method, correct evaluation cases | Interpretation |
| --- | --- | --- |
| 16 | 12/12 → 10/12 | Adding closure and counterexample gates withheld two supported negative claims; reject this formulation pending new development data. |
| 17 | 9/12 → 10/12 | Filtering assistant entries removed one false approval; two explicit user facts/corrections were still withheld. |
| 18 | 12/12 → 12/12 | Three paraphrases added questions without a measured eligibility gain here. |
| 19 | 9/12 → 12/12 | Existence of supporting/refuting arguments exposed all three conflicts missed by the whole-packet questions. |
| 20 | 12/12 → 12/12 | Calibration selected .5, but this separated evaluation set did not change eligibility relative to .8. |

## Failures and assumptions

Experiment 16 did not falsely approve an unsupported claim, but it missed a
complete inventory and an explicit statement that Nova has no search service.
On the complete inventory, closure probability was .78 and presence probability
.63. On the explicit absence statement, closure was .32. A compound semantic
question can be less dependable than a direct support check. No numeric gate
was relaxed after observing these failures.

Experiment 17's baseline saw role-labelled text but had no explicit provenance
filter. It is a diagnostic control, not the current production Scone grounding
policy. The method relies on trusted role metadata and excludes all assistant
entries, even potentially accurate summaries. Both methods withheld a direct
user statement and a user correction. Actual source authority is not inferred.

Experiment 18's three judgments are correlated. Agreement is not independent
corroboration. The mean within-case probability range was .0242, and no case
crossed the .8 boundary between question forms in this run. This easy diagnostic
does not establish that stability gating is useless on harder distributions.

Experiment 19 treats two current, equally authoritative incompatible records as
conflict; it does not decide supersession or which source is true. The baseline
classified those conflicts as uncertain or refuted. The method named conflict
using separate existential judgments. There are four gold statuses, with
uncertain an additional possible output, not a gold category.

Experiment 20 deliberately reports empirical threshold selection, not conformal
prediction. Twelve calibration cases cannot establish a risk guarantee. The
evaluation cases are different from calibration, but remain authored and are
not an untouched external benchmark. Shared evaluation inputs in 18 and 20
must not be counted as independent validation.

Every binary experiment reports coverage, false approvals and missed support,
so a stricter gate cannot hide behind approval precision. Experiments 16 and 18
spend two and three method questions per case respectively; 17 and 19 have
matched question counts. Calibration has a separate twelve-question setup cost.

Most promising follow-up: an independent conflict dataset for experiment 19,
including temporal supersession, disputed authority, and irrelevant opposing
claims. An internal 12/12 is a reason to investigate, not to enable the policy
in the live product.

Primary context: [sufficient context](https://arxiv.org/abs/2411.06037),
[dependency-aware memory repair](https://arxiv.org/abs/2608.10502),
[TANGLE conflicts](https://arxiv.org/abs/2608.13921), and
[TypeSafe consistency](https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook).
