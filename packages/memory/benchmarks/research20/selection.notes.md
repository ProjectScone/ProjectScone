# Selection research diagnostics: IDs 6–10

These are five separate, executable research diagnostics, not novel algorithms,
production retrievers, or evidence of superior RAG. Each uses eight constructed
cases: four target cases, two controls, and two adverse cases. Fixed random seeds
shuffle pools and perturb relevance; the cases remain a small family of designed
fixtures, not eight independent natural-language tasks. No services, APIs, models,
credentials or new dependencies are used.

Policies receive only candidate metadata, supplied premise requirements, supplied
utility groups, or supplied likelihoods. Scoring truth maps, ground-truth claim
labels, and realized probe outcomes are not passed to selection policies. Recorded
observations retain these separately so the scores can be recomputed. Evidence
strengths in ID8 are supplied observations consumed by the same signed-sum answer
rule for both policies; they are not learned or checked against text.

| ID | Mechanism | Baseline → method | Boundary exposed |
| --- | --- | --- | --- |
| 6 | Greedy missing-premise coverage | complete coverage .50 → .75 | Wrong premise hints select irrelevant bridges; top relevance wins adverse cases. |
| 7 | Exact joint subset utility | realized reward .70 → .75 | Correct-looking but false complementary premise destroys pair reward. |
| 8 | Reserve one challenge slot | decision accuracy .50 → .75 | Misleading opposing evidence overturns true claims. |
| 9 | Binary expected information gain | realized log loss .7820 → .9174 bits | Confident wrong likelihoods make the aggregate result worse. |
| 10 | Stop on complete support hints | reads 4.0 → 1.5 | Completeness falls from 1.00 to .75; saved reads are not free accuracy. |

6: Both policies see the same four passages and select two. The method greedily
covers missing supplied premises; the baseline ranks relevance. Known premises and
the required logical chain are supplied, not inferred. FLARE is related active
retrieval work, not the algorithm implemented here.

7: Both select two from the same four passages. The method enumerates six subsets
and rewards completed supplied premise groups. The baseline takes the top two
independent supplied utility estimates. This is exact small-pool optimization,
with combinatorial scaling and more computation than independent ranking. The
budget comparison concerns consumed evidence, not total compute. This tests
complementarity separately from ID6: a pair can earn reward even when neither
member earns any singleton reward.

8: Both consume three passages from one four-candidate pool. The baseline is
explicitly confirmation-only, a weak comparator. The method reserves one slot
for opposing stance, then both aggregate the selected evidence strengths. Stance
discovery, credibility estimation, candidate generation and the cost of another
search are excluded. It is a challenge-budget diagnostic, not proof that any
counterevidence is reliable.

9: Both consume one probe from the same three candidates. The method chooses
binary mutual information under supplied conditional likelihoods; both update
the same .5 prior by Bayes' rule. Expected information gain is reported separately
from realized log loss against the true hypothesis. Outcomes are deliberately
constructed, not draws proving calibrated likelihoods. The adverse likelihoods
produce confident wrong conclusions. This is related to information-gain pruning
but does not reproduce that paper's architecture or evaluation.

10: Both share ranking and a maximum of four reads. The baseline consumes four;
the method stops once supplied premise hints cover every requirement. Actual
coverage is assessed through separate scoring maps. There is no claim of reduced
indexing cost or end-to-end latency; annotation and sufficiency-estimator costs
are not modeled. False sufficiency is deliberately retained as a failure.

Related primary work (URLs supplied and verified by the coordinating track):
- Active Retrieval Augmented Generation (FLARE): https://arxiv.org/abs/2305.06983
- Information Gain Pruning: https://arxiv.org/abs/2601.17532
- Sufficient Context: https://arxiv.org/abs/2411.06037

Validation: six focused behavior/reporting tests cover duplicate-premise avoidance,
complementary pair selection, challenge reservation with no-opposition fallback,
known entropy values, sufficiency boundaries, deterministic output, case retention,
recomputed means and evidence budgets. Strict mypy checks the implementation.
