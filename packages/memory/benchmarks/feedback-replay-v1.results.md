# Recorded feedback as a ranking prior, replayed — 14 September 2026

Recall can add a term for what people said about a passage
(`SCONE_FEEDBACK_WEIGHT`, [`retrieval/feedback_prior.py`](../src/scone_memory/retrieval/feedback_prior.py)).
This replays recorded judgements into a fresh engine and compares MRR with
the weight off and on. It checks two things: questions that paraphrase
judged ones should gain, and questions nobody judged should stay within 0.01.

## Method

- Subjects: [`feedback-replay-v1.json`](feedback-replay-v1.json), 24
  authored subjects, 48 stored passages. Each subject has one passage that
  answers it, one near miss on the same subject, two ways of asking it and
  a paraphrase that is only ever evaluated. Sixteen subjects come in eight
  sibling pairs worded alike: two rate tiers, two storage quotas, two
  clinics, two start dates, two backups, two on-call rotas, two watering
  schedules and two departures. A passage judged for one sibling is
  therefore a close candidate for the other. Eight subjects stand alone.
- Halves alternate (a, b), so every sibling pair has one subject in each.
  The halves were fixed before anything was measured, and the replay runs
  twice, judging each half once.
- Judging: the judged half's two ways of asking are recalled a day apart
  (limit 5) with the weight off. The `kind` judge marks the answering
  passage useful when it is shown. The `strict` judge also marks not useful
  every passage shown above it (all five when it is not shown). A second
  pair of runs judges only the first way of asking, one judgement per
  subject, as a control.
- Evaluation, a day after the last judgement: each question is asked at
  weight 0 and at every weight measured, one weight after another on the
  same engine and the same recorded judgements, reading MRR@10. `judged`
  means the 12 paraphrases of the judged half. `unrelated` means all 36
  questions of the other half (both ways of asking plus the paraphrase).
- Engine: in-memory document store, vector index and event log,
  `HashEmbedder`, defaults otherwise (rank fusion, both lanes). No model.
- Reproduce: `scone_memory.bench.feedback_replay.measure(path, judge=...,
  judgements=...)` and `report(...)`.
  [`tests/benchmarks/test_feedback_replay.py`](../tests/benchmarks/test_feedback_replay.py)
  holds the chosen weight's result on every change.

## How the weight and the bound were chosen

The first design was a term cut at 0.01, with weights 0.002 to 0.01. It
lifted judged paraphrases to MRR 1.0 on both halves and dropped unrelated
questions from 0.8692 to 0.61 (kind judge, both halves): 36 of the 72
unrelated questions fell, 34 of them about a sibling of a judged subject. Rank fusion separates first from
second place by about 0.0003 per lane, so any term that size overrides
the question. The bound was then set from rank fusion itself, before the
sweep below: `MAX_FEEDBACK_BOOST` is what first place is worth over second
when both lanes agree, 2 × (1/61 − 1/62) = 0.000529.

The weights 0.00002 to 0.001 were swept on the replay that judges half a
only (kind and strict judges). The rule, stated before half b was run, was
to take the largest weight whose unrelated MRR stayed within 0.01 of off:
**0.0001**. Half b is its held-out check. The other weights are reported
for both halves so the trade can be seen, but none of them was chosen on
half b.

## Results

Two judgements per subject, MRR@10:

| Judge | Half judged | Set | n | off | 0.00005 | **0.0001** | 0.0002 | 0.0005 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| kind | a | judged | 12 | 0.7153 | 0.7153 | **0.7708** | 0.8125 | 0.8125 |
| kind | a | unrelated | 36 | 0.8611 | 0.8611 | **0.8611** | 0.7755 | 0.7477 |
| kind | b (held out) | judged | 12 | 0.7222 | 0.7222 | **0.7778** | 0.7778 | 0.8194 |
| kind | b (held out) | unrelated | 36 | 0.8773 | 0.8773 | **0.8773** | 0.7778 | 0.7731 |
| kind | both | judged | 24 | 0.7188 | 0.7188 | **0.7743** | 0.7951 | 0.8160 |
| kind | both | unrelated | 72 | 0.8692 | 0.8692 | **0.8692** | 0.7766 | 0.7604 |
| strict | a | judged | 12 | 0.7153 | 0.7153 | **0.7708** | 0.8542 | 0.8542 |
| strict | a | unrelated | 36 | 0.8611 | 0.8611 | **0.8611** | 0.7755 | 0.7477 |
| strict | b (held out) | judged | 12 | 0.7222 | 0.7639 | **0.7778** | 0.8194 | 0.8611 |
| strict | b (held out) | unrelated | 36 | 0.8773 | 0.8773 | **0.8773** | 0.7917 | 0.7778 |
| strict | both | judged | 24 | 0.7188 | 0.7396 | **0.7743** | 0.8368 | 0.8576 |
| strict | both | unrelated | 72 | 0.8692 | 0.8692 | **0.8692** | 0.7836 | 0.7627 |

At 0.0001, four paraphrases rose and none fell, under either judge. Those
four were cafe-tier and northgate-hours when half a was judged, and
team-tier and riverside-hours when half b was. All four belong to sibling
pairs: the prior helped exactly where the near miss and the answer were
tied. The unrelated delta is 0.0000 on both halves.

At 0.0002, 15 unrelated questions fell under either judge (kind: 5 rose;
strict: 8 rose). They were all three questions of team-tier and cafe-tier,
and questions of globex-quota, acme-quota, search-oncall, payments-oncall,
tomatoes-watering, roses-watering and ledger-backup. Each is a sibling of
a judged subject, and each lost its place to the sibling's judged answer.
None of the eight standalone subjects fell at 0.0002. At 0.0005 (kind
judge), inventory-backup, northgate-hours' paraphrase and globex-quota's
other question fell too. Two fresh judgements weigh about 1.93 here, so the
bound cuts the term at 0.000529 from a weight of about 0.00027 up. Above
that, the columns measure the bound, not the weight.

**Corroboration control.** With one judgement per subject, every weight
gives exactly the off ranks on both halves under both judges: 0 rose, 0
fell. The strict judge recorded no judgement against on those first
askings (12 events per half, all useful), so this control tests only that
one useful judgement moves nothing.

**Judgements recorded.** Kind: 24 per half. Strict: 26 (half a) and 28
(half b).

## Cost

With the weight set, every recall reads up to 5,000 judgements from the
event log and folds those of its candidates. Recall latency was measured
with the weight at 0 and at 0.0001, alternating per question, at limit 10.
The setup was the 48 passages above, a SQLite event log of 1,000 or 5,000
synthetic judgements spread over the 48 passages (four in five useful, all
fingerprinted), 48 questions, 3 rounds per pass and three passes. The
machine was shared, at load 60 to 70, so the passes vary. The table gives
the median of the three pass medians:

| Judgements in the log | weight 0 | weight 0.0001 |
|---:|---:|---:|
| 1,000 | 7.40 ms | 24.83 ms |
| 5,000 | 5.66 ms | 79.62 ms |

About half of what remains is the event log decoding 5,000 JSON payloads.
Before timestamps were parsed once per judgement, the same passes gave
36.38 and 116.04 ms with the weight on. That fold alone, run before and
after in one process on the same events, went from a median of 18.90 ms
to 10.30 ms. Nothing is cached between recalls: a judgement recorded by
another process counts on the next recall. The trade is that cost.

## What this does not show

- The corpus is small and authored. The sibling pairs are the case that
  breaks a prior that does not know the question, and there are eight of
  them against eight standalone subjects. A real corpus's share of alike
  subjects decides whether a weight above 0.0001 costs more or less than
  here.
- The gain at the chosen weight is four of the 24 judged paraphrases
  (both halves together). It is a near-tie breaker, not a re-ranker.
- The weight's scale is rank fusion's. `fusion="score"` and
  `"distribution"` produce scores about a hundred times larger, and they
  were not measured.
- Only `HashEmbedder`. A real embedder spaces the vector lane differently;
  rank fusion's spacing does not change, so the bound holds, but the ties
  the prior breaks are different ties.
- A question-aware prior, one that weighs a judgement by how alike the new
  question is to the judged one, is the obvious next step. With queries
  hashed in the event log (the default) there is nothing to compare, and
  siblings worded alike would still look alike.
