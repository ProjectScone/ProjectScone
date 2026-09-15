# Recorded feedback as a ranking prior, replayed — 14 September 2026, re-measured 15 September

Recall can add a term for what people said about a passage
(`SCONE_FEEDBACK_WEIGHT`, [`retrieval/feedback_prior.py`](../src/scone_memory/retrieval/feedback_prior.py)).
This replays recorded judgements into a fresh engine and compares MRR with
the weight off and on. It checks two things: questions that paraphrase
judged ones should gain, and questions nobody judged should stay within 0.01.

Every number below is measured at the engine's defaults after main gave a
hashed embedder's vector lane a hundredth of the text lane's voice (it was
0.25). That change is what moved them; see
[What the vector default changed](#what-the-vector-default-changed).

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
  subject, as a control, and a third pair piles judgements up: six days
  of judging, the two ways of asking each asked three times, the repeats
  with their first space doubled. Judgements count per question (the hash
  its recall recorded), so the same words asked again would replace a
  judgement, not add one; a respelling is another question, as it would be
  from another caller.
- Evaluation, a day after the last judgement: each question is asked at
  weight 0 and at every weight measured, one weight after another on the
  same engine and the same recorded judgements, reading MRR@10. `judged`
  means the 12 paraphrases of the judged half. `unrelated` means all 36
  questions of the other half (both ways of asking plus the paraphrase).
- Engine: in-memory document store, vector index and event log,
  `HashEmbedder`, defaults otherwise (rank fusion, both lanes, the vector
  lane at a voice of 0.01 against the text lane's 1.0). No model.
- Storage: every table up to [Passages stored apart](#passages-stored-apart)
  stores all 48 passages at the same instant, so recency adds the same to
  every candidate. That section stores them an hour or a day apart instead.
- Reproduce: `scone_memory.bench.feedback_replay.measure(path, judge=...,
  judgements=..., stored_hours_apart=..., newest_first=...)` and
  `report(...)`.
  [`tests/benchmarks/test_feedback_replay.py`](../tests/benchmarks/test_feedback_replay.py)
  holds the chosen weight's result on every change.

## What the vector default changed

The first version of this file chose 0.0001 with the vector lane at 0.25.
After main lowered that voice to 0.01 (`HASHED_VECTOR_WEIGHT`, measured on
LongMemEval-S), the replay's MRR at 0.0001 equalled its MRR off under the
kind judge on both halves (0.7014 and 0.7222): the weight moved no rank.
Re-run with `vector_weight=0.25` given explicitly, the replay reproduces
every number of that version's one-instant table (kind judge at 0.0001:
0.7153 to 0.7708 on half a, 0.7222 to 0.7778 on half b, four paraphrases
rose, none fell). Nothing else merged from main moves that table.

It is not that the term is too small for the new scores. Under rank fusion
two candidates adjacent in the text lane sit one text place apart,
1/(60+r) − 1/(61+r), plus or minus the vector lane's say: less when the
vector lane ranks them the other way, more when it agrees. For a place's
disagreement at a voice of 0.25 that was 0.75 to 1.25 of a place, and the
term at 0.0001 (about
0.000193 from two judgements a day or two old) crossed the near-ties the
lanes' disagreement left, without crossing a place both lanes agreed on.
At 0.01 the range is 0.99 to 1.01 of a place, so there are no near-ties
left for a term to settle. Measured at weight 0 on the kind judge's replays:
riverside-hours' paraphrase, whose answer rose from second at 0.00014, sat
0.000259 below the passage above it; of the unrelated questions that fell
at 0.00014, fifteen had their answer first and the sibling's judged answer
second, 0.000262 to 0.000270 below it. One text place at the top is
0.000264. A term below a place moves almost
nothing; a term above it moves a judged passage past a place for every
question it is a candidate for, siblings' questions included. Rescaling the
term against the fused scores does not make a window between those two:
their width is set by the vector lane's voice, not by the scores' size. So
the weight was re-derived for the new default by the rule that chose it.

## How the weight and the bound were chosen

The first design was a term cut at 0.01, with weights 0.002 to 0.01. At the
previous vector voice it lifted judged paraphrases to MRR 1.0 on both halves
and dropped unrelated questions from 0.8692 to 0.61 (kind judge, both
halves): 36 of the 72 unrelated questions fell, 34 of them about a sibling
of a judged subject. Rank fusion separates first from second place by about
0.0003 per lane at full voice, so any term that size overrides the question.
The bound was then set from rank fusion itself, before any sweep:
`MAX_FEEDBACK_BOOST` is what first place is worth over second when both
lanes agree at full voice, 2 × (1/61 − 1/62) = 0.000529. At a hashed
embedder's default voice a place where both lanes agree is worth
1.01 × (1/61 − 1/62) = 0.000267, so the bound is now about two places. It
bounds each candidate's term, not who a candidate can pass: a leader sunk
and a follower lifted close twice the bound, and deeper ranks sit closer
than first and second. The weights chosen sit well under it.

The rule is to take the largest weight whose unrelated MRR stays within
0.01 of off on the replay judging half a (kind and strict judges), with half
b as its check. At the previous voice it picked 0.0001 from a sweep of
0.00002 to 0.001. At the new default the sweep was 0.00002, 0.00003,
0.00004, 0.00005, 0.0001, 0.00011, 0.00012, 0.00013, 0.00014, 0.00015,
0.00017, 0.0002 and 0.0005, and it picks **0.00013**: 0.00014 costs half a
0.11. Half b agrees at 0.00013 and is reported as its check. A finer look
between 0.00013 and 0.00014 is in [The edge](#the-edge); it is not what
chose the weight.

## Results

Two judgements per subject, MRR@10:

| Judge | Half judged | Set | n | off | 0.00005 | 0.0001 | **0.00013** | 0.00014 | 0.0002 | 0.0005 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kind | a | judged | 12 | 0.7014 | 0.7014 | 0.7014 | **0.7153** | 0.8125 | 0.8611 | 0.9583 |
| kind | a | unrelated | 36 | 0.8611 | 0.8611 | 0.8611 | **0.8611** | 0.7477 | 0.7338 | 0.6676 |
| kind | b (held out) | judged | 12 | 0.7222 | 0.7222 | 0.7222 | **0.7222** | 0.7778 | 0.9167 | 0.9583 |
| kind | b (held out) | unrelated | 36 | 0.8727 | 0.8727 | 0.8727 | **0.8727** | 0.7685 | 0.7685 | 0.6944 |
| kind | both | judged | 24 | 0.7118 | 0.7118 | 0.7118 | **0.7188** | 0.7951 | 0.8889 | 0.9583 |
| kind | both | unrelated | 72 | 0.8669 | 0.8669 | 0.8669 | **0.8669** | 0.7581 | 0.7512 | 0.6810 |
| strict | a | judged | 12 | 0.7014 | 0.7014 | 0.7847 | **0.7986** | 0.8542 | 0.8611 | 1.0000 |
| strict | a | unrelated | 36 | 0.8611 | 0.8611 | 0.8611 | **0.8611** | 0.7477 | 0.7338 | 0.6676 |
| strict | b (held out) | judged | 12 | 0.7222 | 0.7222 | 0.8056 | **0.8194** | 0.8333 | 0.9583 | 1.0000 |
| strict | b (held out) | unrelated | 36 | 0.8727 | 0.8727 | 0.8727 | **0.8727** | 0.7685 | 0.7685 | 0.6991 |
| strict | both | judged | 24 | 0.7118 | 0.7118 | 0.7951 | **0.8090** | 0.8438 | 0.9097 | 1.0000 |
| strict | both | unrelated | 72 | 0.8669 | 0.8669 | 0.8669 | **0.8669** | 0.7581 | 0.7512 | 0.6833 |

At 0.00013 no unrelated question fell under either judge, and the unrelated
delta is 0.0000 on both halves. Under the kind judge one paraphrase rose
(lisbon-trip, half a) and none on half b: its answers sit a whole text place
below their near misses, and the term does not cross one. Under the strict
judge three rose per half (alice-start, ledger-backup and lisbon-trip; then
riverside-hours, inventory-backup and parking-permits), since its judgements
against sink the passages that sat above the answers and the two terms
together cross a place. That holds only because the passages share a
creation instant; see [Passages stored apart](#passages-stored-apart).

At 0.00014, nine unrelated questions fell on each half under either judge
(kind: 4 and 2 rose; strict: 5 and 4). On half a they were all three of
team-tier's questions, two of globex-quota's, two of search-oncall's,
inventory-backup's first and tomatoes-watering's first; on half b their
sibling subjects' questions: all three of cafe-tier's, two of
payments-oncall's, acme-quota's second, northgate-hours' third,
ledger-backup's first and roses-watering's first. Sixteen of the eighteen
lost their place to the sibling's judged answer. None of the eight
standalone subjects fell. At 0.0002 one more fell on half a; at 0.0005,
fourteen to sixteen per half.
Two fresh judgements weigh about 1.93 here, so the bound cuts the term at
0.000529 from a weight of about 0.00027 up. Above that, the columns measure
the bound, not the weight.

**Corroboration control.** With one judgement per subject, every weight
measured (0.0001 to 0.0005) gives exactly the off ranks on both halves under
both judges: 0 rose, 0 fell. The strict judge recorded no judgement against
on those first askings (12 events per half, all useful), so this control
tests only that one useful judgement moves nothing. It does not separate
much: with corroboration removed (a useful judgement counting from the
first), the single judgements still give exactly the off ranks at 0.00013
and 0.0002 on both halves. Only the 0.0005 column moves then (6 rose, 9 or
10 fell per half), so the test keeps that column. The rule itself is proven
at unit level in `tests/retrieval/test_feedback_prior.py`.

**Judgements recorded.** Kind: 24 per half. Strict: 26 (half a) and 28
(half b).

### The edge

0.00013 is one text place from a term that costs. Two judgements made just
before a question weigh 2, not the replay's 1.93, so at 0.00013 a fresh pair
gives a term of 0.00026, the replay's term at about 0.000135. Between the grid's
0.00013 and 0.00014:

| Weight | kind a judged | kind b judged | strict a judged | strict b judged | unrelated a | unrelated b |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00013 | 0.7153 | 0.7222 | 0.7986 | 0.8194 | 0.8611 | 0.8727 |
| 0.000132 | 0.7292 | 0.7361 | 0.8125 | 0.8333 | 0.8588 | 0.8704 |
| 0.000135 | 0.7708 | 0.7778 | 0.8542 | 0.8333 | 0.8588 | 0.8704 |
| 0.000136 | 0.7708 | 0.7778 | 0.8542 | 0.8333 | 0.8588 | 0.8565 |
| 0.000138 | 0.7708 | 0.7778 | 0.8542 | 0.8333 | 0.8588 | 0.8519 |
| 0.00014 | 0.8125 | 0.7778 | 0.8542 | 0.8333 | 0.7477 | 0.7685 |

(The unrelated columns are the same under both judges.) From 0.000132 one
unrelated question falls per half: team-tier's third on half a, cafe-tier's
third on half b. At 0.000135 the kind judge's judged paraphrases score what
0.0001 gave them at the previous voice (0.7708 and 0.7778), for one
unrelated question per half. Applied on this finer grid the rule would pick 0.000138,
and half b fails it (0.8727 to 0.8519): past 0.000135 ledger-backup's first
question falls too. A weight picked inside a window a few millionths wide
does not carry to the half it was not picked on, so the weight stays at the
grid's 0.00013, and a fresh pair of judgements already reaches the window
(one unrelated question per half).

## Judgements piling up

A real log piles judgements on popular passages. Before this replay existed
the term grew with every judgement: at the previous vector voice a review
run judging the same two questions over four days took unrelated MRR at
0.0001 from 0.8611 to 0.7755 (half a) and 0.8773 to 0.7778 (half b), and
over six days to 0.7477 and 0.7731. Now only a passage's newest two
judgements each way count, and a question judged again replaces its
judgement, so that run's repeats fold to two. The six-day replay respells
instead, so every judgement is a question of its own and the hold is what
is tested. MRR@10:

| Judge | Half judged | Set | n | off | **0.00013** | 0.00014 | 0.0002 |
|---|---|---|---:|---:|---:|---:|---:|
| kind | a | judged | 12 | 0.7014 | **0.7153** | 0.8125 | 0.8611 |
| kind | a | unrelated | 36 | 0.8611 | **0.8611** | 0.7477 | 0.7338 |
| kind | b | judged | 12 | 0.7222 | **0.7222** | 0.7778 | 0.9167 |
| kind | b | unrelated | 36 | 0.8727 | **0.8727** | 0.7685 | 0.7685 |
| strict | a | judged | 12 | 0.7014 | **0.7986** | 0.8542 | 0.8611 |
| strict | a | unrelated | 36 | 0.8611 | **0.8611** | 0.7477 | 0.7338 |
| strict | b | judged | 12 | 0.7222 | **0.8194** | 0.8333 | 0.9583 |
| strict | b | unrelated | 36 | 0.8727 | **0.8727** | 0.7731 | 0.7731 |

Judgements recorded: kind 72 per half, strict 78 and 84. Every evaluated
recall's record says how many candidates were `held`: up to 12 (kind) and
15 (strict). At 0.00013 no question fell, the same paraphrases rose as with
two judgements, and both judges' rows equal their two-judgement rows, since
a subject's newest two judgements are as old as the two-day replay's. The
kind judge's rows equal them in every column; the strict judge's further
judgements against lift half b's unrelated questions at 0.00014 and 0.0002
(0.7685 to 0.7731).
A term that grew with the pile would cross the place two judgements sit
under: the test holds these rows so that it would fail.

## Passages stored apart

A real store's passages were written at different times. Stored at one
instant, they all get the same recency, and the prior only has to settle
what the lanes leave. Stored apart, recency's small differences break those
ties first, and they decide which near-ties the term crosses.

The same replay with the passages stored an hour or a day apart, the
file's last passage the newest (`oldest first`) or its first
(`newest first`). Kind judge, two judgements per subject, MRR@10:

| Stored | Half judged | judged off | 0.00005 | 0.0001 | **0.00013** | unrelated off | 0.00005 | 0.0001 | **0.00013** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| one instant | a | 0.7014 | 0.7014 | 0.7014 | **0.7153** | 0.8611 | 0.8611 | 0.8611 | **0.8611** |
| one instant | b | 0.7222 | 0.7222 | 0.7222 | **0.7222** | 0.8727 | 0.8727 | 0.8727 | **0.8727** |
| 1 h, oldest first | a | 0.7014 | 0.7014 | 0.7014 | **0.7153** | 0.8611 | 0.8611 | 0.8611 | **0.8611** |
| 1 h, oldest first | b | 0.7222 | 0.7222 | 0.7222 | **0.7500** | 0.8727 | 0.8727 | 0.8727 | **0.8657** |
| 1 h, newest first | a | 0.7014 | 0.7014 | 0.7014 | **0.7292** | 0.8611 | 0.8611 | 0.8611 | **0.8588** |
| 1 h, newest first | b | 0.7222 | 0.7222 | 0.7222 | **0.7222** | 0.8727 | 0.8727 | 0.8727 | **0.8727** |
| 24 h, oldest first | a | 0.5444 | 0.5444 | 0.5583 | **0.5722** | 0.7262 | 0.7262 | 0.7262 | **0.7239** |
| 24 h, oldest first | b | 0.5536 | 0.5536 | 0.5556 | **0.5583** | 0.7417 | 0.7394 | 0.6769 | **0.6676** |
| 24 h, newest first | a | 0.6625 | 0.6625 | 0.7458 | **0.7875** | 0.7824 | 0.7106 | 0.6644 | **0.6644** |
| 24 h, newest first | b | 0.6806 | 0.6806 | 0.7222 | **0.8472** | 0.8273 | 0.8273 | 0.8273 | **0.8273** |

At 0.00013 the unrelated questions that fell were cafe-tier's third and
northgate-hours' third (1 h, oldest first, half b); team-tier's third (1 h,
newest first, half a); riverside-hours' third (24 h, oldest first, half a)
with nine on half b: all three of cafe-tier's, two of acme-quota's, two of
payments-oncall's, ledger-backup's first and roses-watering's first; and
nine a day apart, newest first, on half a: all three of team-tier's, two of
globex-quota's, two of search-oncall's, inventory-backup's first and
tomatoes-watering's first. The strict judge loses the same questions an
hour apart and a day apart oldest first; a day apart newest first it loses
six on half a (0.7824 to 0.7060) and cafe-tier's third on half b, where its
unrelated MRR rises to 0.8366. Its judged columns are equal or higher.

Pooled over both halves at 0.00013 (kind judge), judged paraphrases gained
in every layout: 0.7118 to 0.7326 an hour apart oldest first and to 0.7257
newest first, 0.5490 to 0.5653 and 0.6715 to 0.8174 a day apart. Unrelated
questions lost 0.0035 and 0.0012 an hour apart, 0.0382 a day apart oldest
first and 0.0591 newest first.

The rule that chose 0.00013, re-applied to every layout: take the largest
weight whose unrelated MRR stays within 0.01 of off on half a. It picks
0.00003, since 0.00004 costs half a 0.044 a day apart, newest first, and
0.00005 costs it 0.072 (six questions). 0.00003 holds half b too, but it
lifts no judged paraphrase in any layout under either judge. 0.00002 lifts
none either, and still costs half b a day apart, oldest first, one question
(acme-quota's disk space question: 0.7417 to 0.7394).

So no weight measured passes. A term that does not know the question
crosses whichever near-ties a store's creation times leave, and any weight
large enough to lift a judged question can cost an unrelated one. The
weight stays off by default. 0.00013 is the weight chosen at one instant,
not one shown safe.
[`tests/benchmarks/test_feedback_replay.py`](../tests/benchmarks/test_feedback_replay.py)
holds the 1 h (both judges) and 24 h numbers quoted here.

## Cost

With the weight set, every recall reads up to 5,000 judgements from the
event log and folds those of its candidates. Recall latency was measured
with the weight at 0 and at 0.0001, alternating per question, at limit 10,
before the vector default changed. The read and the fold do not depend on
the weight's value or on the lanes' voices, and they were not re-measured.
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

- No identity is recorded. One caller can corroborate a passage by asking
  two questions, or one question spelled two ways, and lift it for every
  question it is a candidate for. The hold caps what repeating that buys
  at two judgements' worth, and it does not stop the first two.
- Passages stored apart were measured an hour and a day apart in file
  order and its reverse. A real store's creation times are neither, and
  its near-ties fall where they fall.
- The corpus is small and authored. The sibling pairs are the case that
  breaks a prior that does not know the question, and there are eight of
  them against eight standalone subjects. A real corpus's share of alike
  subjects decides whether a weight above 0.00013 costs more or less than
  here.
- The gain at the chosen weight is one of the 24 judged paraphrases under
  the kind judge and six under the strict one (both halves together). It
  is a near-tie breaker, not a re-ranker, and at the hashed default the
  lanes leave it few near-ties.
- The weight's scale is rank fusion's. `fusion="score"` and
  `"distribution"` produce scores about a hundred times larger, and they
  were not measured.
- Only `HashEmbedder`, at its default voice and at 0.25. A real embedder
  keeps the vector lane's full voice, where lanes that disagree by a place
  tie exactly; rank fusion's spacing does not change, so the bound holds,
  but the ties the prior breaks are different ties and the weight was not
  measured there.
- A question-aware prior, one that weighs a judgement by how alike the new
  question is to the judged one, is the obvious next step. With queries
  hashed in the event log (the default) there is nothing to compare, and
  siblings worded alike would still look alike.
