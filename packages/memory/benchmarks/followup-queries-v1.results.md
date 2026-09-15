# Follow-up queries on two-turn pairs — 14 September 2026

A follow-up turn such as "since when?" is searched as asked and, with
`carry`, also with the named terms of the earlier user turn. This
compares second-turn recall with the setting off and with `carry`, and
checks that first turns are untouched. No model ran: `rewrite` needs one
and is covered by tests, not measured here.

## Method

- Pairs: [`followup-pairs-v1.json`](followup-pairs-v1.json), 26 two-turn
  pairs over 80 stored passages (every pair's answers plus 32 distractors
  that share a follow-up's wording with other subjects: other people's
  start dates, other storage quotas, other clinics' walk-in hours). The
  first ten pairs take their subjects from
  [`entity_graph/fixtures-v1.jsonl`](entity_graph/fixtures-v1.jsonl) and
  `tests/fixtures/generation/v1.json`; the other subjects, every second
  turn and every distractor were authored for this set. 23 second turns
  lean on the first; 3 are standalone controls.
- Splits alternate (13 development, 13 held-out) and were fixed before
  anything was measured. The fusion rule was chosen on development only.
- Engine: in-memory document store and vector index, `HashEmbedder`,
  defaults otherwise. Both surfaces at limit 5: `MemoryContext.prepare`
  (a hit is the answer's episode among `references`) and
  `integrations.chat.recall_context` (among the supplied `episode_ids`,
  with a budget every passage fits).
- The first turn is asked alone; the second after the first and the
  assistant reply "Here is what memory holds.", which names nothing.
- Reproduce: `scone_memory.bench.followup.measure(path, split=...)`;
  `tests/benchmarks/test_followup_pairs.py` runs it on every change.

## Results

| Split | Surface | First turn R@5, off | First turn R@5, carry | Second turn R@5, off | Second turn R@5, carry |
|---|---|---:|---:|---:|---:|
| development (13) | context | 13 | 13 | 8 | 13 |
| development (13) | chat | 13 | 13 | 8 | 13 |
| held-out (13) | context | 13 | 13 | 12 | 13 |
| held-out (13) | chat | 13 | 13 | 12 | 13 |
| all (26) | context | 26 | 26 | 20 | 26 |
| all (26) | chat | 26 | 26 | 20 | 26 |

Six second turns were gained with carry: atlas-status, dana-teacher,
tomas-car-age, priya-subject, mei-when and victor-manager. None was lost. A carried query was searched for 17 of the
26 second turns; all three standalone controls were left unchanged.
Every first turn asked as the second turn after another pair's first
turn is left alone too (26 of 26; before the review fix below, 2 had an
unrelated name carried in and a changed top 5).

**Cost.** A carried query is one more recall. Preparing the 26 second
turns five times each on the context surface took a median of 2.28 ms
off and 4.22 ms with carry (p90 6.89 ms and 7.50 ms; 130 preparations
each, alternating order, in-memory stores on a machine shared with other
test runs).

**Fusion choice (development only).** Reciprocal rank fusion of the two
lists at second-query weights 1, 2 and 4 each gave 8/13 on the context
surface — the same as off — because the second query holds the
question's own words, so the question's matches rank in both lists and
outscore the passage only the carried terms found. Interleaving by rank
gave 13/13 with either list first; the question's list first was chosen,
so its best passage keeps the lead. Held-out was measured only with the
chosen rule.

**What carry missed.** Six follow-ups had no query carried:
"What's the escalation path?" and "And when is the half-term break?" have
two telling words and no referring word, so they read as standalone (both
held-out; the rule was not changed after seeing them); "And Lab B?" and
"Is it open on Sundays?" name something of their own ("Lab B",
"Sundays"), so their referring word or shortness does not count; "Is it
encrypted?" and "What about the team tier?" follow first turns whose
subjects are lower-case ("ridge", "the cafe tier"), which carry does not
read as names. Their second turns were found anyway by the question's
own words.

**Review fix (same day).** A review found that a question naming its
own subject was changed by a referring word ("Is there a meeting with
Bob Smith on Friday?" carried the earlier turn's name, because of
"there"), which the three standalone controls, holding no referring
word, could not show. Such a question now leans only through "since
when", "what about", "how about", or a closing "too" or "as well". The
table above was re-measured after that change and is unchanged; the
carried count went from 18 to 17 (greenleaf-sunday, held-out, no longer
carried and still found). The timings above were not re-measured: the
change adds no recall.

## Limits

26 authored pairs over 80 short passages, with a hashed-token embedder:
this shows the mechanism working and the first turn untouched, not a
recall rate on real conversations, where the off baseline, the share of
follow-ups carry recognises, and the cost of a wrong carry are all
unmeasured. The first-turn column is at its ceiling on this set.
