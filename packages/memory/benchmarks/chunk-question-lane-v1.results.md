# The question lane on this repository's documents — 15 September 2026

The question lane (`ingestion/chunk_questions.py`, [docs](../docs/chunk-questions.md))
asks a chat model, once per chunk, for questions the chunk answers, keeps
those whose quote is verbatim in the chunk, and searches them as a lane.
This compares recall with the lane off and on over the same stores, with
questions written by a different prompt than the lane's.

When this was measured the questions were kept in the context index. They
now have an index of their own (after review: a pass could overwrite the
words a chunk is under). The stores here were ingested with the context
lane off, so that index held the questions and nothing else; moved into
the question index on a copy of the stores, `measure` and both `sweep`s
gave the same scores on every question and every weight as below (15
September 2026).

**Result: the lane made recall worse here.** Over 43 evaluation questions,
R@5 fell from 0.953 to 0.837 and MRR from 0.868 to 0.817. Lowering the
lane's fusion weight never beat the lane off on held-out questions; the
best weight tied it. The lane stays off by default.

## Method

- Corpus: 13 documents from `packages/memory/docs` (pdf-ocr, answer-review,
  agent-event-history, directory-sync, chat-exports, followup-queries,
  getting-started, s3-catalog, attachments, space-merge, table-querying,
  http-and-deployment, workflow-pauses), stored with
  `bench.questions.store_corpus` as 233 chunks. SQLite document store and
  vector index, `HashEmbedder` (vector weight 0.25), engine defaults
  otherwise; the context lane is off in both arms.
- Model: `llama3.2-ctx8k` through Ollama at `127.0.0.1:11434`, 300 s timeout,
  on a machine shared with other agents' test runs (load average 15 to 22
  while measuring; not recorded during the builds).
- Evaluation questions: `bench.questions.write_questions` (its own prompt,
  the strict reader, one question per chunk) over two seeded samples of 80
  chunks each. Seed 42 kept 22 questions (45 replies unparsed, 12 quotes not
  in the chunk, 1 call timed out); seed 7 kept 30 (43 unparsed, 6 unquoted,
  1 timed out). 43 distinct questions in all. Those counts are the strict
  reader's as it was then; it has since changed (it no longer reads a
  later bracket when the first list fails), and the sets were not written again. A question is found at rank r
  when the r-th passage returned holds its quote.
- Lane: `engine.build_chunk_questions` over all 233 chunks, 3 per chunk, with
  its own prompt ("the questions a person could later ask … in their own
  everyday words").
- Arms: the same stores opened by an engine with `question_lane=False` and
  one with `question_lane=True`, measured three times in alternating order.
  The scores were identical in every repeat.
- Reproduce, from `packages/memory` with `PYTHONPATH=src`:
  `benchmarks/chunk_question_lane.py build-eval --seed 42 --data D`, the
  same with `--seed 7`, `build-lane --data D`, `measure --data D`, and
  `sweep --data D --choose-seed 42` (and `7`). The stores and question sets
  are not committed.

LongMemEval-S was not used. Stored at the default chunk size, its first
three items hold 921, 935 and 952 chunks, so 20 items would be about
18,600 model calls — at the rate below, over two days of the local model.

## Results

| Questions | Lane | R@1 | R@5 | R@10 | MRR | Not in top 10 |
|---|---|---:|---:|---:|---:|---:|
| all (43) | off | 0.791 | 0.953 | 0.953 | 0.868 | 2 |
| all (43) | on | 0.791 | 0.837 | 0.860 | 0.817 | 6 |
| seed 42 (22) | off | 0.773 | 1.000 | 1.000 | 0.879 | 0 |
| seed 42 (22) | on | 0.773 | 0.864 | 0.864 | 0.818 | 3 |
| seed 7, less seed 42's (21) | off | 0.810 | 0.905 | 0.905 | 0.857 | 2 |
| seed 7, less seed 42's (21) | on | 0.810 | 0.810 | 0.857 | 0.816 | 3 |

Six questions moved up within the top 10 and six moved down. Five of the
six that moved up have a lane question on the answer chunk sharing at least
half its words (Jaccard over the lexical tokens): "What is the maximum size
of a file in MB?" against the lane's "What is the maximum size of a file?",
and "What happens to completed OCR page observations?" asked word for word
by both prompts. None of the six that moved down does. Across all 43, the
median of that overlap is 0.333; 19 are at least 0.5 and 5 have none. The
same model wrote both sets, so the gains are inflated by near-duplicates
and the losses are not.

**Why it lost.** Fusion is by rank (`RRF_K` 60) and the lane is fused at the
context lane's weight, 2.0 (`QUESTION_WEIGHT` now), against 1.0 for text and 0.25
for the hashed vectors. A chunk at rank r in the lane scores 2/(60+r), more
than a chunk first in both the text and vector lanes (1.25/61) down to
about rank 37. In the diagnostic run behind this report, each question that
moved down was overtaken by a chunk the lane placed high whose own questions
were about something else: "What is the difference between `omitted` and
`AgentProgressGap`?" went from rank 2 to 7 behind a chunk placed fourth by the
lane, whose questions ask about `history_omitted`; "What is the purpose of the
`POST /v1/chat-imports` route?" went from rank 1 to out of the top 10 behind
one placed sixth, whose question is "How do I upload a file to a chat?". Four
questions went from rank 1 to out of the top 10 that way.

**Weight sweep (no model).** `sweep` sets the lane's weight for the run
(then `CONTEXT_WEIGHT`, now `QUESTION_WEIGHT`),
scores one seed's questions as the set a weight would be chosen on and the
other seed's remaining questions as held out.

| Weight | Chosen on seed 42 (22), MRR | Held out: seed 7 (21), R@5 / MRR | Chosen on seed 7 (30), MRR | Held out: seed 42 (13), R@5 / MRR |
|---:|---:|---:|---:|---:|
| off | 0.879 | 0.905 / 0.857 | 0.850 | 1.000 / 0.910 |
| 0.01 | 0.902 | 0.905 / 0.857 | 0.867 | 1.000 / 0.910 |
| 0.05 | 0.864 | 0.905 / 0.833 | 0.850 | 0.923 / 0.846 |
| 0.1 | 0.852 | 0.905 / 0.825 | 0.844 | 0.923 / 0.827 |
| 0.25 | 0.838 | 0.905 / 0.813 | 0.831 | 0.846 / 0.816 |
| 0.5 | 0.847 | 0.857 / 0.832 | 0.853 | 0.846 / 0.808 |
| 1.0 | 0.841 | 0.857 / 0.830 | 0.848 | 0.846 / 0.808 |
| 2.0 | 0.818 | 0.810 / 0.816 | 0.838 | 0.846 / 0.769 |

Either way round, the weight chosen is 0.01, where the lane only breaks ties
between chunks the other lanes scored alike, and it gives exactly the lane-off
scores on the held-out questions. Every weight from 0.05 up is below off on
held-out MRR. The weight was not changed: no value measured here is a gain.

**Cost.** Writing the lane took 233 model calls and 2,546 s: 100 calls and
1,093 s per 100 chunks. 535 questions were kept for 209 chunks; 119 pairs
were dropped because the quote was not in the chunk (or under four words),
2 as repeats, and 9 replies could not be read. 197 of the 233 replies were
not one valid list and were read object by object, each read up to its
first object that would not decode; without that reader those chunks
would have kept nothing. The report then called them all lists never
closed, but it could not tell a list cut off before its bracket from a
closed one with a bad object, and the raw replies were not kept. The
reader now in use (`bench.questions.loose_pairs`) passes over a bad object
and reads on, counting it, so it can only add pairs to such replies; how
many it would have added here is not known without calling the model
again. Recall over the 43 questions took a median of 0.260 s
off and 0.281 s on (three alternating runs each).

## What this does not show

- The evaluation questions were written from each chunk with its quote in
  view, so they share the chunk's words: lane off already finds 95% in the
  top 5. The lane is meant for questions in other words than the passage's,
  and this set holds few of those. The unit test of that case
  (`tests/ingestion/test_chunk_questions.py`) passes; no measured set of
  such questions exists yet.
- The sets are what survived the strict reader (88 of 160 replies
  unparsed), so they lean to chunks the model answers in form.
- One small model, one corpus of 233 chunks, 43 questions. A difference of
  one question is 0.023 of R@k on all 43.
