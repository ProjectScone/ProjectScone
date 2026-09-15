# Facts-first synthesis on eight multi-session questions — 15 September 2026

The scoreboard's synthesis row reads "faithful when it speaks, silent too
often", and names per-fact-chunk extraction as the next probe. The
[synthesis modes run](synthesis-modes-v1.results.md) found no mode that
spoke more often than `evidence` (7 of 8 with `llama3.1-ctx8k`). This run
puts the new `facts` mode beside `evidence` on the same eight
LongMemEval-S items, three times, with one local model writing and
judging, and reruns the reference's TreeSummarize on the same passages.

In short: `facts` gave a text on 8 of 8 items and `evidence` on 7 of 8,
and `facts` wrote an answer (its second pass) on 7 of 8. The one item
`evidence` missed is a 600-second model call that never returned, not a
note list that came back empty. `facts` spent 6.5 times the calls, and
the judge scored its texts less faithful (0.762 against 0.905).
Relevancy was 0.000 for every side on every item, so this judge does not
separate the sides on relevancy.

## Method

- Items: LongMemEval-S multi-session questions, `stratified_sample(n=8,
  seed=42)`; the runner asserts they are the scoreboard's eight
  (`gpt4_2ba83207`, `ba358f49`, `c18a7dc8`, `a4996e51`, `a3332713`,
  `60bf93ed`, `2318644b`, `00ca467f`).
- Passages: each item's sessions stored in an in-memory engine with
  `HashEmbedder`, the top 12 recalled passages handed to every side
  unchanged. The retrieved evidence-session share is 0.75. These are
  **not the passages of the synthesis modes run**. Retrieval changed on
  main between that run's commit (`91acf690`) and this one (exact word
  forms on by default, word families weighed at their own idf, and
  more). Per item the two trees' twelve passages share 7 to 10, and
  never in the same order. So neither that run's `evidence` row nor its
  TreeSummarize row is a before/after for this one. TreeSummarize was
  run again here on this run's passages.
- Writer and judge: `llama3.1-ctx8k` (8B, Q4_K_M, fully on the GPU)
  through Ollama at temperature 0. The judge is the writer: a caveat,
  not hidden.
- Our sides: `synthesize_passages` with `max_round_bytes=6000`,
  `max_rounds=16`, `max_sentences=12`, `timeout_s=600`, the limits of
  the modes run. `evidence` packs the 12 passages into about two rounds
  and may fold. `facts` makes one extraction call per passage (12), then
  one answer call. It asks for at most 12 sentences, and the call holds
  at most 6,000 bytes of facts; no item's facts reached that bound
  (`facts.unsent` 0 on every item).
- The reference: LlamaIndex 0.14.24 `TreeSummarize` through the
  comparative runner's adapter to the same model. It made one call per
  item here.
- Runner: [`synthesis_modes.py`](synthesis_modes.py) with the new
  `sides` argument. Command: `synthesis_modes.py ... 8 42
  llama3.1-ctx8k:latest 12 3 evidence,facts`, then the same command with
  `treesummarize` once that run had exited. Both ran from an export of
  commit `6eb11c16`, on 15 September: `evidence,facts` from 09:23 to
  10:51 CDT (wall time 1,971 s, 1,626 s and 1,671 s per run), and
  TreeSummarize from 10:51 to 11:03. Side order alternates item by item
  and shifts one further each run. The box was shared with other jobs,
  so seconds are not compared.
- Judging: each non-empty text is judged by
  `bench.evaluators.faithfulness` (the share of the text's claims a
  passage supports) and `answer_relevancy` (does it address the
  question). Unverified judgments are left out and counted.
- Items with an answer: items whose shown text is non-empty, as the
  modes run counted them. For `facts`, **answer written** is the
  stricter count: items whose second pass produced the shown sentences
  (`folded`). The rest show their checked facts unmerged.
- Quoted share: the notes kept with a checked quote over the notes the
  model returned, summed over the items. For `facts` the notes are the
  extracted facts, so this is facts kept over facts returned.

## Results

Each number is the median of three runs. Every text of every side was
byte-identical across the three runs, and so was every count and score
of `evidence` and `facts`. One TreeSummarize score moved: its text for
`ba358f49` was the same in all three runs, and the judge scored it 0.0
faithful in the first run and 1.0 in the other two.

| side | items with an answer | answer written | faithfulness (judged items) | relevancy (judged items) | quoted share | model calls | passages cited | cited evidence share |
|---|---|---|---|---|---|---|---|---|
| evidence | 7/8 | 7/8 | 0.905 (7) | 0.000 (7) | 0.581 (18/31) | 16 | 15 | 0.427 |
| facts | 8/8 | 7/8 | 0.762 (7) | 0.000 (8) | 0.904 (141/156) | 104 | 26 | 0.708 |
| TreeSummarize | 8/8 | 8/8 | 0.730 (8); 0.605 in one run | 0.000 (8) | n/a | 8 | n/a | n/a |

`facts`, over the eight items:

- Calls: 104, which is 12 extraction calls and one answer call per item.
- Facts extracted: 141 of the 156 facts returned kept a checked quote.
  8 were dropped because the quote was not in the passage and 7 as
  malformed. None could name the wrong passage, since the code, not the
  model, names it.
- Facts used: 48 distinct facts named by the shown sentences.
- Answer sentences dropped for citing no fact: 3.
- Facts left out by the answer call's byte bound: 0.
- Statuses: 7 synthesized and 1 partial.

`evidence` dropped 13 notes (6 unquoted, 1 citing a passage its round
did not hold, 6 malformed), folded on one item, and was 7 synthesized
and 1 unavailable. Unverified judgments: one `facts` faithfulness, on
`gpt4_2ba83207` (the verdict had no claims list).

Per item (run 1; runs 2 and 3 are the same):

| item | evidence | facts: facts returned / kept / used, sentences dropped | facts status |
|---|---|---|---|
| gpt4_2ba83207 | 2 sentences, f 1.0 | 22 / 17 / 11, 0 | answer written, 12 sentences, f unverified |
| ba358f49 | 2 sentences, f 0.5 | 4 / 4 / 1, 0 | answer written, 2 sentences, f 0.5 |
| c18a7dc8 | 3 sentences, f 1.0 | 18 / 16 / 1, 1 | answer written, 2 sentences, f 0.0 |
| a4996e51 | 1 sentence, f 1.0 | 28 / 27 / 8, 0 | answer written, 7 sentences, f 1.0 |
| a3332713 | 5 sentences, f 1.0 | 23 / 23 / 9, 0 | answer written, 12 sentences, f 0.833 |
| 60bf93ed | 1 sentence, f 1.0 | 18 / 16 / 3, 0 | answer written, 4 sentences, f 1.0 |
| 2318644b | nothing: the first round's call ran to the 600 s deadline | 15 / 10 / 10, 2 | partial: the answer's 2 sentences cited no fact, so the 10 facts are shown unmerged, f 1.0 |
| 00ca467f | 2 sentences (folded), f 0.833 | 28 / 28 / 5, 0 | answer written, 3 sentences, f 1.0 |

## The answer prompt was revised before this run

The first `facts` answer prompt asked for sentences each "written from
one or more of the facts". A one-item smoke run on `gpt4_2ba83207`, one
of the eight, sent that call and it ran until it was cancelled at the
600 s deadline. A replay with generation capped at 700 tokens showed
why: the model wrote one sentence per fact, added "but that's not
related to the question" to the facts that were off topic, and then
repeated itself. The prompt was then tried on three multi-session items
**outside** the eight (`0a995998`, `6d550036`, `gpt4_59c863d7`, 300 s
deadline), where the answer call ran to the deadline on 3 of 3. The
revised prompt asks for only the facts that help answer the question,
each thing said once, and at most `max_sentences` sentences. The
extraction prompt was also told to give each fact once and to write no
fact about the passage itself or about what it does not say. On the same
three items the revised prompts wrote an answer on 3 of 3, in 15, 15 and
31 s. The measurement above used only the revised prompts. The smoke run
that led to the revision was on one of the eight; the revised prompts
were checked only on the three items outside them, and no prompt was
changed after the measurement's results were seen.

## What it says

- `facts` gave a text more often than `evidence` here, 8 against 7. The
  difference is one item, `2318644b`. There `evidence`'s first
  6,000-byte call ran to the deadline in all three runs, with nothing
  returned; `facts`'s per-passage calls finished. The rest of that item
  is weak: `facts`'s answer cited no fact, and the 10 facts shown are
  about Tokyo hostels, markets, New York restaurants and a GraphQL query,
  with nothing on Hawaii. Counting only written answers, the two modes
  are level at 7 of 8. This is not the scoreboard's silence (a 4B model
  returning empty note lists). With this 8B model and these limits, no
  item came back empty because the model found nothing. `evidence`'s one
  empty item is a timeout.
- The extraction pass quotes well: 90% of the facts returned kept a
  checked quote, against 58% of `evidence`'s notes. The facts' shown
  sentences reach more evidence sessions (cited evidence share 0.708
  against 0.427). It costs 6.5 times the calls.
- The answer pass is where the faithfulness goes. A shown sentence
  carries the quotes of the facts it names, but nothing checks that the
  sentence says what those facts say. On `a3332713` the answer cites
  real facts ($100, $100, $20, $75) and then writes "You spent a total of
  $295 + $305 = $600 on gifts" and "you still owe $305". The judge scored
  both unsupported. On `c18a7dc8` (no evidence session among the
  passages) it wrote "I am now some number of years older than that",
  scored 0.0. Repeats survive too: on `gpt4_2ba83207` one sentence
  appears twice despite "say each thing once". `verified_accuracy` stays
  false for that reason.
- The extraction still pads. On `2318644b` the shown facts include "You
  can add more fields to the GraphQL query for each pool", for a question
  about hotel prices. On `gpt4_2ba83207` the answer opens with "There is
  no information about which grocery store you spent the most money at",
  citing a fact from `chunk:842`, which is a statement about what the
  passages lack. The revised extraction prompt asks for no such facts.
  The quote check keeps a fact tied to a passage. It does not check that
  the fact bears on the question. (The runner's rows keep the shown text,
  not every extracted fact with its quote, so these are read from the
  shown sentences only.)
- Relevancy is 0.000 for all three sides on all eight items. Even "you
  work up to 50 hours per week during peak campaign seasons", for "How
  many hours do I work in a typical week during peak campaign seasons?",
  scored 0 (the modes run reported this judge's reading of "up to" as a
  maximum rather than a typical week). This judge's relevancy says
  nothing about these sides here.
- Against the dataset's answers (a reading, not a score): on `a4996e51`
  all three sides say 50 hours. On `00ca467f` (answer: 2) TreeSummarize
  says two, and `evidence` says "at least 2" and then that a note
  contradicts it. `facts` lists the two appointments and a physical
  therapy schedule without giving a count. On `gpt4_2ba83207` (Thrive
  Market) no side says that is where the most was spent: `evidence`
  names Thrive Market only as a grocery store.
- Eight items, one judge that is also the writer, and one model. The
  faithfulness gap (0.905, 0.762, 0.730) rests on seven or eight judged
  texts each, and one judge call on an unchanged TreeSummarize text
  moved its run from 0.605 to 0.730.

Not measured: correctness as a score; another model or judge; a sample
larger than eight; the route's own limits (12,000-byte rounds, six
rounds), where `facts` would read six of twelve passages and be
`partial`.
