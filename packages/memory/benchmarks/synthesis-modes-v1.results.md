# Synthesis modes on eight multi-session questions — 15 September 2026

The scoreboard's synthesis row (13 September) read "faithful when it
speaks, silent too often": evidence-first synthesis wrote on 3 of 8
LongMemEval-S multi-session items with `gemma4-e4b`, where the
reference's TreeSummarize wrote on 8 of 8. This run puts the two new
modes, `refine` and `accumulate`, beside `evidence` and TreeSummarize on
the same items, with one local model writing and judging every side.
It ran once on a snapshot of the first commit, and then three more times
on the branch's final synthesis source, side orders rotated between
runs. Our three modes wrote the same texts and got the same scores in all
four runs; TreeSummarize wrote a different text on one item in the last
repeat.

## Method

- Items: LongMemEval-S multi-session questions, `stratified_sample(n=8,
  seed=42)`; the runner asserts they are the scoreboard's eight
  (`gpt4_2ba83207`, `ba358f49`, `c18a7dc8`, `a4996e51`, `a3332713`,
  `60bf93ed`, `2318644b`, `00ca467f`).
- Passages: each item's sessions stored in an in-memory engine with
  `HashEmbedder`, the top 12 recalled passages handed to every side
  unchanged (retrieved evidence-session share 0.875).
- Writer and judge: `llama3.1-ctx8k` (8B, Q4_K_M, fully on the GPU) through
  Ollama at temperature 0. The judge is the writer: a caveat, not hidden.
  This is a different model from the scoreboard's run, so no number here
  is a before/after against that row.
- Our sides: `synthesize_passages` with `max_round_bytes=6000`,
  `max_rounds=16`, `max_sentences=12`, `timeout_s=600`. `evidence` and
  `refine` pack the 12 passages into about two rounds; `accumulate` sends
  one passage per call. The reference: LlamaIndex 0.14.24 `TreeSummarize`
  through the comparative runner's adapter to the same model.
- Per item the four sides run one after another, their order rotated
  item by item. Each non-empty text is judged by
  `bench.evaluators.faithfulness` (share of the text's claims a passage
  supports) and `answer_relevancy` (does it address the question);
  unverified judgments are left out and counted.
- Quoted share: of the sentences the model wrote new, the share kept with
  a checked quote, summed over the items: (`notes.kept` - `notes.carried`)
  / (notes returned - `notes.carried`). `notes.carried` counts the
  sentences a refine rewrite repeats from the answer so far, which are
  checked once already and would otherwise count again; it is 0 in the
  other modes, where the share is kept over returned. Our sides only; the
  reference cites nothing.
- Runner: [`synthesis_modes.py`](synthesis_modes.py). The first run
  (14-15 September, before the review fixes) used a snapshot of the first
  commit's tree before three behaviour-preserving edits (a renamed local,
  one branch folded into an expression, and a duplicate mode check
  removed from `synthesize`, which this runner does not call). The repeat
  (15 September, 02:48-05:18 CDT) ran `synthesis_modes.py ... 8 42
  llama3.1-ctx8k:latest 12 3` from an export of commit `91acf690`, which
  has the review fixes (the answer so far inside a refine round's bound,
  `refine_dropped_carried`) and `notes.carried`. Each repeat run rotates
  every item's side order one further, so a side ran first, second,
  third and fourth on different items in different runs. Wall time
  3,782 s for the first run and 3,068 s, 3,107 s and 2,826 s for the
  repeats, on a box shared with other jobs; seconds are not compared.

## Results

| side | items with an answer | faithfulness (judged items) | relevancy (judged items) | quoted share | model calls | passages read | passages cited | cited evidence share |
|---|---|---|---|---|---|---|---|---|
| evidence | 7/8 | 0.810 (7) | 0.143 (7) | 0.613 (19/31) | 17 | 84 | 13 | 0.531 |
| refine | 7/8 | 0.765 (6) | 0.143 (7) | 0.941 (16/17) | 15 | 84 | 11 | 0.531 |
| accumulate | 6/8 | 0.759 (6) | 0.000 (6) | 0.500 (27/54) | 96 | 96 | 17 | 0.552 |
| TreeSummarize | 8/8 | 0.786 (7), 0.822 in one repeat | 0.000 (8), 0.125 in one repeat | n/a | 8 | 96 given | n/a | n/a |

Dropped notes by reason: `evidence` 9 malformed and 3 citing a passage
the round did not hold; `refine` 1 unknown; `accumulate` 18 malformed, 8
unknown and 1 unquoted. `evidence` folded on 2 items.

Each number in the table is the median of the three repeats, and the
first run's number too. For `evidence`, `refine` and `accumulate` every
text, count and score matched item for item in all four runs, so the
minimum equals the maximum. TreeSummarize matched in the first run and
the first two repeats. In the third repeat it wrote a different text on
`2318644b`, the same figures set out at more length, and the judge
scored that text 0.923 faithful and 1.0 relevant instead of 0.667 and
0.0. That run's TreeSummarize row reads 0.822 and 0.125; its median over
the repeats stays 0.786 and 0.000. The repeat did not record why the
text changed.

`refine`'s second round wrote nothing new. Per round (returned, kept,
carried), its seven second rounds returned exactly the answer so far:
(1, 1, 1), (2, 2, 2), (1, 1, 1), (4, 4, 4), (2, 2, 2), (2, 2, 2),
(4, 4, 4). That is 16 sentences, all carried, none new, and none left
out (`refine_dropped_carried` 0). `evidence`'s second rounds read the
same passages with the notes prompt, returned 14 notes and kept 3, on
`a4996e51` (1) and `2318644b` (2), the two items it folded. So `refine`'s
quoted share is its first round alone, and its first round is
`evidence`'s: 16 of 17 kept. The 0.941 against `evidence`'s 0.613 means
`refine` wrote less, not that it quoted better. On five of the seven
items the text is byte-identical to `evidence`'s.
`refine_kept_prior` was 0 on every item, but it counts only rewrites that
could not be read or kept no checked sentence; a rewrite that repeats
the answer is neither.

Statuses: `evidence` and `refine` 7 synthesized and 1 unavailable;
`accumulate` 6 synthesized and 2 no_evidence. Unverified judgments: one
`refine` faithfulness (the verdict had no claims list) and one
TreeSummarize faithfulness (the judge call failed).

## What it says

- No mode spoke more often than `evidence` here. `refine` answered on the
  same seven items with two fewer calls, because it never folds and its
  second round added nothing to the first round's answer, which is
  `evidence`'s first round. `accumulate` spent 96 calls, 5.6
  times `evidence`'s, and spoke on 6. One passage per call did not stop
  the silence; it gave the model more chances to return malformed notes.
- `evidence` and `refine` send the same first round (the same prompt at
  temperature 0), so with two rounds they differ only in the second, and
  on five of the seven items both answered their texts are the same.
- With this model, refine's prompt kept the answer and dropped what the
  new passages held: 3 checked sentences `evidence` found in the second
  rounds' passages never reached a refine answer. Refine as written here
  is a rewrite that the model treats as a copy.
- The first run predates the review fix that puts the answer so far
  inside `max_round_bytes`. The repeats on the final source give the same
  texts, so the fix did not change these rounds: the second rounds held
  906 to 1,422 bytes of passages, and the carried answers stayed under
  3,800 bytes.
- Relevancy is this judge's reading, and it is strict. Asked about
  "you work up to 50 hours per week during peak campaign seasons" for
  "How many hours do I work in a typical week during peak campaign
  seasons?", it replied that the answer "does not address ... a typical
  week, but rather a maximum number of hours". Only one item scored 1.0
  on any side.
- Faithfulness counts claims the passages support, not whether the
  sentence is right: on `gpt4_2ba83207` every mode wrote "I spent the most
  money at Walmart" beside a quote that was found in its passage, and the
  judge scored it unsupported. The quote check keeps sentences tied to a
  passage; it does not check what the sentence claims.
- On `c18a7dc8` (no evidence session among the 12 passages) the first
  6,000-byte call ran to the 600 s deadline for both `evidence` and
  `refine`, the same prompt twice; nothing bounds how much a model writes.
- With `llama3.1-ctx8k` every side speaks far more often than the
  scoreboard's `gemma4-e4b` run did (7/8 against 3/8 for `evidence`). That
  is the model changing, not a mode. Eight items and one judge do not
  separate 0.810, 0.786, 0.765 and 0.759: one changed TreeSummarize text
  moved its row from 0.786 to 0.822.

Not measured: correctness against the dataset's answers; another model
or judge; a sample larger than eight.
