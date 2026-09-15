# Synthesis modes on eight multi-session questions — 15 September 2026

The scoreboard's synthesis row (13 September) read "faithful when it
speaks, silent too often": evidence-first synthesis wrote on 3 of 8
LongMemEval-S multi-session items with `gemma4-e4b`, where the
reference's TreeSummarize wrote on 8 of 8. This run puts the two new
modes, `refine` and `accumulate`, beside `evidence` and TreeSummarize on
the same items, with one local model writing and judging every side.

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
- Quoted share: notes kept with a checked quote over notes the model
  returned, summed over the items (our sides only; the reference cites
  nothing).
- Runner: [`synthesis_modes.py`](synthesis_modes.py), the script as run
  with the data path made an argument. The measured source was a snapshot
  of the first commit's tree before three behaviour-preserving edits
  (a renamed local, one branch folded into an expression, and a duplicate
  mode check removed from `synthesize`, which this runner does not call).
  Wall time 3,782 s on a box shared with other jobs; seconds are not
  compared.

## Results

| side | items with an answer | faithfulness (judged items) | relevancy (judged items) | quoted share | model calls | passages read | passages cited | cited evidence share |
|---|---|---|---|---|---|---|---|---|
| evidence | 7/8 | 0.810 (7) | 0.143 (7) | 0.613 (19/31) | 17 | 84 | 13 | 0.531 |
| refine | 7/8 | 0.765 (6) | 0.143 (7) | 0.970 (32/33) | 15 | 84 | 11 | 0.531 |
| accumulate | 6/8 | 0.759 (6) | 0.000 (6) | 0.500 (27/54) | 96 | 96 | 17 | 0.552 |
| TreeSummarize | 8/8 | 0.786 (7) | 0.000 (8) | n/a | 8 | 96 given | n/a | n/a |

Dropped notes by reason: `evidence` 9 malformed and 3 citing a passage
the round did not hold; `refine` 1 unknown; `accumulate` 18 malformed, 8
unknown and 1 unquoted. `evidence` folded on 2 items. `refine_kept_prior`
was 0 on every item: no refine round left the answer standing.

Statuses: `evidence` and `refine` 7 synthesized and 1 unavailable;
`accumulate` 6 synthesized and 2 no_evidence. Unverified judgments: one
`refine` faithfulness (the verdict had no claims list) and one
TreeSummarize faithfulness (the judge call failed).

## What it says

- No mode spoke more often than `evidence` here. `refine` matched it on
  answers (7/8) with two fewer calls; `accumulate` spent 96 calls, 5.6
  times `evidence`'s, and spoke on 6. One passage per call did not stop
  the silence; it gave the model more chances to return malformed notes.
- `evidence` and `refine` send the same first round (the same prompt at
  temperature 0), so with two rounds they differ only in the second, and
  on five of the seven items both answered their texts are the same.
- `refine`'s quoted share is not comparable with the others': each
  rewrite repeats the answer's sentences with quotes that were already
  checked, and those are counted again.
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
  separate 0.810, 0.786, 0.765 and 0.759.

Not measured: correctness against the dataset's answers, and more than
one run.
