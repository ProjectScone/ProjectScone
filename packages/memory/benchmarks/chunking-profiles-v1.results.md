# Chunking profiles v1: where the cuts land

Run on 14 September 2026 on the branch that added `chunking_profile`
(parent commit 643caf8c), with

```
PYTHONPATH=src python benchmarks/chunking_profiles.py --corpus docs
```

No model and no store: this measures cut positions at the default target of
700 characters, not answers.

## Output

```
statute-like: 12292 characters, profile statute, seed 20260914
  length     chunks   21  at a genre boundary 11/21 (52%)  split from first child 8/14  fitting unit split 4/6  over 700: 0  mean 585
  structure  chunks   23  at a genre boundary 22/23 (96%)  split from first child 4/14  fitting unit split 6/6  over 700: 0  mean 534
  profile    chunks   27  at a genre boundary 26/27 (96%)  split from first child 0/14  fitting unit split 0/6  over 700: 0  mean 455
  receipt: {'by_size': 1, 'over_target': 0, 'matched': {'part': 3, 'article': 13, 'paragraph': 14, 'letter': 27}, 'began': {'part': 3, 'letter': 8, 'paragraph': 5, 'article': 10}}
Q&A-like: 13190 characters, profile qa, seed 20260914
  length     chunks   26  at a genre boundary 16/26 (62%)  split from first child 0/24  fitting unit split 2/16  over 700: 0  mean 507
  structure  chunks   27  at a genre boundary 10/27 (37%)  split from first child 15/24  fitting unit split 7/16  over 700: 1  mean 488
  profile    chunks   30  at a genre boundary 24/30 (80%)  split from first child 0/24  fitting unit split 0/16  over 700: 2  mean 439
  receipt: {'by_size': 6, 'over_target': 2, 'matched': {'question': 24, 'answer': 24}, 'began': {'heading': 1, 'question': 23}}
docs: 45 documents, 679808 characters
  plain structure: 1372 chunks
  statute  1360 chunks, rules matched {'subsection': 3, 'paragraph': 19}
  paper    1358 chunks, rules matched {'section': 1}
  manual   1360 chunks, rules matched {'task': 1, 'step': 19}
  qa       1358 chunks, rules matched {}
  resume   1358 chunks, rules matched {}
```

## How to read it

Each fixture is cut three ways (by length, by plain structure, under the
profile) and every way is scored against the profile's own units:

- **at a genre boundary**: chunks whose first byte is where one of the
  profile's units begins. The document title before the first unit is never
  one.
- **split from first child**: for the statute, a part or article whose next
  unit is a deeper one (its first clause) landing in a different chunk; for
  the Q&A file, a question and its `A:` line landing in different chunks.
- **fitting unit split**: a part, article or question whose whole subtree is
  no longer than the target, cut across more than one chunk anyway.
- **over 700**: chunks longer than the target. Under the profile the receipt's
  `over_target` agrees with this column. Plain structure's receipt counts only
  tables, so its one Q&A chunk of 722 characters, from the size chunker
  joining a last piece shorter than 120 characters to the one before it, is
  not in its receipt.

## What it shows

- Statute: the share of chunks starting at a boundary does not move (96% both
  ways), because plain structure already starts chunks at `Article` and `(a)`
  lines. What moves is where chunks end: plain structure separates 4 of 14
  parts and articles from their first clause and splits all 6 that fit the
  target; the profile does neither.
- Q&A: plain structure reads `A:` as a boundary of its own and separates 15 of
  24 questions from their answers; the profile separates none, and 80% of its
  chunks start at a question against 37%. The six that do not are second
  pieces of answers longer than the target. Length chunking never separates a
  question from its answer here, because the two sit on adjacent lines, but it
  splits 2 of the 16 pairs that fit.
- Cost: 17% more chunks than plain structure on the statute (27 against 23)
  and 11% more on the Q&A file (30 against 27), because whole subtrees pack
  less tightly than loose units.
- On this repository's own documents, where no genre applies, the rules
  mostly stay quiet: `qa` and `resume` match nothing, `paper` one heading, and
  `statute` and `manual` 19 numbered-list lines. The corpus counts change
  whenever those documents are edited.

Unmeasured: whether a reader answers better from these chunks.
