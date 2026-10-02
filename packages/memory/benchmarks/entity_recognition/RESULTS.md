# Entity recognition on complete public test sets, in Scone's entity types

## Run

- **Date and test sets:** 2 October 2026. Every sentence of two test sets:
  - **OntoNotes 5** (`tner/ontonotes5` test): 8,262 sentences, 11,257 entities, 18 types.
  - **Few-NERD supervised** (`DFKI-SLT/few-nerd` test): 37,648 sentences, 96,902 entities, 66 fine types.
- **Types:** both test sets are mapped onto Scone's types (`data.py`): person, organization, nationality, location,
  facility, product, event, work_of_art, law, language, date, time, money, percent, quantity, ordinal and cardinal.
  Few-NERD names no values (dates, money and so on) and no nationalities, so those are not scored on it.
- **Scoring:** strict. A prediction counts only with the exact tokens and type of a gold span. Precision, recall and
  F1 are micro-averaged over every sentence.
- **Runner:** [`run.py`](run.py). Raw predictions are in `bench-runs/entity-recognition-2026-10-02/`.
- **Speed:** spaCy and GLiNER ran on CPU on an Apple M3 Max. Gemma ran on OpenRouter, 16 sentences at a time, so its
  per-sentence time includes the network.

## Results

| Recognizer | OntoNotes P / R / F1 | Few-NERD P / R / F1 | ms per sentence (median) |
| --- | --- | --- | --- |
| spaCy `en_core_web_trf` (trained on OntoNotes train) | **89.6 / 89.5 / 89.5** | 57.8 / 56.9 / 57.3 | 8.7 / 12.5 |
| spaCy `en_core_web_lg` (no PyTorch) | 84.6 / 85.0 / 84.8 | 45.4 / 44.3 / 44.8 | **2.0 / 2.7** |
| Gemma 4 31B, typed JSON, zero-shot | 71.9 / 71.0 / 71.4 | **67.0 / 71.5 / 69.2** | 839 / 1,616 |
| GLiNER medium v2.1, zero-shot, threshold 0.5 | 30.7 / 49.7 / 38.0 | 50.6 / 63.1 / 56.2 | 28.5 / 32.1 |
| GLiNER large v2.1, zero-shot, threshold 0.5 | 21.7 / 48.9 / 30.1 | 43.3 / 65.0 / 52.0 | 75.1 / 85.2 |

### F1 by type

| Type | OntoNotes: spaCy trf | OntoNotes: Gemma | Few-NERD: spaCy trf | Few-NERD: Gemma |
| --- | ---: | ---: | ---: | ---: |
| person | 94.2 | 83.0 | 83.5 | 85.0 |
| organization (companies and others) | 90.1 | 66.7 | 45.3 | 67.0 |
| nationality / group | 94.3 | 81.4 | not labelled | not labelled |
| location | 95.2 | 84.1 | 63.7 | 70.1 |
| product (tools, software) | 73.3 | 30.4 | 36.2 | 51.7 |
| facility | 75.8 | 38.4 | 26.4 | 52.1 |
| event | 68.3 | 30.4 | 26.1 | 51.0 |
| work of art | 57.1 | 63.8 | 41.5 | 68.8 |
| date | 86.9 | 65.1 | not labelled | not labelled |
| money | 82.9 | 62.1 | not labelled | not labelled |

### Combinations (offline, from the saved predictions)

| Combination | OntoNotes P / R / F1 | Few-NERD P / R / F1 |
| --- | --- | --- |
| spaCy trf first, Gemma fills what it leaves | 86.2 / 90.4 / 88.2 | 55.9 / 60.4 / 58.1 |
| Gemma first, spaCy trf fills | 72.1 / 76.4 / 74.2 | 66.5 / 73.1 / 69.6 |
| only spans both agree on | **95.1** / 67.3 / 78.8 | **79.6** / 51.2 / 62.3 |

## What it shows

- **spaCy's transformer model is the right default for the standard types.** It is local, free and about 9 ms a
  sentence, with 89.5 F1 on held-out OntoNotes. For your examples it scores organization 90.1, nationality 94.3,
  location 95.2 and person 94.2, but product only 73.3. Its CPU-only sibling (`lg`, no PyTorch) is 4x faster at 84.8
  on OntoNotes, but collapses on Few-NERD (organization 29.8, product 8.7).
- **On text unlike its training data, spaCy falls to 57 F1.** Organizations (45) and products and tools (36) suffer
  most. Gemma 4 31B, with no examples, transfers best (69 F1; organization 67, product 52), but it is 100x slower and
  costs per sentence.
- **GLiNER, zero-shot over 17 types at its default threshold, was the weakest here** because of low precision. Its
  thresholds and label wording were not tuned.
- **Where spaCy and Gemma agree, precision is 95.1 on OntoNotes and 79.6 on Few-NERD.** An agreed span is a candidate
  for automatic typing and merging; spans from one recognizer only should stay suggestions.
- **A measurement caveat:** 3,988 of spaCy's Few-NERD entities (and 534 of Gemma's) have edges inside a Few-NERD
  token, and they score as misses. Snapping them to whole tokens may raise both systems' Few-NERD scores.
- **Tokenization:** sentences are read as their tokens joined by single spaces, not as original text, for every
  recognizer alike.

## Cost

Gemma's two runs cost about $1.26 in provider charges (45,910 sentences). Everything else ran locally.
