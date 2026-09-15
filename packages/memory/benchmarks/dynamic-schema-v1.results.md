# Dynamic-schema extraction on this repository's documents — 15 September 2026

The dynamic-schema pass (`ingestion/dynamic_schema.py`,
[docs](../docs/dynamic-schema-extraction.md)) asks a chat model, once per
chunk, for triples with the kind of each end under a suggested
vocabulary, keeps those whose quote stands in the chunk and passes the
distiller's grounding gate, and stores them as proposals. This measures
what it proposes from 20 documentation chunks with a small local model,
under two fixed vocabularies, with new types allowed and with them off.

**Result.** With new types allowed, 92–93% of the whole triples the model
wrote quoted their chunk (55–57% exactly as written; the rest differed
only in whitespace), and the pass proposed 24 triples under Scone's
kind-hint schema and 18 under LlamaIndex's default schema. A strict fixed
schema would have dropped 23 of the 24 and all 18: with new types off the
same vocabularies gave 1 and 0 proposals. By my reading of the quotes, 13
of the 24 and 8 of the 18 proposals state what their sentence says; the
rest have the wrong subject, an inverted direction, a dropped qualifier,
or come from a conditional clause the gate does not treat as unasserted.
The vocabularies are general-purpose (people, places, organisations;
LlamaIndex's products, markets, locations) and the chunks describe
software, so "a strict schema misses almost everything" is a property of
that pairing as much as of the pass.

## Method

- Corpus: every document in `packages/memory/docs` (50 files, including
  this feature's own document) stored with `bench.questions.store_corpus`
  in an in-memory engine at the default chunk size; 1604 chunks of at
  least 200 characters. 20 were sampled with `random.Random(42)`, each
  stored as an episode of its own (one chunk each) in a fresh engine per
  arm. The chunks, from 11 documents: retrieval-and-storage #242, #342,
  #444, #521; text-conversations #15, #23, #25; agent-application-tools
  #3, #13, #17; agent-models #53, #66; conversation-tools #7, #18;
  agent-tools #0; agent-tool-approvals #3; agent-turn-journal #8;
  directory-sync #15; file-ingestion #35; pdf-ocr #34.
- Model: `llama3.2-ctx8k` through Ollama at `127.0.0.1:11434`, asked with
  the reply schema (`OpenAICompatibleChat.complete_structured`, temperature
  0, 2048-token ceiling), 300 s timeout. One run per arm, on a machine
  shared with other agents' jobs (load average 21 to 33). Seconds are
  recorded, not claimed.
- Vocabularies: **Scone** — the ten `entities.kinds.EntityKind` kinds and
  the 54 predicates its kind hints know (`works_at`, `lives_in`, `uses`,
  `depends_on`, …). **LlamaIndex** — the ten entities and ten relations of
  `SchemaLLMPathExtractor`'s defaults as terms (`product`, `technology`,
  `concept`, …; `used_by`, `part_of`, `has`, `is_a`, …).
- Arms: new types allowed, and new types off, per vocabulary; the pass's
  default bounds (10 triples per chunk, 20 new predicates, 10 new kinds).
- The runs were made at commit 7a78d924. The self-reference rule, and the
  review's gate rules (a line break in running text does not end a
  clause; every place in the episode holding the quote's words, in any
  wrapping, is read), were added after them; the numbers below are the
  saved replies replayed through the pass as committed (`benchmarks/dynamic_schema.py replay`, the same
  chunks and replies in call order, a failed call failing again). The
  replay reproduces a run exactly when the rules match: without settling,
  the first run's replay gave its report to the count.
- Reproduce, from `packages/memory` with `PYTHONPATH=src`:
  `benchmarks/dynamic_schema.py run --out S.json --vocabulary scone`, the
  same with `--vocabulary llamaindex --chunks-from S.json`, and `replay
  --out` on each. The saved runs are not committed.

## Results

"Whole" is triples read less malformed ones. "Quoted" is the share of
whole triples whose quote stands in the chunk, after a quote differing
only in whitespace is replaced by the chunk's own span; "as written"
counts only quotes that stood as the model wrote them. "Outside" is
proposals using a kind or predicate outside the suggested vocabulary:
what a strict fixed schema would have dropped.

| Vocabulary | New types | Read | Whole | Quoted | As written | Proposals | Outside | New predicates | New kinds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scone | allowed | 143 | 141 | 0.929 | 0.553 | 24 | 23 | 18 | 6 |
| Scone | off | 160 | 149 | 0.926 | 0.718 | 1 | 0 | 0 | 0 |
| LlamaIndex | allowed | 167 | 167 | 0.922 | 0.569 | 18 | 18 | 16 | 10 |
| LlamaIndex | off | 168 | 168 | 0.911 | 0.619 | 0 | 0 | 0 | 0 |

Why whole triples were not proposed:

| Vocabulary | New types | subject not in quote | object not in quote | predicate not in quote | not an observation | context not asserted | quote not in chunk | self-reference | new type not allowed | new type cut |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Scone | allowed | 39 | 22 | 16 | 13 | 15 | 10 | 2 | – | 0 |
| Scone | off | 50 | 28 | 16 | 12 | 12 | 11 | 3 | 16 | – |
| LlamaIndex | allowed | 72 | 25 | 6 | 17 | 12 | 13 | 2 | – | 2 |
| LlamaIndex | off | 77 | 25 | 8 | 16 | 11 | 15 | 2 | 14 | – |

Bounds and failures: one call in each Scone arm failed with the reply
schema's token ceiling reached (`calls_failed` 1, the chunk yielded
nothing); no reply was unreadable; triples past the per-chunk limit were
dropped 0, 8, 3 and 6 times; the LlamaIndex arm with new types allowed
reached the 10-new-kinds budget and cut 2 triples (`cut_by`:
`triples_per_chunk`, `new_types`). Nothing was restated.

The one proposal that fit Scone's schema, in both arms: (evaluator,
`uses`, SQLite storage) from "evaluator uses temporary SQLite storage".
Nothing fit LlamaIndex's: the model wrote `uses`, which it does not have.

### Proposed vocabulary, new types allowed

Scone: `locates_main_document` (3 proposals), `remains` (3), `are_chosen`,
`are_repaired`, `can_be_visible_before_retirement`, `can_emit`,
`can_skip`, `exceeds`, `holds`, `is_missing`, `is_owned_by`, `read`,
`remains_marked`, `render`, `require`, `requires`, `retain`, `retains`;
kinds `text` (3), `annotation`, `function`, `number`, `parameter`,
`state`. For example, `requires` rests on "Every model-supplied parameter
requires an annotation." and `locates_main_document` on "DOCX, XLSX and
PPTX locate their main document through `_rels/.rels`".

LlamaIndex: `locates_main_document` (3), `can_be_visible`, `can_still`,
`drops`, `has_read_mode`, `holds`, `included`, `is_paged`, `is_read`,
`is_read_by`, `is_refused`, `reports`, `requires`, `retain`, `retains`,
`uses`; kinds `file_format` (3), `file_path` (3), `column`, `migration`,
`records`, `repository`, `run`, `schema`, `shape`, `width`.

The terms are not normalised across tense or number (`require`,
`requires`; `retain`, `retains`), and several embed their object
(`locates_main_document`) or a hedge (`can_skip`). That is what a
reviewer is shown, with the quotes.

### Reading the proposals

Counted as right when the triple says what the quoted sentence says, with
the right subject and direction, from a clause that asserts it. One
reader (the author), not a judged set.

- Scone, new types allowed: 13 of 24. Right: the three
  `locates_main_document` triples, (evaluator, uses, SQLite storage), (run
  request, retains, its chosen width), (sequential records, retain, their
  previous encrypted payload shape), (parallel records, require, a reader
  that supports scheduling), (runtime shutdown, is_owned_by, host),
  (model-supplied parameter, requires, annotation), (tool-use reliability,
  remains, a separate quality requirement), (baseline, remains,
  unreviewed), (passages, are_chosen, in rank order under the byte
  budget), (repository, holds, schema). Wrong, for example: (indexing,
  remains_marked, in flight) and (ceiling, exceeds, 1) from "when …"
  clauses; (model, attends, to the middle of a long context) from "a
  model that attends *least* to the middle"; (mode, remains, a separate
  quality requirement), whose subject is the clause's other noun.
- LlamaIndex, new types allowed: 8 of 18, the same kinds of error, with
  three triples from one sentence about `coverage.read_mode` all wrong.

### Rules measured on the same replies

The first run (made at commit 7d512fb6, before quotes were settled; its
seeded sample was drawn before this feature's document grew by a
paragraph, which moved 9 of the 20 chunks from those above; Scone
vocabulary) was replayed with each rule turned back:

| Rule | New types | Quoted | Proposals | Outside |
|---|---|---:|---:|---:|
| exact quotes only (before settling) | allowed | 0.426 | 5 | 5 |
| settle whitespace | allowed | 0.839 | 18 | 17 |
| settle whitespace, refuse self-reference | allowed | 0.839 | 17 | 16 |
| exact quotes only | off | 0.594 | 0 | 0 |
| settle whitespace (and self-reference) | off | 0.884 | 1 | 0 |

64 of the 89 quotes refused before settling differed from the chunk only
in whitespace: the documents are hard-wrapped and the model joins a
wrapped line with a space. By the reading above, 2 of the 5 proposals
before settling were right and 7 of the 18 after. On the live runs the
self-reference rule removed 2 proposals in each arm with new types
allowed (26 to 24, 20 to 18), all wrong. The clause fix in
`distill._clause_around` (a quote that keeps its full stop is read in its
own sentence) changed no count on these replies. The review's gate rules,
replayed on all three saved runs, changed no proposal and no share: one
triple in the LlamaIndex arm with new types off moved from predicate not
in quote to context not asserted ("replacement is a sequence of
writes,\nnot a global transaction", whose wrapped line's `not` now reads
in its clause; the context check comes first).

## What this does not say

- Nothing about a larger model, other documents, or conversation text.
- Nothing against LlamaIndex's extractors run on these chunks: they were
  not run. Their vocabulary was used; their outputs carry no quote to
  score.
- The correctness counts are one reader's, over 42 proposals.
- The gate's "context not asserted" words miss `when` and `can`; those
  proposals reach review. A clause is found by punctuation, blank lines
  and list, heading and table lines, not parsed: running text whose
  clause ends without a mark reads on into the next line.
