# Dynamic-schema extraction: triples and the types they use, proposed for review

The distiller reads an episode and proposes facts under whatever
predicate wording the model chooses; nothing says which kinds of thing a
space holds or which predicates it expects. A graph built over documents
usually has a vocabulary in mind — a project uses a product, a person
works at an organisation — and text that says something the vocabulary
does not cover.

The dynamic-schema pass asks a chat model, once per chunk, for the
(subject, predicate, object) triples the chunk states, with the kind of
each end, under a suggested vocabulary of entity kinds and predicates. A
switch allows or forbids kinds and predicates outside it. Every triple
quotes the chunk, and what passes is stored as a proposal for a person to
approve or decline, never as a held claim.

It is off by default: it runs only by hand, with a model, and costs one
model call per chunk.

## What is kept

Each reply is an object `{"triples": [...]}` (a bare list is read too)
whose entries are `{"subject", "subject_kind", "predicate", "object",
"object_kind", "quote", "statement_type"}`. A model that can answer to a
JSON schema (`providers.llm.StructuredChatModel`, such as
`OpenAICompatibleChat`) is asked with `dynamic_schema.REPLY_SCHEMA`. An
entry becomes a proposal only when:

1. **It is whole.** Both ends are text, and the predicate and both kinds
   reduce to terms: casefolded, words joined by one underscore, at most 64
   characters (`schema_term`: "Depends On" and "depends-on" are both
   `depends_on`). Otherwise it is `malformed`.
2. **It is among the first `max_triples_per_chunk` entries** of the
   reply. The rest are not read and are counted in `dropped_extra`.
3. **It passes the grounding gate the distiller uses**
   (`distill._grounding_reason`), with the chunk as the source:
   - the quote is present (`missing_quote`), at most 2000 characters
     (`quote_too_long`) and verbatim in the chunk (`quote_not_in_source`)
     — these three are the report's `unquoted`, and `quoted_share` is the
     share of whole triples that had none of them. A quote whose words
     stand in the chunk with other runs of whitespace between them is
     first replaced by the chunk's own span (`settle_quote`) and counted
     in `quotes_settled`: a model copying a hard-wrapped line joins it
     with a space, and the stored quote must be a substring of the
     episode. Case, punctuation and every other character must match;
   - `statement_type` is `observation` (`not_an_observation`);
   - both ends are named in the quote (`subject_not_in_quote`,
     `object_not_in_quote`);
   - the quote's clause holds none of the negating, conditional and
     hedging words the distiller refuses (`not`, `if`, `unless`, `may`,
     `should`, `plans` and others; `when` and `can` are not among them),
     and is not an instruction or a question (`context_not_asserted`);
   - the predicate's words, less glue words, are the quote's
     (`predicate_not_in_quote`). A suggested predicate is therefore used
     only where the quote supports its words: the pass does not map
     "requires" onto a suggested `depends_on`.
4. **Its types are admitted.** A predicate or kind outside the suggested
   vocabulary is refused as `new_type_not_allowed` when new types are off.
   When they are on, a pass admits at most `max_new_predicates` new
   predicates and `max_new_kinds` new kinds; a triple that needs one more
   is `new_type_cut`. A term already admitted in the pass costs nothing
   again.
5. **It is not already on record.** The same subject, predicate and object
   from the same episode, in any status — proposed, held, closed or
   declined — is counted in `restated` and not proposed again, so running
   the pass twice does not double the review queue and a declined proposal
   does not come back. The same triple from another episode is its own
   proposal, as the distiller's would be.

The distiller's rule that two objects for one subject and predicate in one
reply reject each other (`same_source_conflict`) is not applied: documents
state many values at once ("the store keeps episodes, chunks and facts"),
and proposals are not placed in the ledger until approved, where the
predicate's cardinality decides.

## Where proposals go

Through the existing proposal and review path:
`engine.assert_fact(..., proposed=True, origin="extracted",
quote=..., source_episode_id=..., valid_from=episode.created_at)`. A
proposal answers nothing until `approve` (or a batch `decide`) accepts it;
`decline` keeps it out with its reason.

The predicate stored is the term, so a model's `Uses` and the suggested
`uses` are one predicate in the ledger.

## Proposed vocabulary

A predicate or kind outside the suggested vocabulary that a proposal
written in this pass uses is **proposed vocabulary**. The report lists
each in `new_predicates` and `new_kinds` with the number of proposals
using it and up to three example quotes (`examples`, each with its fact
id and chunk id; `examples_cut` counts the rest).
`proposed_outside_schema` counts the proposals using any such term: the
triples a strict fixed schema, like LlamaIndex's `SchemaLLMPathExtractor`
with `strict=True`, would have dropped.

The pass leaves a `dynamic_schema` event holding the report, with the
proposals as fact ids. That event is a receipt, not the store: an event
log may evict (see `entities/vocabulary.py`). The durable record of a new
predicate is the proposals themselves, which keep the predicate, the
quote and the source. Entity kinds have no column in the ledger —
`entities.kinds` infers kinds from predicates — so a proposed kind is
recorded in the report and the event only, and approving a proposal does
not change how any entity's kind is inferred.

## Running it

```python
from scone_memory.ingestion.dynamic_schema import extract_dynamic_schema

report = await extract_dynamic_schema(
    engine, "notes", chat,
    entity_kinds=("person", "organisation", "project", "product"),
    predicates=("works_at", "uses", "depends_on"),
    allow_new_types=True,
)
print(report.text())
```

```sh
SCONE_CHAT_URL=http://127.0.0.1:11434/v1 SCONE_CHAT_MODEL=llama3.2-ctx8k \
  scone --space notes dynamic-schema --kind project --kind product --predicate uses --max-calls 50
```

`--no-new-types` makes the suggested vocabulary the whole schema, and
then at least one kind and one predicate must be named. `--json` prints
the report's record. There is no HTTP route and no `SCONE_*` setting, and
the consolidation worker never runs this pass.

An episode with a proposal from this pass is no longer pending for the
distiller, which reads only episodes no claim cites. Run one or the other
over a space, not both on a timer.

## Bounds, and what the report says when they cut

| Bound | Default | At most | When it cuts |
|---|---|---|---|
| `max_calls` | 200 | 2000 | `chunks_cut` counts chunks not looked at, `resume_after` names the chunk id to pass as `after_chunk` next |
| `max_triples_per_chunk` | 10 (LlamaIndex's default) | 50 | entries past it are `dropped_extra` |
| `max_new_predicates` | 20 | 200 | `rejected_reasons["new_type_cut"]` |
| `max_new_kinds` | 10 | 200 | `rejected_reasons["new_type_cut"]` |
| `MAX_CHUNK_BYTES` | 8000 | | `skipped_long` |
| suggested kinds, suggested predicates | | 200 each | refused before any call |
| `MAX_EXAMPLES` | 3 per term | | `examples_cut` |

`cut_by` names the bounds that cut a pass: `calls`, `chunk_bytes`,
`triples_per_chunk`, `new_types`. A bound out of range is refused before
any model call. A call that fails is counted in `calls_failed`, a reply
with no list of triples in `replies_unparsed`, and the pass goes on to
the next chunk. A chunk whose episode is forgotten while the model
answers gets nothing written and is counted in `chunks_gone`.

## How this differs from the references

LlamaIndex's `DynamicLLMPathExtractor` passes its allowed types to the
model as an initial ontology and keeps every triple it can parse;
`max_triplets_per_chunk` is a request in the prompt, not a limit.
`SchemaLLMPathExtractor` with `strict=True` drops triples outside its
validation schema, silently. Neither asks for or checks a quote, neither
reports what it dropped, and both write straight into the graph. Here the
limit is enforced and counted, every triple must quote its chunk, every
rejection is counted by reason, the new types are reported with their
evidence, and nothing enters the ledger without review.

Measured on this repository's documents with a local model:
[`benchmarks/dynamic-schema-v1.results.md`](../benchmarks/dynamic-schema-v1.results.md).
