# Source-declared table context in embeddings

Enable `SCONE_TABLE_CONTEXT_EMBEDDINGS=1`, or construct `MemoryEngine` with
`table_context_embeddings=True`, to restore missing table headers in chunk
embedding inputs. The engine policy is immutable; select it at construction.
It applies to retained document manifests with source-declared cell/header and
merged-row associations, including supported HTML, Word and XLSX tables.
It does not infer headers from OCR geometry or generate table summaries.

Native table extraction already includes declared header context in ordinary
rows. Scone adds context only for data cells touched by a chunk when their
associated header or row-span text is absent from that chunk. This matters when
a long cell or row is divided across chunks. Header values already in the chunk
are not repeated; header cells themselves do not receive the extra prefix.
The prefix includes only missing source text, without internal source locators
or a duplicate of the cell's contents.

The stored episode, chunk text, UTF-8 byte offsets and provenance remain unchanged.
The prefix affects embedding inputs, not quoted evidence or lexical indexing.
A retrieved fragment can still need evidence expansion to answer a table question;
this setting does not add header text to answer prompts or implement relational
queries over tables.

Before embedding, Scone loads the original and manifest from the record's memory
space, verifies their digests and types, validates cell/header associations, and
checks that the reconstructed extraction matches the episode. Missing sources,
corruption, inconsistent stored chunk text/spans and context-budget excess refuse
indexing before a model call. Additional context is bounded to 8,192 UTF-8 bytes
per chunk and 8 MiB per episode, including the prefix framing. There is no silent
truncation or fallback to different inputs.

Initial ingestion, interrupted-index recovery and explicit vector rebuilding use
the same resolver. Completed embedding checkpoints bind the actual augmented
inputs. Durable document jobs bind the enabled context algorithm in the index-step
revision, so a saved job cannot silently resume with a different policy.

The vector-writer identity records `source-table-headers-v1` when enabled. Existing
vectors must be rebuilt before they can be compared under the new policy. The
existing engine rules apply: deterministic local hash vectors can rebuild on open;
other embedders require explicit `reembed_vectors()`. All models remain selected
by the host. Disabling this setting preserves the previous writer identity and
embedding inputs.

Run the controlled local experiment from `packages/memory`:

```sh
PYTHONPATH=src python benchmarks/table_context.py
```

The experiment measures exact retained cell spans on six synthetic table-value
questions and a final excerpt on six long-cell questions, using hash-256 vector
search alone. It reports both configurations and query-level ranks. These fixtures
exercise lost-header behavior; they are not held-out retrieval or model-accuracy
evaluations. Evaluate your own tables and chosen embedder before enabling the
setting for a production index.
