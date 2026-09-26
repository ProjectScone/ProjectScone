# QASPER development routing comparison, v1

Freeze this protocol and implementation before generating development answers.
Run all 281 papers and 1,005 questions from the official QASPER v0.3 development
split: 4,020 scheduled answers across four arms. No sampling, exclusions, or
post-result policy tuning. Development paper and question IDs must be disjoint
from the earlier official test run; its results remain frozen.

Source: https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
The extracted `qasper-dev-v0.3.json` SHA-256 is
`2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93`.
Inference reads only exported corpus/questions, never gold annotations. Paragraphs
that resemble Markdown headings or fences are enclosed in literal fences without
changing their original bytes. All arms use the same exported paper.

## Arms and matched resources

- `scone_flat`: native per-document vector retrieval.
- `scone_structure`: native hierarchical Jev section routing and automatic fetch.
- `scone_vector_candidates`: native vector-guided section routing and the same
  automatic original/scoped-vector/broad-vector fetch policy.
- `llamaindex`: vector + BM25 reciprocal-rank fusion, one query, no rewriting.

This is known-paper QA, not discovery across a pooled corpus. Every arm receives
the specified paper. Use the same 700-character source-aligned chunks and cached
Nemotron 3 Embed 1B vectors: OpenRouter `nvidia/nemotron-3-embed-1b:free`, 2,048
dimensions, `query: ` / `passage: ` prefixes. Space embedding API requests at
least 3.2 seconds apart. All documents, indexes, and artifacts remain local;
the user authorized these existing hosted model endpoints for this evaluation.

Each arm produces at most 32 candidates within 8,000 bytes. Jev
`typesafe/jev-1.13-20260917` scores the union of unique candidate texts in batches
of at most 64 judgments. Identical texts reuse the same judgment. Final contexts
contain at most five items and 8,000 bytes including source headers. Route cache
size is zero. Alternate the two routing policies' execution order by global
question ordinal and rotate answer-generation arms by that ordinal. Use up to
eight concurrent questions within each paper; process papers sequentially.

Generate independently with `google/gemma-4-31b-it`, temperature zero, no
reasoning, 256 output tokens, and identical instructions. Zero temperature does
not guarantee identical answers. Persist contexts before generation. Resume
requires the same manifest and reuses durable contexts and answers. Keep failed
answers in denominators as zero; output-token-limit failures allow scheduling to
continue. Other provider failures stop after the current bounded batch. An
unfinished generation attempt requires explicit recovery, never silent retry.

## Metrics and provenance

Primary comparison: vector-guided routing minus hierarchical routing. Also
compare it with flat retrieval and LlamaIndex, and report hierarchy against both
controls. Report official answer token F1, supplemental normalized EM, and full
original-paragraph evidence F1/recall with all-evidence and text-only variants.
Require the full paragraph string in context for evidence credit. Retain all
questions, including unanswerable and figure-dependent ones. Check answer-score
parity against the official evaluator independently for each arm.

Use paired paper-cluster bootstrap intervals: 2,000 draws, seed 20260924. Report
failures, effective modes, route reasons/calls, context bytes, routing/fetch,
shared rerank, generation, and stage-sum latency. Report vector preparation
separately; shared rerank is incurred once physically and must not be summed
across arms as run consumption. Report provider usage/cost when supplied; missing
monetary charges remain unknown. These timings describe this provider-backed
benchmark, not production application latency.

Freeze code/protocol/input hashes, environment versions, model IDs, settings, and
the full question/arm schedule in the manifest. Verify artifact hashes and the
complete 4,020-answer schedule after the run. Do not tune this implementation once
generation starts. Any follow-up change requires a new manifest and separately
reported run. Development results do not establish untouched test performance;
model training overlap is unknown.
