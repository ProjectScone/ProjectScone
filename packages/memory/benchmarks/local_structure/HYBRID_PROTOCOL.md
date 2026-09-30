# Local paragraph ranking development experiment, v1

Freeze before inference on the same complete QASPER v0.3 development export:
281 papers, 1,005 questions, previously inspected. Do not tune after seeing these
results. This is development evidence, not a fresh holdout.

Compare native flat vectors, local section expansion, and local paragraph hybrid
ranking without hosted reranking. Hybrid projects the top 32 indexed chunk hits
onto intersecting complete source paragraphs/code/tables, scores each paragraph by
its strongest overlapping vector hit, and fuses that ranking with the top 32
paragraph BM25 matches using equal-weight reciprocal ranks with constant 60.
Use existing native tokenization and BM25 defaults. No query-specific confidence
threshold, learned weights, or gold-dependent filtering.

All arms retrieve at most five output items and pack at most 8,000 UTF-8 bytes
including source headers. For paragraphs that exceed the remaining source budget,
hybrid retains the strongest intersecting vector hit (or the paragraph prefix if
there is no vector hit), bounded to valid UTF-8. Unprojectable heading-only hits
stay in the vector ranking. Skip later overlapping spans. Existing section/flat
modes retain their original policies.
Source offsets must be original. Hybrid adds a paragraph lexical index but no
new embeddings; all arms retain the same source-aligned 700-character vectors.

The runner has no hosted chooser or reranker implementation. Copy the previous
cache locally and abort any embedding miss. Query embeddings are already cached;
report these as warm retrieval/context-packing timings, not full chat latency.
Rotate three-arm order by question ordinal, run serially, and report index/source
preparation separately. Local artifacts only; freeze source/input/cache hashes
and complete schedule, verify them again at completion. Refuse nonempty output
and concurrent writers. An interrupted run is incomplete and is not resumed.

After all inference completes, score complete-paragraph evidence F1 and recall,
including text-only metrics, with official evaluator parity. Count every question.
Compare paired quality with the frozen prior four-arm journal as well: these are
the exact same development questions, but prior hosted latency is a historical
reference from a separate run, not a matched current-provider timing experiment.
Do not infer causal speedup ratios from those historical values. Preserve previous
artifacts unchanged. Report 2,000-draw paired paper-bootstrap intervals, seed
20260924, unadjusted and without repeated-model variance. No generated answers;
no answer-quality claim, no automatic deployment/default change.
