# Answer target feasibility probe, v1

User acceptance target: a complete answer in less than 100 ms, at least 90%
answer correctness, and perfect (100%) supporting-evidence recall. Treat the
deadline as per-request, not a median or first-token target. Cache hits cannot
stand in for first-time questions. No requirement is declared achieved here.

This is a small feasibility diagnostic, not the full accuracy evaluation. Select
32 questions deterministically by SHA-256 of question ID from the existing
annotation-free 1,005-question QASPER development export. Do not inspect answers
to choose questions. Reuse the already-frozen `hybrid_no_rerank` contexts from the
complete local-only run. Thus measured time is **generation only**: retrieval,
query encoding, scheduling and application transport would add work. Failing the
generation-only 100 ms threshold already fails this local pipeline's complete
answer target. Passing it would not establish end-to-end compliance.

Use the already-installed local Ollama `llama3.2-ctx8k:latest` with temperature 0,
256 output tokens, the existing QASPER answer prompt, serial nonstreaming requests,
and one disclosed warmup. Save model metadata/digest, immutable schedule/input/code
hashes, every response, wall time and backend timing counters. Do not use hosted
services, new model downloads, answer caching or gold evidence. No retries. Keep
failed/truncated answers in the denominator. Refuse existing output. Model loading
is reported separately from warm-request timing, not silently counted as free.

After inference completes, score official answer token F1 and normalized exact
match on these 32 questions, with explicit disclosure that neither is a validated
semantic correctness percentage. A 32-item diagnostic cannot certify 90% answer
correctness on the workload. Retain the earlier ten-item synthetic probe only as
a latency sanity check; its 9/10 exact answers are not benchmark accuracy.

Audit evidence representation separately: earlier full-paragraph scoring omitted
caption markers and heading/subparagraph annotations. Classify unmatched gold
strings against original source fields offline. Do not use gold-derived aliases
in retrieval or label a changed evidence metric as a retrieval improvement.
Preserve all earlier source/results artifacts. Caption coverage does not imply
access to figure pixels or table cells, or correct answers about those values.
