# Retained-source budget comparison v1

Keeping ten source anchors instead of five raised complete Hotpot annotation
coverage from **72% to 92%** within the same 8,000-byte context limit. Preparation
remained inexpensive in this run. This measures evidence delivery; answer
accuracy remains unmeasured for these expanded contexts. Defaults stay unchanged.

## Frozen comparison

The [protocol](source-budget-v1.protocol.md) was committed as `ea6a1d9` before
evaluation. All 200 original development questions ran once at each source limit
(5, 8, 10), in rotating order. All configurations retrieved the same 20 candidate
items with the same implicit lane depth of 80, no reranker and no neighboring
passage expansion. Only the retained-source limit changed.

The run used Python 3.14, copied SQLite and Qdrant 1.19.1 stores containing the
original 2,176 paragraphs, and cached BGE-small-en-v1.5 embeddings. Package code,
runner, protocol, original questions, source files, baseline requests, copied
stores and embedding artifacts were fingerprinted and verified unchanged.
No questions were rewritten, no gold filters or seeded facts were used, and no
generation, training or replacement attempts occurred. The reserved 200
questions remain unrun. Labels were read only after all observations were saved.

## Annotation coverage

| Dataset / metric | Five sources | Eight sources | Ten sources |
|---|---:|---:|---:|
| Hotpot (100): supporting-annotation coverage, macro | 86.80% | 93.80% | 96.33% |
| Hotpot: all supporting annotations present | 72% | 86% | 92% |
| SQuAD (100): annotated answer present | 99% | 99% | 99% |
| All 200: annotation coverage, macro | 92.90% | 96.40% | 97.67% |
| All 200: all annotations present | 85.50% | 92.50% | 95.50% |

Against five sources, eight sources produced 14 Hotpot gains and no losses;
ten sources produced 20 gains and no losses. These counts hold for both macro
annotation coverage and complete coverage. SQuAD had no changes.

Every five-source request and retained source sequence matched the original
baseline (200/200). Eight- and ten-source requests and sequences all differed.
All expanded sequences retained the five-source sequence as an exact prefix;
every ten-source sequence also retained the eight-source prefix.
Native top-k rankings were not separately rescored in this experiment.

## Context size and preparation cost

Across 200 questions per configuration:

| Measurement | Five sources | Eight sources | Ten sources |
|---|---:|---:|---:|
| Preparation p50 | 24.17 ms | 25.00 ms | 25.62 ms |
| Preparation p95 | 31.26 ms | 32.58 ms | 33.82 ms |
| Context bytes, mean | 3,578.72 | 5,475.37 | 6,754.88 |
| Context bytes, p50 | 3,603.5 | 5,562 | 6,842 |
| Context bytes, p95 | 4,230 | 6,354 | 7,682 |
| Context bytes, minimum | 2,290 | 3,422 | 4,540 |
| Context bytes, maximum | 4,473 | 6,636 | 7,993 |
| Retained sources, minimum / maximum | 5 / 5 | 8 / 8 | 10 / 10 |
| Distinct documents, mean | 4.69 | 7.48 | 9.34 |
| Distinct documents, minimum / maximum | 3 / 5 | 5 / 8 | 7 / 10 |

The preparation timer stops before the benchmark parses and validates returned
sources. Model construction is excluded. Order rotation reduces ordering bias;
this shared host was not a dedicated performance environment. These timings do
not measure model generation cost, which may increase with the larger contexts.

## Process failure and data completeness

All 600 preparations returned `prepared`, with no recorded preparation errors.
The process subsequently **exited 134** with a native `libc++abi` error:
`recursive_mutex lock failed: Invalid argument`. The precise shutdown cause is
unresolved; this was not a clean process completion.

The complete observation file, final unchanged-input checks and completion
manifest were written before that failure. The separate scorer exited zero and
verified all 600 unique scheduled observations, rotated ordering, unchanged
questions, request digests, source identities, byte/source limits and frozen
artifact hashes. No observations were omitted or rerun. The native failure is
preserved separately from preparation statuses in `process-exit.json` and
`process.log`; it must not be interpreted as a successful runtime lifecycle test.

## Interpretation and artifacts

The five-source limit discarded useful evidence already present in the retrieved
candidates. Retaining more of it improved this sample's literal coverage without
the large CPU cost observed in the [reranking comparison](public-reranking-v1.results.md).
This is a development-set result, not a held-out accuracy claim. Additional
passages can also distract generation. A separately frozen generation comparison
is needed before selecting a new default.

The original literal-match rules remain: Hotpot supporting-sentence substrings
must occur within their annotated document, and SQuAD answer substrings within
an annotated supporting document. Chunk boundaries can undercount useful
evidence. These metrics are not semantic correctness, answer EM/F1, confidence,
or generated supporting-fact/joint scores.

Raw requests, sources, receipts, row metrics, all nine dataset/configuration
aggregates and runtime diagnostics remain outside git in
`bench-runs/source-budget-dev-2026-09-08/`.
An independent audit recomputed all 600 row metrics and all nine aggregates
exactly, including paired changes, statuses, sizes and baseline equality, and
rechecked the frozen artifacts after the process failure.

- Manifest SHA256: `1483a0dfca6bbc682a58cdfa06e7b89adbbd88fc49136f4dd57b5ee79b98e56d`
- Observations SHA256: `219ceb259b758b4294eea5ba81fa3486aae86834048ab16adedea65969e1a87c`
- Scores SHA256: `d51cd22d6c4d99df1ba4912d8b780dc024eb153ad63f49d26da55d8f13915979`
- Scorer SHA256: `f2390bab23e96785bc6800a6ba92b8f9f28afb47dbd3637415ede60e7037f006`
