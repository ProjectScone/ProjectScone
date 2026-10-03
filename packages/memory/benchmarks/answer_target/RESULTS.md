# Strict answer target: feasibility remains unproven

The requested acceptance criteria are a **complete answer below 100 ms**, at least
**90% answer correctness**, and **100% supporting-evidence recall**. These are
joint requirements. None of the evaluated pipelines demonstrates all three.
The deadline is not a median or time-to-first-token target. Answer token F1 and
exact match are reported diagnostics, not substitutes for semantic correctness.

## Local generation on real questions

At commit `120d1105`, the frozen [protocol](PROTOCOL.md) selected 32 questions by
SHA-256 of ID from the complete 1,005-question development export, independently
of gold answers. It used unchanged saved local-hybrid contexts and the installed
local Ollama `llama3.2-ctx8k:latest` on an Apple M3 Max with 36 GiB memory. Inference
made no hosted requests, downloaded no models, and read no gold annotations.

| Measure | Result |
| --- | ---: |
| Complete responses | 32 / 32 |
| Complete responses below 100 ms | **0 / 32** |
| Fastest generation-only request | 336.70 ms |
| Median generation-only request | 547.74 ms |
| p95 / maximum request time | 1,308.06 / 1,703.91 ms |
| Answer token F1 | 35.55% |
| Normalized exact match | 9 / 32 (28.125%) |
| Semantic correctness independently verified | No |

The local request timer includes the complete nonstreaming response, with one
disclosed 186.66 ms warmup. Retrieval, query embedding, scheduling, and application
transport are excluded and would add work. Thus this tested reader already fails
the complete-answer deadline before those stages. This does not prove that every
possible local architecture fails, or estimate full-workload accuracy from 32 cases.
Official QASPER answer-F1 parity was verified after inference completed.

A prior ten-question synthetic code-lookup sanity probe produced 9/10 exact
answers at 60.78 ms median, with every warm request under 100 ms. Its tiny contexts,
short answers, and one observed error make it unsuitable as a 90% benchmark
accuracy claim. Cold model loading/warmup took 1,932.98 ms in that probe. Neither
probe uses precomputed answers or excludes unsuccessful cases.

## Correction to the evidence-coverage diagnosis

The earlier 92.56% available-text oracle ceiling was a ceiling of our existing
paragraph-equality representation, **not proof that all remaining content was
absent from the corpus**. An offline audit classified all 459 annotation entries
not exactly equal to a stored body paragraph:

| Representation difference | Annotation entries |
| --- | ---: |
| Figure/table marker whose remaining caption exactly matches a source caption | 253 |
| Section heading | 129 |
| Substring already present in rendered source | 72 |
| Whitespace difference | 5 |
| No match after these checks | 0 |

These counts include repeated entries across annotations. They are not counts of
distinct missing paragraphs or questions. The dataset documents figure/table
markers, and its baseline reader includes section headings and maps evidence
substrings to source paragraphs. See the [dataset specification](https://huggingface.co/datasets/allenai/qasper/blob/main/README.md)
and [official reader](https://github.com/allenai/qasper-led-baseline/blob/main/qasper_baselines/dataset_reader.py).

The prior result reports accurately describe their frozen full-paragraph metric,
but passing the official evaluator parity check only confirms arithmetic on those
predictions; it does not validate our evidence extraction policy. Correcting
source/evidence identity alignment requires a separately versioned evaluation.
It must not use gold-derived aliases in inference or relabel a measurement change
as retrieval improvement. Existing reports and raw observations remain unchanged.
Caption identity coverage also does not supply figure pixels or table-cell values
needed to answer some questions.

## Consequences for implementation

The current hosted Jev path takes about 344 ms for reranking alone, and the tested
local generative reader also exceeds 100 ms. The next candidate design therefore
needs a faster local answer mechanism or source-derived answers prepared before
queries arrive, with source-version invalidation. Such a design still needs
independent correctness evaluation; neither caching repeated benchmark answers
nor dropping difficult questions can establish the requested target.

First repair and version the evidence identity audit, then evaluate new answer
mechanisms with the entire selected workload, separate first-time queries from
cache hits, count every timeout/failure, and measure the complete request. A
generative fallback exceeding 100 ms must remain a deadline failure. No default
policy or production pipeline was changed by this diagnostic.

## Artifacts and verification

Local results are under `bench-runs/accuracy-target-2026-09-27/real-32-v1/`.
The manifest records the model metadata/digest, exact selected questions, source
and input hashes. Every response, backend timing counter, wall time, failure
status, and warmup is retained. The completion audit verifies immutable inputs
and the exact schedule; the offline scorer verifies official answer-F1 parity.
The parent directory retains `audit_evidence_alignment.py` and
`evidence-alignment-audit.json`, with source and script hashes, for the 459-entry
representation audit.
All five probe tests and the combined 85 focused native/benchmark tests pass;
the new runner passes strict mypy. Production retrieval source is unchanged.

| Artifact | SHA-256 |
| --- | --- |
| Manifest | `bfe894fe878aaa0dce5fbcb84af81b81b88ee66a4b94738047f6942fcf4bae94` |
| Observations | `c29828f43787f63ab421afd731d647a8bb297175599dc043e78d190ca0c8cd80` |
| Completion | `770092335acac699d35d8df8680152b6d880b89543553c3607921e465d2fc964` |
| Offline scorer | `c7093c7a96279ac5aba30fecc468139664430c85751bb227e390968bde7053a3` |
