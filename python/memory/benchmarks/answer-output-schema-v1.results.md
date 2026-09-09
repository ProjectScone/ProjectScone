# Application output schemas — functional probe, 9 September 2026

An explicit application schema increased requested-field compliance from
**2/4 to 4/4** on the same four Gemma functional cases. JSON syntax acceptance
remained **4/4**. This is a synthetic contract check, not a public QA accuracy
result. Three new answers were sentences rather than short entity names;
field validation does not establish answer precision or factual accuracy.

## Setup and change

The baseline is the v3 run recorded in
[the earlier format probe](tool-answer-contract-v1.results.md). The new run
uses commit `2522de4`. Both used `gemma4-e4b-ctx8k:latest`, digest
`858470a535014f5c6847e388069aa64cb4f892e99309c851b59be288eec784cf`, an 8,192-token
context, temperature zero, reasoning disabled and a 2,048-output-token limit.
Python 3.14 ran a fresh SQLite ledger, embedded Qdrant and HashEmbedder. This
does not measure Qdrant server performance or semantic retrieval quality.

The original questions, case order, source text, tool-call limits, one-line
limit, 1,024-byte limit and textual answer instruction were unchanged.
The source remained:

> At the Selene concert, Mara played the oboe. Idris played the cello.

The questions were “Which instrument did Mara play at the Selene concert?”
and “Who played the oboe at the Selene concert?” Each was asked once with an
early-answer opportunity and once with only the initial search allowed,
forcing final writing. The new application requirement was:

```json
{
  "type": "object",
  "properties": {"answer": {"type": "string"}},
  "required": ["answer"],
  "additionalProperties": false
}
```

This schema was supplied in the answer requirements and the provider's
response schema, then enforced by the host before returning text. Each run
froze its code hashes and verified them at terminal completion. No completed
attempt was retried, replaced or edited. Public QA artifacts were not changed
or rescored. No training or fine-tuning was performed for this probe.

## Every observed reply

| Case | Baseline returned JSON | New returned JSON |
| --- | --- | --- |
| Instrument, early | `{"instrument":"oboe"}` | `{"answer":"Mara played the oboe at the Selene concert."}` |
| Instrument, final | `{"answer":"oboe"}` | `{"answer":"Mara played the oboe at the Selene concert."}` |
| Person, early | `{"player":"Mara","instrument":"oboe","concert":"Selene"}` | `{"answer":"Mara played the oboe at the Selene concert."}` |
| Person, final | `{"answer":"Mara"}` | `{"answer":"Mara"}` |

All four new turns used one model call, retained their source and passed source
revalidation. Their `verified_accuracy` value remains false. Source retention
means the recorded evidence is still available, not that the answer is correct.
Early responses contained an answer object inside the action envelope; final
responses contained a direct object. The adapter compacted structural whitespace
and removed the early action envelope, without changing answer values. Raw
provider responses and returned serializations are both retained.

## Observed cost

| Case | Baseline turn latency | New turn latency |
| --- | ---: | ---: |
| Instrument, early | 33.304 s | 3.006 s |
| Instrument, final | 10.723 s | 0.711 s |
| Person, early | 17.562 s | 1.121 s |
| Person, final | 4.920 s | 0.544 s |

These are single observations on a shared workstation with the memory service
still running. They include setup, inference and source checks. The large
difference does not isolate a schema-related speedup; model residency, host
activity and provider caching were not controlled. The sample is too small
for a performance claim or general model-quality conclusion.

## Verification and artifacts

Raw artifacts remain ignored under
`bench-runs/tool-answer-output-schema-smoke-2026-09-09/`. The observations SHA-256
is `598895ebc6c7b27f87fc75a36b771b653d47377d05e9b72ee5aea5c485e8544c`.
The prior v3 observations remain at
`277972b2cb99f81cdb1ed5f66e8d4ed07d826b47a1bfc619b23de90b6b2b4df3`.

Verification checked terminal counts, unchanged questions/source/model inputs,
current code hashes, provider schema equality, raw-to-returned object equality,
host acceptance and source receipts. The runner passed mypy before inference.
Implementation validation passed 329 tests with 294 optional-backend cases
skipped; three changed implementation files passed mypy. Independent review
also compared 192 reference-inlining cases with the standard validator and
approved the fix for `minContains`/`maxContains` coverage.

The next generation issue is choosing the requested answer span while retaining
the correct entity and relation. That needs separate evaluation from JSON
syntax, field compliance and source availability.
