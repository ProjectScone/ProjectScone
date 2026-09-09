# Tool answer contracts — functional smoke probe, 9 September 2026

Object-constrained generation increased accepted JSON replies from **0/4 to
4/4** on the same four small fixtures. This is a synthetic functional probe,
not a public-dataset accuracy result. Only **2/4** constrained replies used the
requested `answer` field; generic JSON-object syntax does not enforce an
application's field schema. All four constrained replies retained their sources
and passed source revalidation. None is marked as verified factual accuracy.

## Setup

Both runs used the installed `gemma4-e4b-ctx8k:latest` model, digest
`858470a535014f5c6847e388069aa64cb4f892e99309c851b59be288eec784cf`, an 8,192-token
context, temperature zero, reasoning disabled, and a 2,048-output-token limit.
Python 3.14 ran a fresh SQLite document store, embedded Qdrant and HashEmbedder.
This does not measure Qdrant server performance or semantic-embedding quality.
The memory service and unrelated workstation activity remained running.

The single synthetic source was:

> At the Selene concert, Mara played the oboe. Idris played the cello.

The two unchanged questions were:

- Which instrument did Mara play at the Selene concert?
- Who played the oboe at the Selene concert?

Each ran once with an early-answer opportunity (four available tool calls) and
once with only the initial search allowed, forcing the final-writing request.
All eight turns used one provider call each. Both revisions received the same
host requirements: JSON object, at most 1,024 UTF-8 bytes, one line, and the
instruction “Return one JSON object with an answer string and no other fields.”

Run v2 used commit `757ada4`, with prompt guidance and a return gate. Run v3
used `7946adc`, adding an object-valued action branch and a JSON-object schema
for the final-writing request. Each run recorded code hashes before inference
and verified them at terminal completion. No completed model attempt was
retried, replaced, or edited. Public QA questions and benchmark artifacts were
not changed or rescored.

## All observed replies

In v2, both early actions returned this answer string:

```text
Mara played the oboe at the Selene concert.
```

Both final-writing requests returned:

````text
```json
{
"answer": "Mara played the oboe at the Selene concert."
}
```
````

The host rejected all four with `tool answer format rejected`. It did not
extract, repair or publish JSON from the fenced responses.

In v3 the generated objects were rendered as follows:

| Case | Returned JSON | Host format gate | Requested field shape |
| --- | --- | --- | --- |
| Instrument, early | `{"instrument":"oboe"}` | Accepted | Wrong key |
| Instrument, final | `{"answer":"oboe"}` | Accepted | Satisfied |
| Person, early | `{"player":"Mara","instrument":"oboe","concert":"Selene"}` | Accepted | Wrong keys |
| Person, final | `{"answer":"Mara"}` | Accepted | Satisfied |

The early responses carried those objects inside the action envelope's
`answer` field; final responses were direct objects. The adapter removes the
action envelope and structural whitespace for the existing text API. It does
not change JSON values, numeric tokens, escapes, key order or string contents.
Raw wire responses and returned serializations are both retained. Syntax
acceptance must not be reported as satisfaction of the textual field request.

## Cost and limitations

| Case | v2 turn latency | v3 turn latency |
| --- | ---: | ---: |
| Instrument, early | 4.789 s | 33.304 s |
| Instrument, final | 2.143 s | 10.723 s |
| Person, early | 2.518 s | 17.562 s |
| Person, final | 2.261 s | 4.920 s |

All four constrained turns were slower. These are single observations on a
shared workstation, including provider setup and source checks; they do not
isolate grammar compilation, inference, or contention. The increased time
needs further measurement. Object generation is used only when callers
explicitly request JSON-object requirements, not for ordinary text turns.

A separate two-request probe changed only the grouping of adjacent system
messages. Both outputs remained fenced JSON. That probe did not establish
that system-message handling caused the failures.

An initial harness attempt failed during setup: first an incorrect import,
then a missing required scope argument in four constructed cases. It issued
zero model requests. Its logs and terminal records are preserved separately;
the corrected v2 runner passed mypy before inference. These setup failures are
not counted as model attempts or silently replaced with successful replies.

## Evidence and follow-up

Raw artifacts remain ignored under
`bench-runs/tool-answer-contract-smoke-2026-09-09{,-v2,-v3}/`.
The grouping probe is inside the v2 directory. Observation hashes are:

- v2: `55a48f52f02d241f7301f3c43ef2c8deaa2aea5247d5ce94d94fbaa635cd8e70`
- v3: `277972b2cb99f81cdb1ed5f66e8d4ed07d826b47a1bfc619b23de90b6b2b4df3`

Verification checked terminal counts, unchanged case/source/model inputs,
current v3 code hashes, raw-to-returned object values, and source receipts.
The implementation passed 363 selected tests plus three additional
intermediate-tool checks, and mypy for four implementation files. Independent
review additionally checked 9,000 deterministic nested-object renderings.

The next contract requirement is an explicit application field schema, with
provider generation constraints and host validation using the same schema.
It must be evaluated separately from factual grounding, answer-type selection,
and public QA accuracy. Existing failed observations remain unchanged.
