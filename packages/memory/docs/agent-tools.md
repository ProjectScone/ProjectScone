# Scoped memory tools and bounded agent turns

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

## Using it from a framework

For model tool calls, `scone_memory.integrations.tools.ToolBox` binds an async
engine to one host-selected space. Its `openai()` and `anthropic()` methods
render the same eight contracts: `search_memory`, `add_memory`, `read_profile`,
`trace_memory`, and the four entity-graph reads `graph_context`,
`explain_entity`, `connect_entities` and `graph_schema`. Hosts can
allowlist a subset. The host executes returned
tool calls with `await box.run(name, arguments)`; installing an adapter does
not automatically enable a tool loop in HTTP Conversations or the MCP server.

```python
from scone_memory.integrations.tools import ToolBox

box = ToolBox(engine, "default", tools=["search_memory", "trace_memory"])
found = await box.run("search_memory", {"query": "who maintains Juniper?"})
if found["ok"] and found["facts"]:
    evidence = await box.run("trace_memory", {
        "seed_fact_id": found["facts"][0]["fact_id"], "max_hops": 3,
    })
```

Tracing returns complete quoted claims with source episode IDs and recorded
origins, directed stored relations, and ordered paths. Object-to-subject
matches are labeled `subject_object`. Traversal uses the ledger's subject
normalization (case folding and collapsed whitespace) and also checks exact
spelling for directly inserted records, sharing the same work limits. It does
not infer aliases or join merely similar names.
Contradictions remain separate evidence, never a path continuation or an
automatically chosen winner. Quote retention is checked; factual accuracy is
not certified. Source text remains untrusted data for the receiving model.

The graph reads are the ones the MCP server offers as `memory_graph_context`,
`memory_entity`, `memory_connections` and `memory_graph_schema`:

- `graph_context` returns what the entity graph records around up to 24
  names, or around the entities a question names. `max_bytes` (512 to
  64,000) bounds the packet text. The candidates and ids around it are
  capped on their own: 24 candidates at most, with names clipped to 120
  characters. With `similar: true`, a question also finds up to three
  entities it resembles by vector, each marked with its score. Only a
  `min_similarity` you pass keeps weak matches out.
- `explain_entity` returns one entity's relations in both directions and
  its values, or the candidates for an ambiguous name.
- `connect_entities` returns the shortest paths between two entities,
  within `max_hops` (1 to 4).
- `graph_schema` returns what the graph is made of: its entity kinds,
  its predicates and the kinds each predicate joins. It is the
  `/v1/graph/schema` JSON, for a model to read before it asks anything.
  `max_bytes` (1,024 to 64,000, default 16,000) bounds the listed
  predicates, and a predicate over 200 characters is shown clipped.

Each reads the current projection of the box's space at one instant. The
first three answer with the JSON that `/v1/graph/context` returns: the
packet text, its status, seeds, candidates and coverage, and the instant
it read at. `graph_schema` answers with the `/v1/graph/schema` JSON.
Every line cites the facts behind it, re-read at that instant, and
coverage says what was left out. A name over 200 characters, a question
over 2,000 characters, or a bound out of range comes back as a result
with `ok: false`.

```python
box = ToolBox(engine, "default", tools=["graph_context", "connect_entities"])
around = await box.run("graph_context", {"question": "who works with Alice Chen?"})
route = await box.run("connect_entities", {"source": "Alice Chen", "target": "Lisbon"})
```

The trace is read-only and bounded: 16 facts, 32 edges, 256 traversal store
calls/candidates, eight paths, and a two-second async timeout. Two additional
revision reads fence native writes. Complete source episodes may be loaded
during point reads. The evidence packet is capped at 64,000 UTF-8 bytes before
the ToolBox envelope. Every source must match any supplied tags. Missing or
ineligible seeds return empty evidence; timeouts, changed revisions, and store
failures return an unavailable result without partial quotes. Direct adapter
writes require adapter transaction discipline. Coverage describes this seed's
bounded neighborhood, never completeness of an answer to an arbitrary query.

## Scoped read tools

For read tools that must preserve a conversation or application's narrower
scope, use `ScopedMemoryTools`:

```python
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope

box = ScopedMemoryTools(engine, "default",
    scope=RecallScope.validated(where={"collection": "manuals"},
                               kind="file", source_prefix="manuals/"),
    exclude_session_id="current-session",
)
schema = box.openai()  # or box.anthropic()
found = await box.run("search_memory", {"query": "Juniper", "limit": 5})
```

This binding offers `search_memory`, `trace_memory`, and `read_memory`. Space, metadata,
source kind/prefix, inclusive source dates and excluded session are fixed by the
host and cannot be supplied or widened in tool arguments. There are no write or
unfiltered profile operations. Search returns retained passage bytes and quoted
fact provenance; it makes one native retrieval request and no assessor-model
call. It uses a 20-candidate / 32,000-byte evidence window, then returns at most
`limit` combined facts and passages (facts first). Omitted output is counted and
marked truncated. Search coverage never certifies completeness for the question.

`read_memory` accepts a retained `chunk_id` and `before`/`after` counts in 0–4
(both default to 1). It returns up to nine exact neighboring chunks from the same
authorized source, preserving their IDs and byte spans. Its optional document
store port, `ChunkWindowLookup.page_chunks`, filters by space, episode and ordinal
before limiting the read to ten rows, including one look-ahead row. Memory,
SQLite, MongoDB, PostgreSQL and Elasticsearch implement this port; older custom
stores return `unsupported_chunk_window` instead of loading every chunk.
Coverage records the returned ordinals and whether later chunks were observed;
it never claims document or answer completeness. Read items use a placeholder
score of `0.0`, not a relevance or confidence estimate. Source validation can
still load the full episode through point reads.

Each call has a configurable 1–30 second cooperative deadline (default 2) and a
512–64,000 UTF-8 byte result cap (default 64,000). Oversized output is refused as a
whole, without clipped quotations or paths. Tracing also retains its stricter
native two-second deadline, checked before accepting output even if a backend
suppressed cancellation. Source point reads may load larger episodes and
cooperative cleanup can exceed deadlines. Errors are content-free codes;
external cancellation propagates. Tool results are checked snapshots, not
permanent evidence: an enclosing answer pipeline must revalidate sources before
publishing an answer based on them. No HTTP tool loop or automatic tool execution
is installed by constructing this binding.

## Bounded model tool turns

For an opt-in model-native tool turn, use the bounded host controller:

```python
import os
from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolLoopLimits
from scone_memory.providers.tool_chat import SelfHostedToolChat

model = SelfHostedToolChat(
    os.environ["SCONE_CHAT_URL"], os.environ["SCONE_CHAT_MODEL"],
    api_key=os.environ.get("SCONE_CHAT_API_KEY"),
)
reply = await EvidenceToolLoop(model, box, limits=ToolLoopLimits(
    max_tool_calls=4, max_tool_rounds=4, timeout_s=120.0,
)).run([{"role": "user", "content": "What is the recorded Juniper dependency chain?"}])
print(reply.text)
print(reply.source_status, reply.evidence_ids)
# Immutable JSON packets retain the passages, quoted claims, directed paths,
# source identifiers, and partial-coverage markers supplied to the model.
packets = reply.evidence_packets
```

## Output requirements

Standalone tool turns can also apply the same host-owned output requirements
as text conversations:

```python
from scone_memory.realtime.answer_requirements import AnswerRequirements

requirements = AnswerRequirements(
    format="json_object", max_bytes=1024, max_lines=1,
    instructions="Return an object with the answer in the dependency field.",
)
reply = await EvidenceToolLoop(
    model, box, initial_search=True, answer_requirements=requirements,
).run([{"role": "user", "content": "What does Juniper depend on?"}])
```

Requirements are validated and copied at construction, supplied on every model
request, and included in the transcript byte budget. Both early answers and
answers after tools are exhausted must satisfy the format, UTF-8 byte and line
limits before `run()` returns. Invalid returned text raises `RuntimeError` with
`tool answer format rejected`; it is not repaired or retried. The
loop's own reply limit still applies. Source revalidation remains mandatory.
The structured adapter's final-writing prompt respects the caller's requested
format instead of requiring prose or an explanation. By itself, `json_object` validates
JSON object syntax; textual
instructions guide the model but are not deterministic semantic checks.
`SelfHostedStructuredToolChat` uses an object-valued answer branch and a final
JSON-object response schema when these requirements request `json_object`.
Valid generated objects are rendered as compact JSON for the text API. Only
whitespace outside strings is removed; number tokens (including precision and
exponents), escapes, string contents and key order are preserved. Fenced JSON,
trailing prose, duplicate keys and wrong value types are rejected by the
provider, not repaired. Raw response formatting can therefore differ from the
returned JSON serialization; record both when evaluating this mode.

Custom providers may implement the optional
`ConstrainedToolModel.complete_with_requirements(messages, tools, requirements)`
capability to apply generation constraints. They receive independent copies
of both the messages and requirements. Providers with only `complete()` remain
supported through prompt guidance and the host's return gate. Omitting
`answer_requirements` preserves the existing SDK return contract and provider
action schema. TextConversation keeps its separate review/repair boundary.
The [recorded Gemma format probe](../benchmarks/tool-answer-contract-v1.results.md)
shows both the improvement in JSON syntax and the remaining field-shape and
latency limitations.

To enforce field names, types and constraints, install
`scone-memory[structured-output]` and add an application schema:

```python
requirements = AnswerRequirements(
    format="json_object", max_bytes=1024, max_lines=1,
    output_schema={
        "type": "object",
        "properties": {"answer": {"type": "string", "minLength": 1}},
        "required": ["answer"],
        "additionalProperties": False,
    },
)
```

The structured adapter sends the same compiled schema for early and final
answers. The host validates the returned object even if a provider ignores
the schema. Other providers receive it through the requirements capability
or prompt, with the same host check. Provider support for individual JSON Schema
keywords varies; unsupported schemas can fail generation without a fallback.

Contracts use Draft 2020-12 validation with an object root. Acyclic local
JSON Pointer references are inlined before embedding; external references,
recursive references, identifiers, dynamic references, anchors, custom dialects
and unknown keywords are rejected during configuration. No schema is fetched.
`format` remains an annotation, without format assertion. Input and expanded
schemas each have a 32 KiB limit; JSON trees are limited to 4,096 values and
32 levels, and schema expansion to 256 nodes and 16 levels. These are bounded
application configurations, not a sandbox or a hard validation CPU deadline.
Schema-mode numbers use exact decimal validation, with at most 4,096 coefficient
digits and an absolute exponent of 4,096. Syntax-only mode keeps its existing
number behavior. Field validation does not verify an answer's factual accuracy.
The [recorded application-schema probe](../benchmarks/answer-output-schema-v1.results.md)
improved requested-field compliance from 2/4 to 4/4 on the same four Gemma
fixtures; three answers still used a full sentence instead of a short entity.

## Initial retrieval and evidence reuse

By default the SDK model chooses search queries. `EvidenceToolLoop(...,
initial_search=True)` first searches the final user message with `limit=5`,
before the first model request. This host-initiated search uses the same scope,
source checks, deadline and byte limits as model-requested tools, and consumes
one tool call. It does not consume a model decision round. Call outcomes identify
their `origin` as `host` or `model`. An unavailable initial search prevents
generation; an empty search is shown to the model without proving absence from
memory. This mode requires a final, nonblank user message of at most 8,000 UTF-8
bytes and fails before storage access if that query is invalid.

Tracing becomes available after retained facts
are discovered; nearby reading becomes available after retained chunks are
discovered. Both accept only IDs returned during that turn. This discovery gate
belongs to the loop; standalone `box.run()` enforces scope without turn history. Host scope still
applies to every expanded source. Every declared call receives a matching tool
response, including denials. Attempts consume the call budget; both results and
denials consume the aggregate tool-byte budget. If complete pairing cannot fit,
the turn fails before another model request. Exhausting calls or rounds permits
one final request with tools disabled; further tool calls are protocol failures.

Within one run, repeated `read_memory` calls reuse retained evidence after fresh
source validation. Matching includes normalized default offsets and windows
with different anchors that cover the same already-read whole source. Whole-source
matching requires contiguous ordinals from zero and explicit untruncated source
coverage; narrower windows remain separate reads. Search and trace calls, failed
or empty reads, and results rejected by the output budget are not cached.
A reused call returns a compact reference to the earlier tool result, reports
`reused=true` in its outcome, and still consumes a call and output bytes. It
does not append duplicate source packets to the receipt or establish completeness.
Source changes or failed revalidation stop the turn before another model request.

`EvidenceToolLoop(..., compact_search_results=True)` optionally compacts an
identical nonempty search result into a reference to its first full result in
the current turn. It is off by default. Every search still executes: this is
presentation compaction, not a retrieval cache. Matching requires the entire
result payload to be byte-for-byte identical, including ranking, ordering,
metadata and coverage. Changed results, empty results and errors are offered
normally. A reference is used only when it is shorter than the full payload.

The reference identifies `tool_result_number`, `new_evidence_count: 0` and
`searched_again: true`; its outcome reports `reused=true`. The earlier receipt
is revalidated before reuse, and every fresh search receipt remains in the
returned evidence packets and final validation set. Attempts still consume
the same call budget; the bytes actually offered consume the tool-byte budget.
This does not establish that the existing evidence is sufficient or correct,
and it does not force the model to answer or to choose a different query.

## Budgets and provider behavior

Defaults bound the transcript to 256,000 bytes, all tool output to 128,000 bytes,
and the final reply to 16,000 bytes. No intermediate text is published. Sources,
chunks, facts, and links are frozen before model consumption and rechecked before
the result is returned. Native revision changes or changed/deleted snapshots fail
the turn. This uses bounded point reads, not a database-wide atomic transaction;
direct adapter writes still need transaction discipline. `box.prepare()` shares
one tool deadline across retrieval and snapshotting. Each later validation pass
is also bounded, within the overall loop deadline.

The self-hosted adapter requires native OpenAI-compatible tool-call support.
It does not execute tool-shaped prose, capture reasoning fields, follow redirects,
use environment proxies, discover/download models, or select a fallback provider.
It rejects incomplete replies, duplicate JSON keys/call IDs, non-finite numbers,
and oversized responses.
HTTP resources are closed before a reply is accepted; cooperative cleanup can
run beyond the deadline, but late success is rejected.

Both tool adapters emit INFO-level `tool_model.started` and `tool_model.finished`
events with a shared call ID. Diagnostics distinguish `native`,
`structured_action`, and `structured_answer` requests, and record request/response
bytes, time to response headers, elapsed time, HTTP status, and the configured
output-token limit. Failures identify HTTP errors, transport errors/timeouts,
request deadlines, response-byte limits, or invalid responses; external
cancellation remains a separate outcome. Public errors remain generic.
An allowlisted finish reason can expose a provider's `length` termination.
Optional `prompt_tokens`, `completion_tokens`, and `total_tokens` are reported
only when the provider supplies bounded nonnegative integers; missing or invalid
values remain unknown, not zero. These counters do not prove context fit or
answer correctness. Logs omit prompts, responses, reasoning, model/endpoint
identifiers, credentials, and raw exception text. Response bytes count decoded
body bytes received before success or failure, not network transfer bytes.

## Structured-action providers

For an explicitly configured model with unreliable native tool calls but JSON
schema support, `SelfHostedStructuredToolChat` implements the same `ToolModel`
interface:

```python
from scone_memory.providers.structured_tool_chat import SelfHostedStructuredToolChat

model = SelfHostedStructuredToolChat(
    os.environ["SCONE_CHAT_URL"], os.environ["SCONE_CHAT_MODEL"],
    api_key=os.environ.get("SCONE_CHAT_API_KEY"),
)
```

It requests one schema-constrained search, trace, read, or answer action, validates it
again on the host, and translates it into the existing bounded tool protocol.
Trace and read seeds must be retained IDs. Scope, execution, source checks, and budgets
remain in the host controller. This adapter does not silently repair JSON or
execute ordinary prose. Tool results are rendered as explicitly marked evidence
messages for providers without native tool-message support. It is opt-in; native
and ordinary chat adapters retain their existing behavior.

## Development findings

Protocol compatibility is not evidence of better answers. In a nine-case
synthetic development comparison on `llama3.2-ctx8k`, structured actions fixed a
malformed native trace-call case, but the model still invented a missing bridge,
reversed a dependency, conflated `painted by` with `depends on`, and answered a
manufacturer question without searching. The checked-in cases are in
`tests/fixtures/tool_action_cases.json`; their expectations require manual
source-grounded adjudication. These are development findings, not a held-out
accuracy score, and do not justify enabling this adapter by default.
