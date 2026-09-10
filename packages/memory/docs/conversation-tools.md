# Tool-based conversations and experiments

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

The controller does not persist workflow checkpoints or automatically capture
chat messages. For conversation history, capture, and public callbacks, opt in
through `TextConversation`:

```python
from scone_memory.realtime.text import TextConversation

conversation = TextConversation(engine, "default", "chat-1",
    tool_model_factory=lambda: SelfHostedToolChat(
        os.environ["SCONE_CHAT_URL"], os.environ["SCONE_CHAT_MODEL"],
        api_key=os.environ.get("SCONE_CHAT_API_KEY"),
    ),
    where={"collection": "manuals"},
    tool_limits=ToolLoopLimits(max_tool_calls=4),
    turn_timeout=120,
)
try:
    reply = await conversation.reply("What is the Juniper dependency chain?")
    receipt = reply["memory_context"]["tool_retrieval"]
finally:
    await conversation.close()
```

## Capture and receipts

Tool mode uses the actual conversation history, with current-session records
excluded from tool searches. It bypasses prompt-based memory preparation and
cannot be combined with independent adaptive retrieval or
extractive selection. The usual conversation deadline and reply/history byte
limits still apply. A public callback receives one final reply after source and
history checks; sources are checked again before assistant capture. A callback
can observe a reply whose later capture fails, so the terminal result remains
the confirmation of completion. ToolModel adapters own per-request resources.

Receipts include call outcomes (including content-free error codes), retained
IDs, source IDs, and fingerprints. Direct replies also contain transient source
packets. Outcome IDs are host-assigned ordinals (`tool-1`, etc.); unknown tool
names are normalized so model-authored strings cannot become cached diagnostics.
Custom `create_conversation_app` runtimes can use this mode today;
cached HTTP receipts discard the packets and rebuild the existing graph from
matching retained source/fact/link fingerprints. Deleted or changed evidence
disappears on refresh. Graph inspection remains bounded and may be partial.
The composed server can select tool mode explicitly for its saved self-hosted
chat connection:

```sh
SCONE_MODEL_CONNECTIONS=~/.scone-memory/model-connections.json
SCONE_CONVERSATIONS_JOURNAL=~/.scone-memory/conversations.db
SCONE_CONVERSATIONS_TOOL_MODE=structured # off (default), native, structured
SCONE_CONVERSATIONS_TOOL_INITIAL_SEARCH=1 # default when tool mode is enabled
SCONE_CONVERSATIONS_TOOL_COMPUTE=0 # opt-in exact arithmetic/counts on quoted inputs
# SCONE_CONVERSATIONS_TOOL_MAX_CALLS=4
# SCONE_CONVERSATIONS_TOOL_MAX_ROUNDS=4
# SCONE_CONVERSATIONS_TOOL_TIMEOUT=120
```

## Serve configuration

Configure the chat connection in the model settings or supply `SCONE_CHAT_URL`
and `SCONE_CHAT_MODEL` as connection defaults. Served tool mode performs the
host-initiated search by default. Set `SCONE_CONVERSATIONS_TOOL_INITIAL_SEARCH=0`
to let the model choose whether to search. SDK `TextConversation` exposes the
same option as `tool_initial_search=True`, with the SDK default remaining false.

## Computation from retained passages

Set `SCONE_CONVERSATIONS_TOOL_COMPUTE=1` to offer `compute_memory` after a
passage has been retrieved in the current turn. SDK callers use
`TextConversation(..., tool_model_factory=..., tool_compute=True)` or
`ScopedMemoryTools(..., enable_computation=True)`. The capabilities endpoint
reports this setting as `tool_retrieval.computation`. It requires an enabled
tool mode and uses the existing call, time, output and final-source-validation
budgets; it adds no model or service dependency.

The tool accepts `operation`, `left` and `right`. Each input is
`{"chunk_id": 123, "quote": "12.5"}` referencing an exact, unique substring of an
authorized passage. `sum`, `product` and `count` use only `left`;
`difference`, `ratio` and `compare` require one input on each side;
`compare_counts` compares the two selected groups. There are at most 16 inputs
total, with quotes up to 512 characters. Arithmetic accepts whole ASCII decimal
tokens (up to 30 integer and 18 fractional digits); expressions, exponents and
thousands separators are rejected. Difference and ratio use left minus/divided
by right. Results are exact decimals or rational strings such as `1/3`, with
original quotes and character offsets. Ambiguous quotes, overlapping source
spans within a group, division by zero, and stale or excluded sources fail
without returning a partial calculation. Direct SDK tool calls check source
access; the agent loop additionally requires prior retrieval of every input.

`count` counts selected spans, **not all entities in a document or corpus**.
Repeated entities in separate mentions are not deduplicated. The model must
choose the correct entities, attributes and compatible units; no unit conversion,
list-completeness check or semantic answer verification is performed. A computed
result still has `verified_accuracy=false`. Structured final writing preserves
and recomputes calculation receipts alongside the original source passages.
These are calculation/source-integrity checks, not evidence of improved model
accuracy on a benchmark.

`native` requires native OpenAI-compatible tool calls; `structured` requires
JSON-schema responses. There
is no automatic protocol or provider fallback. The setting applies to ordinary
text sessions and text sessions using a self-hosted persona; voice keeps its
existing pipeline. Sessions capture the connection at creation, and every turn
gets a fresh tool model. `SCONE_CHAT_THINK` is forwarded when explicitly set.

The whole tool turn has the configured timeout, while each provider request is
bounded by the smaller of that budget and the saved connection timeout. The
existing whole-conversation turn timeout can end it sooner. Calls and rounds
accept 1–16; the tool timeout accepts 0.01–600 seconds. Transcript, output and
reply byte limits retain `ToolLoopLimits` defaults. Startup refuses combinations
with adaptive retrieval, trusted custom model factories or custom
persona catalogs. Those integrations can still bind their own SDK pipelines.

## Evidence rendering

The structured adapter renders tool results as explicitly labeled, untrusted
evidence in plain chat roles. After a completed tool exchange it repeats the
latest actual user question so the provider's answer target does not become the
last source packet. This rendering does not change stored conversation history;
the expanded history is checked against the adapter's 1 MB limit before sending.
Its tool-selection instructions distinguish reading neighboring document chunks
from tracing a mentioned entity's relationships to find a requested attribute.
This guides the model's choice; it does not change retrieval permissions,
increase budgets, or establish that an answer follows from the retained evidence.

When the host disables tools at the end of the turn budget, the structured
adapter switches that final request to prose writing from a dedicated evidence
view. It preserves the actual conversation, source quotations, recorded triples,
origins and validity dates, stored links, and path directions. Equal source
records are deduplicated; conflicting versions and incomplete paths fail before
the request. Ranking scores and machine confidence remain in the diagnostic
receipts, outside the writer's view. Numeric values inside source quotations are
preserved. Coverage limits and retrieval failures remain visible.

This uses the same model and final request, with no additional retrieval or
model call. It removes transient action history rather than generating a summary
of the sources. The projected history is bounded to 1 MB; records are never
sliced to fit. The adapter's 64,000-byte answer limit and the host's configured
reply limit still apply, and original source snapshots are revalidated before
publication. Early `answer` actions remain on the existing JSON path. Final prose
can still misunderstand an attribute or invent a connection; evidence retention
does not certify the generated answer.

`/v1/conversations/capabilities` exposes `tool_retrieval` protocol, budgets and
whether a text connection is configured. This is configuration availability,
not a model-health probe or an accuracy claim. Without a saved chat connection,
text remains unavailable; enabling the mode does not install a model.

## Development experiments and remaining errors

`source_status="retained"` confirms source revalidation, not answer entailment;
`verified_accuracy` remains false. Without initial search, a model can skip
retrieval; either policy can still misunderstand a relation or emit a tool
request as prose. In a synthetic 3B Ollama development
probe, invalid trace arguments and tool-shaped prose prevented a useful answer.
That probe is not a successful accuracy benchmark; tool-use reliability remains
a separate quality requirement before enabling this mode by default.
Before host-initiated search, a two-question served 3B development probe skipped retrieval for a
natural memory question and repeatedly reread one passage when explicitly asked
to use tools, missing a neighboring exception. Successful protocol execution and
retained citations did not make those answers correct. With initial search,
follow-up SQLite and MongoDB/Qdrant probes both returned the requested cutoff
and outage exception for the explicitly guided question. The general question
still omitted the cutoff despite retaining its evidence. These two synthetic
questions used a hash embedder and are integration probes, not a retrieval or
generation accuracy benchmark. A subsequent controlled replay identified plain-role
message ordering as a contributor to the omitted cutoff: restoring the real
question after tool results produced both requested facts with the same three
tool calls. An actual served SQLite probe confirmed both facts for both questions.
The guided question used four calls instead of two, including a duplicate read;
this change does not establish more efficient tool planning. Eight simpler
fixed-evidence cases retained both requested answer details with either rendering.
These are synthetic development observations, not general accuracy guarantees.
Whole-source reuse subsequently reduced the guided probe's last tool result from
1,411 to 260 bytes, skipping its duplicate read preparation after revalidation.
Total tool output fell from 4,392 to 3,241 bytes; four tool attempts remained.
The answer still included the cutoff and outage hold. This measures reduced
duplicate work and context, not better model planning or general accuracy.
Served mode stays opt-in.
