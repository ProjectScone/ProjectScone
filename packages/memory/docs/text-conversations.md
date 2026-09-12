# Native text conversations and memory preparation

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

## Native real-time conversations

Scone owns conversation scheduling, source preparation, public streaming and
transcript capture. Python **3.11+** is required for real-time sessions; the base
memory package still supports Python 3.10. No external conversation framework,
audio model, device or credential is installed or selected by this runtime.

The implementation lives in `scone_memory.realtime`:

- `text.TextConversation`: bounded text turns and public-chunk observation.
- `voice.VoiceSession`: audio input, transcription, interruptions and spoken output.
- `context.MemoryContext`: immutable-scope, transient source preparation.
- `events`: shared `TextDelta`, `ReplyCompleted` and `TextModel` protocol.
- `audio`: PCM packets, speech events and transport/STT/TTS protocols.
- `lifecycle`: cancellation and owned-cleanup handling.

A model is a small provider adapter, not a frame processor:

```python
from scone_memory.realtime.events import TextDelta, ReplyCompleted
from scone_memory.realtime.text import TextConversation

class MyModel:
    async def respond(self, messages):
        # Replace this scripted response with your explicitly configured provider.
        yield TextDelta("Hello.")
        yield ReplyCompleted()  # only after the provider's successful response end

    async def aclose(self):
        # Close any clients owned by this adapter.
        pass

conversation = TextConversation(
    memory, "authorized-space", "session-1", MyModel,
    where={"collection": "manuals"}, kind="file", source_prefix="docs/",
)
try:
    result = await conversation.reply("What does the manual say?")
finally:
    await conversation.close()
```

The host authorizes the memory space and public transcript retention. A synchronous,
zero-argument factory returns a **fresh model per text turn**. Its `respond`
method returns an async iterator supporting `aclose()`. Requests are deep copies;
provider mutations cannot rewrite conversation history. No tools, hidden reasoning
or media events are accepted at this text boundary.

`reply(text, on_text=async_callback)` optionally sends actual, serial public
chunks to an observer before the response ends. These chunks are provisional and
are not necessarily tokens. Backpressure reaches the adapter iterator; this does
not establish buffering guarantees inside a provider's SDK. Empty chunks are
ignored by the observer. Unsupported events, chunks after completion, observer
failures, missing completion and cleanup failures prevent completed capture.

`providers.llm.OpenAICompatibleTextModel` accepts an optional
`max_output_tokens` integer (1–32,768), forwarded as `max_tokens`; omitting it
preserves the provider's configured limit. A reported token-limit cutoff is a
failed reply, even when some text arrived. It cannot be saved as a completed
assistant response.

Each successful result contains `turn_id`, `text`, `user_episode_id`,
`assistant_episode_id`, `memory_context` and `provider_completion="unverified"`.
User text is retained before inference. Assistant text is saved only after
explicit adapter completion, iterator EOF and successful owned-provider cleanup.
Metadata records `integration=scone-text`, role/turn/session, aggregated text and
`completion_evidence=adapter_end_and_stream_closed`. This is not independent
proof of provider correctness, use of memory, or audio playback. Historical
records keep their original metadata; migration never rewrites evidence.

Input is bounded to 32,000 UTF-8 bytes. Defaults are 64,000 reply bytes,
128,000 history bytes and a 30-second turn deadline. One turn runs at a time.
`close()` cancels active work and joins cleanup. External cancellation can permit
another turn only after safe cleanup; interrupted observer or store effects close
the conversation. A submitted user message can remain after a cancelled/failed
reply. An uncertain store acknowledgment must not be automatically retried.
Cooperative cleanup can exceed a deadline; it is not hard process termination.

## Memory preparation

`TextConversation` and `MemoryContext` accept `where`, `kind`,
`source_prefix`, `since` and `until`. Scope is validated/copied at construction
and never broadens after a miss or error. Space authorization belongs to the host.

`await MemoryContext(memory, space, session_id, ...).prepare(messages)` returns
a copied request and a preparation receipt. Only a final plain-text user message
triggers recall. Sources are verbatim JSON values in a labeled, untrusted source
block before the current request; they grant no permissions and are not approved
facts. That block never enters shared history or captured transcripts. The
default 8,000-byte budget omits whole passages rather than silently clipping them.
A low-confidence result supplies no source block.

Recall accepts at most 1,000 characters of query. A longer message, such as a
pasted draft followed by a question, is searched with verbatim excerpts of
itself, up to that limit: questions first (latest first), then the closing and
opening sentences, then the sentences whose words are rarest in the message. An
excerpt that alone exceeds the limit keeps its head and tail, cut at spaces.
Nothing is paraphrased or generated. The receipt's `query_formulation` records
the method, the message and query lengths, and each kept `[start, end)`
character span of the message. Shorter messages are searched as written and
carry no `query_formulation`. The same excerpting applies to
`integrations.chat.recall_context` and the LangChain and LlamaIndex retrievers;
direct `recall`, HTTP, MCP and CLI calls still refuse over-long queries.

`MemoryContext(..., neighbor_chunks=1)` and
`TextConversation(..., neighbor_chunks=1)` optionally read one stored chunk on
each side of a ranked passage. The radius accepts 0..4 and defaults to 0.
Ranked anchors keep priority; neighboring chunks use the remaining byte budget,
with at most 24 total sources. `MemoryContext.limit` still bounds the ranked
anchors. Every added passage retains its original chunk ID, exact text and
source fingerprint; the graph labels it as a nearby passage read. Receipts
include window status, candidate/retained counts and anchor IDs. Window reads
have a separate deadline of at most one second (or `recall_timeout` when lower).
A window failure preserves ordinary ranked evidence; stale or invalid added
evidence is discarded. This option does not expand adaptive selections or
recent-history overviews. Tool conversations use their bounded `read_memory`
operation instead. Added context is not proof of relevance or answer accuracy.

Ranked queries with recalled claims can expand bounded relationships inside the
same scope. With `structured_paths=True` (the default), complete ordered paths
and their quoted claims enter the same context byte budget; related conflicting
evidence is retained together. Optional `path_quotes=True` adds `ordered_evidence`
with verbatim quotes, fact IDs and source episode IDs in traversal order. It is
off by default because extra repetition has not demonstrated a small-model
generation improvement. Paths preserve stored direction and are evidence
connections, not generated conclusions. Missing, deleted or out-of-scope links
cannot complete a path. `structured_paths=False` keeps the prior flat context
for controlled comparisons. Receipts report path counts and expansion coverage;
they do not cache raw paths or source text.

## Paired generation evaluation

Run the paired natural-language evaluator with your installed self-hosted model:

```sh
python -m scone_memory.testing.generation_ablation \
  --fixture tests/fixtures/generation/v1.json \
  --model YOUR_INSTALLED_MODEL --endpoint http://127.0.0.1:11434/v1 \
  --embedding-cache /path/to/cached/fastembed \
  --repeats 2 --timeout 60 --output /path/to/private/new-report.json
```

It uses the actual conversation context builder and synthetic sources, keeps
both successful and failed public replies, and measures evidence coverage
separately from completion-gated phrase checks. Phrase matches do not measure
semantic entailment or unsupported claims. Reports refuse to overwrite an
existing output path. Keep private evaluation reports outside version control.
Add `--ordered-quotes` to test the optional quote projection in the path variant;
the flat baseline remains unchanged.

Add `--tool-mode native` or `--tool-mode structured` to compare the same baseline
against the actual scoped `EvidenceToolLoop`. Tool candidates use the installed
model, 512 output tokens per request, four tool attempts/rounds, and 16,000-byte
transcript, aggregate tool-output and reply limits. Initial retrieval defaults
on; `--no-tool-initial-search` tests model-initiated retrieval. Use `--no-tool-think`
when the selected endpoint supports disabling that option. Tool mode cannot be
combined with the independent adaptive, extractive, or path-projection
candidate options. No fixture answer labels or required quotes reach the model.

Chat-completions adapters translate explicit `think=False` to
`reasoning_effort="none"` and `think=True` to `reasoning_effort="medium"`.
Leaving `think=None` preserves the server default. The endpoint and model must
support the requested reasoning effort; Ollama's native `think` field does not
control reasoning on its OpenAI-compatible endpoint.

Tool candidates can also use `--review-model YOUR_INSTALLED_REVIEWER
--review-quote-mode spans --review-policy require_supported`. The reviewer uses
the same retained-packet review helper as text conversations: it receives the
original tool evidence, may propose one correction, and must confirm that
correction before adoption. Source checks run before/after review and again
before accepting the final result. The original `--timeout` tool budget also
bounds review; `--review-timeout` cannot extend it. Choose both budgets to cover
the installed models. The baseline remains unreviewed. These are retrieval,
generation and review evaluations; they do not run conversation capture.

Tool rows record model-request hashes/bytes, provider-call counts, read reuse,
and post-run coverage of evidence observed at the provider boundary. Failed
generation keeps its observed evidence coverage but receives zero successful
answer credit; successful final source retention is reported separately.
Reviewed rows retain the draft, its elapsed time, review duration and receipt,
and the final accepted text. Rejected review retains the draft for diagnosis
but earns no successful answer credit. Generation-call counts exclude review
calls; the review receipt records its rounds separately. A review verdict is
not a correctness label.
Relation quotes participate in the coverage audit for both variants. Tool
request sizes describe the model-neutral transcript and schemas, not the wire
encoding of a particular provider. This evaluator uses temporary SQLite storage;
omit `--embedding-cache` for a hash-embedder integration check, not a semantic
retrieval benchmark.

Receipts report `prepared`, `empty`, `skipped` or `failed`, source episode/chunk
references, recall event ID when available, context hash/bytes, omissions and
sanitized degradation/error types. They prove preparation—not delivery or use.
Cancellation propagates instead of producing a success receipt.

## Native session interfaces

Implement the `TextModel` protocol above and import `TextConversation` from
`scone_memory.realtime.text`. Audio session and provider interfaces live in
`scone_memory.realtime.voice` and `scone_memory.realtime.audio`.
