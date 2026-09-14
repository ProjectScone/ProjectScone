# scone-client

A thin Python client for the [Scone](https://github.com/ProjectScone/ProjectScone)
HTTP API. One class, `requests` for transport, dataclasses instead of dicts.

## Quickstart

```python
from scone import Scone

with Scone("http://127.0.0.1:7437", "sk-your-key") as memory:   # or $SCONE_URL / $SCONE_API_KEY
    memory.add("the deploy checklist lives in the wiki", tags=["ops"])
    for item in memory.recall("checklist", limit=5, tags=["ops"]):
        print(item.day, item.text)
```

Start the server it talks to with `scone serve` (which needs a `config.toml`
carrying at least one `[[server.keys]]` entry, since a key is bound to
exactly one space).

## API

| Method | Endpoint | Returns |
| --- | --- | --- |
| `add(text, tags=None, source=None, created_at=None)` | `POST /v1/episodes` | `Added(episode_id, deduplicated, chunks)` |
| `recall(query, limit=None, as_of=None, tags=None)` | `GET /v1/recall` | `Recall` (iterable over `Memory`) |
| `facts(include_closed=False)` | `GET /v1/facts` | `list[Fact]` |
| `close_fact(fact_id, reason)` | `POST /v1/facts/{id}/close` | `int` (the closed id) |
| `profile()` | `GET /v1/profile` | `Profile(static_facts, dynamic)` |
| `status()` | `GET /v1/status` | `Status` |
| `tags()` | `GET /v1/tags` | `list[Tag]` |

Every non-2xx answer raises `SconeError`, carrying `.status` (the HTTP code,
or `None` when the request never reached a server) and `.message` (the
server's `{"error": ...}` text, or the plain-text body axum's own extractor
rejections use).

## Response limits

`Scone(..., max_response_bytes=16 * 1024 * 1024)` bounds decoded response bytes
before JSON parsing, including error bodies. Set a larger positive integer when
a known document or agent result needs it. An oversized response raises
`SconeError` with its HTTP status and no retained body. Receiving that error does
not tell you whether a preceding write completed; inspect the durable resource
before deciding to resume or retry it. The client adds no automatic retries.

The transport streams encoded bytes into a bounded buffer and advertises only
`gzip, deflate`. It accepts identity, gzip (up to 1024 members), and wrapped or raw
deflate; other encodings, truncated compressed bodies, and invalid trailers are
refused. Compressed wire bytes are capped at twice `max_response_bytes` plus
64 KiB, and decompression has its own output cap. These are byte limits, not a
limit on the Python objects subsequently created by JSON parsing. The configured
`timeout` still controls connection and idle read timeouts, not a total deadline.

The client sets `Accept-Encoding: gzip, deflate` after copying caller headers;
caller-supplied values cannot opt into an unsupported coding. Successful JSON
responses are not also decoded into an unused error-body string.

Each response is closed, including failed reads. An injected `requests.Session`
remains caller-owned. If a custom adapter or response hook already buffers or
decodes the body, its earlier allocation is outside these bounds; the cached
body is checked before the client parses it. Custom session retry policies are
also caller-controlled.

## Server behavior worth knowing

- **Source and event time are supported.** `add(..., source=..., created_at=...)`
  sends both fields to the Rust server; recall returns the preserved source and
  event date. They are optional, not fabricated when omitted.
- **Deduplication is not an error.** Storing identical text twice answers
  `200` with `deduplicated: true` and no `chunks`, instead of `201`.
- **Profile facts are narrower.** `/v1/profile` omits `valid_from`,
  `valid_until`, and `status`, so those are `None` on a `Fact` from there.
- **Engine errors have HTTP categories.** Missing facts return `404`, invalid
  engine input returns `422`, and unrecognized keys return `401`. Transport and
  extractor errors can have other shapes; the client preserves their status
  and handles plain-text bodies. A blank query is also refused locally.

## Install

```sh
pip install -e ".[test]"
```

## Tests

```sh
python -m pytest
```

Unit tests drive the client against a stub `http.server` in a thread. The
integration tests build `scone-cli` in release mode, start a real
`scone serve` on a free port with a temp data dir, and run a full round trip;
they skip themselves when `cargo` is unavailable or the build fails. Set
`CARGO_TARGET_DIR` to build somewhere other than `<repo>/target`.

```sh
python -m pytest -m "not integration"   # unit tests only
```


## Contributing and citation

See the framework [contribution guide](../../CONTRIBUTING.md),
[citation formats](../../CITING.md), and [citation metadata](../../CITATION.cff).
Research and academic use must credit Mark Sturman, JudgeHuman and ProjectScone
as required by the included [license](LICENSE).

## Agent workflow resources

The independently installed client includes typed catalog, plan, and run
resources for hosts that advertise the corresponding `agents.*` capabilities.
`expected_space` verifies responses and mutation admission; it does not override
the space assigned to the bearer key.

```python
from scone import Scone, ModelTask, TaskPlan

with Scone("http://127.0.0.1:7437", api_key="your-space-key") as memory:
    agents = memory.agents(expected_space="alpha")
    choices = agents.catalog()
    # Use agent/model IDs actually returned by this host's catalog.
    plan = TaskPlan("research", (
        ModelTask("answer", "researcher", "local-careful", "Answer with evidence."),
    ))
    saved = agents.save_plan(plan, expected_revision=0)
    progress = agents.start("research-1", plan=saved, question="What changed?")
    original = agents.request("research-1")
    agents.status("research-1").match(original)
```

`agents.plans()` and `agents.runs()` return one bounded page with an explicit
`next_after` cursor. `agents.plan(id)`, `agents.policy()`, and
`agents.cancel(run_id)` provide inspection and cancellation. `HumanInput` tasks
and `HandoffPlan`/`HandoffAgent` preserve their native plan formats and explicit
model choices. Resource construction is local; reads and plan saves do not
execute a model. Only an explicit `start` submits a run. Errors never trigger an
automatic retry, resume, or alternate model selection.

Catalogue choices also expose the host's configured application tools through
`AgentChoice.tools`: an immutable tuple of `ToolChoice(name, description,
revision)`, or `None` when a legacy host does not report tool selections. An
empty tuple means the agent has no application tools. Shared descriptions are
validated once and resolved to each agent's selected names; tool schemas,
handlers, and private configuration are not exposed. Reading this metadata
does not invoke tools or require an additional capability beyond
`agents.catalog`. Tool selection remains host configuration, not a plan edit.

New workflow response models reject malformed scalars, mismatched identities,
and changed acknowledgement bindings. The client preserves server HTTP errors
and refuses redirects. JSON is encoded as UTF-8 so valid multibyte inputs do not
expand into ASCII escapes beyond the server's request budget. Python 3.9 remains
supported, and the distribution includes `py.typed` for static type checking.

Human inputs use separate reply and execution operations:

```python
pending, = agents.inputs("run-1")
answered = agents.respond(pending, response="Proceed with the careful analysis.")
# Persist this continuation ID and selected reply for an explicit retry.
status = agents.continue_run(
    "run-1", continuation_id="approval-1", responses=(answered,),
)
```

A reply alone never resumes a run. The client checks fresh input identities
before each write and confirms the selected activation after continuation.
Transport failures and uncertain acknowledgements raise `SconeError`; the client
never retries a write automatically. Callers can read `inputs()` and `status()`
without execution, then explicitly retry the original response or continuation
ID and selection. A successful continuation confirms admission, not completion.
The native server remains responsible for atomic revision and source checks.

For a real local agent API contract test, point `SCONE_TEST_NATIVE_PYTHON` at
an interpreter with the repository's native framework and API dependencies:

```sh
SCONE_TEST_NATIVE_PYTHON=/path/to/native/python python -m pytest -q tests/test_native_agents.py
```

The test launches an isolated loopback server, restarts it after saving a reply,
and verifies that only the explicitly selected model executes after continuation.
It does not contact a model provider or the live memory service.

Durable document jobs retain their original upload and extraction request:

```python
jobs = client.document_jobs(expected_space="alpha")
formats = jobs.formats()
original = jobs.upload(b"# Notes\n\nAda studies stars.", media_type="text/markdown")
status = jobs.start("import-1", attachment_id=original.attachment_id, filename="notes.md")
request = jobs.request("import-1")
status = jobs.status("import-1")
# Once completed, read currently verified original/manifest/episode identities.
result = jobs.result("import-1")
```

Callers choose when to poll. Starting an existing import only returns its status;
it never implicitly resumes a stopped attempt. For an explicit recovery, read the
current request and pass that exact snapshot to `jobs.resume(request)`. Cancellation
uses `jobs.cancel(request)`. Both operations compare the saved request before writing,
send its control revision, and verify the acknowledgement afterward. A stale snapshot
raises `SconeError`; no automatic retry occurs. Document recovery can explicitly retry
an interrupted extraction/index operation whose outcome is unknown.

`jobs.list(limit=20, after=page.next_after)` reads retained history. `jobs.result()`
does not run extraction, even when source verification previously failed or the
attempt limit has been reached. Missing or currently unverifiable results raise an
error. The result's extraction filename is checked against the immutable job request;
the original blob may retain a different first-upload filename after deduplication.

For PDF OCR, pass `pdf_ocr=PdfOcr("missing_text", "columns_ltr")` to `start()` after
checking the host's `formats.pdf_ocr_available` and advertised choices. Users select
OCR behavior; the host owns the parser implementation, revision, and processing limits.
Upload and job admission are separate explicit writes. Uploaded bytes must fit the
host's advertised input limit and the client's 25 MiB ceiling.

Model tasks can carry an explicit output contract on hosts advertising
`agents.output_requirements`. A schema additionally requires `agents.output_schema`:

```python
from scone import ModelTask, TaskPlan, TaskAnswerRequirements

requirements = TaskAnswerRequirements(
    instructions="Return a concise JSON object with a summary.",
    format="json_object",
    max_bytes=4000,
    max_lines=20,
    output_schema={
        "type": "object",
        "$defs": {"summary": {"type": "string"}},
        "properties": {"summary": {"$ref": "#/$defs/summary"}},
        "required": ["summary"],
        "additionalProperties": False,
    },
)
plan = TaskPlan("summarize", (
    ModelTask("summary", "research", "careful", "Summarize the retained evidence.",
              answer_requirements=requirements),
))
saved = agents.save_plan(plan, expected_revision=0)
```

`TaskAnswerRequirements` defaults to empty instructions, `max_bytes=64000`,
`max_lines=None`, and `format="text"`. Instructions accept up to 8000 UTF-8 bytes;
byte limits are strict integers in `1..128000`, and an optional line limit is a
strict integer in `1..1000`. A schema requires `json_object` and must be a JSON
object within 32768 compact serialized UTF-8 bytes, depth 32, and 4096 value
nodes including the root. The SDK preserves authored `$defs`, `$ref`, and other
schema content in a detached, deeply immutable snapshot. `to_json()` returns a
fresh mutable copy. It performs bounded JSON checks, not full JSON Schema
validation or compilation; the configured native host owns execution validation.

The same contract can constrain the final answer of a handoff workflow:

```python
from scone import HandoffAgent, HandoffPlan

handoff = HandoffPlan(
    "research-handoff", "research",
    (HandoffAgent("research", "careful", ("writer",)),
     HandoffAgent("writer", "careful")),
    max_handoffs=1,
    answer_requirements=requirements,
)
saved = agents.save_plan(handoff, expected_revision=0)
```

Saving a handoff contract requires `agents.handoffs.output_requirements`, in
addition to ordinary handoff support; a schema also requires `agents.output_schema`.
The contract applies to the final answer. Intermediate handoff answers remain text.
For JSON output, the native model protocol carries an object in the final envelope;
the public result's `final.text` remains a string. The SDK does not run models or
rewrite answers, and a handoff-limit result still has no final answer.

An absent task or handoff contract is omitted from the wire for compatibility with older
hosts. Saving a contracted plan requires the relevant advertised capabilities
before any write, and its acknowledgement must preserve the authored contract.
Output contracts guide and constrain model output; they do not establish factual
accuracy. Result reads preserve the server's returned text and retain the same
final source-verification ordering.

Completed agent outputs are typed and checked against the saved execution request:

```python
from scone import TaskResult, HandoffResult, ModelOutput, HumanOutput

result = agents.result("run-1")
if isinstance(result, TaskResult):
    for output in result.results.values():
        if isinstance(output, ModelOutput):
            print(output.model_id, output.text, output.evidence_ids)
        elif isinstance(output, HumanOutput):
            print("Human reply", output.text, output.activation_id)
elif isinstance(result, HandoffResult):
    print(result.status, result.final.text if result.final else None)
```

Model receipts retain the selected model, binding, dependency IDs, call counts,
and immutable evidence packets. Packet identifiers and path references must agree
with the returned records; nested packet data is read-only. A human output must
match a freshly read activated reply and cannot carry model/evidence fields.
A handoff-limit result preserves actual hops and has no final answer.

`result()` performs only reads. For interactive runs it reads input receipts first,
then requests the source-verified result last. It never runs another model or
restarts a workflow. The server verifies current source access, retained content,
and native entity-classification rules; client-side packet checks establish wire
consistency and do not independently prove that a model's answer is true. A deleted
or unavailable source is an error, rather than a fallback to an old result.

To read recorded provider token counts, opt in on hosts advertising `agents.usage`:

```python
result = agents.result("run-1", include_usage=True)
if isinstance(result, TaskResult):
    outputs = tuple(value for value in result.results.values() if isinstance(value, ModelOutput))
else:
    outputs = tuple(hop.output for hop in result.hops)
for output in outputs:
    usage = output.usage
    if usage is not None:
        print(output.model_id, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        for call in usage.calls:
            print(call.prompt_tokens, call.completion_tokens, call.total_tokens)
```

The default request is unchanged and exposes `ModelOutput.usage` as `None`.
Opt-in reads send `include_usage=true` and require a usage field on every model
output. `null` means the legacy journal has no recorded usage. A non-null
`ToolTokenUsage` contains exactly `model_calls` immutable `ModelTokenUsage`
reports, at most 17. Each report has three optional strict integer counts in
`0..1_000_000_000`; contradictory totals or malformed reports are refused.
Unknown categories remain `None`. Each category total is available only when
all calls report it; a total is never inferred from the other categories.
Aggregate counts may exceed the per-call limit. Human outputs cannot carry usage.

Usage belongs to actual model outputs. For handoffs, iterate `hops` once: `final`
is a checked copy of the last hop, not an additional model call. Reused steps
retain their original reports; reading a result does not consume model tokens.
These are provider assertions for recorded completed outputs, not billing totals
for failed or uncertain attempts, token estimates, or measures of answer quality.
Opt-in reads preserve the same final source-verification read and never execute
or retry a workflow.

### Configured local directory scans

A native host with `SCONE_DIRECTORY_SYNC_CONFIG` advertises `documents.sync`.
The standalone client discovers only collections available to its key; it cannot
supply an arbitrary filesystem root. All methods perform one explicit operation.
Reads, reconnects and repeated starts with the same ID do not resume work.

```python
from scone import Scone

with Scone(api_key="local-space-key") as memory:
    sync = memory.directory_sync(expected_space="alpha")
    collection = next(item for item in sync.collections() if item.collection_id == "notes")
    run = sync.start("notes-scan-001", collection=collection, delete_missing=False)
    # Keep this run ID if admission is not confirmed. Check before retrying.
    current = sync.status(run.record.run_id)
    print(current.status, current.active_elsewhere, current.outcome_unknown)
```

Start includes the discovered configuration digest; the host refuses stale
configuration before execution. `sync.list(limit=20, after=cursor)` reads a bounded
history page, and `sync.request(run_id)` reads the immutable intent and current
revision. Resume with a freshly read `SyncStatus` using `sync.resume(status)`;
`sync.cancel(status)` requests cancellation with the same revision guard. An
acknowledgement validates the actual transition, not merely a successful HTTP code.
Refresh status until the worker is idle before resuming after cancellation.
Completed and partial scans require a new run ID for another scan.

`sync.results(run_id, limit=20, after=index)` returns typed source receipts and
scan issues with a numeric next cursor. Results bind the expected space/run and
validate counts and ordering. They describe the completed scan; they do not assert
that an episode is still retained. Python preserves full 64-bit episode identities
and marks escaped filesystem diagnostics separately from source paths. No method
automatically follows pages, retries a write, downloads a model, or resumes a run.

### Retained video evidence and frame interpretation

Read a sampled video's evidence, download a verified frame, or explicitly ask the
server's selected self-hosted vision model about that frame:

```python
from scone import Scone

with Scone('http://127.0.0.1:7437', 'your-space-key', timeout=150) as memory:
    videos = memory.video_documents(expected_space='research')
    source = videos.catalogue(episode_id=7)
    frame = source.frames[0]
    print(source.frame_time(frame.ordinal))  # exact fractions.Fraction seconds
    png = videos.frame(source, frame.ordinal)
    description = videos.interpret(source, frame.ordinal,
                                   prompt='Describe the visible scene and state uncertainty.')
    print(description.model, description.text)
```

Choose the vision connection through the server's Models settings. Each explicit
interpretation request uses the current saved selection. The result carries the
source catalogue and frame and has `persisted=False`; it does not write facts,
embeddings or searchable visual descriptions. There is no implicit retry or
inference on catalogue reads, frame downloads or host restart. Configure the
client timeout for your own model latency; the host also bounds interpretation.

Catalogues preserve original and manifest attachment identities, integer frame
ordinals and presentation timestamps, rational time bases, sampling coverage and
UTF-8 OCR regions. `frame_time()` keeps fractions exact, including large or
negative stream offsets. PNG downloads check the recorded byte length, SHA-256,
frame headers and image dimensions. The client checks the source catalogue before
and after a download or interpretation, refusing changed source evidence or space.
These checks observe current server state; they are not an atomic transaction.

The API requires the native video catalogue/frame routes; interpretation also
requires `documents.video.understand`. Older hosts refuse unsupported operations.
The client remains independently installable on Python 3.9+, with no decoder,
OCR, model or native-framework dependencies. Model output is unsaved and may be
incorrect; sampled frames do not describe everything between those frames.


### Exact tool approvals

Hosts advertising `agents.approvals` support review of the exact pending tool
call, including its selected model and literal canonical arguments. Reads and
decisions do not execute the tool. After reviewing the call:

```python
pending, = agents.approvals("run-1")
print(pending.call.model_id, pending.call.tool_name, pending.call.arguments())
decided = agents.decide_tool(pending, decision="approve")  # or "deny"
continuation = agents.continue_tools(
    "run-1", continuation_id="review-1", decisions=(decided,),
)
```

Decision requests require a review/full key; continuation requires a write/full
key. The authenticated host records the actor. SDK validation binds the selected
records and their decision hashes to the returned immutable activation receipt.
It never retries an ambiguous mutation automatically. An explicit retry with the
same activation ID and selection preserves the original receipt without replaying
a completed tool call. `RunStatus.paused_steps` identifies resumable tool pauses;
unknown outcomes remain non-replayable. Human responses still use the separate
`continue_run(..., responses=...)` method.

The console approval interface is not included. The native protocol is exercised
across process restarts by `tests/test_native_agent_approvals.py`, using the same
`SCONE_TEST_NATIVE_PYTHON` setting as the other native contract tests.


### Agent execution history

Hosts advertising `agents.history` provide typed cursor replay:

```python
from scone import ProgressEvent, ProgressGap, CollectionEvent

page = agents.history("run-1", limit=50)
if not page.available:
    print("No retained observations are available")
if page.omitted is not None:
    print("History positions removed by retention:", page.omitted)
for entry in page.items:
    event = entry.event
    if isinstance(event, ProgressEvent):
        print(entry.step_id, event.model_id, event.kind, event.elapsed_s)
    elif isinstance(event, ProgressGap):
        print("Unobserved sequences:", event.first_sequence, event.last_sequence)
    elif isinstance(event, CollectionEvent):
        print(event.kind, event.observed_events, event.lost_events, event.terminal_kind)

# Persist the cursor to reconnect later, including after a server restart.
if page.next_after is not None:
    later = agents.history("run-1", after=page.next_after)
```

`HistoryPage`, `HistoryEntry` and their event objects are immutable. The decoder
checks the expected space/run, saved task or reachable handoff selection, exact
model binding, event shape, timestamps, finite timing, collection identity and
cursor/page continuity. Unknown metadata fields and private diagnostic strings
are rejected. Built-in memory tool names remain valid observations. Legacy entries
without collection/activation IDs remain readable; no timing is fabricated.

Producer loss (`ProgressGap`) and storage retention (`page.omitted`) are distinct.
Collection completion means observation ended, including for a failed, paused or
cancelled turn; it does not prove workflow success or that every event was saved.
An empty tail page retains its cursor. Purged or changed histories require a fresh
observation read, and server errors propagate without retries or model selection
changes. Every call is read-only and checks fresh capability/request metadata
before the server's final current-source and recipient verification.

Cursor shape and position checks do not let the client authenticate the server's
HMAC independently. The host remains responsible for authorization and encrypted
journal integrity. History contains execution metadata, never prompts, tool
arguments, answer text or hidden reasoning. Use `agents.result()` for a currently
verified final answer. `history()` reads one bounded page and does not poll or
resume the agent automatically; `stream_history()` below follows the run live.

`tests/test_native_agent_history.py` verifies typed replay across two process
restarts with the selected model called once, both with and without an initial
native memory search. Set `SCONE_TEST_NATIVE_PYTHON` as described above to run it.

### Live history over SSE

Hosts advertising `agents.history` also publish the same verified pages as
`text/event-stream` frames. `stream_history()` opens that route and yields
`HistoryPage` objects as the server writes them:

```python
from scone import CollectionEvent

with agents.stream_history("run-1", limit=50) as stream:
    for page in stream:
        for entry in page.items:
            print(entry.position, entry.step_id, type(entry.event).__name__)
        if any(isinstance(e.event, CollectionEvent) and e.event.kind == "collection_finished"
               for e in page.items):
            break
cursor = stream.cursor  # last page's next_after, kept for a later resume

# After a disconnect, a client error or a server restart, continue from there.
with agents.stream_history("run-1", after=cursor) as stream:
    for page in stream:
        ...
```

Every page passes the same decoder as `history()` — space, run, model binding,
event shape, cursor/page continuity — and the frame's SSE `id` must equal the
page's `next_after`, so a resume cursor always names a page the client actually
decoded. The `after` cursor is sent both as the query parameter and as
`Last-Event-ID`, which the server requires to agree. Keep-alive comments are
ignored; the server's `end` frame stops iteration; an `error` frame raises
`SconeError("history stream refused: <reason>")`. The stream ending mid-frame is
an invalid response.

The connection is closed when the `with` block exits, including on an exception
or an early `break`. Reads are bounded by the client's `timeout`, so a stalled
server raises `SconeError` instead of hanging; each line is bounded at 1 MiB and
the whole stream at `max_response_bytes`. `stream_history()` never re-opens the
connection on its own: reconnecting is the caller's decision, made with the
cursor the object exposes. This is execution metadata delivered as it is written,
not answer-token streaming.

`tests/test_native_agent_history_stream.py` follows a real run to completion over
a loopback server, checks positions are contiguous and no private text is
delivered, then restarts the server and resumes from the kept cursor.

### The answer as it is written

Hosts advertising `agents.text_stream` publish the text a running step is
writing. `stream_answer()` opens that route and yields events as the host
publishes them:

```python
from scone import Ended, Gap, Terminal, TextDelta, Withdrawn

with agents.stream_answer("run-1", "answer") as stream:
    for event in stream:
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, Withdrawn):
            print("\n[what was written so far was not the answer]")
        elif isinstance(event, Gap):
            print(f"\n[fell behind: resuming at {event.next_sequence}]")
        elif isinstance(event, Terminal):
            print(f"\n[run {event.status}; read the receipt]")
        elif isinstance(event, Ended):
            print(f"\n[{event.reason}]")
cursor = stream.cursor  # the last sequence seen, for after= on a reconnect
```

`TextDelta` carries one delta and its sequence; `Withdrawn` says text
streamed before a tool turn was not the answer; `Gap` says the reader fell
behind the host's bounded window and names the next sequence it can read;
`Terminal` says the run has a receipt (`read_receipt` is always true);
`Ended` says the observation window closed. Sequences are contiguous from
the cursor -- a gap is the only sanctioned jump -- and a frame's SSE `id`
must name its sequence, because it is what a reconnect sends back. A frame
with fields beyond its own, a terminal without a receipt, an unknown kind,
a stream cut mid-frame, or a host that goes quiet past the client's
`timeout` raises `SconeError`; an `error` frame raises
`SconeError("answer stream refused: <reason>")`.

What arrives is provisional. The host keeps it in memory only while the
step runs; after the run, or after a restart, a reader is given `Terminal`
and no text, and `agents.result()` returns the verified answer. Prompts,
tool arguments and reasoning are never in this stream: a host whose model
turn cannot stream, or whose answer is structured, delivers the accepted
answer as one delta once it is known.

`tests/test_native_agent_answer_stream.py` follows a real run's answer over
a loopback server, model to receipt.

## Conversations

The conversation service keeps a session's lifecycle in a journal with a
revision every command must name, and each turn as a receipt under a
client-chosen request id that stays `pending` until it settles. The
client puts that on the wire without inventing anything: a command names
its `request_id` and `expected_revision`, so a retry is the same command
and a stale one is a `ConversationConflict` carrying the revision the
server holds.

```python
conversations = client.conversations(expected_space="alpha")
caps = conversations.capabilities()            # text_configured, streaming, turn_cancellation, ...
session = conversations.create(request_id="open-1")
receipt = conversations.submit(session.session_id, request_id="ask-1", text_="hello",
                               expected_revision=session.revision)   # status == "pending"
with conversations.stream_reply(session.session_id, "ask-1") as stream:
    for event in stream:               # ReplyDelta, ReplyGap, ReplyTerminal, ReplyEnded
        ...                            # stream.cursor is what a reconnect sends back
settled = conversations.wait(session.session_id, "ask-1", timeout=30)   # the receipt is the answer
page = conversations.transcript(session.session_id)                     # newest first, opaque before= cursor
history = conversations.events(session.session_id, after=0)             # the journal, by revision
conversations.stop(session.session_id, request_id="stop-1", expected_revision=settled_session.revision)
conversations.delete(session.session_id)                                # closed sessions only
```

What arrives on the stream is provisional: `text` frames carry a
sequence as their id, `gap` says the reader fell behind the host's
window, `terminal` says the turn has a receipt (`read_receipt`), and
`end` says the window closed. A frame whose id does not name its
sequence, a sequence that skips without a gap, a terminal for another
turn, or a stream cut inside a frame is refused. Nothing reconnects on
its own; `stream_reply(..., after=stream.cursor)` resumes, sending the
cursor as both `after=` and `Last-Event-ID`. `cancel()` returns the
receipt only when it says `cancelled`; `delete()` and `cancel()` check
the service's capabilities first and refuse plainly when it lacks them.
