# Conversation HTTP service and lifecycle journal

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

## Authenticated conversation API (optional, single-process)

`scone_memory.api.conversations.create_conversation_app` exposes the journal and
a configured text runtime under `/v1/conversations`, with the existing native
memory routes mounted at the same origin. Install the `api` extra and use Python 3.11+ for native real-time sessions.
The caller owns the open engine; the ASGI lifespan owns its journal and runtime
tasks. This factory does not launch a server or select a provider for you.

## Launching the service

The CLI can launch that service without a custom ASGI entry point:

```sh
# Uses the existing SCONE_API_KEY / SCONE_API_KEYS and native store settings.
# Create the journal's parent directory first; never use the native memory DB.
scone-memory serve-conversations --journal ./conversation-sessions.db --history-only

# Explicit model opt-in, from a trusted Python module on your import path:
scone-memory serve-conversations --journal ./conversation-sessions.db --model-factory my_models:create
```

The same service can ride the memory server instead of a second port. Naming
a journal composes it onto `serve`'s origin, so `/v1/conversations/*`, the
memory routes, the packaged pages and the consolidation worker share one
process; leave it unset and `serve` is the memory-only server it always was:

```sh
# Composed: same keys, same store, one origin. History-only without a factory.
SCONE_CONVERSATIONS_JOURNAL=./conversation-sessions.db scone-memory serve
SCONE_CONVERSATIONS_JOURNAL=./conversation-sessions.db SCONE_CONVERSATIONS_MODEL_FACTORY=my_models:create scone-memory serve
```

A composed host can also serve a saved persona catalog. `SCONE_CONVERSATIONS_PERSONAS`
names a JSON array of Persona documents (`scone_memory.realtime.persona.Persona`,
schema 1: id, name, instructions, and exact reply/transcription/speech/activity
choices); `SCONE_CONVERSATIONS_REGISTRY` names a trusted zero-argument callable
returning the `ProviderRegistry` that admits those choices. Every persona is bound
at startup, so a choice the registry does not register stops `serve` with exit 2
naming the persona and the stage. Clients read `GET /v1/conversations/personas`
(ids, names, provider/model labels, `text_ready`, `voice_ready`; never the
instructions), pick one with `persona` on `POST /v1/conversations`, and see
`{"id", "name", "fingerprint", "current"}` in every session receipt. The listing
carries a `revision` and each persona a `fingerprint` of its configuration; send
`persona_fingerprint` with the create to be refused (409) if the operator changed
that persona since the client displayed it. With a catalog and no bare model
factory, a session must name a persona; nothing is chosen for it. `voice_ready`
is false for every persona until a browser audio transport exists.

Voice rides the same service. `POST /v1/conversations` with `"mode": "voice"` and
a `persona` creates a session that waits (`created`) for its audio socket,
`WebSocket /v1/conversations/{sid}/audio`. The first text frame is
`{"type": "hello", "key": "<bearer>", "sample_rate": 16000, "channels": 1}` (a
browser cannot set a bearer header on a WebSocket; the key is never in a URL);
a bad key, session, mode, state or format gets `{"type": "error", "reason"}` and
close 1008, and an accepted hello gets `{"type": "ready"}` while the session runs.
Then binary frames are PCM s16le in that format both ways; outgoing frames carry a
header (turn id length, turn id, uint32 sample rate, uint8 channels) so a
`{"type": "clear", "turn_id"}` control can drop a turn's buffered playback on
interruption; `{"type": "end"}` or closing the socket ends the input and the
session (`ended`), a provider or format failure fails it, and `POST /stop` works
while it runs. Both sides are captured to memory and read back through the
transcript route. `GET /v1/conversations/capabilities` reports `"voice": true`
and the catalog reports `voice_ready` on such a host.

On a composed host the pages carry no baked key even with a single configured
key (the tab asks for one), `GET /v1/capabilities` reports
`features.conversations: true`, and `GET /v1/conversations/capabilities` says
whether text is configured. A journal that is the memory database, or a factory
that does not load, stops `serve` with exit status 2 before it listens.

`my_models:create` must be a synchronous, zero-argument callable returning a
**fresh native `TextModel` adapter per turn**. Importing the module executes trusted
Python code; use only operator-controlled modules. The launcher checks the
callable without invoking it; the text runtime invokes it when a turn runs.
Provider credentials, dependencies and client lifecycle belong to that factory.
No default provider is selected, and `SCONE_CHAT_*` consolidation settings do not
configure the conversation model. Configured remote stores or embedders can
still perform their normal network access at startup or during memory requests.

`--history-only` does not import a model adapter or accept new text sessions. It permits
inspection of saved conversations and retains the authenticated native memory
routes; it is **not a read-only memory server**. Browser pages are hosted by the independent ProjectScone-Webapp repository. The CLI uses SQLite persistence by default,
and the same host/port environment settings as `serve`. Stop the other server or
choose a different `SCONE_PORT` before launching. This command does not start a
consolidation worker. It builds and serves native memory on one event loop and
closes owned backends after shutdown. Journal locking requires Linux/macOS and a
local filesystem. The configured Scone launcher enables public-text SSE;
history-only mode does not. Voice, video and live-provider certification remain separate work; the existing `serve` command is unchanged.

```python
from scone_memory.api.conversations import create_conversation_app
from scone_memory.realtime.text import TextConversation

# memory, space_keys and model_factory are explicit server configuration.
# Never use your native memory database as the conversation journal.
app = create_conversation_app(
    memory, space_keys, "conversation-sessions.db",
    lambda space, sid: TextConversation(memory, space, sid, model_factory),
    public_text_streaming=True,  # known-compatible native runtime; default is False
)
```

## Public-text stream (optional)

`public_text_streaming=True` opts every configured runtime into
`reply(text, on_text=async_callback)`. Custom factories default to off and must
explicitly implement that public-text contract. The Scone CLI launcher enables
it for its known runtime. Capabilities advertise `streaming: true` and
`text_stream: {transport: "sse", replay: "active_window", max_bytes: 65536,
max_chunks: 256}`. `reply_transport: "poll"` still describes final receipts.

After the ordinary idempotent turn POST, open
`GET /v1/conversations/{sid}/turns/{request_id}/stream` with the **Authorization
bearer header**. Use authenticated fetch or an HTTP client; do not put a key in
the URL. The response is `text/event-stream`, `Cache-Control: no-store`, with
proxy buffering disabled. JSON `data` frames preserve literal text safely:

| Event | Data | Meaning |
| --- | --- | --- |
| `text` | `{sequence, text, provisional: true}` | Public chunk, with matching SSE `id`; not a token or saved-reply receipt. |
| `gap` | `{after, next_sequence}` | Earlier chunks left the bounded window. Do not synthesize the missing text. |
| `terminal` | `{request_id, status, read_receipt: true}` | Fetch the existing turn receipt for final status and available saved text. |
| `end` | `{request_id, reason, read_receipt: true}` | Window unavailable, service shutting down, or session deleted; not proof of completion. |

Reconnect with `after=<last-sequence>` or `Last-Event-ID`; both must agree if
provided. The cursor is a nonnegative signed64-bit integer scoped to this turn.
Invalid, unknown or duplicate query fields return 422; a cursor ahead of an active
window returns 409. Missing or cross-space records return 404, missing auth 401,
and a disabled stream 501. Reads never start or retry the model. A disconnected
reader does not cancel the turn; use the existing cancel/stop commands.

Only the most recent 256 chunks and 65,536 UTF-8 bytes of an **active** turn are held.
Oversized single chunks fail the turn instead of being silently truncated.
Invalid observation is latched even if a custom runtime catches the callback
error. Custom runtime side effects are not rolled back; configure capture so
it does not persist unfinished replies as complete.
Readers have no private chunk queue. A 10-second comment heartbeat keeps idle
connections observable. Terminal/cancel/stop clears provisional text; reconnects
after completion or restart yield receipt information, not reconstructed chunks.
Final text is read from the retained episode, so forgetting it cannot expose a
stale streaming copy. Text already sent to clients cannot be revoked.

This window is not a durable event journal or delivery acknowledgement. It does
not bound upstream provider queues; connection counts, rates and slow
client limits belong in the serving ASGI/proxy configuration. The React consumer
is separate work; enabling the API does not change the deployed browser bundle.

The owned CLI server calls `app.state.begin_conversation_shutdown()` **before**
draining HTTP tasks, so idle streams receive `end: service_shutdown` and close.
It also sets a 5-second graceful HTTP-drain limit for stalled sends. Custom hosts
must call that hook on the service event loop before waiting for open responses,
and configure their own bounded drain policy. Lifespan cleanup alone runs too
late on servers that wait for SSE connections first. The hook closes stream
windows and new-command admission; lifespan still owns runtime cleanup. This is
cooperative cleanup, not a guarantee that an external provider has stopped.

## Per-session recall scope

To let an API caller narrow recall for each session, opt in explicitly:

```python
app = create_conversation_app(
    memory, space_keys, "conversation-sessions.db", None,
    scoped_runtime_factory=lambda space, sid, scope: TextConversation(
        memory, space, sid, model_factory, **scope.kwargs()
    ),
)
```

Then `POST /v1/conversations` may include, for example,
`"recall_scope": {"kind": "file", "where": {"collection": "manuals"}, "source_prefix": "docs/"}`.
The vocabulary is `where`, `kind`, `source_prefix`, `since`, `until`; it never
selects another authorized space. Constraints are validated and normalized before
creation, persisted for the session's life, and returned by inspect and list.
Changing the scope with the same create request ID conflicts, including after a
restart. Dates are inclusive; reversed ranges are rejected. An empty
`source_prefix` still requires an episode to have a source; omit the field to
include sourceless episodes. Empty results never
cause the native Scone runtime to broaden its recall. Scope limits retrieved
knowledge, not capture or the session's own conversation history.

The scoped factory receives an immutable `scone_memory.recall_scope.RecallScope`.
`scope.kwargs()` returns fresh native recall arguments. This factory takes
precedence over the legacy factory, including for an empty scope. It is trusted
server configuration: custom runtimes must actually enforce the constraints,
and any additional server policy must remain enforced rather than be replaced
by client filters. Two-argument factories remain supported but reject nonempty
scope with HTTP 422. The additive `recall_scope: true` capability is advertised
only for an explicitly configured scoped factory. A freshly packaged webapp
uses that capability to offer memory selection at session creation and display
the persisted filters afterward. Older deployed bundles need a separate rebuild
and deployment; backend capability alone does not update the browser UI.

## Webapp and authenticated controls

The independent [Webapp repository](https://github.com/ProjectScone/ProjectScone-Webapp)
hosts `/memory`, `/playground`, and conversation pages, and proxies the API.
This service provides JSON, SSE, and WebSocket routes only. Configure provider
credentials on the server and use HTTPS beyond loopback.

Bearer keys determine the space. Provider credentials never belong in request
bodies or query strings. Send `POST /v1/conversations` with a stable `request_id`
and `capture: true`; use the returned `session_id` and lifecycle `revision` for
controls. `POST /v1/conversations/{sid}/turns` accepts `request_id`, `text` and
`expected_revision`, returning a 202 receipt. Poll
`GET /v1/conversations/{sid}/turns/{request_id}` for the result. Repeating an
identical request returns its receipt without invoking the model again; changing
its input conflicts. Revisions track session lifecycle, not turn numbering.

When capabilities explicitly report `turn_cancellation: true`, the workspace
offers **Cancel reply** for a pending turn. It sends
`POST /v1/conversations/{sid}/turns/{request_id}/cancel` and checks the existing
receipt without resubmitting the message. A cancelled receipt stays cancelled,
even if a custom runtime returns late. Cancellation is local, not proof that an
external provider stopped processing; the submitted message remains captured.

The Scone text runtime can accept another turn after model cancellation only
when owned processor cleanup succeeds. Interrupted capture or failed cleanup
interrupts the session instead. Custom runtimes must explicitly expose
`closed is False` after cancellation to allow continuation; absent or uncertain
state is treated conservatively. The UI preserves the next draft and waits for
verified session readiness before enabling Send.

`POST /v1/conversations/{sid}/stop` takes `request_id` and `expected_revision`.
Cleanup continues if the requesting browser disconnects. A matching stop retry
returns current state (possibly `stopping`), not a fabricated original event.
`GET /v1/conversations/{sid}/events` exposes durable lifecycle event receipts;
`GET /v1/conversations` lists bounded pages using `after`/`limit`.
`GET /v1/conversations/{sid}/transcript` reads the latest native message page
attributed to that session, in chronological order. `limit` defaults to 200 and
accepts 1–200. When `has_more` is true, pass the opaque `next_before` value as
`before` to read an older page. The cursor binds the authorized space and session
to a timestamp/episode-ID boundary, so deleting that boundary message does not
break navigation. It checks session ownership before reading memory and does not
reconstruct reply receipts. Pages reflect current retained data, not a frozen
snapshot; backdated imports can change older pages. Response size is bounded,
but the current engine still scans the space's history to find matching episodes.

With explicit `transcript_pagination: true`, the React workspace shows 50 messages
per page and provides Older, Newer and Latest controls. Polling re-reads the selected
page without changing your position or resubmitting work. A server without this
capability remains readable but does not expose unsupported navigation.

Turn receipts survive service recreation in the journal. A receipt's `status`
is separate from `result_state`: `available` includes the authorized saved reply;
`forgotten` means its episode is gone; `unavailable` means no reply episode can
be resolved; `unreadable` means a temporary storage read failure. Completed turns
remain completed when their text is absent. Pending receipts may have a null
`result_state`. Retry receipt reads, never automatically resubmit model work.
Both live and recovered receipts resolve text from the native episode so forgetting
it also removes the text from subsequent receipt reads.

`GET /v1/conversations/{sid}` includes `latest_request_id`, calculated from
journal acceptance order, and `active_request_id`, which names only current
in-process work. Either can be null. Use the latest ID to discover an outcome
after reopening; do not infer chronology from UUIDs or the paginated turn list.
`GET /v1/conversations/{sid}/turns?after=&limit=` lists scoped receipts in request-ID
order, with limits from 1 to 200. Latest means most recently accepted, not delivered
or successfully completed. Receipt availability still depends on memory retention.

When conversation capabilities explicitly report `session_deletion: true`, the
React workspace offers **Delete conversation** for verified closed sessions. It
requires confirmation and sends `DELETE /v1/conversations/{sid}`. A successful
204 removes the session, lifecycle events, turn receipts and message episodes
not held by another conversation's receipts. Imported context is not deleted.
Running sessions return 409: end the conversation first. This is not an erasure
request to external providers or backups, and it does not reclaim process-local
creation capacity. If the response is uncertain, the UI offers a read-only status
check, never an automatic delete retry. A remaining session may be partially
deleted after a storage failure; inspect it before confirming another attempt.

The Scone text runtime gives each session, turn and speaker a distinct capture
identity, so repeated identical messages remain separate transcript occurrences.
Custom runtime factories must also preserve session attribution; content-only
deduplication can make their metadata-based transcripts incomplete.

Restart never resubmits a model request: abandoned sessions become interrupted,
while captured episodes follow the configured engine's persistence. Transcript
history is not automatically restored into a new model session. Defaults admit 100 create IDs per process
(including failed starts and terminal sessions), and 100 turns per session.
Matching retries remain available at capacity; new work returns 429.

This first service requires one worker on Linux/macOS and an application-owned
local directory. A persistent canonical-path `.owner.lock` sidecar prevents
multiple service owners; hard-linked journals are refused. Do not delete or
replace either file while running. This is not distributed leasing, protection
against a hostile filesystem owner, or support for uncooperative runtimes.

Capabilities explicitly report configured text, polling, no voice/video, and no
token stream. `runtime_factory=None` reports text unavailable and refuses starts.
Tests connect HTTP controls to native Scone scheduling and native memory using a
scripted model; they do not certify a live provider. The React page supports
start, send, saved transcripts, source inspection, recovered outcomes and stop.
Provider certification, process-crash recovery evaluation, Rust HTTP memory adapters and
voice/video remain required product work. Existing live servers are unchanged.

## Conversation lifecycle journal (service foundation)

`SessionJournal` persists explicit session state and ordered lifecycle receipts
in a **separate SQLite database**. It does not start a conversation, call a model,
or expose browser routes. Select a new path, never your native memory database:

```python
from scone_memory.session_journal import SessionJournal

with SessionJournal("conversation-sessions.db") as journal:
    created = journal.create("default", "create-1", mode="text")
    session_id = created["session_id"]
    # A runtime would issue this only when it actually starts the session.
    started = journal.transition("default", session_id, "start-1", "start", 1)
    page = journal.events("default", session_id, after=0, limit=100)
```

The caller must authorize the space; this is storage, not authentication.
Request IDs are opaque identifiers, not messages or credentials. Matching retries
return their original receipt, even after the session advances. Reusing an ID
for a different command or acting on an old revision raises `Conflict`; its
`revision` is the current **session** revision, not a memory-space revision.
State and event insertion commit together. Replay returns `events`, `next_after`
and `has_more`, with page limits of 1–200.

`create(..., recall_scope={...})` records immutable session recall constraints.
`get()` and `sessions()` expose the canonical `recall_scope` mapping. Journal
schema 3 upgrades schema 1 and 2 transactionally, retaining existing sessions,
events and turn receipts; older sessions receive an empty scope and retain their
original create replay signature. Back up the journal before upgrading: older
versions of Scone cannot open a newer schema. Runtime enforcement is separate
from persistence and is supplied by the scoped factory described above.

States are created, running, stopping, ended, failed and interrupted. Terminal
sessions cannot restart. Reopening preserves recorded state; `running` does not
prove a provider task is still alive. Runtime ownership and crash reconciliation
are not implemented by this journal. The text/voice mode field is metadata, not
a claim that either runtime is configured. No transcripts or media are stored.

Unrelated and unsupported-version databases are refused. Use one owned connection
per serialized caller, not one connection shared across threads. Separate
connections serialize writes through SQLite. Protect the selected path with
appropriate filesystem permissions. Authenticated session APIs, model execution,
the conversation UI, transport/reconnect handling and voice/video remain separate
work; the existing memory server is not a conversation server.
