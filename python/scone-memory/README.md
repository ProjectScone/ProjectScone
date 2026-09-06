# scone-memory

The memory layer for LLM applications, RAG pipelines and agents, in Python.
Every model call, retrieval step and agent in a system writes what happened
and what is true into one place, and reads it back scoped, dated and
explained.

Two kinds of memory live side by side:

- **Episodes**: what happened. Notes, chat turns, documents, tool output,
  kept verbatim and split into chunks that point back into the original
  bytes. Nothing stored is ever rewritten.
- **Facts**: what is true. Subject, predicate, object, and the interval
  over which it held. A new fact closes the one it supersedes and keeps the
  reason; asking about a past date returns what was true then.

The engine is written against three small protocols (documents, vectors,
embedder), so the same code runs in-process with no server, or against
MongoDB and Qdrant behind FastAPI. The in-process stores are the reference
implementation; the database adapters pass the same contract tests.
Vector indexes: in-memory, SQLite, Qdrant, Chroma (embedded or server),
LanceDB (embedded), Milvus (Milvus Lite embedded or server), PostgreSQL
with pgvector, Redis with RediSearch, Elasticsearch. Document stores: in-memory, SQLite, MongoDB, PostgreSQL,
Elasticsearch. Evidence: in-memory, SQLite, MongoDB, PostgreSQL,
Elasticsearch. With `SCONE_DOCUMENTS=postgres` or `=elasticsearch` the
vectors and the evidence log default to the same database through one
connection.

## Install

```sh
pip install scone-memory                 # core, in-process stores
pip install 'scone-memory[mongo,qdrant]' # database adapters
pip install 'scone-memory[postgres]'     # PostgreSQL + pgvector for documents, vectors and evidence (SCONE_POSTGRES_URL)
pip install 'scone-memory[chroma]'       # Chroma vectors (SCONE_VECTORS=chroma; SCONE_CHROMA_PATH or SCONE_CHROMA_URL)
pip install 'scone-memory[lancedb]'      # LanceDB vectors (SCONE_VECTORS=lancedb, SCONE_LANCEDB_PATH)
pip install 'scone-memory[redis]'        # Redis Stack vectors (SCONE_VECTORS=redis, SCONE_REDIS_URL)
pip install 'scone-memory[milvus-lite]'  # Milvus vectors, embedded (SCONE_VECTORS=milvus, SCONE_MILVUS_URI=./milvus.db); [milvus] for a server
pip install 'scone-memory[elasticsearch]' # Elasticsearch 8 for documents, vectors and evidence (SCONE_ELASTICSEARCH_URL)
pip install 'scone-memory[api]'          # FastAPI server
pip install 'scone-memory[local-embed]'  # bge-small ONNX, same model as the Rust core
```

## In-process

```python
import asyncio
from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder

async def main():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember("default", "Moved to Lisbon in March; the flat is on Rua Augusta", created_at="2024-03-02")
    await engine.assert_fact("default", "mark", "lives_in", "Lisbon", valid_from="2024-03-02")
    result = await engine.recall("default", "which street is my flat on")
    for item in result.items:
        print(item.score, item.created_at[:10], item.text)
    print([f.object for f in result.facts])

asyncio.run(main())
```

`HashEmbedder` is deterministic and dependency-free; it tracks word overlap,
not meaning. Use `LocalEmbedder()` (fastembed, bge-small) or
`RemoteEmbedder(base_url, model)` (any OpenAI-compatible `/embeddings`,
including Ollama) for real semantics.

## Server

```sh
export SCONE_API_KEY=change-me
export SCONE_DOCUMENTS=mongo  SCONE_MONGO_URL=mongodb://localhost:27017
export SCONE_VECTORS=qdrant   SCONE_QDRANT_URL=http://localhost:6333
export SCONE_EMBEDDER=local
scone-memory
```

The HTTP surface is the same as the Rust `scone serve`, so the
[`scone-client`](../scone-client) package (moving from `clients/python`) and
the MCP setup work against either. The bearer key decides the space; a key can never read
outside the space it was issued for.

| Route | What |
|---|---|
| `POST /v1/episodes` | remember; `{content, tags?, source?, created_at?, kind?}`; unknown fields are refused |
| `DELETE /v1/episodes/{id}` | forget |
| `GET /v1/recall?q&limit&as_of&tags&where&history&kind&source_prefix&since&until` | hybrid recall plus the facts that held at `as_of`; `history=true` adds the closed facts that came before them; `kind`, `source_prefix` (literal text), `since` and `until` (inclusive) narrow the candidates the way the Rust engine does |
| `GET /v1/facts?all&as_of` · `POST /v1/facts` · `POST /v1/facts/{id}/close` | the fact ledger |
| `GET /v1/profile` · `GET /v1/tags` · `GET /v1/status` · `GET /healthz` | overviews |

Errors are `{"error": "..."}` with 401, 404 or 422.

## What recall returns

Items carry `score` (rank within this query; the top item is always 1.0)
and `similarity` (cosine from the vector lane, when that lane saw the
chunk). Two lanes, vector and lexical, are fused by reciprocal rank with a
small recency term, capped at two chunks per episode. A lane that fails is
named in `degraded` and the other lane still answers. `context_reduction`
is the share of the space's bytes that were left behind.

`top_similarity` is the best cosine the vector lane saw for the query.
With `SCONE_SIMILARITY_FLOOR` set (a cosine, e.g. `0.45`), a recall whose
best hit falls below it, or that finds nothing, carries
`low_confidence: true` so the reader can decline to answer from weak
evidence; without a floor the field is `null` and nothing is judged. The
floor has no default because the right value depends on the embedder:
`scone-memory bench` prints, for each candidate floor, how many
no-evidence questions it would catch and how many answerable ones it
would wrongly withhold, and that sweep is where a floor comes from.

`history=true` (CLI `--history`) adds, for every matched fact, the closed
facts that held before it for the same subject and predicate, oldest
first, each with its interval and closing reason, bounded by `as_of`.

## What survives a crash

A `remember` marks the episode's identity in the document store before
it writes, and clears the mark only after the rows and the vectors are
all durable. On the next open the engine finishes or forgets whatever
was cut off in between: chunks are rebuilt from the stored content when
they are missing, vectors are re-embedded when they are missing, and a
mark with no episode behind it is dropped. Each open that had anything
to repair records one `recover` event with the counts. The guarantee:
an episode you can see is complete, or it is absent; it is never
searchable by one lane and not the other. This holds on every document
store below (the recovery contract in `tests/test_contract.py` plays the
crash at each step of the write).

## Stores and what each one promises

Every store below runs the same 37 contract tests (`tests/test_contract.py`)
and every evidence sink the same 8 (`tests/test_events.py`). "Verified"
says how: embedded means in-process in the test suite and CI; container
means against a real server in Docker locally and as a CI service.

| Store | Documents | Vectors | Evidence | Verified | Notes |
|---|---|---|---|---|---|
| in-memory | yes | yes | yes | embedded | reference implementation |
| SQLite | yes | yes | yes | embedded | FTS5 lexical lane; WAL; schema stamped, additive steps from v5 (quote column, inflight table) |
| MongoDB | yes | | yes | container (local; CI when `SCONE_TEST_MONGO_URL` is set) | `$text` lexical lane; TTL retention |
| PostgreSQL + pgvector | yes | yes | yes | container | tsvector lexical lane, HNSW cosine, one pool for all three |
| Elasticsearch 8 | yes | yes | yes | container | BM25 lexical lane, float32 HNSW (int8 would round cosine), refresh per write |
| Qdrant | | yes | | embedded (local mode) and container | payload filters server-side |
| Redis Stack | | yes | | container | TAG/NUMERIC prefilters inside the KNN query |
| Chroma | | yes | | embedded; server by URL | width recorded in collection metadata |
| LanceDB | | yes | | embedded | SQL predicates, quotes doubled |
| Milvus | | yes | | embedded (Milvus Lite); server by URI | filter expressions with JSON literals |
| any LangChain VectorStore | | yes | | embedded (`InMemoryVectorStore`, FAISS) | needs a `filter_builder` and a stated `score`; see below |

Intentional differences: lexical scores are each store's own (BM25,
`$text`, `ts_rank`); only their order reaches the fusion, so ranking
agrees across stores while the raw numbers do not. Retention is a TTL
index on MongoDB and a clock-driven sweep elsewhere. No store migrates
another build's data: each stamps a schema version and refuses a
mismatch, SQLite excepted for the one recorded step.

## Preserve an original image from the CLI

```sh
scone-memory remember source-note.txt --image screenshot.png \
  --space my-project --source manual-import \
  --meta session_id=my-session --json
```

The note is searchable text; the image is an original attachment, not OCR or a
model-generated description. The CLI reads only the explicitly selected file,
never paths mentioned inside the note or an agent transcript. One image can be
linked per invocation; `--jsonl` and `--image` cannot be combined. Files must be
nonempty regular files up to 25 MB. PNG, JPEG, GIF and WebP are identified by
their byte signatures rather than filename extensions. Signature recognition
does not validate decoding or bound decoded image dimensions.

The command stores exact bytes through the configured blob store, links them to
the episode and reads back the link before reporting success. With `--json`, the
receipt additionally contains the selected attachment's digest, type, byte count
and basename; text-only receipt shape is unchanged. Identical note text may reuse
an episode. Repeated identical images reuse their content-addressed attachment.
Source and session metadata follow the existing episode deduplication rules;
reusing a note does not create a new session record or replace its metadata.

This CLI opens the configured native stores, **not** the browser's HTTP server.
To inspect the result in the webapp, both must use the same authorized memory
space and document/blob configuration. SQLite defaults keep originals in an
attachment directory beside the database; `SCONE_BLOB_DIR` selects another
location. In-memory blob configuration does not persist originals across runs.

Image storage and episode linking are separate operations. An error after a
write can leave bytes or an episode behind; the command reports an unconfirmed
save and does not retry. Inspect the store before repeating it. There is no
automatic host image capture, missing-image reconstruction or Rust CLI parity
claim. Export/import still carries references, not a portable copy of image bytes.

## Any LangChain VectorStore as the vector index

```python
from langchain_community.vectorstores import FAISS   # or Pinecone, Weaviate, PGVector, Azure Search, ...
from scone_memory.backends import LangChainVectorIndex

from scone_memory.backends.langchain import LangChainVectorIndex as Bridge

def faiss_filter(space, as_of_ts, tags, where):          # FAISS hands a callable the metadata dict
    return lambda meta: Bridge._matches(meta, space, as_of_ts, tags, where)

index = LangChainVectorIndex(score="cosine_similarity", filter_builder=faiss_filter)
index.bind(FAISS(embedding_function=index.embeddings, index=faiss.IndexFlatIP(dim), docstore=InMemoryDocstore(),
                 index_to_docstore_id={}, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, normalize_L2=True))
engine = MemoryEngine(documents, index, embedder)
```

The bridge makes the three things that differ between stores explicit
rather than guessing: vectors reach the store through `index.embeddings`
(or `add_embeddings` where the store has it); scope filters need a
`filter_builder(space, as_of_ts, tags, where)` that returns the store's
own filter, and without one the bridge over-fetches and filters on its
metadata, refusing (the lane reads as degraded, the lexical lane still
answers) whenever the window filled with out-of-scope candidates before
`limit` matches were found; and `score` names what the store's number
means (`cosine_similarity`, `cosine_distance`, `unit_l2_squared`, or the
default `unknown`, under which the order still drives fusion but no
similarity is shown and no confidence verdict is derived). The contract
runs through the bridge over `langchain_core`'s `InMemoryVectorStore`,
filtered and post-filtered, and over FAISS (inner product on unit
vectors, a callable filter over the metadata dict), which is the recipe
above.

## Using it from a framework

Framework adapters live under `scone_memory.integrations`; each needs its
framework installed (`pip install 'scone-memory[langchain]'`,
`[llamaindex]`, `[openai-agents]`) and says so if it is missing.

```python
from scone_memory import SyncMemoryEngine
from scone_memory.integrations.langchain import SconeRetriever, SconeChatMessageHistory

memory = SyncMemoryEngine.from_env()
retriever = SconeRetriever(memory=memory, space="default", limit=5, where={"user_id": "mark"}, include_facts=True)
docs = retriever.invoke("where is the deploy runbook")   # Documents with episode_id, score, similarity, lanes in metadata
history = SconeChatMessageHistory(memory, "default", session_id="chat-1", extra={"user_id": "mark"})
history.add_messages([...])                              # one episode per message, in order, recallable like any memory
```

`scone_memory.integrations.llamaindex.SconeRetriever(memory, space, ...)`
returns `NodeWithScore` nodes (`retrieve` needs a `SyncMemoryEngine`,
`aretrieve` takes either); `scone_memory.integrations.openai_agents.SconeSession(engine, space, session_id)`
is a `Session` for the Agents SDK runner (`get_items`, `add_items`,
`pop_item`, `clear_session`).

Conversation turns are stored one episode each with `session_id`, `role`
and `seq` metadata. A plain text message is stored as its text so recall
reads well over the transcript; anything else (tool calls, structured
content, extra fields) is stored verbatim as JSON. Either way what was
added is what comes back. Turns are deduplicated by position, not text
(`Record.dedup_key`), so the second "ok" in a conversation is a second
turn; a dump carries each episode's identity, so a re-imported transcript
keeps its repeats.

### Pipecat public transcripts

Install `pip install 'scone-memory[pipecat]'` in a **Python 3.11+** environment.
This optional extra pins the tested Pipecat 1.8.1 release; base Scone still
supports Python 3.10 and does not import or install Pipecat.

Provision NLTK's `punkt_tab` resource during environment/image setup and set
`NLTK_DATA` to its directory. Otherwise Pipecat's tokenizer warm-up may attempt a
download at runtime, even in a text-only pipeline:

```sh
python -m nltk.downloader -d ./nltk_data punkt_tab
export NLTK_DATA="$PWD/nltk_data"
```

For an existing Pipecat `LLMContextAggregatorPair`, attach before running:

```python
from scone_memory.integrations.pipecat import SconePipecatMemory

capture = SconePipecatMemory(engine, "default", session_id="conversation-1")
capture.attach(context_aggregators)
try:
    await runner.run()  # your configured Pipecat pipeline, including its cleanup
    capture.raise_if_failed()  # finishing a conversation does not prove capture succeeded
finally:
    capture.detach()

result = await engine.recall("default", "telescope", where={"session_id": "conversation-1"})
```

Only user `on_user_turn_message_added` and assistant `on_assistant_turn_stopped`
events become conversation episodes. User records have `capture_status=context_message`;
assistant records have `aggregated` or `interrupted`. A user context message may
be a segment rather than a complete turn. An assistant aggregation can be flushed
at shutdown: it does **not** prove normal response completion or audio playback.
Pipecat's supplied text is preserved, not the original audio or raw token sequence.
Thought events, interim transcription frames, tools and whole context snapshots
are not subscribed to. Empty events are counted without fabricating memory text.

The episode retains session, role, capture identity, observation sequence and
the supplied timestamp/user ID when available. Source timestamps describe what
Pipecat reported, not an independently verified clock. Each adapter instance has
a new capture identity; repeated text and reconnects stay distinct. This does not
deduplicate replay after a crash or provide a durable delivery queue.

Capture runs in Pipecat's event tasks, serializing writes within an adapter.
`write_timeout` defaults to 5 seconds (including waiting for another write) and
`max_pending` to 128 admitted callbacks. A timeout, cancellation, overflow or
storage failure latches `capture.error`; subsequent nonempty events increment
`unrecorded_count` instead of silently resuming past the gap. Poll that error
during a live session and call `raise_if_failed()` after Pipecat cleanup. An
already-running write may still take effect; inspect stored records before
retrying. The limits do not bound Pipecat's own event dispatch queue, and timeouts
require cancellation-cooperative dependencies. `stored_count` counts acknowledged
writes, not proof of crash-safe storage on every backend.

This is a Python-native transcript integration, not a configured voice service:
browser devices, transport/provider setup, audio/video recording,
Rust HTTP interoperability and graph session-event wiring
remain separate work. Never use transcripts as approved facts without review.
See [the isolated pipeline example](examples/pipecat_memory.py) for a runnable,
credential-free capture-and-recall demonstration using clearly labeled fixtures.

### Pipecat request memory

The same optional extra includes `SconeMemoryContextProcessor`. Place it after
the user aggregator and before a compatible text LLM service:

```python
from pipecat.pipeline.pipeline import Pipeline
from scone_memory.integrations.pipecat_context import SconeMemoryContextProcessor

memory_context = SconeMemoryContextProcessor(
    engine, "default", session_id="conversation-1",
    where={"collection": "manuals"},  # optional metadata filter inside the space
    limit=5, max_context_bytes=8000, recall_timeout=2.0,
)
# `pair` is your LLMContextAggregatorPair; `llm` is your configured text service.
pipeline = Pipeline([pair.user(), memory_context, llm, pair.assistant()])
```

This is a native Python adapter, not an HTTP client. The caller must authorize
the fixed space and filters. `session_id` labels receipts; it neither grants
access nor restricts recall to that session. Without `where`, recall searches
the fixed space. Use a knowledge collection filter when current conversation
capture should not feed the same prompt back into retrieval.

For a latest plain-text user message, the processor recalls source passages and
copies the request context. It inserts a user-level, explicitly untrusted JSON
source block immediately before that request, preserving source text, episode /
chunk IDs, source identity, creation time and capture status. It does **not**
modify the shared aggregator history or turn retrieved material into a captured
user message. Tools and tool choice are carried into the copied request.

The byte budget covers the UTF-8 source block, not the entire model request or
its token count. Oversized passages are omitted whole; they are never silently
clipped. A configured engine low-similarity warning suppresses the block.
Otherwise ranking alone does not establish relevance, approval or truth. The
JSON boundary labels untrusted data; it is not a guarantee against prompt injection.

Each forwarded context frame carries `metadata["scone_memory"]`: status,
request/session/source-frame IDs, selected references, exact-block SHA-256 and
byte count, omitted count, and the engine's recall event ID when available.
Statuses are `prepared`, `empty`, `skipped` or `failed`; cancelled and superseded
requests are discarded and reported through `last_receipt`. A failed recall
passes the original request with a failed receipt, not cached or invented memory.
Surface that state in your application. `last_error` holds the actual local
exception; frame receipts contain only its type, not raw diagnostic text.

Receipts establish preparation/forwarding, **not provider delivery or model
use**. Frame metadata and `last_receipt` are not a durable delivery journal. A
configured Scone event log separately records retrieval. Recall timeouts are
cooperative; interruptions and a changed current request discard stale results.

Multimodal inputs, tool continuations and non-user final messages pass through
without recall. Provider compatibility and tool-loop context continuity are not
certified by the pipeline tests. Live voice/video, durable delivery, Rust HTTP
interoperability and a browser session service remain unfinished.

Run `python examples/pipecat_context.py` after the NLTK setup above for an
[isolated source-to-request-to-capture example](examples/pipecat_context.py).
It uses real Pipecat scheduling and native Scone retrieval, but an explicitly
scripted responder instead of a model; all fixture data stays in process memory.

### Pipecat text conversation runtime (service foundation)

`scone_memory.integrations.pipecat_text.PipecatTextConversation` runs role-separated
text exchanges through a supplied Pipecat model processor. It requires the same
optional Pipecat environment and NLTK provisioning described above. Supply an
already-open native engine, an authorized space and session ID, and a **factory**
that creates a fresh compatible `FrameProcessor` for each turn. The processor's
cleanup must release any clients it owns; a provider class is not automatically
certified merely because it is a Pipecat processor.

```python
from scone_memory.integrations.pipecat_text import PipecatTextConversation

# memory is your open native engine; model_factory is your configured factory.
conversation = PipecatTextConversation(
    memory, "default", "conversation-1", model_factory,
    where={"collection": "manuals"},
)
try:
    first = await conversation.reply("What does the calibration manual say?")
    followup = await conversation.reply("Explain the next step.")
finally:
    await conversation.close()
```

Each result contains `turn_id`, `text`, `user_episode_id`,
`assistant_episode_id`, `memory_context`, and `provider_completion="unverified"`.
The model receives previous user/assistant messages plus a transient source block.
Neither that source block nor model-side context mutations rewrite stored history.
Public user and assistant text becomes scoped conversation episodes. Thought
frames are not captured. Assistant metadata says `aggregated` and records
`completion_evidence=response_end_frame`: that frame alone cannot prove a provider
stopped successfully, so it is not labeled as verified provider completion.

One turn may run at a time. Input is bounded to 32,000 UTF-8 bytes; defaults are
64,000 reply bytes, 128,000 history bytes, and a 30-second turn deadline. History
budget counts JSON-encoded role/text messages; the retrieved source block has its
own budget. Turn errors, pipeline errors/timeouts, unsupported tool flow and
interruptions prevent a successful result and close the instance. `close()`
cancels an active turn. A submitted user message can remain after a failed reply;
a timed-out store write may have committed and is not automatically retried.
Deadlines rely on cooperative providers/stores, not hard process termination.

History is in-process, not automatically restored after restart. This module has
no session authentication, browser routes, durable request-id replay, live output
stream or audio/video transport. Tests use real Pipecat scheduling with an
explicitly scripted processor—not live inference. Provider-specific completion,
retry policy and client cleanup, the authenticated service, React controls and
Rust HTTP interoperability remain release gates.

### Authenticated conversation API (optional, single-process)

`scone_memory.api.conversations.create_conversation_app` exposes the journal and
a configured text runtime under `/v1/conversations`, with the existing native
memory routes mounted at the same origin. Install the `api` extra; Pipecat-backed
runtimes additionally need the optional Pipecat environment described above.
The caller owns the open engine; the ASGI lifespan owns its journal and runtime
tasks. This factory does not launch a server or select a provider for you.

```python
from scone_memory.api.conversations import create_conversation_app
from scone_memory.integrations.pipecat_text import PipecatTextConversation

# memory, space_keys and model_factory are explicit server configuration.
# Never use your native memory database as the conversation journal.
app = create_conversation_app(
    memory, space_keys, "conversation-sessions.db",
    lambda space, sid: PipecatTextConversation(memory, space, sid, model_factory),
    console=True,  # optional same-origin packaged React workspace; no keys embedded
)
```

With `console=True`, the service serves `/memory`, `/playground`,
`/conversations` and `/conversations/{session_id}` directly, including browser
refreshes. `/` opens the same application. Pages contain no configured keys:
enter a Scone space key in the connection dialog; it stays in tab memory and
must be supplied again after a full reload. Provider credentials stay in server
configuration. Hosting defaults to off; enabling pages does not configure a
model, launch a server or enable voice/video. Use a freshly packaged Webapp
build and HTTPS when deploying beyond loopback.

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

The Pipecat text runtime can accept another turn after model cancellation only
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

The Pipecat text adapter gives each session, turn and speaker a distinct capture
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
Tests connect HTTP controls to real Pipecat scheduling and native memory using a
scripted model; they do not certify a live provider. The React page supports
start, send, saved transcripts, source inspection, recovered outcomes and stop.
Provider certification, process-crash recovery evaluation, Rust HTTP memory adapters and
voice/video remain required product work. Existing live servers are unchanged.

### Conversation lifecycle journal (service foundation)

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

## Running it as a service

```sh
docker build -t scone-memory .                      # server image; add --build-arg EXTRAS=api,mongo,qdrant,postgres,local-embed for the ONNX embedder
SCONE_API_KEY=change-me docker compose up --build   # MongoDB + Qdrant + scone-memory on :7437, data in named volumes
scripts/compose-smoke.sh                            # brings the stack up, round-trips an episode, tears it down
```

The image refuses to start without `SCONE_API_KEY`. Its `/data` volume
holds the SQLite file when no database is configured, and the embedder's
model cache.

## Tests

```sh
python -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest
SCONE_TEST_MONGO_URL=mongodb://localhost:27017 SCONE_TEST_QDRANT_URL=http://localhost:6333 .venv/bin/pytest
```

Behavioural tests are proven to fail before they are trusted:
`PROVE_RUNNER=".venv/bin/pytest -q" ../../scripts/prove-test.sh <file> <needle> <replacement> <test>`
breaks the code, requires red, restores, requires green.
