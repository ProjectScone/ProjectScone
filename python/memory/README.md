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

Inspect the retained originals using the episode ID returned by `remember`:

```sh
scone-memory attachments 42 --space my-project
scone-memory attachments 42 --space my-project --json
```

This lists the episode's linked attachment IDs, media types, byte counts and
filenames. JSON returns one object with `space`, `episode_id` and `attachments`
(an empty array for an existing text-only episode). Missing, forgotten or
other-space episodes fail with exit status 2, not an empty success. This is a
metadata read: it does not download, decode, or verify the current blob bytes,
and does not describe image contents. Use the attachment ID with the native
`engine.attachment(space, id)` API or an authorized `GET /v1/attachments/{id}`
request when you want the original bytes.

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
memory.close()                                          # finish owned loop work before exiting
```

Use `SyncMemoryEngine` as a context manager, or call `close()` when finished.
Closing first rejects new calls, then cancels and drains tasks on its dedicated
loop, finalizes async generators and waits for that loop's default executor.
Pending calls report cancellation after their async cleanup completes. A call
that already completed is not retroactively cancelled, and cancellation is not
a rollback of writes that reached storage.

`close(timeout=5.0)` bounds how long the calling thread waits. If cleanup or an
executor worker has not finished, it raises `TimeoutError` and leaves the loop
draining; new work remains rejected. Release any externally blocked work and call
`close()` again to wait for the same shutdown. Repeated/concurrent closes do not
cancel cleanup again. Blocking facade calls, including close, cannot be made
from its own event-loop thread; use the async engine there instead. Failed
construction also shuts down its worker loop.

This lifecycle owns the wrapper's loop, not arbitrary document/vector/backend
clients. Backend-resource ownership and explicit client closing remain the
caller's responsibility; successful facade shutdown alone does not certify that
every remote pool or external thread has been released.

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

### Native real-time conversations

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

#### Memory preparation

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

Receipts report `prepared`, `empty`, `skipped` or `failed`, source episode/chunk
references, recall event ID when available, context hash/bytes, omissions and
sanitized degradation/error types. They prove preparation—not delivery or use.
Cancellation propagates instead of producing a success receipt.

See [the executable native example](examples/realtime_conversation.py). It uses
real Scone memory and scheduling with a scripted provider, not live inference.

#### Migration from the experimental framework adapters

The three former `integrations/pipecat*.py` modules and their optional dependency
have been removed. Replace processor factories with the `TextModel` protocol
above and import `TextConversation` from `scone_memory.realtime.text`.
Audio imports are now `scone_memory.realtime.voice` and
`scone_memory.realtime.audio`. No compatibility layer pretends a framework
processor is a native Scone provider. Existing HTTP routes, result field names,
saved transcripts and request-replay semantics remain unchanged.

### Scone Voice (native sessions)

Scone owns the audio runtime in `scone_memory.realtime.voice.VoiceSession`: its event types,
bounded input queue, turn lifecycle, interruption, response/speech sequencing and
memory capture. It uses standard Python `asyncio` and the native Scone engine.
**No Pipecat installation, import, probe environment or framework scheduler is
required.** Python 3.11+ is needed for this runtime's structured deadlines; the
base package and independent Rust/Python CLIs retain their existing requirements.

The native provider interfaces live in `scone_memory.realtime.audio`:

| Interface | Host adapter implements |
| --- | --- |
| `AudioTransport` | `receive()` audio stream, `send(chunk, turn_id)`, `clear(turn_id)`, `aclose()` |
| `SpeechRecognizer` | `transcribe(audio)` yielding speech-start and transcript events, `aclose()` |
| `VoiceModel` | `respond(messages)` yielding public text deltas and explicit completion, `aclose()` |
| `SpeechSynthesizer` | `synthesize(text)` yielding PCM chunks, `aclose()` |
| `SpeechActivityDetector` | optional `detect(chunk)` returning a strict boolean, `aclose()` |

Stream methods return async iterators supporting `aclose()` (async generators
work). Transport audio uses Scone's frozen `AudioChunk(pcm, sample_rate, channels)`:
signed 16-bit little-endian interleaved PCM, explicit 8–192 kHz sample rate, mono
or stereo. Adapters handle required format conversion; Scone does not silently
resample. Methods and factories must cooperate asynchronously and never block
the event loop. Resources must be fresh and distinct per session.

```python
from scone_memory.realtime.voice import VoiceSession

# These factories are configured by your trusted host, not supplied as code or
# credentials by a browser. They implement the Scone interfaces above.
session = VoiceSession(
    memory, "authorized-space", "voice-session-1",
    transport_factory=audio_transport_factory,
    stt_factory=speech_recognizer_factory,
    model_factory=language_model_factory,
    tts_factory=speech_synthesizer_factory,
    capture=True,
    where={"collection": "manuals"},
    session_timeout=1800,
    turn_timeout=30,
)
try:
    await session.run()
finally:
    await session.close()
```

The host obtains participant consent and authenticates the connection before
starting. `capture=True` authorizes public transcript writes; the flag alone does
not prove consent. No microphone, provider, credential, model download or endpoint
is selected implicitly. This is a native session—not a browser signaling server.

`SpeechStarted()` interrupts current output. A new final `Transcript` also
supersedes any active reply. Interim/empty transcripts do not trigger memory
writes or model calls. Scone cancels and drains the old response, asks transport
to clear its output by turn ID, and rejects late output from the old generation.
That clear request is not proof previously played audio was unheard. Input EOF
must be consumed by the recognizer and drains the final reply.

An optional `activity_factory` supplies a local speech detector independently of
the recognizer. A false→true speech transition queues an interruption before the
same PCM chunk reaches transcription. Activity and recognizer events share a
serialized controller, including duplex recognizers that consume PCM in another
task. Input remains bounded; no
samples are removed or resampled. The detector is closed with the other session
resources, including on failure or cancellation. Without it, the recognizer's
speech-start events continue to own early interruption.

#### Personas and independent provider selection

`realtime.persona.Persona` is a frozen, versioned configuration containing a name,
instructions and independent reply, transcription, speech and optional activity
choices. It chooses an existing model/voice; it does not train or clone a voice.
Serialize with `model_dump_json()` and load with `model_validate_json()`. Unknown
fields, unsupported schema versions and blank instructions are rejected. Changing
the speech selection does not change instructions, other models or memory scope.

```python
from scone_memory.realtime.persona import Persona
from scone_memory.realtime.providers import ProviderRegistry

# These are operator-defined IDs, not installed provider defaults.
persona = Persona.model_validate({
    "schema_version": 1, "id": "juniper", "name": "Juniper",
    "instructions": "Be concise. Explain the source behind each answer.",
    "reply": {"provider": "local", "model": "reply-v1"},
    "transcription": {"provider": "transcriber", "model": "speech-v1"},
    "speech": {"provider": "voice-a", "model": "tts-v1", "voice": "alto"},
    "activity": None,
})

# Host-created factories close over credentials. Nothing in a persona imports
# Python code, selects network endpoints, or grants a memory space/recall scope.
registry = ProviderRegistry(
    reply={("local", "reply-v1"): language_model_factory},
    transcription={("transcriber", "speech-v1"): speech_recognizer_factory},
    speech={("voice-a", "tts-v1", "alto"): speech_synthesizer_factory},
)
bound = registry.resolve(persona)  # checks EVERY choice; creates no resources
session = bound.voice(memory, "authorized-space", "persona-session-1",
                      transport_factory=audio_transport_factory, capture=True,
                      where={"collection": "manuals"})
# Or bound.text(memory, "authorized-space", "text-session-1", where=...).
```

The registry admits exact `(provider, model)` pairs and, for speech, exact
`(provider, model, voice)` triples. There is no fallback to another provider.
Each host should expose only the choices that user may use. A bound text session
constructs no audio resources and obtains a fresh selected model per turn.
Credentials and PCM compatibility remain adapter responsibilities; successful
binding is not a remote availability or compatibility check.

Direct Deepgram, OpenAI, Cartesia, ElevenLabs and Silero adapters, an HTTP persona
catalog and browser voice selection are separate pending integrations. This
native composition API does not advertise browser voice as available.

Only public `TextDelta` and `ReplyCompleted` events are accepted from the model.
Scone groups text into speech segments; synthesis and output are awaited before
requesting further model output, providing backpressure at those interfaces.
Reply-end and audio-end are different: a reply is retained only after explicit
completion, stream closure, successful synthesis and output acceptance. A missing
completion or empty synthesis fails the turn. `send()` acceptance is **not** proof
a person heard the audio. Tool and hidden-reasoning events have no accepted type;
adapters must never relabel private reasoning as public text.

Final user text is saved before response generation. Successful assistant text
becomes a `conversation` episode with capture/session/turn IDs, speaker/role and
`representation=aggregated_text`. Assistant metadata identifies
`completion_evidence=adapter_end_and_output_accepted` and `playback=unverified`.
Interrupted partial replies are not retained as completed. A cancelled or timed-out
write is unconfirmed and stops the session; inspect storage before retrying.
There is no automatic uncertain-write replay.

Recall scope (`where`, `kind`, `source_prefix`, `since`, `until`) is validated
and frozen before factories run. Recalled text is source-referenced, marked
untrusted, bounded and inserted only into a copy of the current request—not
shared history or transcript memory. `last_memory_receipt` reports prepared,
empty or failed lookup, not provider use. Lookup failure can continue without
recalled material; capture/output/provider failures cannot report success.
Raw audio and source blocks are not retained by this integration.

The session is single-use. `state` is `new`, `starting`, `running`, `ended`,
`interrupted` or `failed`. Observe the `run()` task alongside the `started`
event because startup can fail. `stored_count` counts acknowledged writes.
Call `close()` from a host task, not from a provider callback. Repeated Close
or caller cancellation joins the same owned cleanup. The deadline requests
shutdown; noncooperative resource cleanup can delay return. No hard process
termination is claimed.

Defaults: 8 queued input packets, 64,000 bytes per PCM packet, 32,000 bytes per
transcript, 64,000 reply bytes, 128,000 JSON-encoded history bytes, bounded recall,
30 seconds per response and 30 minutes per session. The input producer may hold
one additional packet while the queue is full. These bounds do not control
provider-internal queues, network buffers or physical playback.

Run the native regressions in the ordinary project environment:

```sh
python/memory/.venv/bin/python -m pytest python/memory/tests/test_voice.py -q
```

Tests use scripted protocol adapters and real isolated Scone memory, not live
recognition/model calls. Concrete provider adapters, authenticated browser audio
transport, React voice controls and video remain release gates. HTTP capabilities
still advertise `voice: false` until that entire path works. The text runtime and voice runtime share Scone-owned public events and scoped
context preparation; neither depends on an external conversation framework.

### Browsing retained sources

`GET /v1/sources?limit=25&kind=file&before=123` enumerates retained episodes in
descending episode-ID order. Omit `kind` to browse all kinds; omit `before` to
start at the newest ID. This is inventory, not semantic search, source-date order,
or a frozen database snapshot. Newer inserts appear on refresh; deleting a page's
boundary record does not invalidate its `before` value. Keep the same kind filter
while following `next_before`.

The response has `items`, `has_more` and nullable `next_before`. Each item contains
`episode_id`, `kind`, `source`, `created_at`, `byte_count` (UTF-8 stored text),
`preview` (at most 500 Unicode scalar values), and `preview_truncated`. A preview
is not an original file; retrieve retained text with `GET /v1/episodes/{id}` and
original media through the separate attachment routes. The bearer key selects
the space; query/body fields cannot change it. Page limits are 1–100; an invalid
or unknown query field is rejected. Existing `GET /v1/episodes?ids=...` remains a
separate bounded batch read.

Native async and sync engines expose `source_page(space, before=None, limit=25,
kind=None)`, returning a `SourcePage` of full episode records. Built-in document
stores implement the optional `EpisodeInventory.page_episodes` port. Custom
stores without it advertise `episodes.list: false` and HTTP returns 501; there is
no fallback that scans the whole export or disguises ranked recall as inventory.
Adapters can run `scone_memory.testing.contract_inventory` with their usual
engine fixture. In-memory inventory scans resident entries with bounded selection;
database adapters apply scope, kind, ID boundary, ordering and limit in their
native queries. Read cost and remote-store certification are separate from the
bounded response contract. The Documents browser UI is subsequent work.

### Authenticated conversation API (optional, single-process)

`scone_memory.api.conversations.create_conversation_app` exposes the journal and
a configured text runtime under `/v1/conversations`, with the existing native
memory routes mounted at the same origin. Install the `api` extra and use Python 3.11+ for native real-time sessions.
The caller owns the open engine; the ASGI lifespan owns its journal and runtime
tasks. This factory does not launch a server or select a provider for you.

The CLI can launch that service without a custom ASGI entry point:

```sh
# Uses the existing SCONE_API_KEY / SCONE_API_KEYS and native store settings.
# Create the journal's parent directory first; never use the native memory DB.
scone-memory serve-conversations --journal ./conversation-sessions.db --history-only --console

# Explicit model opt-in, from a trusted Python module on your import path:
scone-memory serve-conversations --journal ./conversation-sessions.db --model-factory my_models:create --console
```

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
routes; it is **not a read-only memory server**. `--console` opts into the packaged
React pages, with no keys embedded. The CLI uses SQLite persistence by default,
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
    console=True,  # optional same-origin packaged React workspace; no keys embedded
    public_text_streaming=True,  # known-compatible native runtime; default is False
)
```

#### Public-text stream (optional)

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

To let an API caller narrow recall for each session, opt in explicitly:

```python
app = create_conversation_app(
    memory, space_keys, "conversation-sessions.db", None,
    scoped_runtime_factory=lambda space, sid, scope: TextConversation(
        memory, space, sid, model_factory, **scope.kwargs()
    ),
    console=True,
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
