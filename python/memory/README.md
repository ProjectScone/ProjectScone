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
A third field gives a key a role:
`SCONE_API_KEYS=r:default:read,w:default:write,v:default:review,f:default:full`.
`read` only reads; `write` remembers, links and forgets but never decides a
claim; `review` approves, declines, excludes and includes but never adds;
`full` (the default, and always what `SCONE_API_KEY` gets) does everything.
A refused request is a 403 naming the role. The conversation service holds
the same rule, including the audio socket's hello.

| Route | What |
|---|---|
| `POST /v1/episodes` | remember; `{content, tags?, source?, created_at?, kind?}`; unknown fields are refused |
| `DELETE /v1/episodes/{id}` | forget |
| `GET /v1/spaces/{space}/impact` · `DELETE /v1/spaces/{space}?confirm={space}` | delete a whole space (`scone-memory delete-space --confirm <space>`, `--dry-run` for the preview): the preview says what would go, the deed removes it, in order, attachment holds (bytes only when no other space holds them), vectors, chunks, episodes, links, claims, events, tombstones and the revision; it needs the `full` role and the name repeated, and afterwards the key answers 404 on every route, so a key left in config cannot re-create what was erased |
| `GET /v1/recall?q&limit&as_of&tags&where&history&kind&source_prefix&since&until` | hybrid recall plus the facts that held at `as_of`; `history=true` adds the closed facts that came before them; `kind`, `source_prefix` (literal text), `since` and `until` (inclusive) narrow the candidates the way the Rust engine does |
| `GET /v1/facts?all&as_of` · `POST /v1/facts` · `POST /v1/facts/{id}/close` | the fact ledger |
| `POST /v1/consolidate` `{scope: distill \| derive}` | one consolidation pass by hand over this key's space: `distill` runs the worker's pass (extraction, retention and, with `SCONE_DERIVE=1`, derivation), `derive` only the derivation pass, which proposes claims that follow from the claims held, each with its premises as `derived_from` links and no quote (`scone-memory derive`); 501 without a model. `GET /v1/status` carries `pending_derivation` (groups not yet sent at their current membership) and `derivation` on/off |
| `GET /v1/profile` · `GET /v1/tags` · `GET /v1/status` · `GET /healthz` | overviews; the profile's `static_facts` are the claims that hold now (one not yet valid, ended, or excluded stays out) and its `recent` is `dynamic` with its evidence, one `{episode_id, excerpt, created_at}` per entry, most recent by the episode's own time first, the same rule and shape the Rust engine serves |

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

### Indexed fact recall

SQLite accelerates lexical fact lookup with a derived token index. Query token
overlap, confidence/ID ordering, historical validity, and source filters retain
the ledger scan's semantics. Facts outside the requested source scope cannot
consume the result limit. The engine checks returned facts against current ledger
records; an unavailable or invalid index falls back to scanning and reports
`fact_index: unavailable` in recall degradation. Other document stores retain
their existing scan unless they implement the optional `IndexedFactSearch` port.

SQL triggers record fact edits made by older clients as well as this library.
The next lookup refreshes pending terms for that space. Existing databases incur
an initial backfill; missing or incompatible derived objects are rebuilt on open.
The index does not change the ledger schema version. Historical-chain retrieval
still scans, and common query terms can still require sorting many matching
postings. This index does not accelerate vector retrieval or model generation.

Measure exact result parity and warm lookup latency on disposable synthetic
ledgers, with initial indexing reported separately:

```sh
python -m scone_memory.testing.fact_search_benchmark \
  --sizes 1000 10000 50000 --repeats 5 --output /path/to/new-report.json
```

The diagnostic includes sparse terms, common terms, and a source filter that
rejects most higher-ranked matches. It reports full ledger rows materialized and
fact point reads; it does not claim constant-time lookup or generation accuracy.
No model, network connection, or application database is used.

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

### Optional S3 attachment storage

Install `scone-memory[aws]` and select pre-provisioned resources explicitly:

```sh
SCONE_BLOBS=s3
SCONE_S3_BUCKET=your-attachment-bucket
SCONE_DYNAMODB_BLOB_TABLE=your-blob-metadata-table
SCONE_AWS_REGION=us-east-1
SCONE_S3_PREFIX=attachments/
```

S3 stores attachment bytes; DynamoDB stores their space ownership, episode
references, and cleanup journals. This is an attachment backend, not a DynamoDB
replacement for the document ledger, conversation journal, or retrieval index.
Configure those stores separately. `SCONE_BLOBS=auto` preserves existing defaults;
`file` requires `SCONE_BLOB_DIR`, and `memory` explicitly selects ephemeral bytes.
Contradictory storage settings fail at startup.

The adapter uses the AWS SDK credential chain; prefer workload IAM roles.
Constructing it does not resolve credentials or contact AWS. Supply a private,
versioned S3 bucket dedicated to attachments with default SSE-KMS encryption and a DynamoDB table with
string partition key `pk` and string sort key `sk`. Configure table encryption,
PITR, and least-privilege access to that table, bucket prefix, and KMS key.
The role also needs `s3:ListBucket` on the dedicated bucket so S3 HEAD can report
missing objects as 404 during interrupted-upload recovery; 403 is treated as an
error, never proof that bytes are absent.
Deployment container instructions are in [`deploy/aws`](../../deploy/aws/README.md).
Reusable Terraform configuration lives in the repository-root `terraform/`
directory. Supply deployment values through the environment; private `.env`
files, populated variable files, plans, and state stay ignored.

Each S3 upload has an immutable generation key. DynamoDB transactions publish
ownership and persist deletion intents before S3 cleanup. Cleanup targets the
recorded version, so a delayed delete cannot remove a newly published generation.
S3 and DynamoDB do not provide a shared atomic transaction: failed operations
raise and retain recovery state. Operators can call
`await engine.blobs.recover_uploads(space, after=cursor, limit=100)` on this
adapter to fence unpublished uploads and retry cleanup. Each call returns a
`RecoveryPage` with `visited` and `next_cursor`; follow the cursor until it is
`None` to complete a sweep. Repeat sweeps after paused writers finish; an attempt
is not proof that no late write can arrive. Pending upload intents are retained
for that purpose. No automatic background recovery is configured.

This initial adapter has explicit capacity bounds: 25 MiB per attachment, 1,024
held attachments per space, 1,024 episode references per attachment, and 300 KiB
per metadata item. Over-budget operations fail; these limits are not a claim of
unbounded storage capacity. Partition reads are strongly consistent and use
queries, not table scans. SDK calls run off the event loop with bounded waits;
cancellation cannot interrupt an already running SDK request.

Run emulator tests with `pip install -e '.[aws-test]'` followed by
`pytest tests/test_aws_blobs.py tests/test_aws_blob_config.py`. Tests use synthetic
resources and credentials through Moto; they do not provision AWS resources or
measure AWS throughput.

## LlamaIndex and LangChain workflows

The optional `llamaindex` and `langchain` extras can coexist over one Scone
engine. `scone_memory.integrations.llamaindex.SconeRetriever` returns scored
nodes; `scone_memory.integrations.langchain.SconeRetriever` returns documents,
and `SconeChatMessageHistory` provides explicit session history.

The packaged [composition API](src/scone_memory/integrations/composition.py) runs
LlamaIndex retrieval inside a LangChain Runnable workflow while preserving
source text, chunk/episode IDs and the application's authorized scope. Its
`retrieve_without_tracing` helper disables hosted tracing per invocation and
needs no model service. These retriever adapters are separate from the optional
vector-store bridge below.

Additional package APIs provide [query evidence](src/scone_memory/retrieval/evidence_graph.py),
[bounded reranking](src/scone_memory/retrieval/reranking.py),
[structural context](src/scone_memory/retrieval/structural.py),
[recorded multi-hop retrieval](src/scone_memory/retrieval/multihop.py), and
[encrypted workflow checkpoints](src/scone_memory/agents/workflow.py).
The [retrieval workflow builder](src/scone_memory/agents/retrieval.py) composes
both frameworks with retained-source checks.

`GET /v1/recall?graph_analysis=true` adds bounded community, hub and bridge
analysis to the scoped query result. Set `evidence_graph=true` as well to receive
the nodes and retained-source relationships behind those IDs; both options share
one graph build. The pure Python API is
`scone_memory.retrieval.graph_analysis.analyze_evidence_graph`.
Its versioned algorithm analyzes unique undirected recorded relationships and
reports directional hub counts separately. Coverage and omissions describe the
supplied graph, not the entire memory store; connectivity is not confidence.
Analysis failures leave ordinary recall available with an explicit unavailable
status. Neither option enables external services or model calls.

The [self-hosted reranking evaluator](src/scone_memory/testing/self_hosted_reranking.py)
and [Qdrant comparison](src/scone_memory/testing/qdrant_comparison.py) are runnable
package modules. They use isolated fixtures and record failures as well as
successful retrieval; fixture scores do not establish general answer accuracy.
The root [.env.example](../../.env.example) lists supported settings without
credentials. Keep private values in ignored `.env.local` with mode `0600`;
`scripts/serve-self-hosted.sh --check` validates its format before an explicit launch.

### Offline cross-encoder reranking

Install `scone-memory[offline-rerank]` to rank retained passages with a dedicated
CPU model instead of asking a chat model to assign relevance scores:

```python
from scone_memory.providers.offline_reranker import OfflineCrossEncoderReranker

reranker = OfflineCrossEncoderReranker(
    "/srv/models/ms-marco-MiniLM-L-6-v2",
    model_name="Xenova/ms-marco-MiniLM-L-6-v2",
)
memory = await MemoryEngine(
    documents, vectors, embedder, reranker=reranker,
    rerank_limit=16, rerank_timeout=2,
).open()
result = await memory.recall(
    "authorized-space", "Who approves external data exports?",
    limit=3, candidate_limit=32,
)
```

The directory must already contain the model's ONNX weights, configuration, and
tokenizer files. Provision those artifacts separately; the adapter makes no
download or inference network calls and never loads remote Python model code.
Supported plain ONNX models are `Xenova/ms-marco-MiniLM-L-6-v2`,
`Xenova/ms-marco-MiniLM-L-12-v2`, and `BAAI/bge-reranker-base`. Other models can
use the existing caller-supplied `Reranker` interface.
`model_identity` records artifact hashes for reproducible runs. Runtime settings
also accept `SCONE_RERANKER_CROSS_ENCODER_DIR` and
`SCONE_RERANKER_CROSS_ENCODER_MODEL`; these are mutually exclusive with the
existing trusted `SCONE_RERANKER_FACTORY` option.

Scores are raw model logits used for ordering, not confidence or a support
threshold. Negative scores can still identify the best available evidence.
Candidate depth, scope checks, result size, and reranking budgets remain owned by
Scone's retrieval pipeline. Every full query/passage pair must fit the configured
token limit and the model's declared limit. Oversized pairs fail reranking rather
than silently scoring a truncated prefix; the existing retrieval fallback and
failure trace remain visible. This does not expand a model's context window.

Inference runs off the event loop, with one active CPU job per adapter. Cancelling
the awaiting request does not stop an already running ONNX operation; its result
is discarded and its slot stays occupied until it finishes.

The isolated evaluator accepts `--cross-encoder-dir /srv/models/MODEL` together
with `--model MODEL_NAME`, `--embedding-cache EXISTING_BGE_CACHE`, and
`--output NEW_REPORT.json`. It compares ordinary fusion, expanded candidate
retrieval, and offline reranking on the same synthetic cases. It measures passage
retrieval, not generated-answer accuracy.

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

For model tool calls, `scone_memory.integrations.tools.ToolBox` binds an async
engine to one host-selected space. Its `openai()` and `anthropic()` methods
render the same four contracts: `search_memory`, `add_memory`, `read_profile`,
and `trace_memory`. Hosts can allowlist a subset. The host executes returned
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
origins, directed stored relations, and ordered paths. Exact object-to-subject
matches are labeled `subject_object`; they are not inferred semantic links.
Contradictions remain separate evidence, never a path continuation or an
automatically chosen winner. Quote retention is checked; factual accuracy is
not certified. Source text remains untrusted data for the receiving model.

The trace is read-only and bounded: 16 facts, 32 edges, 256 traversal store
calls/candidates, eight paths, and a two-second async timeout. Two additional
revision reads fence native writes. Complete source episodes may be loaded
during point reads. The evidence packet is capped at 64,000 UTF-8 bytes before
the ToolBox envelope. Every source must match any supplied tags. Missing or
ineligible seeds return empty evidence; timeouts, changed revisions, and store
failures return an unavailable result without partial quotes. Direct adapter
writes require adapter transaction discipline. Coverage describes this seed's
bounded neighborhood, never completeness of an answer to an arbitrary query.

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
and oversized responses. Logs contain outcome and elapsed time, not message text.
HTTP resources are closed before a reply is accepted; cooperative cleanup can
run beyond the deadline, but late success is rejected.

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

Protocol compatibility is not evidence of better answers. In a nine-case
synthetic development comparison on `llama3.2-ctx8k`, structured actions fixed a
malformed native trace-call case, but the model still invented a missing bridge,
reversed a dependency, conflated `painted by` with `depends on`, and answered a
manufacturer question without searching. The checked-in cases are in
`tests/fixtures/tool_action_cases.json`; their expectations require manual
source-grounded adjudication. These are development findings, not a held-out
accuracy score, and do not justify enabling this adapter by default.

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

Tool mode uses the actual conversation history, with current-session records
excluded from tool searches. It bypasses prompt-based memory preparation and
cannot be combined with independent adaptive retrieval, answer review, or
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
# SCONE_CONVERSATIONS_TOOL_MAX_CALLS=4
# SCONE_CONVERSATIONS_TOOL_MAX_ROUNDS=4
# SCONE_CONVERSATIONS_TOOL_TIMEOUT=120
```

Configure the chat connection in the model settings or supply `SCONE_CHAT_URL`
and `SCONE_CHAT_MODEL` as connection defaults. Served tool mode performs the
host-initiated search by default. Set `SCONE_CONVERSATIONS_TOOL_INITIAL_SEARCH=0`
to let the model choose whether to search. SDK `TextConversation` exposes the
same option as `tool_initial_search=True`, with the SDK default remaining false.

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
with adaptive retrieval, answer review, trusted custom model factories or custom
persona catalogs. Those integrations can still bind their own SDK pipelines.

The structured adapter renders tool results as explicitly labeled, untrusted
evidence in plain chat roles. After a completed tool exchange it repeats the
latest actual user question so the provider's answer target does not become the
last source packet. This rendering does not change stored conversation history;
the expanded history is checked against the adapter's 1 MB limit before sending.
Its tool-selection instructions distinguish reading neighboring document chunks
from tracing a mentioned entity's relationships to find a requested attribute.
This guides the model's choice; it does not change retrieval permissions,
increase budgets, or establish that an answer follows from the retained evidence.

`/v1/conversations/capabilities` exposes `tool_retrieval` protocol, budgets and
whether a text connection is configured. This is configuration availability,
not a model-health probe or an accuracy claim. Without a saved chat connection,
text remains unavailable; enabling the mode does not install a model.

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
combined with the independent adaptive, review, extractive, or path-projection
candidate options. No fixture answer labels or required quotes reach the model.

Tool rows record model-request hashes/bytes, provider-call counts, read reuse,
and post-run coverage of evidence observed at the provider boundary. Failed
generation keeps its observed evidence coverage but receives zero successful
answer credit; successful final source retention is reported separately.
Relation quotes participate in the coverage audit for both variants. Tool
request sizes describe the model-neutral transcript and schemas, not the wire
encoding of a particular provider. This evaluator uses temporary SQLite storage;
omit `--embedding-cache` for a hash-embedder integration check, not a semantic
retrieval benchmark.

Receipts report `prepared`, `empty`, `skipped` or `failed`, source episode/chunk
references, recall event ID when available, context hash/bytes, omissions and
sanitized degradation/error types. They prove preparation—not delivery or use.
Cancellation propagates instead of producing a success receipt.

#### Adaptive evidence retrieval

For applications with an explicit question plan, the SDK also provides a
model-free `StructuredEvidenceAssessor`. Each requirement asks for recorded
values of an exact subject/predicate, subjects of an exact predicate/object,
a simple directed path of one exact predicate to a named endpoint, or a recorded
attribute reached through explicitly allowed predicates. Every requirement needs a complete witness before
its verdict is `sufficient`; this verdict describes the supplied plan and
records, not semantic truth or general question-answer accuracy.

```python
from scone_memory.retrieval.structured_evidence import (
    EvidenceRequirement, StructuredEvidenceAssessor,
)
from scone_memory.retrieval.adaptive import AdaptiveRetriever
from scone_memory.retrieval.recall_scope import RecallScope

question = "Which dependency path connects aster to denver?"
assessor = StructuredEvidenceAssessor(question, (
    EvidenceRequirement(kind="path", subject="aster", predicate="depends on",
                        object="denver", max_hops=3),
))
result = await AdaptiveRetriever(memory, assessor).retrieve(
    "authorized-space", question,
    scope=RecallScope.validated(where={"collection": "manuals"}),
)
```

The application authors the requirements and binds them to the exact question.
There is no automatic intent parser or implicit HTTP activation. Identity matching
is literal: synonyms, case differences and alternate predicates need an explicit
application mapping. Terminal, negative, and whole-index completeness claims are
unsupported. A missing path means no witness within the supplied candidate and
hop bounds; it does not prove there is no path in the knowledge store.

For “Who uses Polaris?”, leave the subject unknown instead of reversing the
stored relation:

```python
users_of_polaris = EvidenceRequirement(kind="fact", predicate="uses", object="Polaris")
# Matches “Juniper uses Polaris”, not “Polaris uses Juniper”.
```

Fact requirements must name at least one endpoint; path requirements still need
both. These lookups inspect the bounded candidate set and retain all matching
subjects plus competing recorded values around their witnesses. They do not
claim to enumerate every user across an entire index.

For an attribute whose owning entity is not known in advance, use
`reachable_fact`:

```python
office_location = EvidenceRequirement(
    kind="reachable_fact", subject="invoice", predicate="located in",
    via=("assigned to", "managed by"), max_hops=3,
)
# Requires invoice -> team -> office, then office's recorded location.
# A team name, an unrelated office, or a missing bridge cannot satisfy it.
```

`via` contains 1–8 allowed exact predicates, which may occur in any order or
repeat along a route. The final `predicate` must be distinct from them.
At least one bridge is required; use `fact` for a direct attribute. `max_hops`
counts both bridge facts and the final attribute fact (2–6 total). Omit `object`
to retain recorded values, or supply it to require a particular value while
keeping competing observations. Breadth-first traversal keeps one shortest
supporting route per reachable subject and continues looking for other matching
subjects within the bounds. It does not enumerate every alternative route.
The full witnesses and competing values form one atomic group per requirement.
An attribute owner need not be a graph leaf: this contract cannot establish an
“ultimate” destination unless the application's relation semantics justify it.
It supplies recorded evidence connections, not a synthesized transitive fact.
This is available to both the adaptive assessor and the quote selector below;
it does not automatically change ordinary conversations or model tool choices.

The assessor accepts 1–8 requirements and at most 100 candidates / 128,000 UTF-8
candidate bytes. Path-edge work defaults to 256 and is configurable up to 2,048;
exhaustion returns `uncertain`, preserving other witnessed requirements. It emits
up to three follow-up search queries for missing requirements. Competing recorded
values around witnesses remain together in atomic groups; no winner is inferred.
The native retriever still owns candidate discovery, scope, timing, and source
revalidation. Invalidated selected groups are omitted together. This does not
turn a passage that looks like a triple into a stored fact.

The same explicit plan can govern quote-based answer selection with
`StructuredEvidenceSelector`. It requires all fact/path witnesses, including
competing recorded values, to fit in at most three whole cards. A missing bridge,
reversed path, or different predicate produces no selection. The answer remains
the original quoted records and citations; no model writes the public answer.

```python
from scone_memory.realtime.structured_selector import StructuredEvidenceSelector
from scone_memory.realtime.text import TextConversation

question = "Which dependency path connects aster to denver?"
requirements = (
    EvidenceRequirement(kind="path", subject="aster", predicate="depends on",
                        object="denver", max_hops=3),
)
conversation = TextConversation(memory, "authorized-space", "planned-answer-1",
    evidence_selector=StructuredEvidenceSelector(question, requirements),
    evidence_answer_policy="required",
    where={"collection": "manuals"},
)
try:
    reply = await conversation.reply(question)
finally:
    await conversation.close()
```

This is a fixed, application-authored question plan, not automatic intent
understanding. For another question, the application must supply its corresponding
plan. The selector checks only offered cards: missing or budget-omitted evidence
cannot establish a global negative. Path checks match recorded triples; they do
not certify extraction correctness, quote entailment, causation, or source truth.
`verified_accuracy` remains false. The synthetic tool-action development cases
also exercise this selector with hand-authored plans; passing them is a contract
check, not a natural-language generation accuracy result.

Selections from this selector are atomic: if the entire selected quote set
exceeds the answer byte limit, the renderer abstains instead of publishing a
partial chain. Other selectors may request this behavior through
`EvidenceSelection(atomic=True, card_ids=...)`. Receipts expose
`atomic_selection` and count output-budget omissions.

`evidence_answer_policy="required"` needs an evidence selector and never falls
back to generation. Empty or skipped retrieval returns a recorded abstention
with `source_status="none"`; failed preparation returns an error. No generation
provider is needed. The default `"when_available"` policy retains normal
generation when no memory is prepared. This SDK policy and selector are opt-in;
the default server configuration does not automatically create question plans.

An optional bounded loop assesses retrieved evidence, keeps selected records,
and searches for missing information before generation. The host supplies an
`EvidenceAssessor`; the core fixes the memory space and session filters for every
search. Model output can select existing IDs and propose queries, but cannot
change authorization or run tools. Candidate, query, round, byte and time limits
are independent. Source changes during assessment invalidate the affected
evidence; errors and timeouts return an explicit uncertain result.

```python
from scone_memory.providers.evidence_assessor import SelfHostedEvidenceAssessor
from scone_memory.providers.llm import OpenAICompatibleTextModel
from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever
from scone_memory.realtime.text import TextConversation

endpoint = "http://inference.home.arpa:11434/v1"
model = "my-installed-model"
adaptive = AdaptiveRetriever(
    memory, SelfHostedEvidenceAssessor(endpoint, model, timeout=30),
    limits=AdaptiveLimits(max_rounds=3, max_queries=6, timeout_s=30.0),
)
conversation = TextConversation(
    memory, "authorized-space", "session-1",
    lambda: OpenAICompatibleTextModel(endpoint, model, timeout=45, trust_env=False),
    where={"collection": "manuals"}, adaptive_retriever=adaptive,
    recall_timeout=30, turn_timeout=90,
)
```

Use and close the conversation as above. The retriever must bind the same
engine; `recall_timeout` must cover its deadline, and the full turn also needs
time for generation. Greetings and overview retrieval keep their existing flow.
The assessor's `sufficient` verdict is a fallible model judgment, not an answer
accuracy guarantee. This option remains off by default and does not automatically
enable itself in the HTTP service when a model is loaded.

The standard `serve` launcher can mount it explicitly for custom-model,
saved-connection and persona **text** sessions:

```dotenv
SCONE_ADAPTIVE_RETRIEVAL=1
SCONE_ADAPTIVE_URL=http://127.0.0.1:11434/v1
SCONE_ADAPTIVE_MODEL=YOUR_INSTALLED_MODEL
SCONE_ADAPTIVE_TIMEOUT=15
SCONE_ADAPTIVE_MAX_ROUNDS=3
SCONE_ADAPTIVE_MAX_QUERIES=6
SCONE_ADAPTIVE_CANDIDATE_LIMIT=20
SCONE_ADAPTIVE_MAX_EVIDENCE_BYTES=16000
SCONE_ADAPTIVE_GRAPH_HOPS=3
# SCONE_ADAPTIVE_API_KEY=  # Only for an authenticated assessor endpoint.
```

This requires `SCONE_CONVERSATIONS_JOURNAL` and a separately configured text
model or persona to reply. Assessor credentials are independent of generation,
extraction and answer review. The full conversation-turn timeout still includes
retrieval, generation and any answer review; allocate enough time for all enabled
stages. Greetings and overview queries keep their existing routing. Voice is
unchanged. Graph hops default to zero (disabled); enabling 1..6 hops uses the
native `MultiHopLimits` defaults for the other per-expansion work bounds.

Served adaptive retrieval uses `retain_verified` for assessment failures and
empty selections, plus `original_and_selected` to retain the original query's
verified pool alongside later selection. These policies are described below;
none establishes answer accuracy. They preserve scope and source checks across
every round. `adaptive_retrieval` in conversation capabilities reports configuration,
budgets and graph hops independently of reply-model availability. Turn context
receipts report rounds, queries, fallback, truncation and graph work. These are
current-process context receipts, not a durable reconstruction after restart.
When embedding `create_conversation_app`, its optional `adaptive_retriever` must
bind the same engine; custom text factories must accept and honor the native
`adaptive_retriever` and `recall_timeout` keywords.

The default `failure_policy="retain_verified"` recovers from assessor errors,
invalid decisions, and assessment timeouts by independently rechecking the last
bounded candidate snapshot. It reserves `min(1 second, timeout_s / 4)` within the
existing deadline for that check; it does not retry the model or run new searches.
Changed, deleted, out-of-scope, or unverifiable evidence is omitted. Retrieval
and source-verification failures still return no evidence. Cancellation propagates.

Recovered evidence has `status="uncertain"`, `evidence_basis="verified_candidates"`,
and `fallback_status="retained"`; the original sanitized assessment error remains
visible. This is an unassessed candidate pool, not a sufficient answer or a model
selection. Recovery preserves host-known atomic groups and groups from prior valid
decisions only; a failed response cannot establish new groups. Native context
receipts expose the basis, fallback status, and delivery completeness separately. Use
`failure_policy="empty"` when any assessment failure should discard all evidence.

A valid assessment can also return `insufficient` or `uncertain` with no selected
records. The separate default `empty_selection_policy="retain_verified"` retains
the final offered candidate snapshot after source revalidation. Its basis is
`unselected_candidates`, its round still records zero model-selected records,
and `fallback_status` remains `not_used`. The insufficiency or uncertainty stays
visible; keeping a known partial route does not establish its missing endpoint.
Candidate retention does not establish relevance either: the bounded pool may
include distractors that the assessor did not select.

This policy applies only when the final valid decision selected nothing. It does
not resurrect earlier pools discarded during follow-up searches, or replace a
nonempty selection that later loses its sources. Existing scope, deadline, byte,
and atomic-group checks still apply. Use `empty_selection_policy="empty"` to
preserve model-only selection, independently of assessment-failure handling.

To protect against later searches drifting away from the original question,
opt into `evidence_policy="original_and_selected"`. The retriever saves the first
verified query pool, including any enabled graph expansion, and combines it with
the final model selection. Reciprocal rank fusion gives each lane a vote using
`1 / (60 + rank)`; whole atomic components compete by their strongest member's
score. Ties use original-query order first. Packing uses the existing candidate
and UTF-8 byte limits, so either lane can lose records when the combined pool
does not fit. Scores indicate ranking, not relevance or factual confidence.

The host revalidates the union before packing. Original snapshots take precedence
for shared IDs; a later search cannot replace a changed original source under
the same identity. Verification reads at most two bounded pools within the
existing deadline. Each input lane is capped at 128,000 serialized bytes; the
combined output still uses the configured, potentially smaller context budget.
Valid atomic contracts learned in earlier rounds continue to apply across both
lanes, so restoring original evidence cannot expose a surviving group fragment.
`original_query_ids` and `model_selected_ids` distinguish the
origins of returned records and may overlap. Native context filters those origin
lists again after its own packing. A model's `sufficient` verdict describes its
selection; it does not certify the blended evidence or generated answer. Losing
selected evidence to validation or packing downgrades the result to `uncertain`.

This policy keeps the original pool even when a successful workflow ends with an
empty selection; it takes precedence over terminal empty-selection handling.
Assessor errors still use the separate failure policy. Follow-up retrieval and
selection behavior are unchanged. The default remains `evidence_policy="model_selected"`.
Compare with `--adaptive-evidence-policy original_and_selected` in the evaluator;
neither this option nor the adaptive strategy is automatically enabled in HTTP.
The explicit served configuration above selects `original_and_selected`.

For a controlled comparison against existing compact paths, add
`--adaptive-model YOUR_INSTALLED_MODEL --baseline-paths --adaptive-timeout 30
--adaptive-rounds 3` to the generation evaluator. Each row records the selected
variant and adaptive diagnostics; frozen answer checks remain separate from
source coverage and manual semantic review. The assessment transport timeout
uses the requested adaptive budget; the retriever enforces the remaining total
budget across all calls. Reports record both limits and the failure policy; use
`--adaptive-failure-policy empty` to compare the explicit empty-on-failure behavior.
Use `--adaptive-empty-selection-policy empty` for the separate valid-empty-selection
comparison; the report records both policies.
Receipts distinguish assessment timeouts, provider failures and invalid model decisions without
including raw provider errors or source text.

For optional atomic relation selection, construct the assessor with
`group_relations=True, max_evidence_bytes=16000`. The pure
`retrieval.evidence_groups.build_evidence_groups` helper groups supplied facts
by exact object-to-subject matches, preserving branches and cycles. It does not
invent semantic links or search beyond the supplied candidate pool. Existing
stored-link kinds are still handled by the separate graph expansion stage.

To gather missing connecting facts **before** assessment, pass
`graph_limits=MultiHopLimits(...)` to `AdaptiveRetriever` (import it from
`scone_memory.retrieval.multihop`). The host expands verified recall seeds using
bounded stored-link reads and exact object-to-subject joins, within the same
space, source filters, session exclusion, and adaptive deadline. With graph
expansion enabled, all candidate sources and facts share the engine clock
boundary; future-created sources are excluded. It verifies expanded source
records before disclosing them to the assessor.

This option prioritizes connected fact components within the adaptive candidate
and byte budgets. Exact components become host-owned atomic groups: partial
model selections or later source loss omit the whole group. These known groups
also survive an assessor failure, so fallback can retain a route gathered before
assessment. Use `group_relations=True` on the self-hosted assessor to let the
model select the components directly. Stored links retain their kinds and
orientation in the separate evidence graph; an exact component is not a claim
of causation or an inferred answer.

`graph_limits=None` keeps expansion disabled. Enable it in the generation
comparison with `--expand-relations`; reports record the graph limits and
per-expansion coverage and work. Graph limits apply to each expansion; the
adaptive round cap bounds their number and the total deadline bounds the run.
`store_calls` counts traversal and its source revalidation; additional adaptive
checks are bounded by the candidate count and deadline. Reaching a hop,
candidate, store-call, node, edge, or byte limit leaves explicit incomplete coverage. Even an exhausted
reachable graph does not establish query completeness. Graph expansion uses
bounded adjacency reads; it does not search beyond the fixed scope or infer unrecorded links. Initial recall retains its existing backend
behavior, including the current fact-seed lookup implementation.

A selected group expands to all of its original evidence IDs. The model-neutral
`EvidenceDecision.selected_groups` contract carries that requirement through
source revalidation and conversation packing: if any member changes or cannot
fit, the whole group is omitted. Independent ungrouped evidence can still be
used. Atomic membership is by evidence ID: separately recalled chunks remain
independent, even when they quote a grouped fact. This does not guarantee atomic
delivery of equivalent source content across representations. Receipts expose
group omissions; a complete group does not prove that the answer is sufficient
or correct. Grouping stays off by default.

The adapter's `max_evidence_bytes` bounds the serialized evidence array including
group metadata and joins. It is separate from the core's input-evidence budget
and the conversation's final context budget; configure all three explicitly.
Add `--group-relations` to the adaptive evaluator command to compare this mode
against ordinary compact paths. Every input record is represented once or
construction fails; no source or group is silently clipped to fit.

See [the executable native example](examples/realtime_conversation.py). It uses
real Scone memory and scheduling with a scripted provider, not live inference.

#### Optional answer review

A complete evidence path does not guarantee that a model follows it correctly.
The native conversation can review its public draft against the delivered memory
packet and attempt one correction:

```python
from scone_memory.providers.answer_reviewer import SelfHostedAnswerReviewer
from scone_memory.realtime.answer_review import AnswerReviewLimits

conversation = TextConversation(
    memory, "authorized-space", "reviewed-session",
    lambda: OpenAICompatibleTextModel(endpoint, model, timeout=45, trust_env=False),
    answer_reviewer=SelfHostedAnswerReviewer(endpoint, model, timeout=20),
    review_limits=AnswerReviewLimits(timeout_s=20.0, max_rounds=2),
    review_policy="report",
    turn_timeout=90,
)
```

The first review may identify unsupported claims, contradictions, incomplete
answers, or broken paths and propose a replacement. Scone adopts that replacement
only after a second review reports it supported. Evidence IDs must come from the
delivered packet, and issue quotations must match the draft exactly. Malformed
reviews never trigger an automatic repair call. The reviewer sees the current
question, draft, and memory packet; it does not receive system instructions or
the full conversation history.

When enabled, review buffers the draft. The observer receives the final text
once; only that text enters conversation history and assistant capture. Without
a reviewer, existing streaming behavior is unchanged. Greetings and other turns
without prepared memory skip this memory-specific review.

`report` retains the original draft if review fails or remains uncertain and its
sources can still be validated. `require_supported` rejects an eligible memory
reply unless review reports support. Both policies reject stale or unavailable
sources. The returned `answer_review` receipt records the outcome, correction,
issue codes, and source status separately; `verified_accuracy` is always false.
A supported review is a model judgment, not independent proof of correctness.

Review uses one deadline, at most two model calls, and a reserve for final source
checks. Preparation and review share the same budget; each bounded source-read
pass also has a one-second cap. Source checks use the original context revision,
fixed scope, exact records, and immutable snapshots. Even an unrelated native
write changes the revision and can conservatively invalidate review. The full
conversation timeout must cover retrieval, generation, review, and capture.

For an isolated comparison, add `--review-model YOUR_INSTALLED_MODEL
--review-timeout 20 --review-policy report` to the generation evaluator. Only the
candidate is reviewed. Reports preserve its public draft, final answer, review
receipt, and separate draft/review timing; gold answers never enter review.
For the standard HTTP server, configure the reviewer explicitly alongside
`SCONE_CONVERSATIONS_JOURNAL` and a text model or persona catalog:

```dotenv
SCONE_ANSWER_REVIEW_POLICY=require_supported
SCONE_ANSWER_REVIEW_URL=http://127.0.0.1:11434/v1
SCONE_ANSWER_REVIEW_MODEL=YOUR_INSTALLED_MODEL
SCONE_ANSWER_REVIEW_TIMEOUT=20
# SCONE_ANSWER_REVIEW_API_KEY=  # Only when your reviewer requires authentication.
```

`off` is the default; `report` and `require_supported` follow the policies above.
The reviewer has its own endpoint, model and optional credential; it never
borrows the chat/extraction key. Configuration does not install a model. The
standard `serve` launcher applies review to custom-model, saved-connection and
persona **text** sessions. Voice sessions are unchanged. The authenticated
conversation capabilities endpoint reports `answer_review.configured` and
`answer_review.policy`, independently of text-model availability.

Reviewed text arrives as one final public delta. Failed required reviews save
no assistant reply and expose a content-free `answer_review` diagnostic in the
current process's turn receipt. Lifecycle failure messages survive restart;
full per-turn review/context receipts are not reconstructed after restart.
When embedding `create_conversation_app` directly, pass a `ConversationReview`
from `scone_memory.runtime.conversation_review`; custom runtime factories must
accept and honor its `answer_reviewer`, `review_policy` and `review_limits`
keywords. The standard server supplies compatible native factories.

The review endpoint must support structured JSON output with `anyOf` and `const`.
Status-specific branches prevent a constrained decoder from returning, for
example, `supported` alongside a proposed correction. Host validation still
checks quotations, evidence IDs and response bounds; schema validity does not
establish that the review judgment is correct.

Evaluate the reviewer separately from answer generation with labeled drafts:

```sh
python -m scone_memory.testing.answer_review_evaluation \
  --fixture tests/fixtures/answer_review/v1.json \
  --output /tmp/scone-review-baseline.json \
  --endpoint http://127.0.0.1:11434/v1 --model YOUR_INSTALLED_MODEL
```

The included original synthetic cases are **development** cases covering direct
facts, invented facts, justified abstention, missed answers, unresolved conflicts
and incomplete routes. Add separately held-out cases before making quality claims.
Each call reviews the original draft once; proposed revisions are recorded only
as a boolean and are not adopted. This isolates reviewer judgment from generation
and correction quality. Labels, categories and splits never enter model inputs.

Reports separate false approvals, false rejections, abstentions and failures,
with summaries by split and category. Approval precision uses approved drafts as
its denominator; false-approval rate uses all labeled unacceptable drafts;
acceptable-answer recall uses all labeled acceptable drafts. Decision coverage
counts only `supported`/`needs_revision`, and labeled agreement counts correct
decisions over **all** observations. An unavailable or uncertain review never
counts as a correct rejection. Missing denominators are `null`, not perfect scores.
The report omits source/draft/revision text and provider error messages. It includes
a SHA-256 of the normalized fixture, labels, timing and enum diagnostics. Output
must be a new file. Only the explicitly selected model is called; an optional
credential comes from `SCONE_ANSWER_REVIEW_API_KEY`.
Timeouts rely on cooperative asynchronous providers; a late result earns no
credit even if a provider swallows cancellation. The evaluator cannot forcibly
interrupt blocking synchronous work inside a custom provider.

#### Optional extractive answers

For memory questions where exact recorded wording matters, a model can select
evidence instead of composing a free-form answer:

```python
from scone_memory.providers.evidence_selector import SelfHostedEvidenceSelector

conversation = TextConversation(
    memory, "authorized-space", "extractive-session",
    lambda: OpenAICompatibleTextModel(endpoint, model, timeout=45, trust_env=False),
    evidence_selector=SelfHostedEvidenceSelector(endpoint, model, timeout=20),
    evidence_answer_timeout=20,
    turn_timeout=90,
)
```

When memory is prepared, Scone builds bounded source cards and makes at most one
selection call. The model returns up to three known card IDs; Scone renders the
original quotations and source IDs. No free-form generator runs on this path.
Complete supplied paths and connected contradictions stay together in a card;
their constituent passages cannot bypass that grouping. Whole cards are omitted
when they exceed a budget. A path records ordered statements and stored link
directions; it does not assert a new transitive relationship or infer an endpoint.
An exact same-episode passage duplicate is removed only after its standalone
claim card fits the budget. Different sources, text, and claims stay separate;
`deduplicated_card_count` is distinct from budget omissions.

Selection cards also carry typed claim triples, their origins and source IDs,
and ordered path steps. This preserves route identity when several claims quote
the same paragraph. A structural object-to-subject match remains distinct from
a stored relationship and its traversal direction. These fields count toward
the evidence byte budget; they do not change the public quotation rendering or
establish that a quotation entails a stored claim. Existing custom selectors
can still construct cards without this optional metadata.

Sources are checked before and after selection against the original scope,
revision, and records. Preparation, selection, and validation share one deadline.
Stale sources, invalid selections, or provider failures suppress the answer. A
valid empty selection returns a fixed no-support message. Only the final rendered
text enters history, capture, and the text callback.

The `evidence_answer` receipt lists delivered cards and evidence IDs, omissions,
and source status. `verified_accuracy` remains false: exact quotations prevent
new model-authored claims, but the model can still select irrelevant or incomplete
evidence. This is an extractive answer mode, not a guarantee of fluent generation
accuracy. Turns without prepared memory use the normal model and streaming flow.
An evidence selector and answer reviewer cannot be enabled together. Custom
selectors implement the `EvidenceSelector` protocol; their lifecycle belongs to
the caller. This native option is not automatically enabled on HTTP routes.

Add `--evidence-selector-model YOUR_INSTALLED_MODEL --evidence-answer-timeout 20`
to the generation evaluator to enable it for the candidate only. Reports separate
context evidence coverage from `selected_evidence_coverage`, record the actual
answer mode, and count free-form generation calls. Gold labels are used only for
scoring after selection. Quoting sources can inflate lexical scores; compare
selection relevance and evidence coverage separately from generative accuracy.

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

## Direct speech providers

Install `pip install 'scone-memory[speech]'` to use the native
`scone_memory.providers.speech.CartesiaSpeech` and `ElevenLabsSpeech` adapters.
They implement `realtime.audio.SpeechSynthesizer` directly; no Pipecat or provider
orchestration SDK is required. Select the provider, model and voice explicitly:

```python
import os
from contextlib import aclosing
from scone_memory.providers.speech import ElevenLabsSpeech

async def speak(text, play_pcm):
    # The caller supplies play_pcm(AudioChunk). No device is opened implicitly.
    async with aclosing(ElevenLabsSpeech(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        model=os.environ["SCONE_SPEECH_MODEL"],
        voice=os.environ["SCONE_SPEECH_VOICE"],
        sample_rate=24000,
    )) as speech:
        async with aclosing(speech.synthesize(text)) as audio:
            async for chunk in audio:
                await play_pcm(chunk)
```

For Cartesia use `CartesiaSpeech` with `CARTESIA_API_KEY`; the constructor options
are the same. Pass a fresh adapter factory to `VoiceSession(tts_factory=...)` or
register it under the exact `(provider, model, voice)` in `ProviderRegistry`.
Keep keys in operator configuration, never persona JSON or browser settings.
Instantiating an adapter makes no request; consuming `synthesize` sends text to
the selected provider and may incur charges. Provider retention and account/model
availability still apply; Scone does not claim zero retention at either provider.

Output is mono signed 16-bit little-endian PCM at the selected sample rate. No
resampling, compression decoding, voice fallback or automatic request retry is
performed. Supported rates are 8000, 16000, 22050, 24000, 44100 and 48000 Hz;
provider/account availability may be narrower. Output chunks default to at most
4096 bytes and the utterance cap defaults to 24 MB. `timeout` limits HTTP I/O
inactivity, not total generation time; `VoiceSession` supplies the turn deadline.
Close each iterator on interruption and the adapter at session end. Only one
utterance may be active per adapter; it can serve successive utterances.

The adapters follow [Cartesia's bytes API](https://docs.cartesia.ai/api-reference/tts/bytes)
(pinned version `2026-08-14`) and [ElevenLabs' streaming API](https://elevenlabs.io/docs/api-reference/text-to-speech/stream).
Transport-contract and native-runtime tests use scripted HTTP peers, not paid
provider calls. These adapters alone do not enable a browser microphone, audio
playback, speech recognition or a complete live voice conversation.

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
