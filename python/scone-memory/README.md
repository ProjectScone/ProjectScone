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
| `GET /v1/recall?q&limit&as_of&tags&where&history` | hybrid recall plus the facts that held at `as_of`; `history=true` adds the closed facts that came before them |
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

## Stores and what each one promises

Every store below runs the same 37 contract tests (`tests/test_contract.py`)
and every evidence sink the same 8 (`tests/test_events.py`). "Verified"
says how: embedded means in-process in the test suite and CI; container
means against a real server in Docker locally and as a CI service.

| Store | Documents | Vectors | Evidence | Verified | Notes |
|---|---|---|---|---|---|
| in-memory | yes | yes | yes | embedded | reference implementation |
| SQLite | yes | yes | yes | embedded | FTS5 lexical lane; WAL; schema stamped, one additive step from v5 |
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

Three adapters live under `scone_memory.integrations`; each needs its
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
