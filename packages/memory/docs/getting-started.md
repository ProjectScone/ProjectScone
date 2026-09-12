# Installation and in-process memory

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

## scone-memory

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
  reason; asking about a past date returns what was true then. The order
  facts arrive in never decides what held. A claim stated again from a
  later day is kept as an affirmation of the fact that holds. If a late
  backfill then cuts that fact short, the claim resumes from the day it
  was stated again.

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

See [the architecture guide](../ARCHITECTURE.md) for the engine's component boundaries,
storage ports, source-validation rules and lifecycle guarantees.

[Document duplicate review](../docs/document-deduplication.md) detects copied
passages and optional embedding-based paraphrases using existing memory indexes.
Reports include source spans, review notifications, and explicit search projections
for keeping content, suppressing copied passages, or excluding a document while
retaining its original. Semantic inspection defaults on and can be disabled.

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

Scone's ONNX embedding and offline reranking paths set `ORT_DISABLE_TELEMETRY=1`
before importing the runtime and disable telemetry through its API. If your host
imports ONNX Runtime or FastEmbed first, export that variable before starting
the process. A later API call cannot undo earlier initialization events; see
[ONNX Runtime's telemetry documentation](https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md).

## Ask the graph

Facts form an entity graph that is read, not guessed: every line an answer
gives cites the facts behind it, read again before it is shown.

```python
import asyncio
from scone_memory import MemoryEngine, InMemoryDocumentStore, InMemoryVectorIndex, HashEmbedder
from scone_memory.entities.changes import graph_changes
from scone_memory.entities.context import graph_context
from scone_memory.entities.match import graph_match

async def main():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.assert_fact("default", "alice chen", "works_at", "Acme Robotics", valid_from="2020-01-01T00:00:00Z")
    await engine.assert_fact("default", "alice chen", "works_at", "Globex", valid_from="2024-03-01T00:00:00Z")
    await engine.assert_fact("default", "globex", "based_in", "Lisbon", valid_from="2019-01-01T00:00:00Z")
    # What the graph records around a name, one cited line per item.
    print((await graph_context(engine, "default", names=["alice chen"])).text)
    # Who works somewhere based in Lisbon?
    found = await graph_match(engine, "default", [
        {"subject": "?who", "predicate": "works_at", "object": "?org"},
        {"subject": "?org", "predicate": "based_in", "object": "Lisbon"}])
    print(found.text)
    # What changed since the start of 2024?
    print((await graph_changes(engine, "default", since="2024-01-01T00:00:00Z")).text)

asyncio.run(main())
```

Among what it prints:

```
hop 1: alice chen works_at Globex [fact 2]
hop 2: Globex based_in Lisbon [fact 3]
row: ?who = alice chen (person) ent:…; ?org = Globex (organisation) ent:… [facts 2, 3]
moved: alice chen works_at Acme Robotics → Globex [facts 1, 2]
```

The same reads are HTTP routes (`/v1/graph/context`, `/v1/graph/match`,
`/v1/graph/changes`), MCP tools and `scone graph` commands. See
[Retrieval and storage](retrieval-and-storage.md) for every graph route,
the report, the drawings and their bounds.

## PDF ingestion

The optional `pdf` extra adds native text-layer PDF ingestion, retained originals,
and page provenance for search spans. See [PDF ingestion](../docs/pdf-ingestion.md)
for the API, limits and coverage labels. The optional `pdf-ocr` extra adds
[scanned PDF ingestion](../docs/pdf-ocr.md) with an explicitly configured Tesseract
recognizer, retained word regions and no model downloads or generative LLM.

## Image context and entities

Index alt text, captions and attributed metadata beside retained images, with
explicit entity IDs, aliases and evidence references. Search returns the original
image reference and its source context without invoking a vision model.
See [image context and entity retrieval](../docs/image-context.md) for the native and
HTTP APIs, HTML extraction, provenance, and validation limits.

See [storage adapters](../docs/storage-adapters.md) for OpenSearch, ElastiCache and
AWS integration boundaries, and [scaling validation](../docs/scaling-validation.md)
for measured performance, configurable Qdrant search effort and capacity targets.
