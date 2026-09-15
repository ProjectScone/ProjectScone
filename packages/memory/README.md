# scone-memory

Python memory infrastructure for RAG applications and agents. Store original
sources, retrieve scoped evidence, follow recorded relationships, and give a
model the context it needs to answer. Run the engine inside your application or
expose it through HTTP and MCP.

- **Episodes** retain notes, documents, conversation turns and tool output, with
  chunks that point back to the source bytes.
- **Facts** record subject–predicate–object claims, provenance and validity dates,
  including superseded claims and proposals awaiting review.
- **Retrieval** combines lexical and vector search with optional reranking,
  graph traversal, neighboring passages and bounded evidence gathering.
- **Conversations** add model adapters, public streaming, output contracts and
  optional answer review. Model availability and mounted services are separate.

Retrieval scores are ranking signals. Retained citations prove where evidence
came from; they do not establish that a generated answer is correct.

See [named agents and model selection](docs/agent-models.md) for explicit per-agent
LLM choices, scoped tool execution and workflow checkpoint identity.

## Install

Python **3.14** is the primary development runtime. The base library supports
Python 3.10+; real-time conversations require 3.11+.

From this repository's root:

```sh
python -m pip install -e ./packages/memory
# Select the capabilities you need:
python -m pip install -e './packages/memory[api,qdrant,pdf]'
```

The core quickstart needs no database server, model download or API key.
See [installation and embedding choices](docs/getting-started.md) for optional
adapters. Models, service endpoints and credentials are selected by the host.

## Quickstart

```python
import asyncio
from scone_memory import (
    HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine,
)

async def main():
    memory = await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
    ).open()
    try:
        await memory.remember(
            "demo", "Juniper's launch checklist requires a rollback plan.",
            source="notes/launch.txt",
        )
        result = await memory.recall("demo", "launch checklist")
        for item in result.items:
            print(item.text)
    finally:
        await memory.close()

asyncio.run(main())
```

`HashEmbedder` is a deterministic word-overlap fixture, not a semantic model.
Choose provisioned embedding models for semantic retrieval. In-memory stores
are ephemeral; use SQLite or explicitly configured database adapters for
persistence. See [recall, storage and recovery](docs/retrieval-and-storage.md).

## Serve the API

With the `api` extra installed, this starts persistent SQLite memory on loopback:

```sh
export SCONE_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export SCONE_DOCUMENTS=sqlite
export SCONE_VECTORS=sqlite
export SCONE_SQLITE_PATH=./memory.db
export SCONE_EMBEDDER=hash
scone-memory serve
```

The default address is `http://127.0.0.1:7437`. Send the key as a bearer token;
keys select an authorized memory space and role. This configuration does not
start inference or mount a conversation service.

See [HTTP routes and deployment](docs/http-and-deployment.md),
[conversation service setup](docs/conversation-service.md), and the independent
[Python HTTP client](../scone-client/README.md). The
[Webapp](https://github.com/ProjectScone/ProjectScone-Webapp) and
[Rust implementation](https://github.com/ProjectScone/ProjectScone-Rust) have
separate repositories and release lifecycles.

## Architecture and guides

The [architecture guide](ARCHITECTURE.md) describes the engine's boundaries,
storage ports, source validation and resource ownership.

| Task | Guide |
|---|---|
| Choose stores and understand recovery | [Recall and storage](docs/retrieval-and-storage.md), [storage adapters](docs/storage-adapters.md), [S3 catalog](docs/s3-catalog.md) |
| Ingest Office, structured data and text with restart checkpoints | [File ingestion and format coverage](docs/file-ingestion.md) |
| Import WhatsApp, Telegram, Discord and Slack exports as conversations | [Chat exports](docs/chat-exports.md) |
| Reconcile local files, changed revisions and managed deletions | [Incremental directory ingestion](docs/directory-sync.md) |
| Resume interrupted source deletion across stores | [Source cleanup and recovery](docs/retirement-catalog.md) |
| Ingest PDFs, scans and page provenance | [PDF ingestion](docs/pdf-ingestion.md), [PDF OCR](docs/pdf-ocr.md) |
| Preserve images and search attributed context | [Attachments](docs/attachments.md), [image context and entities](docs/image-context.md), [image embedding lane](docs/image-embedding-lane.md) |
| Review duplicate documents without discarding originals | [Document duplicate review](docs/document-deduplication.md) |
| Compose LlamaIndex, LangChain and reranking | [Framework integrations](docs/integrations.md) |
| Give agents scoped search, trace and read tools | [Agent tools](docs/agent-tools.md), [tool-based conversations](docs/conversation-tools.md) |
| Build text sessions and gather missing evidence | [Text conversations](docs/text-conversations.md), [adaptive retrieval](docs/adaptive-retrieval.md), [follow-up queries](docs/followup-queries.md) |
| Constrain, review or quote an answer | [Answer review and output contracts](docs/answer-review.md) |
| Add voice, personas and provider adapters | [Voice conversations](docs/voice-conversations.md) |
| Expose authenticated sessions and retain their lifecycle | [Conversation service](docs/conversation-service.md) |
| Give an OpenAI-compatible app memory by changing its base URL | [OpenAI-compatible chat](docs/openai-compatible-chat.md) |
| Measure retrieval, generation and capacity | [QA experiments](benchmarks/README.md), [scaling validation](docs/scaling-validation.md) |

The repository [.env.example](../../.env.example) lists supported configuration.
Optional integrations are explicit; installing an extra does not provision a
service or establish its availability.

## Measurements and experiments

The [public QA baseline](benchmarks/public-qa-v1.results.md) recorded **600
responses**: each model answered the same **200 original questions**, split
between HotpotQA and SQuAD, over 2,176 source paragraphs. Questions and settings
were frozen before inference; model weights were not fine-tuned for the run.

| Model | Exact match | Token F1 | Inference p50 / p95 |
|---|---:|---:|---:|
| Gemma 4 E4B | 64.0% | 72.9% | 2.29 / 5.88 s |
| Llama 3.2 3B | 59.0% | 68.7% | 1.46 / 3.83 s |
| Llama 3.1 8B | 61.0% | 72.5% | 3.36 / 11.88 s |

These are sampled development-set results, not universal accuracy or an official
leaderboard submission. The report separates source retrieval, context coverage,
answer scores, abstentions and latency. Full protocols, dataset provenance,
follow-up gains **and regressions** remain in the [experiment index](benchmarks/README.md).
For example, [repeated-search compaction](benchmarks/public-search-compaction-v1.results.md)
reduced offered tool bytes **28.11%** while exact match stayed **56.5%** across
200 paired questions; it remains disabled by default.

Recent [storage and parsing measurements](docs/scaling-validation.md):

| Experiment | Recorded result | Scope |
|---|---|---|
| HTML image context | Median 44.45 → 2.90 ms | Fixed 300-image document; seven parser runs |
| Qdrant HNSW effort, `ef=32` → `128` | Recall@10 94.17% → 100%; median 1.862 → 1.975 ms | 20,000 synthetic 64-D vectors; unfiltered queries |
| S3 attachment-link verification | 1 MiB GET → HEAD with zero body bytes | Moto request/byte counts; supported versioned checksums |

Code retrieval, measured with `scone bench-code` on this package's own source
(273 files, 737 documented functions, hash embedder, k=5):

| Asked | Cut at declarations | Cut by length |
|---|---:|---:|
| The docstring, as written | 94% found their own definition | 95% |
| "what &lt;the name&gt; does" | 44% | 44% |
| A returned chunk that is a whole declaration | 40% | 35% |

Read that honestly: **cutting code at its declarations does not find more.** What
it changes is what comes back — a whole function rather than the end of one and
the start of the next — which is what makes a citation quotable, and it is why
every recalled chunk now carries its byte span, its lines and the declaration
holding it. The gap worth closing is the second row: asked in a person's words,
two in five questions never return the function's own definition, and that is an
embedder question rather than a chunker one. The corpus is this repository, so it
is a measurement of behaviour, not a held-out benchmark result.

Synthetic nearest-neighbor recall is separate from semantic retrieval and answer
accuracy. Emulator byte counts are not AWS throughput. No ten-million-vector
capacity result is claimed; the detailed guide retains the validation targets,
filter-specific measurements and limitations.

## Development, security and citation

From the repository root, run the framework tests after installing the test extra:

```sh
python -m pip install -e './packages/memory[test]'
PYTHONPATH=packages/memory/src python -m pytest packages/memory/tests -q
```

The [test category guide](tests/README.md) explains suite organization. The
[contribution guide](../../CONTRIBUTING.md) covers required type checks,
optional backend tests and cross-repository contracts. Test counts describe
software checks; they are not retrieval or generation accuracy scores.

Keep credentials in ignored environment files or host configuration, never in
source, prompts or browser payloads. Restrict storage access by space, configure
transport security beyond loopback, and treat retrieved content as untrusted.
[HTTP and deployment guidance](docs/http-and-deployment.md) explains bearer roles
and explicitly configured services.

Research and academic use must credit **Mark Sturman**, **JudgeHuman**, and
**ProjectScone** under the included [license](LICENSE). See
[MLA/EasyBib, APA, Chicago and BibTeX examples](../../CITING.md), plus
[machine-readable citation metadata](../../CITATION.cff). Exact commits and
individual file references are not required. The license is custom and
MIT-derived, not the unmodified MIT License.
