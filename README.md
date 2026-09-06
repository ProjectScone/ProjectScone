<p align="center"><strong>🥐 Scone</strong></p>

<p align="center">
  <strong>Evidence-grounded memory for humans, agents, and applications. First-class Rust and Python libraries, local execution, and self-hosting.</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/scone-cli">CLI</a> ·
  <a href="https://crates.io/crates/scone-core">Rust library</a> ·
  <a href="python/scone-memory">Python library</a> ·
  <a href="https://crates.io/crates/scone-ffi">C ABI</a> ·
  <a href="https://github.com/DrDrewCain/ProjectScone/releases">Releases</a>
</p>

---

Scone helps software build on what it has already learned: preserve source
material, retrieve relevant evidence, and inspect dated claims and corrections.
The ambition is continuity across assistants, applications, and storage providers
without repeatedly explaining the same context. Portability and consistency are
engineering promises to test, not consequences of storing something as JSON.

Rust and Python are first-class native libraries today. Python also has a
distinct [HTTP client](clients/python). Neither native library requires the
other: there is no automatic Rust loading or silent Python fallback. They share
intended memory semantics, **not yet a common database or full export contract**.
The shared behavioural specification (private for now) records each rule's
status; the [conformance baseline](#cross-language-conformance) tests a limited
episode-transfer profile against both products.

Typical workflows: an agent resumes project work through MCP; a Python pipeline
ingests and recalls scoped evidence; a person inspects memory and corrects an
outdated claim. Explicit team membership and sharing controls remain future work.
Stored claims, including model extractions, are not independently verified truth.

| | |
|---|---|
| 🧠 **Temporal memory** | Claims with validity intervals and recorded closure reasons. Rust and Python both maintain a fact ledger; their backfill behavior is not yet fully conformant. Rust also supports decay and strengthening. |
| 🧮 **Computed answers (Rust)** | Retrieve dated evidence and compute date differences with a derivation. A historical 40-question Rust experiment measured 47.5% versus 37.5% for generated answers; this is not a Python result or general answer-quality claim. |
| 🕰️ **Time travel** | `search --as-of 2026-03-15` selects records valid at that time according to the current ledger. It does not reconstruct what the store believed before later corrections arrived. |
| 🔍 **Hybrid search** | BM25, vectors, facts, and recency fused in one query, with provenance on every result. Local ONNX embeddings by default; it works on a plane. |
| 🏷️ **Tags** | Tag anything on the way in (`--tag research`), then retrieve only that: papers, a client, one knowledge base. Works on the CLI, MCP, and HTTP surfaces. |
| 👤 **Profiles** | Identity facts + recent activity in one call, on the CLI, MCP, and HTTP surfaces. |
| 📉 **Context economy** | Recall reports returned versus stored bytes. Byte reduction is not measured token reduction, and neither establishes answer quality. |
| 📦 **Portable & embeddable** | Native export/import rebuilds derived indexes. Full fact/history transfer between the Rust and Python products is unsupported pending schema/provenance work. The existing C ABI exposes a limited note/recall surface. |

## Use Scone

<table><tr><td width="33%" valign="top">

### 🧑‍💻 I use AI tools

Give Claude Code (or any MCP client) persistent memory across sessions.

**[→ Agent memory](#give-your-ai-memory)**

</td><td width="33%" valign="top">

### 🔧 I'm building

Use the native Rust or Python library, the C ABI subset, or an HTTP client.

**[→ Build with Scone](#build-with-scone)**

</td><td width="33%" valign="top">

### 🖥️ I run my own infra

One binary, one config file, Bearer keys each bound to a space.

**[→ Self-host](#self-host)**

</td></tr></table>

## Quickstart

Install with Homebrew:

    brew install DrDrewCain/scone/scone

Or grab a prebuilt binary from
[Releases](https://github.com/DrDrewCain/ProjectScone/releases). Either
way, no Rust toolchain is needed:

    scone setup claude-code         # also: claude-desktop, cursor, vscode,
                                    # windsurf, zed, codex, opencode, cline
    scone add --note "hello, memory"

Building from source instead:

    cargo install scone-cli

    scone add --note "changed the oil on the truck"
    scone search "vehicle maintenance"      # semantic hit, fully offline
    scone watch ~/notes --once              # ingest a directory
    scone add paper.pdf --tag research      # PDFs become searchable text
    scone add --url https://example.com/post --tag reading
    scone distill                           # extract temporal facts (any LLM)
    scone facts list --all                  # history, with closure reasons
    scone search "tools" --as-of 2026-03-15T00:00:00Z
    scone search "attention" --tag research # narrow to what you tagged
    scone tags                              # tags in this space, with counts
    scone ask "when did I switch deploy targets?"
    scone profile                           # identity + recent activity
    scone status                            # stores, counts, index health
    scone export > memory.jsonl             # your memory is portable

The following CLI commands and configuration are for the Rust product.
Fact extraction can use a configured LLM (Ollama, OpenAI-compatible, or
Anthropic), or your host agent through MCP. Episodic search does not require
an answering/extraction model. `~/.scone/config.toml`:

    [llm]
    provider = "ollama"
    model = "llama3.1:8b"

Running local models on a laptop, two lessons from our own benchmarks:
derive a bounded-context variant (`ollama create llama3.1-ctx8k` from a
Modelfile with `PARAMETER num_ctx 8192`) so an 8B reserves ~6GB instead
of 22GB and stops tripping macOS memory kills, and wrap anything
long-running in `caffeinate -i` so idle sleep cannot end a job hours in.
Reasoning models (Gemma 4 and kin) need thinking disabled for short
extraction calls or they return empty answers under token caps; scone
sends Ollama's `think: false` when configured to.

Expect a local model to retrieve well and answer imperfectly. Retrieval
lands the right session for most questions while a small local reader
converts far fewer of them into correct answers, and the gap is
reasoning over evidence rather than finding it. Two attempts to close it
by asking more of the model, an evidence-chaining prompt and an
extract-then-answer reader, measured as losses of 10 and 13 points.
Those runs sampled at the server's default temperature, and a later
check found that noise alone moves results by about that much, so both
verdicts are withdrawn until they are repeated with sampling pinned.
What has held up under that stricter measurement is the opposite move:
handing the reader fewer, better memories, and doing arithmetic for it
rather than asking it to.

We are not quoting an end-task accuracy at the moment, because the
figure we published was measured with a harness that handed the reader
undated context. A quarter of that benchmark is arithmetic over dates,
so those questions were unanswerable as served, by any reader. The
packaging is fixed and the measurements are being re-run; the number
will come back with the conditions stated. For answer quality beyond a
small model's ceiling, point the answer step at a larger model, or let
your coding agent do the reading through the MCP server.

## See what it remembers

    scone ui        # opens a console at http://127.0.0.1:7438

Search your memory, inspect the claims distilled from it, and close one
that is wrong with a reason attached. Closing does not delete source data;
the claim remains in historical views. Excluding a claim from recall and
deleting its underlying data are distinct operations, not synonyms for closure.
An explicit exclude operation is not implemented yet. The Rust console binds to
loopback only and mints a key that lives as long as the process, so it
authenticates like every other client rather than opening a private
door into the store. It is one file with no build step and no network
calls, so it works on a plane like everything else here.

## Connect what you already write in

    scone connect notion --token secret_abc     # or: github, slack, google-drive
    scone sync                                  # pull everything connected
    scone search "retention policy" --tag notion

| Connector | What it pulls | Credential |
|---|---|---|
| `notion` | Pages you shared with the integration | Internal integration token |
| `github` | Issues and pull requests the token can see | Personal access token |
| `slack` | Messages from channels the bot is in | Bot token |
| `google-drive` | Google Docs, exported as text | OAuth access token |

Tokens are read from `SCONE_<PROVIDER>_TOKEN` first, so nothing has to
touch disk; `connect` otherwise stores them in `~/.scone/connectors.toml`
with 0600 permissions. Each sync is incremental, dated by the source's
own timestamps rather than when the sync ran, and tagged with the
provider so you can retrieve one source at a time. Re-syncing is cheap:
content already stored is recognized and skipped.

## Give your AI memory

    claude mcp add scone -- scone --space myproject mcp
    scone setup claude-code-hooks   # real-time: inject memory each prompt,
                                    # capture the session when it ends

| Tool | What it does |
|---|---|
| `memory_store` | Save an observation; duplicates are recognized, facts distill immediately when an LLM is configured. |
| `memory_recall` | Hybrid recall with your profile prepended, every line dated; `as_of` for time travel, `tags` to narrow it. |
| `memory_facts_about` | What's currently known about an entity (aliases resolved). |
| `memory_pending` | Episodes awaiting fact extraction. Your agent reads them. |
| `memory_store_facts` | Your agent submits what it extracted; the engine applies contradiction closure and provenance. |
| `memory_forget` | Close a fact with your reason. Recorded, never deleted. |

Each `--space` partitions memory: one per project, per client, or per team.
Pass `--space auto` and the name comes from the git remote, so everyone
who clones a repo derives the same space name, whether they cloned over ssh
or https. This does not synchronize separate stores or grant access to a team.
External bearer keys are bound to spaces; metadata filters are not permissions.
Retrieved content is data, never authority to change access or execute tools.

The last two tools are how Scone extracts facts without an API key. Your
agent already reads well and you already pay for it, so it does the
distillation on your existing subscription: `memory_pending` hands it the
episodes, it reasons over them, `memory_store_facts` submits the result.
The engine still owns the invariants; the agent only proposes. Configure
an LLM instead if you want extraction to run unattended.

## Build with Scone

### Rust

    cargo add scone-core

```rust
let mut engine = Engine::open(dir, Box::new(OnnxEmbedder::new(cache)?))?;
let space = auth::resolve(&mut engine, "notes", true)?;
engine.ingest(&space, IngestInput::Note { text: "…".into() })?;
let pack = engine.recall(&space, "what do I know about X", &RecallOpts::default())?;
```

| API | Purpose |
|---|---|
| `Engine::ingest` | Store content: chunked, embedded, indexed, queued for distillation |
| `Engine::recall` | Hybrid retrieval: facts, cited chunks, context economy |
| `Engine::distill` | Drain the queue through your LLM into temporal facts |
| `Engine::profile` | Identity facts + recent activity |
| `Engine::export_jsonl` / `import_jsonl` | Rust JSONL episodes, aliases and fact history; not a shared cross-product archive schema |
| `scone-ffi` | Limited C ABI: open/close, add note, recall JSON, error and string ownership; not every Rust method |

### Python

Install the native library from this checkout (no Rust toolchain required):

```sh
python -m pip install -e './python/scone-memory'
# Add extras such as [api], [mcp], [mongo,qdrant], or [local-embed] as needed.
```

```python
import asyncio
from scone_memory import (
    HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine,
)

async def main():
    memory = await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()
    ).open()
    await memory.remember(
        "project", "Use the documented migration checklist.",
        source="notes://migration-decision",
    )
    result = await memory.recall("project", "migration checklist")
    for item in result.items:
        print(item.source, item.text)

asyncio.run(main())
```

These in-memory stores are ephemeral. `HashEmbedder` is a deterministic
word-overlap baseline, not a semantic model. Choose persistent stores and a
local or remote embedder for your deployment. The [Python library](python/scone-memory)
provides async `MemoryEngine`, a blocking `SyncMemoryEngine`, document/vector/model
protocols, FastAPI, MCP and CLI integrations. Python-to-Python episode/fact
export/import includes provenance-ID remapping and repeat-import tests; missing
source records and differing fact annotations still require care. Database files
are not interchangeable with Rust's store.

The [separate HTTP client](clients/python) uses `from scone import Scone`.
It connects to a server; it is not the native `scone_memory` engine. The servers
have overlapping routes but differ in accepted fields and error behavior.

### Current capability boundaries

| Capability | Rust | Python | Agreement / limitation |
|---|---|---|---|
| Native storage | SQLite + tantivy + usearch | In-memory, SQLite; Mongo document and Qdrant vector adapters | Deliberate deployment choices; performance measured separately |
| Stored chunk offsets | UTF-8 bytes | UTF-8 bytes | Half-open spans; chunk boundaries differ. Pre-release Python stores check schema versions and reject incompatible stores; no automatic migrations |
| Episode kinds | note, file, conversation, observation, connector | Same vocabulary | Tested transfer subset: note, file, connector; chat/web are rejected, not aliased |
| Temporal facts | Intervals, reasons, extraction, computation | Intervals, reasons, extraction, backfill tests | No full temporal parity claim; origin/review remains incomplete |
| Scope / filtering | Spaces and tags | Spaces, tags, metadata `where` | Metadata is filtering, not an authorization boundary |
| Local embedding | ONNX; persisted model ID/dimension pin | ONNX; adapter dimension checks | Same-width model identity enforcement remains a Python gap |
| HTTP / MCP / CLI | Available | Available | Overlap is not full wire/API conformance |
| C ABI | Available, limited surface | No native dependency | Broader bindings require a measured justification |
| Full archive exchange between products | Unsupported | Unsupported | Different hashes, provenance, statuses and fields |
| Retrieval-quality measurements | Historical Rust runs below | No corresponding baseline reported here | Never attribute Rust results to Python |

### Cross-language conformance

The [shared fixture](crates/scone-core/tests/fixtures/episodes-v1.json) contains
literal expected Unicode content, source, kind and canonical UTC millisecond
timestamps. Native tests check scoped dedup and stored byte spans; cross-runtime
tests transfer actual exports in both directions and into nonempty destinations.
They do not require identical hashes, local IDs, chunk boundaries or rankings.

```sh
cargo test -p scone-core --test shared_contract
cargo build -p scone-core --example episode_roundtrip
cd python/scone-memory
SCONE_TEST_RUST_ROUNDTRIP="$(pwd)/../../target/debug/examples/episode_roundtrip" \
  .venv/bin/pytest -q tests/test_cross_language.py
```

Install the Python test dependencies and optional Qdrant adapter first. Without
`SCONE_TEST_RUST_ROUNDTRIP`, cross-runtime cases explicitly skip; native Python
cases still run. The example is **test-only**, not a migration tool: it refuses
facts, aliases, nonempty tags/metadata, unsupported kinds and unknown fields.
This narrow profile is not general lossless export compatibility. Packaged
installation and OS/Python-version matrices remain separate verification work.

## Self-host

    # ~/.scone/config.toml
    [server]
    listen = "127.0.0.1:7437"
    [[server.keys]]
    key = "sk-alice"
    space = "alice"

    scone serve

`POST /v1/episodes` · `GET /v1/recall` (`?as_of=`, `?tags=`) ·
`GET /v1/facts` · `POST /v1/facts/{id}/close` · `GET /v1/profile` ·
`GET /v1/status` · `GET /v1/tags`. Every key is bound to exactly one
space, and the server refuses to start keyless.

## How it works

In Rust, SQLite is the source of truth; tantivy (BM25) and usearch (HNSW) are
derived, rebuildable indexes (`scone doctor --rebuild`). Ingestion is two
lanes: episodic (synchronous, offline-complete) and semantic (async LLM
distillation that never blocks a write). Four invariants are property-tested:
chunks reassemble exactly; no two active facts share subject+predicate;
contradiction closes intervals, never deletes; every fact carries provenance.

Python orchestrates document stores, vector indexes and embedders through
protocols. Compensating rollback across stores is not a crash-atomic transaction;
deletion recovery, cancellation and concurrent operations need stronger tested
guarantees. Rust's rebuildable indexes have their own persistence/recovery model.
Neither architecture alone proves the other one's reliability.

## Measurement status

Historical **Rust-only** baseline: LongMemEval-S session-level all-evidence
Recall@15 81.0%, any-evidence 94.0%, byte context reduction 97.8%. The run used
all 500 questions, including 30 abstention items that the
[official retrieval evaluator](https://github.com/xiaowu0162/LongMemEval)
excludes. It is not directly comparable to the official 470-case metric,
not an answer-accuracy result, and not a Python benchmark or leaderboard ranking.
These are previously recorded results, not reruns of the current commit.

Historical Rust measurement on Apple Silicon (criterion): recall ~300 µs over 5k chunks,
3.6 ms end-to-end including local query embedding, ingest 2.9 ms/note.
Python latency, memory use and retrieval quality must be measured separately
under comparable workloads. Future reports must record dataset/model settings,
exclusions and failures, with retrieval, answering, abstention, measured tokens,
bytes, latency, storage and ingestion cost reported separately.

## License

MIT. Built by studying what came before and keeping the receipts.
