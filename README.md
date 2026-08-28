<p align="center"><strong>🥐 Scone</strong></p>

<p align="center">
  <strong>Local-first memory engine for humans and AI agents. Your memory, on your machine, at memory speed.</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/scone-cli">CLI</a> ·
  <a href="https://crates.io/crates/scone-core">Engine crate</a> ·
  <a href="https://crates.io/crates/scone-ffi">C ABI</a> ·
  <a href="https://github.com/DrDrewCain/ProjectScone/releases">Releases</a>
</p>

<p align="center">
  <strong>98% Recall@15 on LongMemEval-S with 97.7% context reduction — measured fully on-device, 135 ms p50, no datacenter.</strong><br/>
  <em>(50-item run, session-id ground truth; full-dataset verification in progress — every number we publish comes from an executed run.)</em>
</p>

---

Your AI forgets everything between conversations, and the memory services
that fix it want your memories in their cloud. Scone fixes both: a complete
memory engine — episodic + semantic, temporal, searchable — that runs
entirely on your machine and is embeddable in anything.

| | |
|---|---|
| 🧠 **Temporal memory** | Facts extracted from what you store, with validity intervals. Contradictions close the old fact with a reason — history stays queryable. Stale facts decay; recalled facts strengthen. |
| 🕰️ **Time travel** | `search --as-of 2026-03-15` answers "what did I believe in March?" — validity is a WHERE clause, not a version-chain walk. |
| 🔍 **Hybrid search** | BM25 + vectors + facts + recency fused in one query, with provenance on every result. Local ONNX embeddings by default — works on a plane. |
| 👤 **Profiles** | Identity facts + recent activity in one call, on the CLI, MCP, and HTTP surfaces. |
| 📉 **Context economy** | Every recall reports bytes returned vs stored ("97.7% saved") — token optimization is a product surface, not a benchmark footnote. |
| 📦 **Portable & embeddable** | `scone export` → JSONL with full fact history. C ABI (`include/scone.h`) embeds the engine in any language. Zero TypeScript. |

## Use Scone

<table><tr><td width="33%" valign="top">

### 🧑‍💻 I use AI tools

Give Claude Code (or any MCP client) persistent memory across sessions.

**[→ Agent memory](#give-your-ai-memory)**

</td><td width="33%" valign="top">

### 🔧 I'm building

Embed the engine as a Rust crate or through the C ABI; or call the HTTP API.

**[→ Build with Scone](#build-with-scone)**

</td><td width="33%" valign="top">

### 🖥️ I run my own infra

One binary, one config file, Bearer keys each bound to a space.

**[→ Self-host](#self-host)**

</td></tr></table>

## Quickstart

    cargo install scone-cli

    scone add --note "changed the oil on the truck"
    scone search "vehicle maintenance"      # semantic hit, fully offline
    scone watch ~/notes --once              # ingest a directory
    scone distill                           # extract temporal facts (any LLM)
    scone facts list --all                  # history, with closure reasons
    scone search "tools" --as-of 2026-03-15T00:00:00Z
    scone profile                           # identity + recent activity
    scone export > memory.jsonl             # your memory is portable

The semantic lane uses whatever LLM you configure — Ollama, OpenAI-compatible,
or Anthropic — and pauses loudly when none is set; episodic search never
needs one. `~/.scone/config.toml`:

    [llm]
    provider = "ollama"
    model = "llama3.1:8b"

## Give your AI memory

    claude mcp add scone -- scone --space myproject mcp

| Tool | What it does |
|---|---|
| `memory_store` | Save an observation; duplicates are recognized, facts distill immediately when an LLM is configured. |
| `memory_recall` | Hybrid recall with your profile prepended; `as_of` for time travel. |
| `memory_facts_about` | What's currently known about an entity (aliases resolved). |
| `memory_forget` | Close a fact with your reason — recorded, never deleted. |

Each `--space` is an isolated brain: one per project, per client, per team.

## Build with Scone

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
| `Engine::recall` | Hybrid retrieval → facts + cited chunks + context economy |
| `Engine::distill` | Drain the queue through your LLM into temporal facts |
| `Engine::profile` | Identity facts + recent activity |
| `Engine::export_jsonl` / `import_jsonl` | Full-fidelity portability |
| `scone-ffi` | The same engine via C ABI, from any language |

## Self-host

    # ~/.scone/config.toml
    [server]
    listen = "127.0.0.1:7437"
    [[server.keys]]
    key = "sk-alice"
    space = "alice"

    scone serve

`POST /v1/episodes` · `GET /v1/recall` · `GET /v1/facts` ·
`POST /v1/facts/{id}/close` · `GET /v1/profile` · `GET /v1/status` — every
key is bound to exactly one space; the server refuses to start keyless.

## How it works

SQLite is the single source of truth; tantivy (BM25) and usearch (HNSW) are
derived, rebuildable indexes (`scone doctor --rebuild`). Ingestion is two
lanes: episodic (synchronous, offline-complete) and semantic (async LLM
distillation that never blocks a write). Four invariants are property-tested:
chunks reassemble exactly; no two active facts share subject+predicate;
contradiction closes intervals, never deletes; every fact carries provenance.

Measured on Apple Silicon (criterion): recall ~300 µs over 5k chunks,
3.6 ms end-to-end including local query embedding, ingest 2.9 ms/note.

## License

MIT. Built by studying what came before and keeping the receipts.
