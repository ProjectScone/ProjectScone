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
  <strong>94.0% Recall@15 on the full LongMemEval-S dataset (all 500 questions, every type) with 97.8% context reduction, fully on-device. No datacenter.</strong><br/>
  <em>(Session-id ground truth, any-evidence; all-evidence 81.0%. Full per-type breakdown in the project ledger. Every number we publish comes from an executed run.)</em>
</p>

---

Your AI forgets everything between conversations, and the memory services
that fix it want your memories in their cloud. Scone fixes both: a complete
memory engine, episodic and semantic, temporal and searchable, that runs
entirely on your machine and embeds in anything.

| | |
|---|---|
| 🧠 **Temporal memory** | Facts extracted from what you store, with validity intervals. A contradiction closes the old fact with a recorded reason, so history stays queryable. Stale facts decay; recalled facts strengthen. |
| 🕰️ **Time travel** | `search --as-of 2026-03-15` answers "what did I believe in March?". Validity is a WHERE clause, not a version-chain walk. |
| 🔍 **Hybrid search** | BM25, vectors, facts, and recency fused in one query, with provenance on every result. Local ONNX embeddings by default; it works on a plane. |
| 🏷️ **Tags** | Tag anything on the way in (`--tag research`), then retrieve only that: papers, a client, one knowledge base. Works on the CLI, MCP, and HTTP surfaces. |
| 👤 **Profiles** | Identity facts + recent activity in one call, on the CLI, MCP, and HTTP surfaces. |
| 📉 **Context economy** | Every recall reports bytes returned vs stored ("97.7% saved"). Token optimization is a product surface, not a benchmark footnote. |
| 📦 **Portable & embeddable** | `scone export` writes JSONL with full fact history. The C ABI (`include/scone.h`) embeds the engine in any language. Zero TypeScript. |

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

The semantic lane uses whatever LLM you configure (Ollama, OpenAI-compatible,
or Anthropic) and pauses loudly when none is set. Episodic search never
needs one. `~/.scone/config.toml`:

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
lands the right session for 90 to 100% of questions in our benchmarks
while an 8B reader converts under half of them into a correct answer, and
the gap is entirely reasoning over evidence, not finding it. Attempts to
close it by asking more of the model backfired: an evidence-chaining
prompt cost 10 points, and splitting the work into extract-then-answer
passes cost 13 and zeroed out temporal questions. Small readers improve
when you shrink their job, not when you add structure to it. For answer
quality beyond that ceiling, point the answer step at a larger model, or
let your coding agent do the reading through the MCP server.

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

Each `--space` is an isolated brain: one per project, per client, per team.

The last two tools are how Scone extracts facts without an API key. Your
agent already reads well and you already pay for it, so it does the
distillation on your existing subscription: `memory_pending` hands it the
episodes, it reasons over them, `memory_store_facts` submits the result.
The engine still owns the invariants; the agent only proposes. Configure
an LLM instead if you want extraction to run unattended.

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
| `Engine::recall` | Hybrid retrieval: facts, cited chunks, context economy |
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

`POST /v1/episodes` · `GET /v1/recall` (`?as_of=`, `?tags=`) ·
`GET /v1/facts` · `POST /v1/facts/{id}/close` · `GET /v1/profile` ·
`GET /v1/status` · `GET /v1/tags`. Every key is bound to exactly one
space, and the server refuses to start keyless.

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
