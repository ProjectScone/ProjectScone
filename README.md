# Scone

A memory engine, built the way engines should be built: a Rust library first,
a single-binary CLI + daemon second, a self-hostable API third — all one core.

Scone ingests what you and your AI agents encounter (episodic memory), distills
it into temporal facts that know when they were true (semantic memory), and
returns budgeted, provenance-cited context on demand — fully offline by default,
frontier models when you want them.

Status: M2 complete - episodic AND semantic engines work offline
end-to-end. Episodic: add / search / status / doctor --rebuild, hybrid
BM25+vector recall, local ONNX embeddings, SQLite as the single truth.
Semantic: distill extracts temporal facts (subject-predicate-object with
validity intervals) via a pluggable LLM (Ollama / OpenAI-compatible /
Anthropic / none), contradictions close the old interval with a reason
instead of deleting, and search --as-of answers "what did I believe in
March". M3: scone mcp serves that memory to any MCP agent — memory_store,
memory_recall, memory_facts_about, memory_forget — space-scoped and
input-bounded, with immediate fact distillation when an LLM is configured.
M4: scone serve is a self-hostable multi-user HTTP API (Bearer keys, each
bound to one space), and scone-ffi ships a C ABI (include/scone.h) so any
language can embed the engine. Measured on-device (Apple Silicon, criterion):
recall over a 5k-chunk corpus in ~300 µs, 3.6 ms end-to-end including local
query embedding, ingest at 2.9 ms/note - no datacenter involved.

Give Claude Code persistent memory:

    cargo build --release
    claude mcp add scone -- $PWD/target/release/scone --space myproject mcp

    scone add --note "mark switched from pnpm to bun"
    scone distill                      # facts, via your configured LLM
    scone facts list --all             # history, with closure reasons
    scone search "mark" --as-of 2026-03-15T00:00:00Z   # time travel

    cargo build --release
    scone add --note "changed the oil on the truck"
    scone search "vehicle maintenance"   # semantic hit, fully offline

License: MIT
