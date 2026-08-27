# Scone

A memory engine, built the way engines should be built: a Rust library first,
a single-binary CLI + daemon second, a self-hostable API third — all one core.

Scone ingests what you and your AI agents encounter (episodic memory), distills
it into temporal facts that know when they were true (semantic memory), and
returns budgeted, provenance-cited context on demand — fully offline by default,
frontier models when you want them.

Status: M1 complete - the episodic engine works offline end-to-end:
scone add / search / status / doctor --rebuild, hybrid BM25+vector recall
with local ONNX embeddings (bge-small-en-v1.5), SQLite as the single source
of truth. Next: M2, the semantic lane (temporal facts, contradiction
closure, decay). See docs/superpowers/specs/ for the design spec and
memory/ for the predecessor study that informs it.

    cargo build --release
    scone add --note "changed the oil on the truck"
    scone search "vehicle maintenance"   # semantic hit, fully offline

License: MIT
