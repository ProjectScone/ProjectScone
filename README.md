# Scone

A memory engine, built the way engines should be built: a Rust library first,
a single-binary CLI + daemon second, a self-hostable API third — all one core.

Scone ingests what you and your AI agents encounter (episodic memory), distills
it into temporal facts that know when they were true (semantic memory), and
returns budgeted, provenance-cited context on demand — fully offline by default,
frontier models when you want them.

Status: pre-alpha, under active design. See docs/superpowers/specs/ for the
design spec and memory/ for the knowledge base that informs it.

License: MIT
