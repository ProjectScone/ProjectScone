# Working on Scone

Rust and Python are first-class products. Keep their documented behavioral
promises aligned without assuming identical implementations or performance.

## Repository layout

```text
crates/
  scone-core/       Rust engine and native conformance tests
  scone/            Rust CLI, HTTP/MCP integrations and console
  scone-ffi/        C ABI subset and ownership/error tests
  scone-bench/      Rust evaluation harness
python/
  scone-memory/    Native async/sync engine, adapters, API/MCP and console
  scone-client/    HTTP client (distribution: scone-client; import: scone)
scripts/           Development utilities, including mutation proofs
.github/workflows/ CI and release workflows
```

The shared episode fixture lives inside `crates/scone-core/tests/fixtures/` so
it ships with the Rust source package. Python checkout tests consume that same
corpus. Normal Python library use does not require Rust or this fixture.

Build outputs (`target/`, package `build/`, `dist/`, virtual environments and
caches) are ignored. Local research/design notes (`memory/`, `docs/`), datasets
and benchmark results are private and ignored; do not force-add them. They are
not runtime dependencies. Never move an active database or benchmark output as
part of source-layout cleanup.

## Rust checks

From the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Some tests use local embedding models and may need an initial download. For
the deterministic episode contract, no model API/download is needed:

```sh
cargo test -p scone-core --test shared_contract --test portability
cargo build -p scone-core --example episode_roundtrip
```

Use debug builds during normal development. Coordinate release builds and
resource-heavy benchmarks with anyone already using the machine.

## Native Python checks

```sh
cd python/memory
python -m venv .venv
.venv/bin/python -m pip install -e '.[test,qdrant]'
.venv/bin/python -m pytest -q
```

For actual cross-runtime transfer, first build the Rust example above, then:

```sh
SCONE_TEST_RUST_ROUNDTRIP="$(pwd)/../../target/debug/examples/episode_roundtrip" \
  .venv/bin/python -m pytest -q tests/test_cross_language.py
```

Without that variable, the cross-runtime cases explicitly skip. To exercise
MongoDB or Qdrant **server** fixtures, install the relevant extras and provide
`SCONE_TEST_MONGO_URL` / `SCONE_TEST_QDRANT_URL` for disposable test services.
These tests create and clean test databases/collections: never use production
credentials or endpoints. Qdrant local mode is not server performance evidence.

## Python HTTP client checks

```sh
cd python/scone-client
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q -m 'not integration'
```

Integration tests start a real Rust server and may compile a release binary;
inspect their setup and coordinate resources before running them. The directory
move does not change the distribution name `scone-client` or `from scone import
Scone`.

## Changes and measurement

Agree on a small milestone and explicit file ownership before parallel edits.
Preserve unrelated changes; stage explicit files, not whole shared directories.
Run affected native, API/client and package checks. New adapters must pass
behavioral tests or document unsupported semantics; implementing a protocol is
not enough. Derive expected results independently and prove important tests
fail when the guarded behavior is removed (`scripts/prove-test.sh`), preferably
in an isolated copy when another agent is editing the engine.

Separate vector-neighbor recall, evidence retrieval, answering and abstention.
Compare speed at matched retrieval quality; retain raw timings, resource limits,
dataset/model versions and failures. Report Rust and Python separately. Byte
reduction is not measured token reduction. Do not publish, deploy, spend on
services, or destructively clean user data without the agreed authorization.
