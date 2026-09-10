# Framework tests

Run the complete framework suite from `packages/memory`:

```sh
python -m pytest -q tests
python -m mypy --config-file mypy.ini src/scone_memory
```

Tests are grouped by the behavior they cover. Every category is collected by the
full command above; the directories do not separate fast tests from integration
tests or silently exclude service checks.

| Directory | Coverage |
|---|---|
| `agents/` | Tool execution, evidence selection, answer review and orchestration |
| `api/` | HTTP authentication, request validation, response contracts and routes |
| `backends/` | Store adapters, filtering, indexing, integrity and recovery |
| `benchmarks/` | Evaluation harness correctness and scoring contracts |
| `conversations/` | Text sessions, durable history, streaming and memory context |
| `ingestion/` | Documents, images, OCR, parsing, deduplication and workers |
| `integrations/` | Package boundaries, external interfaces and examples |
| `media/` | Audio, voice pipelines and telephony transports |
| `memory/` | Shared engine contracts, facts, events, retention and lifecycle |
| `providers/` | Model/embedding transports, rerankers and authentication |
| `retrieval/` | Hybrid search, graph traversal, adaptive retrieval and context |
| `runtime/` | Configuration, launchers, CLI commands and diagnostics |

For a targeted run, use a category or file:

```sh
python -m pytest -q tests/ingestion
python -m pytest -q tests/backends/test_qdrant_index_recovery.py
```

`conftest.py` supplies the shared backend matrix. `fixtures/` retains common input
data; `paths.py` resolves those files and repository fixtures independently of
category depth. Cross-category helpers use explicit relative imports, so running
one category does not rely on another category being collected first.

Live-service tests require explicit `SCONE_TEST_*` endpoints; absent services are
skipped. Set only endpoints for provisioned test instances. Fixtures use isolated
namespaces and clean up their own data. A passing mock contract is not a live
service compatibility or performance result.

Benchmark harness tests belong here; experiment results and their methods are
documented in the [scaling report](../docs/scaling-validation.md) and
[package documentation](../README.md). Test pass counts are not retrieval recall
or generated-answer accuracy.
