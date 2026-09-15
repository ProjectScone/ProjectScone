# HTTP API, deployment and tests

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

## Server

```sh
export SCONE_API_KEY=change-me
export SCONE_DOCUMENTS=mongo  SCONE_MONGO_URL=mongodb://localhost:27017
export SCONE_VECTORS=qdrant   SCONE_QDRANT_URL=http://localhost:6333
export SCONE_EMBEDDER=local
scone-memory serve
```

The [`scone-client`](../../scone-client) package talks to the framework's HTTP
API. Shared wire contracts with the independent Rust implementation are tested
separately; this is not a claim of complete cross-runtime feature parity.
The bearer key decides the space; a key cannot read outside its authorized space.
A third field gives a key a role:
`SCONE_API_KEYS=r:default:read,w:default:write,v:default:review,f:default:full`.
`read` only reads; `write` remembers, links and forgets but never decides a
claim; `review` approves, declines, excludes and includes but never adds;
`full` (the default, and always what `SCONE_API_KEY` gets) does everything.
A refused request is a 403 naming the role. The conversation service holds
the same rule, including the audio socket's hello.

| Route | What |
|---|---|
| `POST /v1/documents/pdf` · `GET /v1/episodes/{id}/pdf?chunk_id=...` | Parse an uploaded PDF text layer and inspect retained page evidence; see [PDF ingestion](pdf-ingestion.md) for dependency capabilities and bounds |
| `POST /v1/episodes` | remember; `{content, tags?, source?, created_at?, kind?}`; unknown fields are refused |
| `DELETE /v1/episodes/{id}` | forget |
| `GET /v1/spaces/{space}/impact` · `DELETE /v1/spaces/{space}?confirm={space}` | delete a whole space (`scone-memory delete-space --confirm <space>`, `--dry-run` for the preview): the preview says what would go, the deed removes it, in order, attachment holds (bytes only when no other space holds them), vectors, chunks, episodes, links, claims, events, tombstones and the revision; it needs the `full` role and the name repeated, and afterwards the key answers 404 on every route, so a key left in config cannot re-create what was erased |
| `GET /v1/recall?q&limit&as_of&tags&where&history&kind&source_prefix&since&until` | hybrid recall plus the facts that held at `as_of`; `history=true` adds the closed facts that came before them; `kind`, `source_prefix` (literal text), `since` and `until` (inclusive) narrow the candidates the way the Rust engine does |
| `GET /v1/facts?all&as_of` · `POST /v1/facts` · `POST /v1/facts/{id}/close` | the fact ledger |
| `POST /v1/consolidate` `{scope: distill \| derive}` | one consolidation pass by hand over this key's space: `distill` runs the worker's pass (extraction, retention and, with `SCONE_DERIVE=1`, derivation), `derive` only the derivation pass, which proposes claims that follow from the claims held, each with its premises as `derived_from` links and no quote (`scone-memory derive`); 501 without a model. `GET /v1/status` carries `pending_derivation` (groups not yet sent at their current membership) and `derivation` on/off |
| `POST /v1/openai/chat/completions` | chat completions with memory: recalls from the key's space with the latest user message, injects the passages as one delimited system block, answers with the configured model (`SCONE_CHAT_URL`, `SCONE_CHAT_MODEL`) and keeps the turn; a `scone` field names what was recalled, injected and kept; `stream: true` is refused with a 422 in this version; see [OpenAI-compatible chat](openai-compatible-chat.md) |
| `GET /v1/profile` · `GET /v1/tags` · `GET /v1/status` · `GET /healthz` | overviews; the profile's `static_facts` are the claims that hold now (one not yet valid, ended, or excluded stays out) and its `recent` is `dynamic` with its evidence, one `{episode_id, excerpt, created_at}` per entry, most recent by the episode's own time first, the same rule and shape the Rust engine serves |

Errors are `{"error": "..."}` with 401, 404 or 422.

## Running it as a service

```sh
docker build -t scone-memory .                      # server image; add --build-arg EXTRAS=api,mongo,qdrant,postgres,local-embed for the ONNX embedder
SCONE_API_KEY=change-me docker compose up --build   # MongoDB + Qdrant + scone-memory on :7437, data in named volumes
scripts/compose-smoke.sh                            # brings the stack up, round-trips an episode, tears it down
```

The image refuses to start without `SCONE_API_KEY`. Its `/data` volume
holds the SQLite file when no database is configured, and the embedder's
model cache.

## Tests

```sh
python -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest
SCONE_TEST_MONGO_URL=mongodb://localhost:27017 SCONE_TEST_QDRANT_URL=http://localhost:6333 .venv/bin/pytest
```

Behavioural tests are proven to fail before they are trusted:
`PROVE_RUNNER=".venv/bin/pytest -q" ../../scripts/prove-test.sh <file> <needle> <replacement> <test>`
breaks the code, requires red, restores, requires green.


## Contributing and citation

See the framework [contribution guide](../../../CONTRIBUTING.md),
[citation formats](../../../CITING.md), and [citation metadata](../../../CITATION.cff).
Research and academic use must credit Mark Sturman, JudgeHuman and ProjectScone
as required by the included [license](../LICENSE).
