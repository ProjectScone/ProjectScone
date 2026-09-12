# scone-client

A thin Python client for the [Scone](https://github.com/ProjectScone/ProjectScone)
HTTP API. One class, `requests` for transport, dataclasses instead of dicts.

## Quickstart

```python
from scone import Scone

with Scone("http://127.0.0.1:7437", "sk-your-key") as memory:   # or $SCONE_URL / $SCONE_API_KEY
    memory.add("the deploy checklist lives in the wiki", tags=["ops"])
    for item in memory.recall("checklist", limit=5, tags=["ops"]):
        print(item.day, item.text)
```

Start the server it talks to with `scone serve` (which needs a `config.toml`
carrying at least one `[[server.keys]]` entry, since a key is bound to
exactly one space).

## API

| Method | Endpoint | Returns |
| --- | --- | --- |
| `add(text, tags=None, source=None, created_at=None)` | `POST /v1/episodes` | `Added(episode_id, deduplicated, chunks)` |
| `recall(query, limit=None, as_of=None, tags=None)` | `GET /v1/recall` | `Recall` (iterable over `Memory`) |
| `facts(include_closed=False)` | `GET /v1/facts` | `list[Fact]` |
| `close_fact(fact_id, reason)` | `POST /v1/facts/{id}/close` | `int` (the closed id) |
| `profile()` | `GET /v1/profile` | `Profile(static_facts, dynamic)` |
| `status()` | `GET /v1/status` | `Status` |
| `tags()` | `GET /v1/tags` | `list[Tag]` |

Every non-2xx answer raises `SconeError`, carrying `.status` (the HTTP code,
or `None` when the request never reached a server) and `.message` (the
server's `{"error": ...}` text, or the plain-text body axum's own extractor
rejections use).

## Response limits

`Scone(..., max_response_bytes=16 * 1024 * 1024)` bounds decoded response bytes
before JSON parsing, including error bodies. Set a larger positive integer when
a known document or agent result needs it. An oversized response raises
`SconeError` with its HTTP status and no retained body. Receiving that error does
not tell you whether a preceding write completed; inspect the durable resource
before deciding to resume or retry it. The client adds no automatic retries.

The transport streams encoded bytes into a bounded buffer and advertises only
`gzip, deflate`. It accepts identity, gzip (up to 1024 members), and wrapped or raw
deflate; other encodings, truncated compressed bodies, and invalid trailers are
refused. Compressed wire bytes are capped at twice `max_response_bytes` plus
64 KiB, and decompression has its own output cap. These are byte limits, not a
limit on the Python objects subsequently created by JSON parsing. The configured
`timeout` still controls connection and idle read timeouts, not a total deadline.

The client sets `Accept-Encoding: gzip, deflate` after copying caller headers;
caller-supplied values cannot opt into an unsupported coding. Successful JSON
responses are not also decoded into an unused error-body string.

Each response is closed, including failed reads. An injected `requests.Session`
remains caller-owned. If a custom adapter or response hook already buffers or
decodes the body, its earlier allocation is outside these bounds; the cached
body is checked before the client parses it. Custom session retry policies are
also caller-controlled.

## Server behavior worth knowing

- **Source and event time are supported.** `add(..., source=..., created_at=...)`
  sends both fields to the Rust server; recall returns the preserved source and
  event date. They are optional, not fabricated when omitted.
- **Deduplication is not an error.** Storing identical text twice answers
  `200` with `deduplicated: true` and no `chunks`, instead of `201`.
- **Profile facts are narrower.** `/v1/profile` omits `valid_from`,
  `valid_until`, and `status`, so those are `None` on a `Fact` from there.
- **Engine errors have HTTP categories.** Missing facts return `404`, invalid
  engine input returns `422`, and unrecognized keys return `401`. Transport and
  extractor errors can have other shapes; the client preserves their status
  and handles plain-text bodies. A blank query is also refused locally.

## Install

```sh
pip install -e ".[test]"
```

## Tests

```sh
python -m pytest
```

Unit tests drive the client against a stub `http.server` in a thread. The
integration tests build `scone-cli` in release mode, start a real
`scone serve` on a free port with a temp data dir, and run a full round trip;
they skip themselves when `cargo` is unavailable or the build fails. Set
`CARGO_TARGET_DIR` to build somewhere other than `<repo>/target`.

```sh
python -m pytest -m "not integration"   # unit tests only
```


## Contributing and citation

See the framework [contribution guide](../../CONTRIBUTING.md),
[citation formats](../../CITING.md), and [citation metadata](../../CITATION.cff).
Research and academic use must credit Mark Sturman, JudgeHuman and ProjectScone
as required by the included [license](LICENSE).

## Agent workflow resources

The independently installed client includes typed catalog, plan, and run
resources for hosts that advertise the corresponding `agents.*` capabilities.
`expected_space` verifies responses and mutation admission; it does not override
the space assigned to the bearer key.

```python
from scone import Scone, ModelTask, TaskPlan

with Scone("http://127.0.0.1:7437", api_key="your-space-key") as memory:
    agents = memory.agents(expected_space="alpha")
    choices = agents.catalog()
    # Use agent/model IDs actually returned by this host's catalog.
    plan = TaskPlan("research", (
        ModelTask("answer", "researcher", "local-careful", "Answer with evidence."),
    ))
    saved = agents.save_plan(plan, expected_revision=0)
    progress = agents.start("research-1", plan=saved, question="What changed?")
    original = agents.request("research-1")
    agents.status("research-1").match(original)
```

`agents.plans()` and `agents.runs()` return one bounded page with an explicit
`next_after` cursor. `agents.plan(id)`, `agents.policy()`, and
`agents.cancel(run_id)` provide inspection and cancellation. `HumanInput` tasks
and `HandoffPlan`/`HandoffAgent` preserve their native plan formats and explicit
model choices. Resource construction is local; reads and plan saves do not
execute a model. Only an explicit `start` submits a run. Errors never trigger an
automatic retry, resume, or alternate model selection.

New workflow response models reject malformed scalars, mismatched identities,
and changed acknowledgement bindings. The client preserves server HTTP errors
and refuses redirects. JSON is encoded as UTF-8 so valid multibyte inputs do not
expand into ASCII escapes beyond the server's request budget. Python 3.9 remains
supported, and the distribution includes `py.typed` for static type checking.

Human inputs use separate reply and execution operations:

```python
pending, = agents.inputs("run-1")
answered = agents.respond(pending, response="Proceed with the careful analysis.")
# Persist this continuation ID and selected reply for an explicit retry.
status = agents.continue_run(
    "run-1", continuation_id="approval-1", responses=(answered,),
)
```

A reply alone never resumes a run. The client checks fresh input identities
before each write and confirms the selected activation after continuation.
Transport failures and uncertain acknowledgements raise `SconeError`; the client
never retries a write automatically. Callers can read `inputs()` and `status()`
without execution, then explicitly retry the original response or continuation
ID and selection. A successful continuation confirms admission, not completion.
The native server remains responsible for atomic revision and source checks.

For a real local agent API contract test, point `SCONE_TEST_NATIVE_PYTHON` at
an interpreter with the repository's native framework and API dependencies:

```sh
SCONE_TEST_NATIVE_PYTHON=/path/to/native/python python -m pytest -q tests/test_native_agents.py
```

The test launches an isolated loopback server, restarts it after saving a reply,
and verifies that only the explicitly selected model executes after continuation.
It does not contact a model provider or the live memory service.

Durable document jobs retain their original upload and extraction request:

```python
jobs = client.document_jobs(expected_space="alpha")
formats = jobs.formats()
original = jobs.upload(b"# Notes\n\nAda studies stars.", media_type="text/markdown")
status = jobs.start("import-1", attachment_id=original.attachment_id, filename="notes.md")
request = jobs.request("import-1")
status = jobs.status("import-1")
# Once completed, read currently verified original/manifest/episode identities.
result = jobs.result("import-1")
```

Callers choose when to poll. Starting an existing import only returns its status;
it never implicitly resumes a stopped attempt. For an explicit recovery, read the
current request and pass that exact snapshot to `jobs.resume(request)`. Cancellation
uses `jobs.cancel(request)`. Both operations compare the saved request before writing,
send its control revision, and verify the acknowledgement afterward. A stale snapshot
raises `SconeError`; no automatic retry occurs. Document recovery can explicitly retry
an interrupted extraction/index operation whose outcome is unknown.

`jobs.list(limit=20, after=page.next_after)` reads retained history. `jobs.result()`
does not run extraction, even when source verification previously failed or the
attempt limit has been reached. Missing or currently unverifiable results raise an
error. The result's extraction filename is checked against the immutable job request;
the original blob may retain a different first-upload filename after deduplication.

For PDF OCR, pass `pdf_ocr=PdfOcr("missing_text", "columns_ltr")` to `start()` after
checking the host's `formats.pdf_ocr_available` and advertised choices. Users select
OCR behavior; the host owns the parser implementation, revision, and processing limits.
Upload and job admission are separate explicit writes. Uploaded bytes must fit the
host's advertised input limit and the client's 25 MiB ceiling.

Completed agent outputs are typed and checked against the saved execution request:

```python
from scone import TaskResult, HandoffResult, ModelOutput, HumanOutput

result = agents.result("run-1")
if isinstance(result, TaskResult):
    for output in result.results.values():
        if isinstance(output, ModelOutput):
            print(output.model_id, output.text, output.evidence_ids)
        elif isinstance(output, HumanOutput):
            print("Human reply", output.text, output.activation_id)
elif isinstance(result, HandoffResult):
    print(result.status, result.final.text if result.final else None)
```

Model receipts retain the selected model, binding, dependency IDs, call counts,
and immutable evidence packets. Packet identifiers and path references must agree
with the returned records; nested packet data is read-only. A human output must
match a freshly read activated reply and cannot carry model/evidence fields.
A handoff-limit result preserves actual hops and has no final answer.

`result()` performs only reads. For interactive runs it reads input receipts first,
then requests the source-verified result last. It never runs another model or
restarts a workflow. The server verifies current source access, retained content,
and native entity-classification rules; client-side packet checks establish wire
consistency and do not independently prove that a model's answer is true. A deleted
or unavailable source is an error, rather than a fallback to an old result.

### Configured local directory scans

A native host with `SCONE_DIRECTORY_SYNC_CONFIG` advertises `documents.sync`.
The standalone client discovers only collections available to its key; it cannot
supply an arbitrary filesystem root. All methods perform one explicit operation.
Reads, reconnects and repeated starts with the same ID do not resume work.

```python
from scone import Scone

with Scone(api_key="local-space-key") as memory:
    sync = memory.directory_sync(expected_space="alpha")
    collection = next(item for item in sync.collections() if item.collection_id == "notes")
    run = sync.start("notes-scan-001", collection=collection, delete_missing=False)
    # Keep this run ID if admission is not confirmed. Check before retrying.
    current = sync.status(run.record.run_id)
    print(current.status, current.active_elsewhere, current.outcome_unknown)
```

Start includes the discovered configuration digest; the host refuses stale
configuration before execution. `sync.list(limit=20, after=cursor)` reads a bounded
history page, and `sync.request(run_id)` reads the immutable intent and current
revision. Resume with a freshly read `SyncStatus` using `sync.resume(status)`;
`sync.cancel(status)` requests cancellation with the same revision guard. An
acknowledgement validates the actual transition, not merely a successful HTTP code.
Refresh status until the worker is idle before resuming after cancellation.
Completed and partial scans require a new run ID for another scan.

`sync.results(run_id, limit=20, after=index)` returns typed source receipts and
scan issues with a numeric next cursor. Results bind the expected space/run and
validate counts and ordering. They describe the completed scan; they do not assert
that an episode is still retained. Python preserves full 64-bit episode identities
and marks escaped filesystem diagnostics separately from source paths. No method
automatically follows pages, retries a write, downloads a model, or resumes a run.
