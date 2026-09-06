# scone-client

A thin Python client for the [Scone](https://github.com/DrDrewCain/ProjectScone)
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
