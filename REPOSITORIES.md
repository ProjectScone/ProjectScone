# Repository separation

The framework remote is https://github.com/ProjectScone/ProjectScone.git.
Webapp and Rust were extracted from framework commit
`0deab557ffea4fea6693bc35740fc8f8bf3ae0cd`, preserving history for their selected
paths in independent repositories:

- https://github.com/ProjectScone/ProjectScone-Webapp
- https://github.com/ProjectScone/ProjectScone-Rust

The framework keeps the native Python engine, HTTP client, API/MCP services,
backend adapters, tests and deployment modules. Webapp owns the TypeScript
application, browser tests, frontend packaging and HTTP/SSE/WebSocket proxy.
Rust owns Cargo packages, native fixtures and Rust release workflows.

## Migration

The former `python/` directory is now `packages/`, with `packages/memory`
and `packages/scone-client`. Package names and Python imports are unchanged.
CI, Docker build contexts, helper scripts and test paths use the new location.

Python `create_app`, `create_conversation_app` and the conversation launcher
no longer accept browser hosting arguments (`console`, `console_key`,
`local_console_key`, `reload_pages`). `--console`, `SCONE_UI_DEV` and
`SCONE_RELOAD_PAGES` are removed. Browser URLs on the Python API return JSON 404.

Run the independent Webapp host against the Python API; follow its README for
manual authentication and explicit loopback bootstrap configuration. Existing
memory databases and conversation journals retain their paths and formats.
The split does not move, reset, or convert operational data.

Generated HTML is no longer Python package data. Rust retains its existing
embedded snapshot, with explicit artifact update instructions in its repository.
Cross-language tests use explicitly configured external executables; no Python
install or test command builds a Rust sibling implicitly.

## Validation at separation

- Python 3.14: 4,944 passed, 2,447 optional integration cases skipped.
- Focused API/server lifecycle: 86 passed, including real TCP tests.
- HTTP client: 31 passed, 7 unconfigured Rust integration cases skipped.
- Explicit cross-runtime probe: 43 passed, 30 optional backend cases skipped.
- Fresh wheel installation: API health, authentication and absent UI assets verified.
- Changed-module mypy: 28 errors also present on source main; no new diagnostics.
- Rust minimal workspace: 251 passed, 1 ignored; formatting and both Clippy profiles passed.
- Rust default tests: linking stopped by disk exhaustion; no pass claimed.

Deployment image definition now builds only Python. A new Docker image was not
built during the split; the disposable smoke probe has been updated for API-only
routing. No cloud resources were provisioned.
