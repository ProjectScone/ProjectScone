# Contributing

The Python framework and HTTP client live here. The Webapp and Rust engine
have independent repositories, dependency locks, and CI. Do not add build-time
imports or implicit sibling-directory lookups between them.

## Layout

- `packages/memory`: native async/sync framework, HTTP/MCP, runtime and tests.
- `packages/scone-client`: separately packaged HTTP client.
- `tests/fixtures`: versioned HTTP and prompt contract fixtures.
- `deploy` and `terraform`: explicitly configured deployment modules.

Use feature branches and preserve per-commit history when merging. Never commit
secrets, generated bundles, environments, model weights, or private datasets.

## Python checks

```sh
python -m pip install -e './packages/memory[test]'
cd packages/memory
PYTHONPATH=src python -m pytest -q
python -m build
```

Run the HTTP client tests separately from `packages/scone-client` after installing
its `[test]` extra. Python 3.14 is the primary runtime; CI also covers supported
compatibility versions and separately configured storage adapters. Check types
for changed modules with mypy and record pre-existing errors separately.

## Cross-repository checks

Cross-runtime tests are optional only when no external executable is supplied.
Build `episode_roundtrip` in the Rust repository, then set
`SCONE_TEST_RUST_ROUNDTRIP` to its absolute path and run
`scripts/episode-conformance.sh --ci`. Python keeps its own versioned copy of
`episodes-v1.json`; coordinate contract changes across repositories.

Set `SCONE_TEST_RUST_BINARY` to a built Rust CLI to exercise HTTP client and
prompt-hook interoperability. A configured invalid binary fails validation;
tests never silently compile another repository. These narrow transfer tests
are not proof of full archive or database compatibility.

## Evaluation

Keep questions and source data unchanged. Record dataset revisions, model and
sampling settings, failures, exclusions, retrieval coverage, answer metrics,
latency, and resource use. Byte reduction is not token reduction or accuracy.

## Review and compatibility

Keep changes scoped to a feature branch. Describe the concrete behavior change,
validation and remaining limits in the pull request. Preserve individual commits
when merging. HTTP clients, Webapp and Rust releases are independent: describe
wire-contract changes and coordinate versioned fixtures rather than depending on
another repository's checkout or generated output.

Use typed public interfaces, validate input at API boundaries, preserve scope
checks and source provenance, and test meaningful failure and cancellation
paths. New provider integrations must remain explicitly configured. Tests use
synthetic fixtures or appropriately attributed public data; never include private
conversations, credentials, model weights or operator environment files.

## Attribution and licensing

Contributions use this repository's [LICENSE](LICENSE). Preserve third-party
notices and identify the provenance of reused material. Do not present code
from another project as original work. See [CITING.md](CITING.md) for MLA,
APA, Chicago and BibTeX examples and [CITATION.cff](CITATION.cff) for metadata.
The project credit is ProjectScone contributors, JudgeHuman, ProjectScone.
