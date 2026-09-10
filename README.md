# ProjectScone

A self-hosted Python RAG and memory framework for agents and applications.
A JudgeHuman project by ProjectScone contributors.

Scone retains source material, retrieves evidence, and tracks claims,
relationships, and corrections over time. Compose document stores, vector
indexes, model providers, retrieval pipelines, and agent workflows through
Python protocols, or expose the framework through HTTP and MCP.

## Repositories

- **Framework and Python HTTP client:** this repository.
- **Web application:** [ProjectScone-Webapp](https://github.com/ProjectScone/ProjectScone-Webapp).
- **Rust engine, CLI and C ABI:** [ProjectScone-Rust](https://github.com/ProjectScone/ProjectScone-Rust).

Each repository builds independently. The Python API does not bundle or serve
browser pages. Run the Webapp host alongside it to proxy authenticated HTTP,
SSE, and WebSocket requests. Rust is not a Python installation dependency.

## Quickstart

```sh
python -m pip install -e './packages/memory[api]'
```

```python
import asyncio
from scone_memory import (
    HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine,
)

async def main():
    memory = await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()
    ).open()
    try:
        await memory.remember("project", "Use the migration checklist.",
                              source="notes://migration-decision")
        result = await memory.recall("project", "migration checklist")
        for item in result.items:
            print(item.source, item.text)
    finally:
        await memory.close()

asyncio.run(main())
```

The in-memory stores are ephemeral; `HashEmbedder` is a deterministic test
baseline. Choose persistent storage and a semantic embedder for real retrieval.
See the [framework documentation](packages/memory/README.md) for SQLite, Qdrant,
other backend adapters, scoped retrieval, temporal facts, provenance, ingestion,
conversations, multimodal providers, and framework integrations.

## Serve the API

Copy [.env.example](.env.example) to an ignored `.env.local`, set your own
space key and storage paths, then use `scripts/serve-self-hosted.sh`.
The API listens on port 7437 by default. `/healthz` reports process health;
`/v1/capabilities` reports configured features and requires a space key.
A configured model alone does not mount the conversations service: configure
its separate journal and runtime as described in the framework documentation.

The [Python HTTP client](packages/scone-client) uses `from scone import Scone`.
The native async/sync engine uses `scone_memory`; it does not require a server.
Infrastructure and provider connections are explicitly configured. Nothing in
this repository provisions or enables a hosted service automatically.

## Measurement and development

[Public QA results](packages/memory/benchmarks/public-qa-v1.results.md) record
200 unchanged HotpotQA/SQuAD questions across three self-managed models:
64% exact match for Gemma E4B, 61% for Llama 3.1 8B, and 59% for Llama 3.2 3B.
These are recorded development runs, not a benchmark of every current commit
or a guarantee of answer accuracy. Reports separate retrieval coverage,
generation quality, latency, and resource use.

See [CONTRIBUTING.md](CONTRIBUTING.md) for tests and cross-repository conformance.
Multimodal document parsing, durable extraction, and broader evaluation remain
active development work; see the maintained capability baseline for gaps.

## License

For the repository layout, local test setup and contribution workflow, see
[CONTRIBUTING.md](CONTRIBUTING.md).

Current versions use the [ProjectScone Research Attribution License](LICENSE),
a custom MIT-derived license with a mandatory research and academic citation
condition. This is not the unmodified MIT License. Third-party materials retain
their own licenses; earlier versions released under MIT retain those terms.

### Research and academic citation

If you use ProjectScone or a substantial portion of its code for research or
academic work, you **must cite this repository** in resulting publications,
theses, reports, presentations, and other publicly shared research outputs.
For datasets or software artifacts, include the citation in accompanying
documentation. Credit **ProjectScone contributors** as the author, **JudgeHuman** as the
company, and **ProjectScone** as the project, and include the repository URL.
Use MLA, APA, Chicago, or another recognized academic citation style; citations
generated using EasyBib or BibTeX are accepted. Exact versions, commit hashes,
and individual file references are **not required**. See [LICENSE](LICENSE)
for the complete condition.

Full citation guidance is in [CITING.md](CITING.md).

Citation metadata is available in [CITATION.cff](CITATION.cff), which GitHub uses
for its **Cite this repository** action. Ready-to-copy examples follow.

**MLA / EasyBib (MLA)**

> ProjectScone contributors. *ProjectScone*. JudgeHuman, 2026,
> https://github.com/ProjectScone/ProjectScone.

**APA**

> ProjectScone contributors. (2026). *ProjectScone* [Computer software]. JudgeHuman.
> https://github.com/ProjectScone/ProjectScone.

**Chicago**

> ProjectScone contributors. *ProjectScone*. JudgeHuman, 2026. Computer software.
> https://github.com/ProjectScone/ProjectScone.

**BibTeX**

```bibtex
@misc{projectscone2026,
  author       = {ProjectScone contributors},
  title        = {{ProjectScone}},
  year         = {2026},
  howpublished = {JudgeHuman},
  note         = {Computer software},
  url          = {https://github.com/ProjectScone/ProjectScone}
}
```
