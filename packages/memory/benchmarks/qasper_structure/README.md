# Full QASPER comparison

Evaluate native structure-address retrieval against a native vector control and
installed LlamaIndex hybrid retrieval. See [PROTOCOL.md](PROTOCOL.md) for the
frozen resource budgets, task scope and scoring limitations. The complete first
run is recorded in [RESULTS.md](RESULTS.md): improved evidence retrieval, no clear
answer-accuracy win, and higher latency.

The separate four-arm development evaluation is documented in
[DEV_PROTOCOL.md](DEV_PROTOCOL.md) and [DEV_RESULTS.md](DEV_RESULTS.md).
Vector-guided routing reduced retrieval time versus hierarchy and improved evidence
over the controls, without establishing an answer-accuracy advantage over LlamaIndex.
Export with `split='dev'`, run with `--split dev`, and score with
`development=True`; the full schedule is 281 papers, 1,005 questions, and 4,020
answers. Development and test artifacts use separate directories and manifests.

Use the isolated LlamaIndex benchmark environment, not Scone's runtime. Install
the optional benchmark dependencies used by `matched_qa`, including
`llama-index-retrievers-bm25`. Run from the repository root with
`PYTHONPATH=packages/memory/src:packages/memory/benchmarks`.

1. Download the official archive linked in the protocol. Export the complete
   `qasper-test-v0.3.json` using `qasper_structure.data.export(raw_path, output)`.
   The inference bundle contains no gold annotations.
2. Load the private environment through `scripts/local_env.py`. Run
   `python -m qasper_structure.run --dataset /path/to/export --output /path/to/run`.
   The runner requires all 416 papers and 1,451 questions. Source/input digests,
   model configuration, package versions and all 4,353 scheduled answers are frozen.
3. Resume an interrupted run with the same command plus `--resume`. Existing
   answers, including failures, are retained. Prepared contexts are reused exactly.
   An attempt interrupted before its observation requires explicit recovery;
   the runner refuses to silently resample an answer.
4. Verify `completion.json` artifact hashes, manifest source/input hashes and
   original raw-data hash before invoking
   `qasper_structure.scoring.score(raw_path, observations_path)`. Missing answers
   are rejected by default. `allow_partial=True` is progress-only and never a ranking.

Embedding vectors, model receipts, contexts and answers stay in the local run
directory. Do not commit these generated files or API credentials. Routing tokens
are reported where available; unavailable provider charges must not be invented.
