# Jev in Scone RAG

Jev evaluates text and returns typed decisions. It does not produce embedding
vectors or generate an answer. Scone can therefore use it between retrieval and
answer generation: lexical/vector/graph retrieval finds candidates, Jev ranks
their relevance, and the selected chat model writes the answer from evidence.

This integration is original Scone code against TypeSafe's published API
contract and the observed OpenRouter decisions endpoint. No reference framework
code is copied.

## Implemented

`scone_memory.providers.jev.JevReranker` implements the existing asynchronous
reranking interface. One request carries the query, candidate passage text and
one independent yes/no relevance question per passage. Local passage identifiers
are repeated in the instructions because TypeSafe says question-map keys are
not inputs to the underlying model. Replies must supply exactly one finite
probability for every candidate. The adapter records the resolved model in
`evaluate()` results; `rerank()` returns the scores expected by Scone.

Only retained passages that pass Scone's scope checks reach the adapter. Jev
cannot add candidates, change access controls, write memories, authorize tools
or approve an agent-created tool. Invalid responses, service failures and
timeouts retain baseline retrieval ordering and appear as a failed rerank trace.
Cancellation propagates. There are no automatic retries or alternate models.

Relevance probabilities are ranking signals, not proof that a passage is true
or that the final answer is correct. The model's constrained output format also
does not establish resistance to prompt injection.

## Use

The existing host factory supports Jev without a new service or database:

```dotenv
SCONE_RERANKER_FACTORY=scone_memory.providers.jev:create_reranker
SCONE_JEV_MODEL=~typesafe/jev-latest
SCONE_JEV_API_KEY_ENV=OPENROUTER_API_KEY
SCONE_RERANK_TIMEOUT=5
```

Set the named key variable privately on the server. If the key is already stored
under another name, set `SCONE_JEV_API_KEY_ENV` to that name. These settings
explicitly send query and authorized passage text to OpenRouter/TypeSafe;
storage remains local. Restart the host after changing its environment. Merely
installing or importing the adapter makes no network calls.

For an in-process engine:

```python
from scone_memory.providers.jev import JevReranker

engine.reranker = JevReranker(api_key=server_key)
engine.rerank_timeout = 5
result = await engine.recall(space, query, candidate_limit=32, limit=5)
```

`~typesafe/jev-latest` is the requested moving alias. Use a versioned
`typesafe/jev-…` identifier for a repeatable evaluation. `result.rerank` reports
whether ranking actually applied; individual items carry `rerank_score`.

## Live integration evidence — 2026-09-19

The [downloaded public QA evaluation](../benchmarks/jev-public-qa-v1.results.md)
now provides the larger comparison: all 200 reserved questions across four
Scone retrieval arms, with 800 completed observations. Hybrid support recall@5
rose from 85.75% to 97.00%, with median latency increasing from 47 ms to 543 ms.
The report includes regressions, per-dataset results, the HashEmbedder baseline
limitation and a live Jev-to-Gemma conversation/restart check.

`benchmarks/jev_retrieval.py` ran four synthetic questions through authenticated
`GET /v1/recall`, using local SQLite, HashEmbedder and lexical retrieval.
The same candidates and return limit were used for baseline and Jev.

| Question | Baseline top passage correct | Jev top passage correct | Jev recall latency |
| --- | --- | --- | --- |
| Cedar maintainer | Yes | Yes | 567 ms |
| Retention exception | Yes | Yes | 400 ms |
| Acquisition direction | No | Yes | 383 ms |
| Rescheduled rollout | No | Yes | 465 ms |

The alias resolved to `typesafe/jev-1.13-20260917`. Correct top-ranked passages
increased from 2/4 to 4/4. The rollout passage containing an instruction to print
score 1 received 0.04. These are integration smoke results, not a general
benchmark, a calibration study or a prompt-injection robustness claim.
Raw results and the SQLite fixture were saved outside git at
`bench-runs/jev-retrieval-20260919/run-1/` in the project workspace.

Run another independent smoke with a new output directory:

```sh
PYTHONPATH=src python benchmarks/jev_retrieval.py --output /tmp/jev-retrieval-run
```

## Next integration points to evaluate

These are proposed uses, not shipped features:

| RAG/agent decision | Jev primitive | Host responsibility |
| --- | --- | --- |
| Choose lexical, vector or graph retrieval | Choice | Offer supported routes; keep authorization unchanged |
| Decide whether retrieved evidence is sufficient | Noul | Evaluate abstention quality before using any threshold |
| Judge duplicate or conflicting memory proposals | Noul/Choice | Retain provenance and require review for consequential writes |
| Classify repetitive agent behavior | Choice | Combine with deterministic loop detection; allow human escalation |
| Rank competing graph paths for a query | Score | Preserve path provenance and verify edge direction |

Next quality evidence should compare recall/MRR and answer correctness on a
held-out corpus with a non-Chinese embedding model, including missing evidence,
negation, relationship direction, long passages and adversarial instructions.
Jev cannot recover a relevant passage omitted by first-stage retrieval. A
lexical/graph-only pipeline is possible, but its candidate recall must be measured
before removing embeddings.

## Primary references

- [TypeSafe System One](https://docs.typesafe.ai/concepts/system-one): model role and limits.
- [TypeSafe API](https://docs.typesafe.ai/api): state, questions and typed answers.
- [OpenRouter Jev 1.13](https://openrouter.ai/typesafe/jev-1.13): served model.
- [OpenRouter Jev alias](https://openrouter.ai/~typesafe/jev-latest): moving selection.
