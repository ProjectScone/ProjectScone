# Counterfactual memory contracts: research prototype

Status: experimental, explicitly imported, not connected to the webapp or serving
runtime. No external RAG framework, agent SDK, local language model, or cloud
storage is required. Semantic judgments use direct TypeSafe Jev. The storage
journey uses native Scone SQLite and the configured Qwen embedding API.

## Question and design

Can an answer retain an executable description of the evidence withdrawals that
would invalidate it, so a future withdrawal does not require model inference?

An ordinary answer cache remembers a response for one state. This prototype
evaluates counterfactual states before they occur, then stores their judgments.
It also exposes the minimal sets of source origins whose removal loses support.
This is bounded semantic memoization, not a proof, causal discovery, or a new
theory of truth maintenance. An ordinary memoization cache is equally fast after
it has already seen a particular state.

For example:

* Two independent records establish Maya as deployment owner: either can survive
  alone; removing both loses support.
* One record says the release captain owns deployment, another identifies Maya
  as captain: losing either breaks the inference.
* A record and its copied recap share an origin: withdrawing that origin removes
  both. The caller supplies known provenance; the model does not discover it.
* Conflicting records can become unambiguous after a withdrawal. Support is not
  assumed monotonic in the amount of evidence.

The compiler enumerates every subset of up to five provenance families: at most
32 worlds. Twenty records and 16 KB of source text bound each packet. Each
nonempty world receives separate support and contradiction Noul questions. The
questions run in one direct Jev request; each question contains only its world's
evidence. Common state contains the question and candidate claim, not all source
text. Empty evidence is deterministically insufficient.

Code uses fixed thresholds: at least .8 for yes, at most .2 for no. Support and
contradiction both high means conflict. Intermediate judgments remain uncertain.
Only `supported` authorizes reuse. Probabilities are model outputs, not calibrated
guarantees on this task. No threshold or prompt was tuned after seeing v1 results.

Each contract binds the exact question, proposed claim, context key, source IDs,
origin grouping, source content, model, policy version and expiry. A caller must
supply a complete current evidence snapshot. Any addition, edit, partial origin
withdrawal, context change, or expiry requests recompilation. There is no inferred
generalization to an untested world. Requested worlds outside the original packet
are rejected at the provider boundary. Missing or malformed provider judgments
never turn into support.

## What the diagnostic measures

Eight hand-authored synthetic cases define 60 evidence worlds, including eight
empty controls. Labels are written in `memory_contract_cases.py`; they are never
sent to Jev. Cases cover redundant support, copies, two-hop inference, conflict,
compound claims, absence versus negation, entity scope and proposed decisions.

For each case:

1. Compile all worlds in one batched Jev request.
2. Evaluate each corresponding current snapshot locally.
3. Ask the same Jev model the same two questions about that world's evidence
   again, as a fresh-check baseline.
4. Record both classifications, raw probabilities, reported tokens, provider
   model and wall-clock timings. Preserve all disagreements.

Every nonempty state is visited once. This intentionally measures the best use
case for anticipatory compilation; unused worlds would waste work. Compilation
always precedes baseline checks, so order, provider caching and network conditions
are confounders. There is no random assignment, confidence interval, external
benchmark, retrieval task, answer generation, or trained novelty detector here.
The sixty worlds are correlated variants of eight cases, not sixty independent
questions. Classification accuracy must not be called end-to-end RAG accuracy.

The reported break-even count divides total compilation latency by average fresh
check latency and number of contracts. It is a latency estimate under this run's
workload, not a price estimate or guaranteed production saving. Token totals are
reported separately; no unverified dollar price is assumed.

## Reproduction

From the repository root, using an environment with the memory package's test and
remote-embedding dependencies:

```sh
PYTHONPATH=packages/memory/src python scripts/local_env.py \
  --env-file /absolute/path/to/.env.local -- python \
  packages/memory/benchmarks/memory_contracts.py --output /tmp/contracts.json

PYTHONPATH=packages/memory/src python scripts/local_env.py \
  --env-file /absolute/path/to/.env.local -- python \
  packages/memory/benchmarks/memory_contract_storage.py \
  --directory /tmp/new-contract-storage-directory

PYTHONPATH=packages/memory/src python -m pytest -q packages/memory/tests/experimental
```

The environment loader requires an owned mode-0600 file. Required variables:
`TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`; the storage
journey additionally reads `SCONE_EMBED_URL`, `SCONE_EMBED_MODEL` and
`SCONE_EMBED_API_KEY`. It creates a new SQLite database and refuses to reuse an
existing directory. It never opens the user's live memory store.

Raw output is local and excluded from source control. Model request errors abort
the run without producing a success summary; completed case records are saved
incrementally. A stopped run is incomplete, not evidence of accuracy.

## Closest research and scope of originality

* [RAGONITE (2024)](https://arxiv.org/abs/2412.10571) already studies
  counterfactual evidence attribution in conversational RAG. Evidence ablation
  is not an invention of this experiment.
* [BeliefMem (2026)](https://arxiv.org/abs/2605.05583) retains alternative
  conclusions with probabilities and updates them as observations arrive.
  Keeping uncertain memory is already an active research direction.
* [Dependency-guided rollback repair (2026)](https://arxiv.org/abs/2608.10502)
  repairs affected memory and execution after diagnosed faults, preserving
  independent support. It reports recovery of 85.3% versus 77.3% for its strongest
  comparison on a controlled benchmark. Those are that paper's results, not Scone's.
* [Ingest-time semantic compilation (2026)](https://arxiv.org/abs/2608.20845)
  motivates maintaining a compiled semantic substrate to reduce repeated reading.

The Scone hypothesis is that **precomputed, provenance-grouped failure conditions
can become a useful native memory primitive**. This implementation tests that
combination on a small controlled workload. It establishes neither priority nor
superiority over these systems, LlamaIndex, or production RAG frameworks.

## Boundaries before any production integration

The exponential compiler is unsuitable for large evidence packets. The next
research step is selecting a small set of informative interventions while
measuring errors against the exhaustive table, followed by an untouched external
temporal-memory benchmark. Full rebuild and ordinary lazy memoization baselines
must include unused compilation work and realistic update distributions.

Production also requires atomic scope-aware snapshots, source retirement and
permission events, encrypted decision storage, and eviction of derived artifacts
on forget. The serialized prototype contains its claim and original evidence;
deleting a source from SQLite does not erase that separate research artifact.
`from_json` validates shape and bounds, not authenticity. Never accept a contract
from an untrusted client. The storage demo uses only synthetic public fixtures.

Time, user identity, permissions and policy changes must be reflected in the
host-supplied context key. `now` and expiry belong to the trusted host. The cache
does not discover new facts, determine whether a source is trustworthy, or
guarantee that the host has supplied all relevant evidence. These are reasons
the prototype remains outside the live answer path.
