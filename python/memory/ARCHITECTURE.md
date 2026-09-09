# Python memory architecture

`MemoryEngine` is the public entry point for storage and memory operations.
Retrieval components depend on typed ports rather than on an engine instance.

| Component | Responsibility |
|---|---|
| `core/validation.py` | Shared names, metadata, tags, timestamps and retention rules |
| `retrieval/recall.py` | Lane execution, fusion, source verification, optional reranking and result assembly |
| `retrieval/fact_recall.py` | Validated indexed fact lookup, scan fallback and historical facts |
| `retrieval/episode_scope.py` | Final episode scope checks shared by passage and fact retrieval |
| `retrieval/reranking.py` | Bounded adapter calls, output validation and fallback ordering |
| `realtime/context.py` | Pack retrieved evidence into a bounded model context with provenance |
| `memory/engine.py` | Coordinate the public API and remaining ingestion, lifecycle and graph operations |

For each `MemoryEngine.recall` call, the engine constructs a `RecallRuntime` from
its current document store, vector index, embedder and ranking configuration.
It also supplies clock, event-emission and query-evidence callbacks. A standalone
host can provide those dependencies directly to the recall component.

Retrieval configuration references are captured when a call starts. Changes to
the engine's configured dependencies apply to subsequent calls. This does not
freeze stored records: a source can still be deleted or replaced during an
awaited reranker call, and retained-source checks still run before delivery.
Bound evidence callbacks remain live.

The refactor preserves candidate depth, filter semantics, fusion ordering,
scope verification, cancellation propagation and degraded-mode reporting.
Reranker failures retain baseline ordering; ranking scores do not become
probabilities of correctness. Indexed facts are checked against retained records
and fall back to a scan when the optional index is unavailable or invalid.

Existing public engine methods and imports of shared validation helpers remain
available. Private helper delegation is not a customization interface; hosts
should supply the documented storage ports and reranker adapters.

The engine decomposition is ongoing. Ingestion and recovery, fact lifecycle,
retention/deletion, activity graphs and import/export still need their own
boundaries. Moving those operations must preserve storage ordering, revision
semantics and the existing public API.

The retrieval extraction was checked against the contract and scope suites and
an actual Qdrant replay of 200 unchanged public questions. With identical storage
and a fixed clock, complete model requests and recall results matched the prior
implementation on all 200 questions. This is behavior-equivalence evidence,
not a new answer-accuracy benchmark.
