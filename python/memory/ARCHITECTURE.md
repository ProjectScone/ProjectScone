# Python memory architecture

`MemoryEngine` is the public entry point for storage and memory operations.
Retrieval, ingestion and archive components depend on typed ports rather than
on an engine instance.

| Component | Responsibility |
|---|---|
| `core/validation.py` | Shared names, metadata, tags, timestamps and retention rules |
| `ingestion/records.py` | Batch records, recovery reports and content identity helpers |
| `ingestion/batch.py` | Chunking, deduplication, embedding, ordered writes, rollback and crash recovery |
| `ingestion/jobs.py` | Durable batch receipts, progress, request lookup, pagination and cancellation |
| `retrieval/recall.py` | Lane execution, fusion, source verification, optional reranking and result assembly |
| `retrieval/fact_recall.py` | Validated indexed fact lookup, scan fallback and historical facts |
| `retrieval/episode_scope.py` | Final episode scope checks shared by passage and fact retrieval |
| `retrieval/activity_graph.py` | Read activity and retained sources into a graph of recorded relationships |
| `retrieval/activity_facts.py` | Coordinate indexed fact reads with a shared budget and validated partial results |
| `retrieval/graph.py` | Graph values and typed edge construction without storage reads |
| `retrieval/reranking.py` | Bounded adapter calls, output validation and fallback ordering |
| `realtime/context.py` | Pack retrieved evidence into a bounded model context with provenance |
| `memory/archive.py` | Export original records and import with identity, source and link remapping |
| `memory/fact_placement.py` | Place temporal claims, preserve restatement identity and close covered intervals |
| `memory/fact_review.py` | Human decisions, historical batch ordering and fact visibility changes |
| `memory/catalog.py` | Source inventory, fact selection, profiles, metadata aggregates and status reads |
| `memory/retention.py` | Integrity inspection, expiry, forget receipts, tombstones and space deletion |
| `memory/engine.py` | Coordinate the public API, attachments, replacement and relationship validation |

For each `MemoryEngine.recall` call, the engine constructs a `RecallRuntime` from
its current document store, vector index, embedder and ranking configuration.
It also supplies clock, event-emission and query-evidence callbacks. A standalone
host can provide those dependencies directly to the recall component.

Retrieval configuration references are captured when a call starts. Changes to
the engine's configured dependencies apply to subsequent calls. This does not
freeze stored records: a source can still be deleted or replaced during an
awaited reranker call, and retained-source checks still run before delivery.
Bound evidence callbacks remain live.

Batch ingestion and recovery similarly receive an `IngestionRuntime` with
document/vector stores, an embedder, clock, chunk target and callbacks for
embedding text and event emission. The internal batch entry point expects the
engine to check space access first; recovery runs over its configured stores
during engine startup. The engine retains public remember events,
attachments and replacement semantics. It reexports `Record`,
`RecoveryReport`, content identity helpers and the existing batch constant.

All fresh records are embedded before the first document write. Each write's
inflight marker precedes its episode, chunks and vectors; marker removal follows
vector persistence. Ordinary write errors roll back the partial batch. Recovery
completes interrupted episodes from their retained content or clears orphan
markers. This extraction preserves those operations and their ordering; it does
not add cross-store transactions or change cancellation/recovery semantics.

Ingestion jobs receive a `JobRuntime` with document storage, clock, event and
bound ingestion/receipt callbacks. Recording and reading capabilities remain
separate; unsupported operations retain their existing refusals. Searchable
and consolidated timestamps describe separate milestones. Cancelling a job
preserves searchable records and completed consolidation receipts. Request IDs
return recorded jobs without ingesting another batch; this lookup is not a
transactional reservation across concurrent callers. Page limits use the
current engine configuration when each call starts.

Retention receives a `RetentionRuntime` with document/vector/blob/event ports,
a clock and bound source/deletion callbacks. Its functions own read-only
integrity checks, impact previews, expiry and deletion. The engine retains its
public entry points and supplies fresh dependency references per call. Stored
records are not frozen across awaits; bound callbacks remain live.

Expiry compares parsed UTC instants and breaks ties by episode ID before
applying the batch limit. Timestamp offsets and equivalent textual spellings
cannot change which oldest source is selected. A dry run reports the due count
without deleting records. Expiry calls the bound forget method so an existing
host guard is still honored. Forgetting preserves claims as historical ledger
entries and records the absence of their evidence through tombstones.

Whole-space deletion releases attachment holds, removes documents and vectors,
then purges the event trail. Another space's attachment holds keep shared bytes
available. A durable space-deletion marker prevents later ingestion from
recreating the space. Deletion still spans independent stores: a failure can
leave earlier writes applied, propagates to the caller, and is not a
distributed transaction or an automatic cleanup retry.

Archive import receives an `ArchiveRuntime` with the document store, clock and
normal ingestion callback. The engine retains space validation and deleted-space
guards. Export depends only on the document store and yields episodes, facts and
unique links; chunks and vectors are derived again through ingestion on import.
`ImportSummary` and existing identity helper imports remain engine aliases.

An imported source ID is remapped before fact identity is compared. Distinct
retained sources, origins or quotes remain distinct facts, preserving link ends.
Repeating an import does not create an unexcluded copy of a fact suppressed by
the target. Tombstoned content stays omitted unless resurrection is explicit.
Store-local supersession IDs retain their existing import behavior; this is not
a complete cross-store snapshot or a transaction across document/vector stores.

Activity graph assembly receives the current document store and optional event
log. The engine validates the space and clamps the event/provenance window before
dispatch. The builder hydrates retained sources and recorded capture events,
preserves session/episode focus, and distinguishes sources omitted for room from
sources that no longer exist. Retrieval edges retain their lane/rank labels;
similarity is never promoted to a factual relationship. Graph data structures
and edge helpers remain separate from storage reads and graph analysis.

Temporal placement receives a `FactPlacementRuntime` containing the current
document store, clock and bound event/placement/truncation callbacks. Assertion
and approval retain their engine entry points and relationship or review
validation. Standalone hosts can bind the component's `place` and `truncate`
functions directly to a document store. The engine preserves its private
placement hooks and legacy helper aliases for existing callers.

Placement includes closed ledger intervals when identifying restatements,
overlaps and successors. Backfilled claims cannot overwrite fresher intervals;
shortening an interval preserves a person's closure reason. Proposals remain
outside the held ledger until approval. Source quotes are validated before
storage, and event payloads and revision ordering are unchanged. This boundary
does not add transaction isolation between concurrent assertions or a bound on
the number of rival facts read for a subject and predicate.

Review and visibility changes receive a `FactReviewRuntime` containing storage,
clock, event and bound decision callbacks. Approval, rejection, exclusion,
inclusion and manual closure keep their public engine methods. Batches validate
the starting revision, apply in historical order and return per-ID outcomes in
caller order. Cancellation can leave earlier decisions applied, and event
failures can follow persisted changes; this is not a transactional batch.

Fact selection uses the optional `GraphFactReader` capability in
`core/graph_read.py`, coordinated by `retrieval/activity_facts.py`. The default
budget is 400 facts (maximum 2,000), shared across source groups with one-row
lookahead. Stores filter by space and optional source before the indexed limit;
the graph never falls back to full-ledger listing. Focused reads visit sources
in episode-ID order, then order selected facts by fact ID. All fact statuses
remain visible. Unvisited source groups conservatively mark the result partial.

`facts_truncated` and `fact_read_status` distinguish bounded partial results,
unsupported readers, failed snapshots and graphs that did not read facts.
Malformed or failed reads discard the fact snapshot; cancellation propagates.
The coordinator applies a two-second cooperative deadline to fact reads.
Event and additional source windows remain separate. Incident-link reads and
the total returned node count are not globally bounded by the fact budget;
the graph is not an atomic snapshot across document and event stores.

The refactor preserves candidate depth, filter semantics, fusion ordering,
scope verification, cancellation propagation and degraded-mode reporting.
Reranker failures retain baseline ordering; ranking scores do not become
probabilities of correctness. Indexed facts are checked against retained records
and fall back to a scan when the optional index is unavailable or invalid.

Catalog queries receive a document store directly. Profile reads also receive a
clock; status reads receive a typed identity callback evaluated after the
revision read, so current embedder/store/vector labels remain live. The engine
reexports `Profile`, `RecentActivity` and source-walk constants and preserves its
public query signatures. Its source-page wrapper forwards the current walk
settings rather than freezing them when the engine is created.

Catalog source pages retain descending-ID order, space and kind filters,
metadata filtering across batches, and continuation when the read budget runs
out. Counts, scopes, pending distillation and fact selection retain their
existing scans and status rules. This separation adds neither indexed aggregate
queries nor an atomic snapshot across catalog reads.

Existing public engine methods and imports of shared validation helpers remain
available. Private helper delegation is not a customization interface; hosts
should supply the documented storage ports and reranker adapters.

The engine decomposition is ongoing. Relationship validation, attachment handling,
replacement and derivation coordination still need their own
boundaries. Moving those operations must preserve storage ordering, revision
semantics and the existing public API.

The retrieval extraction was checked against the contract and scope suites and
an actual Qdrant replay of 200 unchanged public questions. With identical storage
and a fixed clock, complete model requests and recall results matched the prior
implementation on all 200 questions. This is behavior-equivalence evidence,
not a new answer-accuracy benchmark.

Ingestion checks exercise direct component calls, batch rollback, UTF-8 offsets,
deduplication and recovery from writes interrupted at different stages. The
engine contract suite also runs those behaviors with a real Qdrant server.

Archive checks cover direct component calls, renamed spaces, shifted IDs,
distinct fact provenance, repeated import, UTF-8 chunk rebuilding and explicit
resurrection. Contract tests run with in-memory, SQLite, embedded Qdrant and a
self-managed Qdrant server; real cross-language transfer remains a separate,
explicitly configured test.

Activity graph tests cover direct port calls, stored capture/source/recall/feedback
edges, focused sessions, time/space scope, omitted versus missing sources, bounded
capture hydration and cancellation. The same public graph payload continues to
feed graph analysis and the HTTP endpoints.

Lifecycle checks combine job cancellation, source retrieval, evidence
invalidation, preserved claims, tombstones, whole-space deletion and shared
attachments. The same workflow runs over in-memory, SQLite and embedded Qdrant
in the focused suite; additional configured backends use the contract fixture.
Timezone-order regressions exercise bounded expiry with noncanonical clock
timestamps. This refactor reduces `engine.py` from 1,326 to 1,132 lines while
retaining its public method signatures and documentation.
