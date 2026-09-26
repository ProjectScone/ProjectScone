# Native structure-address retrieval experiment

Vector retrieval remains central. Document headings provide a versioned address
space, and Jev chooses whether to fetch original sections directly or search
passages inside them. This is an opt-in per-document Python API. It does not
replace the running webapp's retrieval pipeline or claim a win over LlamaIndex.

## Retrieval flow

1. The caller supplies a **current, authorized** Markdown document snapshot.
   Native parsing keeps original UTF-8 offsets and content-bound section IDs.
2. Nemotron encodes the question while Jev routes through section menus.
   Each menu includes descendant headings so broad titles do not hide the useful
   topic. Choice outputs are restricted to supplied addresses plus no-match.
3. Broad vector retrieval always runs. Jev then selects one of three fetch modes:
   - `original`: fetch original bytes of selected sections, including subsections;
   - `section_vector`: search within selected addresses and combine those results
     with broad vector results using reciprocal rank fusion;
   - `flat_vector`: keep broad vector retrieval.
4. If a whole section exceeds the evidence budget, use scoped vector retrieval.
   No-match, invalid judgments, routing timeouts, and scoped backend errors retain
   the broad vector baseline. Cancellation propagates.

Routing uses a bounded beam of three paths and at most eight rounds. Path scores
are length-normalized heuristic scores, not calibrated end-to-end confidence.
Overlapping addresses are resolved in descending score order. A weaker parent
must not replace a stronger child with the entire book. Budget checks use UTF-8
bytes. Truncated vector passages retain exact source offsets; original-section
fetches are whole sections, never silently truncated.

## Native API

```python
from scone_memory.embedders.remote import RemoteEmbedder
from scone_memory.providers.jev_sections import JevSectionChooser
from scone_memory.retrieval.section_routing import SectionRouter, SectionSnapshot
from scone_memory.retrieval.structured_document import StructuredDocumentIndex

snapshot = SectionSnapshot.from_markdown(space, document_id, markdown)
embedder = RemoteEmbedder(
    'https://openrouter.ai/api/v1', 'nvidia/nemotron-3-embed-1b:free',
    api_key=embedding_key, dim=2048, query_prefix='query: ',
    document_prefix='passage: ', trust_env=False,
)
try:
    index = await StructuredDocumentIndex.build(snapshot, embedder)
    async with JevSectionChooser(api_key=jev_key) as chooser:
        router = SectionRouter(chooser)
        result = await index.retrieve(query, snapshot, router, chooser, mode='auto')
        # result.evidence: selected original source spans
        # result.baseline: independently retained broad vector evidence
        # result.route / result.fetch: actual decisions, models, usage and timings
finally:
    await embedder.close()
```

The index defaults to Scone's local in-memory vector backend and accepts the
native `VectorIndex` port. It uses a separate namespace; it does not migrate an
existing production collection. Reuse the index, router and provider for queries.
Callers own embedder and backend lifecycles, authorization, and replacement of
old snapshots when documents change. A cached route is never an access grant.
For a multi-document corpus, discover candidate documents through ordinary
vector retrieval before invoking this per-document path. That corpus-level
integration remains future evaluation work, not an existing feature here.

Cache keys include scope, document identity, exact source, query, routing settings
and provider/prompt definition. Changed content cannot reuse old routes. A changed
snapshot cannot query old document vectors: rebuild its index. This does not
claim semantic truth tracking of every downstream answer or decision. Route
cache is bounded, in-memory only; no sensitive evidence cache is written to disk.

## Reproducible live integration check

Run from the repository root, using the existing private credential loader:

```sh
PYTHONPATH=packages/memory/src:packages/memory/benchmarks \
python scripts/local_env.py --env-file /absolute/path/to/.env.local -- \
python -m structure_routing.run --output /absolute/path/to/fresh-output
```

This runs **all six questions in the shipped fictional fixture**, in all four
modes, with actual hosted Nemotron and Jev. It records input/code hashes, source
spans, decisions, token usage, fallbacks and stage timings. All modes use identical
chunk vectors and the same evidence limits. `evidence_contains` labels are used
only after retrieval; they never enter the provider state.

This is functional evidence coverage, not a six-question substitute for a full
benchmark, not generated answer accuracy, and not evidence of a LlamaIndex win.
Flat mode pays cold query embedding; later modes reuse that vector. Auto pays
cold routing; forced modes reuse that route. Do not compare their wall times as
independent end-to-end system latencies. Route and fetch stages are reported
separately to make their added work visible.

The runner intentionally accepts only the shipped public fixture. It stores
public evidence locally; use a reviewed encrypted store before persisting
sensitive query/source receipts in a broader evaluation. It will refuse to
reuse a nonempty output directory. An interrupted run leaves partial observations
without a completed summary; do not present these as complete.

## Relationship to STAIR

The inspiration is structure as an address space, not replication of a trained
Differentiable Search Index. Jev is not trained on these documents; Scone still
uses vector search, keeps original text, and fetches grounded source bytes.
No external RAG framework is imported by the native implementation.

The original SearchTome link returned `repository_expired` on September 24, 2026.
The discovered `SGK86/searchtome-subset` dataset explicitly describes an independent
scaled-down reconstruction. Neither is required for this native experiment, and
neither has been presented as a completed full SearchTome evaluation.

Before promoting this path, evaluate all questions in a frozen, structure-rich
corpus against Scone's existing retrieval and an appropriately configured
LlamaIndex reference, using matched models, corpus access and evidence budgets.
Report answer EM/F1, evidence recall, source bytes, failure counts, API cost and
latency distributions. Include misleading headings, cross-section questions and
source updates. A known-book experiment must not be compared with corpus-wide
retrieval without disclosing that difference.

References: [STAIR](https://arxiv.org/html/2609.03874v1),
[TypeSafe hierarchy cookbook](https://docs.typesafe.ai/cookbooks/hierarchical_classification),
[Choice API](https://docs.typesafe.ai/primitives/choice).
