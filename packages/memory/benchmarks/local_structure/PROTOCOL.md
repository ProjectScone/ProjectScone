# Local structure retrieval development experiment, v1

Freeze the implementation before running all 281 papers / 1,005 questions of the
existing QASPER v0.3 development export. This is a development experiment on a
previously inspected split, not a new holdout. No tuning after this run starts.

Compare four arms: native flat vectors, native vector-guided Jev routing, native
`mode='local_structure'`, and installed LlamaIndex vector + BM25 fusion. Native
local retrieval uses source-original sections when at most 2,000 bytes, otherwise
expands hits to intersected complete paragraphs/code/table blocks when the expanded
span fits that ceiling. Preserve raw chunks when expansion is too large; the
2,000-byte bound is an expansion ceiling, not a limit on unexpanded UTF-8 chunks.
Drop later overlapping spans. Existing limits still enforce candidate count and
aggregate bytes. No routing/fetch model calls occur on this path. Routing remains
available explicitly; no confidence threshold or automatic escalation is claimed.

Reuse the existing Nemotron 3 Embed 1B vectors exactly from a local copy of the
completed run's SQLite cache. Abort on a cache miss; make no new embedding API
calls. Retain original source-aligned 700-character indexed chunks. Source, input,
configuration, package versions, and initial cache hashes are frozen in a manifest.

Each arm gets at most 32 candidates and 8,000 candidate bytes. Score the union of
unique candidate texts with the existing pinned OpenRouter Jev
`typesafe/jev-1.13-20260917`, batches at most 64. Identical text shares judgments.
Each arm packs up to five items / 8,000 bytes including source headers. Rotate
native retrieval order by global question ordinal; vector-guided route cache is
zero. Use eight concurrent questions within each paper; papers run sequentially.
All artifacts stay local. The existing hosted benchmark endpoint authorization
applies; no service integration is added.

Measure retrieval and retrieval-plus-shared-rerank stage sums, p50/p95, route/fetch
calls and times, source bytes, retained usage receipts, and complete-paragraph
all-evidence/text-only F1 and recall. Report source/index preparation separately
from warm retrieval; query embeddings are already cached. Each arm is charged
shared rerank time for comparison, but count its physical cost only once. Larger
contexts may increase later generation latency; no answer generation occurs, so
make no answer-accuracy, chat-latency, or full end-to-end improvement claim.

Raw gold is read only by offline scoring after the complete inference journal is
frozen. Count every scheduled question; rerank provider failures retain explicit
failed rows and score zero for every arm. Native route/fetch failures preserve
the existing broad-vector fallback. Persist all four contexts, model receipts,
and route reasons. Resume requires matching manifest and retains saved rows,
including failures. Unpersisted paper work can repeat after interruption and must
be disclosed; saved rows are never replaced. Reject concurrent writers with a
local lock. Audit all 1,005 unique question identities and 4,020 arm contexts,
source/input/artifact hashes and byte caps. Check evidence F1 parity against the
official evaluator. Report paired paper-cluster bootstrap intervals (2,000 draws,
seed 20260924); primary comparison local minus vector-guided, controls flat and
LlamaIndex. Intervals are unadjusted and do not estimate repeated-model variance.
