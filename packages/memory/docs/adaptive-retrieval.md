# Adaptive retrieval and explicit evidence requirements

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

## Adaptive evidence retrieval

For applications with an explicit question plan, the SDK also provides a
model-free `StructuredEvidenceAssessor`. Each requirement asks for recorded
values of an exact subject/predicate, subjects of an exact predicate/object,
a simple directed path of one exact predicate to a named endpoint, or a recorded
attribute reached through explicitly allowed predicates. Every requirement needs a complete witness before
its verdict is `sufficient`; this verdict describes the supplied plan and
records, not semantic truth or general question-answer accuracy.

```python
from scone_memory.retrieval.structured_evidence import (
    EvidenceRequirement, StructuredEvidenceAssessor,
)
from scone_memory.retrieval.adaptive import AdaptiveRetriever
from scone_memory.retrieval.recall_scope import RecallScope

question = "Which dependency path connects aster to denver?"
assessor = StructuredEvidenceAssessor(question, (
    EvidenceRequirement(kind="path", subject="aster", predicate="depends on",
                        object="denver", max_hops=3),
))
result = await AdaptiveRetriever(memory, assessor).retrieve(
    "authorized-space", question,
    scope=RecallScope.validated(where={"collection": "manuals"}),
)
```

## Question-bound evidence requirements

The application authors the requirements and binds them to the exact question.
There is no automatic intent parser or implicit HTTP activation. Identity matching
is literal: synonyms, case differences and alternate predicates need an explicit
application mapping. Terminal, negative, and whole-index completeness claims are
unsupported. A missing path means no witness within the supplied candidate and
hop bounds; it does not prove there is no path in the knowledge store.

Applications that already hold authorized `EvidenceCandidate` records can inspect
each requirement directly, without another model call:

```python
assessment = await assessor.assess_with_coverage(question, candidates)
for row in assessment.coverage:
    print(row.requirement, row.witness_ids, row.followup_query)
```

`witness_ids` identify the exact matching facts and connecting path;
`selected_ids` also retain competing values around those witnesses. The report
includes every requirement and missing query, even though the adaptive decision
limits follow-up queries to three. `work_used` counts path-edge examinations;
candidate indexing and direct lookups have separate input bounds. A row can have
a positive witness and `work_exhausted=True` when traversal found evidence before
running out of budget. The aggregate decision remains `uncertain` in that case.
These diagnostics contain requirements and record IDs, without copying candidate
text. They do not replace scope authorization or source-retention validation.
The existing `assess()` interface returns the same report's `decision`.

## Frontier expansion and relation direction

To search from an already reached entity when a route is incomplete, opt into
`followup_strategy="bridge"` on `StructuredEvidenceAssessor`. For example, if
the requirement needs `invoice → team → office → location` and the first round
finds only the first two edges, the follow-up can search `office located in`.
The partial route is selected as one atomic group so adaptive retrieval carries
it into the next round and revalidates its sources. It is reported separately in
`bridge_ids`, with no positive witness and an `insufficient` (or budget-exhausted
`uncertain`) verdict. Only the complete original requirement can be sufficient.

This deterministic strategy keeps up to three reached frontier entities per
missing requirement, prioritizing deeper routes and breaking depth ties in
candidate/traversal order. A known continuation replaces its prefix as a search
frontier; another branch stays eligible. It reserves a hop for the missing
relation and retains competing values around each chosen path. Coverage reports
the primary `followup_query` and up to two `alternative_queries`. The decision
still issues at most three queries, prioritizing each requirement's primary gap
before branch alternatives. It does not exhaustively search every branch.
Names longer than the requirement identity limit are never truncated;
without a usable bridge it falls back to the original requirement query.
The default `"requirement"` strategy keeps the original query/selection behavior.

When a round has multiple follow-up queries, adaptive retrieval divides its
remaining candidate and byte capacity among the pending queries. Unused capacity
remains available to later queries. This prevents an early result window from
filling the entire pool before another query contributes. The total budgets stay
unchanged; `query_evidence_share` records omissions caused by that allocation.

For “Who uses Polaris?”, leave the subject unknown instead of reversing the
stored relation:

```python
users_of_polaris = EvidenceRequirement(kind="fact", predicate="uses", object="Polaris")
# Matches “Juniper uses Polaris”, not “Polaris uses Juniper”.
```

Fact requirements must name at least one endpoint; path requirements still need
both. These lookups inspect the bounded candidate set and retain all matching
subjects plus competing recorded values around their witnesses. They do not
claim to enumerate every user across an entire index.

For an attribute whose owning entity is not known in advance, use
`reachable_fact`:

```python
office_location = EvidenceRequirement(
    kind="reachable_fact", subject="invoice", predicate="located in",
    via=("assigned to", "managed by"), max_hops=3,
)
# Requires invoice -> team -> office, then office's recorded location.
# A team name, an unrelated office, or a missing bridge cannot satisfy it.
```

`via` contains 1–8 allowed exact predicates, which may occur in any order or
repeat along a route. The final `predicate` must be distinct from them.
At least one bridge is required; use `fact` for a direct attribute. `max_hops`
counts both bridge facts and the final attribute fact (2–6 total). Omit `object`
to retain recorded values, or supply it to require a particular value while
keeping competing observations. Breadth-first traversal keeps one shortest
supporting route per reachable subject and continues looking for other matching
subjects within the bounds. It does not enumerate every alternative route.
The full witnesses and competing values form one atomic group per requirement.
An attribute owner need not be a graph leaf: this contract cannot establish an
“ultimate” destination unless the application's relation semantics justify it.
It supplies recorded evidence connections, not a synthesized transitive fact.
This is available to both the adaptive assessor and the quote selector below;
it does not automatically change ordinary conversations or model tool choices.

The assessor accepts 1–8 requirements and at most 100 candidates / 128,000 UTF-8
candidate bytes. Path-edge work defaults to 256 and is configurable up to 2,048;
exhaustion returns `uncertain`, preserving other witnessed requirements. It emits
up to three follow-up search queries for missing requirements. Competing recorded
values around witnesses remain together in atomic groups; no winner is inferred.
The native retriever still owns candidate discovery, scope, timing, and source
revalidation. Invalidated selected groups are omitted together. This does not
turn a passage that looks like a triple into a stored fact.

## Atomic quote selection

The same explicit plan can govern quote-based answer selection with
`StructuredEvidenceSelector`. It requires all fact/path witnesses, including
competing recorded values, to fit in at most three whole cards. A missing bridge,
reversed path, or different predicate produces no selection. The answer remains
the original quoted records and citations; no model writes the public answer.

```python
from scone_memory.realtime.structured_selector import StructuredEvidenceSelector
from scone_memory.realtime.text import TextConversation

question = "Which dependency path connects aster to denver?"
requirements = (
    EvidenceRequirement(kind="path", subject="aster", predicate="depends on",
                        object="denver", max_hops=3),
)
conversation = TextConversation(memory, "authorized-space", "planned-answer-1",
    evidence_selector=StructuredEvidenceSelector(question, requirements),
    evidence_answer_policy="required",
    where={"collection": "manuals"},
)
try:
    reply = await conversation.reply(question)
finally:
    await conversation.close()
```

This is a fixed, application-authored question plan, not automatic intent
understanding. For another question, the application must supply its corresponding
plan. The selector checks only offered cards: missing or budget-omitted evidence
cannot establish a global negative. Path checks match recorded triples; they do
not certify extraction correctness, quote entailment, causation, or source truth.
`verified_accuracy` remains false. The synthetic tool-action development cases
also exercise this selector with hand-authored plans; passing them is a contract
check, not a natural-language generation accuracy result.

Selections from this selector are atomic: if the entire selected quote set
exceeds the answer byte limit, the renderer abstains instead of publishing a
partial chain. Other selectors may request this behavior through
`EvidenceSelection(atomic=True, card_ids=...)`. Receipts expose
`atomic_selection` and count output-budget omissions.

`evidence_answer_policy="required"` needs an evidence selector and never falls
back to generation. Empty or skipped retrieval returns a recorded abstention
with `source_status="none"`; failed preparation returns an error. No generation
provider is needed. The default `"when_available"` policy retains normal
generation when no memory is prepared. This SDK policy and selector are opt-in;
the default server configuration does not automatically create question plans.

## Model-assessed retrieval

An optional bounded loop assesses retrieved evidence, keeps selected records,
and searches for missing information before generation. The host supplies an
`EvidenceAssessor`; the core fixes the memory space and session filters for every
search. Model output can select existing IDs and propose queries, but cannot
change authorization or run tools. Candidate, query, round, byte and time limits
are independent. Source changes during assessment invalidate the affected
evidence; errors and timeouts return an explicit uncertain result.

```python
from scone_memory.providers.evidence_assessor import SelfHostedEvidenceAssessor
from scone_memory.providers.llm import OpenAICompatibleTextModel
from scone_memory.retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever
from scone_memory.realtime.text import TextConversation

endpoint = "http://inference.home.arpa:11434/v1"
model = "my-installed-model"
adaptive = AdaptiveRetriever(
    memory, SelfHostedEvidenceAssessor(endpoint, model, timeout=30),
    limits=AdaptiveLimits(max_rounds=3, max_queries=6, timeout_s=30.0),
)
conversation = TextConversation(
    memory, "authorized-space", "session-1",
    lambda: OpenAICompatibleTextModel(endpoint, model, timeout=45, trust_env=False),
    where={"collection": "manuals"}, adaptive_retriever=adaptive,
    recall_timeout=30, turn_timeout=90,
)
```

Use and close the conversation as above. The retriever must bind the same
engine; `recall_timeout` must cover its deadline, and the full turn also needs
time for generation. Greetings and overview retrieval keep their existing flow.
The assessor's `sufficient` verdict is a fallible model judgment, not an answer
accuracy guarantee. This option remains off by default and does not automatically
enable itself in the HTTP service when a model is loaded.

`AdaptiveRetriever(..., include_search_history=True)` additionally supplies a
private `EvidenceAssessmentContext` to an assessor implementing
`assess_with_context(question, candidates, context)`. The self-hosted assessor
supports this interface; existing `assess(question, candidates)` adapters remain
unchanged when the option is off. Incompatible adapters are rejected before
retrieval when history is enabled.

The context records completed query strings, round numbers, degraded-retrieval
flags, additions to each bounded candidate pool, and remaining query/round
budgets. It helps an assessor distinguish attempted searches from missing
information, following the research-history concept in the RAGFlow reference.
Zero additions can mean duplicates, filtering or exhausted capacity; it does
not establish absence or completeness. Counts describe search-time observations,
not currently retained evidence or globally new facts. Only the freshly checked
candidate snapshot can support a selection. History stays within the individual
run and is not added to public diagnostics; assessor input remains untrusted data.
The provider bounds serialized history to 64,000 UTF-8 bytes and keeps one model
request per assessment. No extra search, retry or sufficiency verdict is forced.
This is an optional capability, not a measured answer-accuracy improvement.

## Serve configuration

The standard `serve` launcher can mount it explicitly for custom-model,
saved-connection and persona **text** sessions:

```dotenv
SCONE_ADAPTIVE_RETRIEVAL=1
SCONE_ADAPTIVE_URL=http://127.0.0.1:11434/v1
SCONE_ADAPTIVE_MODEL=YOUR_INSTALLED_MODEL
SCONE_ADAPTIVE_TIMEOUT=15
SCONE_ADAPTIVE_MAX_ROUNDS=3
SCONE_ADAPTIVE_MAX_QUERIES=6
SCONE_ADAPTIVE_CANDIDATE_LIMIT=20
SCONE_ADAPTIVE_MAX_EVIDENCE_BYTES=16000
SCONE_ADAPTIVE_GRAPH_HOPS=3
# SCONE_ADAPTIVE_SEARCH_HISTORY=1  # Optional attempted-query context for the assessor.
# SCONE_ADAPTIVE_API_KEY=  # Only for an authenticated assessor endpoint.
```

This requires `SCONE_CONVERSATIONS_JOURNAL` and a separately configured text
model or persona to reply. Assessor credentials are independent of generation,
extraction and answer review. The full conversation-turn timeout still includes
retrieval, generation and any answer review; allocate enough time for all enabled
stages. Greetings and overview queries keep their existing routing. Voice is
unchanged. Graph hops default to zero (disabled); enabling 1..6 hops uses the
native `MultiHopLimits` defaults for the other per-expansion work bounds.

Served adaptive retrieval uses `retain_verified` for assessment failures and
empty selections, plus `original_and_selected` to retain the original query's
verified pool alongside later selection. These policies are described below;
none establishes answer accuracy. They preserve scope and source checks across
every round. `adaptive_retrieval` in conversation capabilities reports configuration,
budgets, graph hops and `search_history` independently of reply-model availability. Turn context
receipts report rounds, queries, fallback, truncation and graph work. These are
current-process context receipts, not a durable reconstruction after restart.
When embedding `create_conversation_app`, its optional `adaptive_retriever` must
bind the same engine; custom text factories must accept and honor the native
`adaptive_retriever` and `recall_timeout` keywords.

## Failure and recovery policies

The default `failure_policy="retain_verified"` recovers from assessor errors,
invalid decisions, and assessment timeouts by independently rechecking the last
bounded candidate snapshot. It reserves `min(1 second, timeout_s / 4)` within the
existing deadline for that check; it does not retry the model or run new searches.
Changed, deleted, out-of-scope, or unverifiable evidence is omitted. Retrieval
and source-verification failures still return no evidence. Cancellation propagates.

Recovered evidence has `status="uncertain"`, `evidence_basis="verified_candidates"`,
and `fallback_status="retained"`; the original sanitized assessment error remains
visible. This is an unassessed candidate pool, not a sufficient answer or a model
selection. Recovery preserves host-known atomic groups and groups from prior valid
decisions only; a failed response cannot establish new groups. Native context
receipts expose the basis, fallback status, and delivery completeness separately. Use
`failure_policy="empty"` when any assessment failure should discard all evidence.

A valid assessment can also return `insufficient` or `uncertain` with no selected
records. The separate default `empty_selection_policy="retain_verified"` retains
the final offered candidate snapshot after source revalidation. Its basis is
`unselected_candidates`, its round still records zero model-selected records,
and `fallback_status` remains `not_used`. The insufficiency or uncertainty stays
visible; keeping a known partial route does not establish its missing endpoint.
Candidate retention does not establish relevance either: the bounded pool may
include distractors that the assessor did not select.

This policy applies only when the final valid decision selected nothing. It does
not resurrect earlier pools discarded during follow-up searches, or replace a
nonempty selection that later loses its sources. Existing scope, deadline, byte,
and atomic-group checks still apply. Use `empty_selection_policy="empty"` to
preserve model-only selection, independently of assessment-failure handling.

## Preserving original candidates

To protect against later searches drifting away from the original question,
opt into `evidence_policy="original_and_selected"`. The retriever saves the first
verified query pool, including any enabled graph expansion, and combines it with
the final model selection. Reciprocal rank fusion gives each lane a vote using
`1 / (60 + rank)`; whole atomic components compete by their strongest member's
score. Ties use original-query order first. Packing uses the existing candidate
and UTF-8 byte limits, so either lane can lose records when the combined pool
does not fit. Scores indicate ranking, not relevance or factual confidence.

The host revalidates the union before packing. Original snapshots take precedence
for shared IDs; a later search cannot replace a changed original source under
the same identity. Verification reads at most two bounded pools within the
existing deadline. Each input lane is capped at 128,000 serialized bytes; the
combined output still uses the configured, potentially smaller context budget.
Valid atomic contracts learned in earlier rounds continue to apply across both
lanes, so restoring original evidence cannot expose a surviving group fragment.
`original_query_ids` and `model_selected_ids` distinguish the
origins of returned records and may overlap. Native context filters those origin
lists again after its own packing. A model's `sufficient` verdict describes its
selection; it does not certify the blended evidence or generated answer. Losing
selected evidence to validation or packing downgrades the result to `uncertain`.

This policy keeps the original pool even when a successful workflow ends with an
empty selection; it takes precedence over terminal empty-selection handling.
Assessor errors still use the separate failure policy. Follow-up retrieval and
selection behavior are unchanged. The default remains `evidence_policy="model_selected"`.
Compare with `--adaptive-evidence-policy original_and_selected` in the evaluator;
neither this option nor the adaptive strategy is automatically enabled in HTTP.
The explicit served configuration above selects `original_and_selected`.

## Controlled comparisons and graph groups

For a controlled comparison against existing compact paths, add
`--adaptive-model YOUR_INSTALLED_MODEL --baseline-paths --adaptive-timeout 30
--adaptive-rounds 3` to the generation evaluator. Each row records the selected
variant and adaptive diagnostics; frozen answer checks remain separate from
source coverage and manual semantic review. The assessment transport timeout
uses the requested adaptive budget; the retriever enforces the remaining total
budget across all calls. Reports record both limits and the failure policy; use
`--adaptive-failure-policy empty` to compare the explicit empty-on-failure behavior.
Use `--adaptive-empty-selection-policy empty` for the separate valid-empty-selection
comparison; the report records both policies.
Receipts distinguish assessment timeouts, provider failures and invalid model decisions without
including raw provider errors or source text.

For optional atomic relation selection, construct the assessor with
`group_relations=True, max_evidence_bytes=16000`. The pure
`retrieval.evidence_groups.build_evidence_groups` helper groups supplied facts
by exact object-to-subject matches, preserving branches and cycles. It does not
invent semantic links or search beyond the supplied candidate pool. Existing
stored-link kinds are still handled by the separate graph expansion stage.

To gather missing connecting facts **before** assessment, pass
`graph_limits=MultiHopLimits(...)` to `AdaptiveRetriever` (import it from
`scone_memory.retrieval.multihop`). The host expands verified recall seeds using
bounded stored-link reads and ledger-normalized object-to-subject joins, within the same
space, source filters, session exclusion, and adaptive deadline. With graph
expansion enabled, all candidate sources and facts share the engine clock
boundary; future-created sources are excluded. It verifies expanded source
records before disclosing them to the assessor.

In-memory, SQLite, MongoDB, PostgreSQL and Elasticsearch document stores expose
the bounded subject and incident-link reads used by this traversal, plus link
lookups for source revalidation. Candidate reads return at most 129 records,
in identifier order, including the traversal's lookahead record. PostgreSQL
uses indexes on space, subject or link endpoint, and identifier; Elasticsearch
uses exact keyword filters and bounded search sizes. The host still checks
source scope and validity for each candidate before retaining it. These reads
do not equate a connected route with an entailed answer.

This option prioritizes connected fact components within the adaptive candidate
and byte budgets. Exact components become host-owned atomic groups: partial
model selections or later source loss omit the whole group. These known groups
also survive an assessor failure, so fallback can retain a route gathered before
assessment. Use `group_relations=True` on the self-hosted assessor to let the
model select the components directly. Stored links retain their kinds and
orientation in the separate evidence graph; an exact component is not a claim
of causation or an inferred answer.

`graph_limits=None` keeps expansion disabled. Enable it in the generation
comparison with `--expand-relations`; reports record the graph limits and
per-expansion coverage and work. Graph limits apply to each expansion; the
adaptive round cap bounds their number and the total deadline bounds the run.
`store_calls` counts traversal and its source revalidation; additional adaptive
checks are bounded by the candidate count and deadline. Reaching a hop,
candidate, store-call, node, edge, or byte limit leaves explicit incomplete coverage. Even an exhausted
reachable graph does not establish query completeness. Graph expansion uses
bounded adjacency reads; it does not search beyond the fixed scope or infer unrecorded links. Initial recall retains its existing backend
behavior, including the current fact-seed lookup implementation.

A selected group expands to all of its original evidence IDs. The model-neutral
`EvidenceDecision.selected_groups` contract carries that requirement through
source revalidation and conversation packing: if any member changes or cannot
fit, the whole group is omitted. Independent ungrouped evidence can still be
used. Atomic membership is by evidence ID: separately recalled chunks remain
independent, even when they quote a grouped fact. This does not guarantee atomic
delivery of equivalent source content across representations. Receipts expose
group omissions; a complete group does not prove that the answer is sufficient
or correct. Grouping stays off by default.

The adapter's `max_evidence_bytes` bounds the serialized evidence array including
group metadata and joins. It is separate from the core's input-evidence budget
and the conversation's final context budget; configure all three explicitly.
Add `--group-relations` to the adaptive evaluator command to compare this mode
against ordinary compact paths. Every input record is represented once or
construction fails; no source or group is silently clipped to fit.

See [the executable native example](../examples/realtime_conversation.py). It uses
real Scone memory and scheduling with a scripted provider, not live inference.
