# Answer review, output contracts and extractive answers

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

## Optional answer review

A complete evidence path does not guarantee that a model follows it correctly.
The native conversation can review its public draft against the delivered memory
packet and attempt one correction:

```python
from scone_memory.providers.answer_reviewer import SelfHostedAnswerReviewer
from scone_memory.realtime.answer_review import AnswerReviewLimits

conversation = TextConversation(
    memory, "authorized-space", "reviewed-session",
    lambda: OpenAICompatibleTextModel(endpoint, model, timeout=45, trust_env=False),
    answer_reviewer=SelfHostedAnswerReviewer(endpoint, model, timeout=20),
    review_limits=AnswerReviewLimits(timeout_s=20.0, max_rounds=2),
    review_policy="report",
    turn_timeout=90,
)
```

The first review may identify unsupported claims, contradictions, incomplete
answers, or broken paths and propose a replacement. Scone adopts that replacement
only after a second review reports it supported. Evidence IDs must come from the
delivered packet, and issue quotations must match the draft exactly. Malformed
reviews never trigger an automatic repair call. The reviewer sees the current
question, draft, and memory packet. Explicit `answer_requirements`, when configured,
are shared with it; other system instructions and full conversation history are
not supplied.

When enabled, review buffers the draft. The observer receives the final text
once; only that text enters conversation history and assistant capture. Without
a reviewer or answer requirements, existing streaming behavior is unchanged. Greetings and other turns
without prepared memory skip this memory-specific review.

`report` retains the original draft if review fails or remains uncertain and its
sources can still be validated. `require_supported` rejects an eligible memory
reply unless review reports support. Both policies reject stale or unavailable
sources. The returned `answer_review` receipt records the outcome, correction,
issue codes, and source status separately; `verified_accuracy` is always false.
A supported review is a model judgment, not independent proof of correctness.

## Output contracts

For consumers that need a constrained answer, configure an output contract:

```python
from scone_memory.realtime.answer_requirements import AnswerRequirements

requirements = AnswerRequirements(
    instructions="Return only the requested entity name, without explanation.",
    max_bytes=160,
    max_lines=1,
)
conversation = TextConversation(
    memory, "authorized-space", "short-answer-session",
    lambda: OpenAICompatibleTextModel(endpoint, model, timeout=45, trust_env=False),
    answer_requirements=requirements,
    answer_reviewer=SelfHostedAnswerReviewer(endpoint, model, timeout=20),
    turn_timeout=90,
)
```

The same requirements guide generation and review. Code enforces nonblank text,
UTF-8 byte and line limits before publication or assistant capture; a trailing
line break counts as another line. `format="json_object"` additionally requires
one strict JSON object without fences, duplicate keys, or NaN/Infinity constants.
An optional `output_schema` adds the field validation described above, using
the same requirements for generation and review. Neither syntax nor schema
validation establishes instruction compliance or factual accuracy. Scone never
truncates or rewrites an answer to make it pass.

Requirements work without review and apply to ordinary generation, native tools,
extractive answers, and abstentions. Output is buffered until checked, including
turns with no retrieved memory. Incompatible output raises a content-free
`RuntimeError`; the user's message remains recorded, but the rejected assistant
text is not published or captured and the conversation closes. Budget checks
include requirements added to the system message.

An invalid proposed revision is rejected before a confirmation call. Under
`report`, a compliant original may still be returned; an invalid original is
withheld under either policy. `answer_review.format_status` describes the returned
text (`unchecked`, `satisfied`, or `rejected`). The `answer_format_rejected` error
can also identify a rejected proposal when the returned original is compliant.
Format acceptance does not establish support: `require_supported` still requires
a supported review and retained sources.

Standalone callers can pass `requirements=requirements` to `review_answer` or
`review_tool_answer`; they must inspect the receipt and withhold rejected formats
as well as stale/unavailable sources. Custom reviewers used with requirements
must implement `review_with_requirements(question, answer, evidence, evidence_ids,
requirements)` in addition to `review`. Legacy reviewers continue working when no
requirements are configured. These are programmatic host options; the default
HTTP server does not expose per-conversation output contracts yet.

Review uses one deadline, at most two model calls, and a reserve for final source
checks. Preparation and review share the same budget; each bounded source-read
pass also has a one-second cap. Source checks use the original context revision,
fixed scope, exact records, and immutable snapshots. Even an unrelated native
write changes the revision and can conservatively invalidate review. The full
conversation timeout must cover retrieval, generation, review, and capture.

## Reviewer configuration and comparisons

For an isolated comparison, add `--review-model YOUR_INSTALLED_MODEL
--review-timeout 20 --review-policy report` to the generation evaluator. Only the
candidate is reviewed. Reports preserve its public draft, final answer, review
receipt, and separate draft/review timing; gold answers never enter review.
For the standard HTTP server, configure the reviewer explicitly alongside
`SCONE_CONVERSATIONS_JOURNAL` and a text model or persona catalog:

```dotenv
SCONE_ANSWER_REVIEW_POLICY=require_supported
SCONE_ANSWER_REVIEW_URL=http://127.0.0.1:11434/v1
SCONE_ANSWER_REVIEW_MODEL=YOUR_INSTALLED_MODEL
SCONE_ANSWER_REVIEW_TIMEOUT=20
SCONE_ANSWER_REVIEW_QUOTE_MODE=text
# SCONE_ANSWER_REVIEW_API_KEY=  # Only when your reviewer requires authentication.
```

`off` is the default; `report` and `require_supported` follow the policies above.
The reviewer has its own endpoint, model and optional credential; it never
borrows the chat/extraction key. Configuration does not install a model. The
standard `serve` launcher applies review to custom-model, saved-connection and
persona **text** sessions. Voice sessions are unchanged. The authenticated
conversation capabilities endpoint reports `answer_review.configured` and
`answer_review.policy`, independently of text-model availability.

## Span-based review

`SCONE_ANSWER_REVIEW_QUOTE_MODE=spans` selects an alternative structured review
protocol. The host gives the reviewer numbered draft excerpts; each issue
selects one excerpt instead of copying its text. The host returns that exact
text through the existing `answer_quote` field. Only an `incomplete_answer`
omission can select no span. Unknown spans, extra fields and foreign evidence
identifiers are rejected. Set `quote_mode="spans"` on `SelfHostedAnswerReviewer`
for the same behavior in the SDK. The default `text` protocol is unchanged.

The span catalog preserves the whole draft, with at most 128 excerpts of at
most 2,000 characters each. Sentence/newline boundaries are preferred; long
units are split, and highly fragmented drafts use fixed-size excerpts to stay
bounded. These are mechanical spans, not claims inferred by another model.
The catalog adds input text but no model requests. It prevents altered draft
quotations, not incorrect review judgments; measure correct-answer rejection
as well as false approval before choosing a reviewer or protocol. The isolated
review evaluator accepts `--quote-mode spans` and records the selection in its
report. Compare both modes using the same model, corpus and limits.

Native and structured tool conversations can use the same review settings.
Review receives the exact retained tool packets, including quoted claims,
ordered paths, source identifiers and coverage limits. It performs no new
retrieval. Even a tool turn with no retained evidence is reviewed, so an early
answer cannot bypass a required review. `report` permits the original draft
after an uncertain or failed review only when source validation succeeds;
`require_supported` withholds it. A proposed replacement is accepted only after
a second supported verdict. Review remains a fallible model judgment.

The tool turn's original deadline also bounds review and its source checks;
review cannot start a fresh tool budget. Accepted replacements must fit both
the tool and conversation reply limits, plus the history limit. Source changes
block publication regardless of policy, and sources are checked again after
the public callback before capture. Leave time for review when configuring the
tool and whole-conversation budgets. This adds up to two reviewer requests;
the tool receipt's `model_calls` counts generation/tool decisions, while
`answer_review.rounds` reports review requests separately.

Reviewed text arrives as one final public delta. Failed required reviews save
no assistant reply and expose a content-free `answer_review` diagnostic in the
current process's turn receipt. Lifecycle failure messages survive restart;
full per-turn review/context receipts are not reconstructed after restart.
When embedding `create_conversation_app` directly, pass a `ConversationReview`
from `scone_memory.runtime.conversation_review`; custom runtime factories must
accept and honor its `answer_reviewer`, `review_policy` and `review_limits`
keywords. The standard server supplies compatible native factories.

The review endpoint must support structured JSON output with `anyOf` and `const`.
Status-specific branches prevent a constrained decoder from returning, for
example, `supported` alongside a proposed correction. Host validation still
checks quotations, evidence IDs and response bounds; schema validity does not
establish that the review judgment is correct.

## Reviewer evaluation and recorded limitations

Evaluate the reviewer separately from answer generation with labeled drafts:

```sh
python -m scone_memory.testing.answer_review_evaluation \
  --fixture tests/fixtures/answer_review/v1.json \
  --output /tmp/scone-review-baseline.json \
  --endpoint http://127.0.0.1:11434/v1 --model YOUR_INSTALLED_MODEL
```

The included original synthetic cases are **development** cases covering direct
facts, invented facts, justified abstention, missed answers, unresolved conflicts
and incomplete routes. Add separately held-out cases before making quality claims.
`tests/fixtures/answer_review/contrastive.json` adds 20 synthetic development
drafts in ten pairs: each pair shares the same question and sources but contains
one acceptable and one unacceptable answer. It covers negation, unit conversion,
ownership changes, cross-source routes, causality, quantifiers, inventory scope,
tables, conflicting observations and instructions embedded in source text.
Pass that path to `--fixture` to compare reviewers on these cases. Reporting a
low false-approval rate alone is insufficient: a reviewer that rejects both
members of every pair accepts none of the correct answers.
`tests/fixtures/answer_review/completeness.json` adds eight development drafts
where both members contain supported statements, but only one answers every
explicitly requested part. It covers a receiver's storage medium, owner plus
date, conditional action plus deadline, and attribution of conflicting values.
These cases distinguish answering the question from merely saying something
true about the sources. They are not held-out evidence of general accuracy.
Each call reviews the original draft once; proposed revisions are recorded only
as a boolean and are not adopted. This isolates reviewer judgment from generation
and correction quality. Labels, categories and splits never enter model inputs.

Reports separate false approvals, false rejections, abstentions and failures,
with summaries by split and category. Approval precision uses approved drafts as
its denominator; false-approval rate uses all labeled unacceptable drafts;
acceptable-answer recall uses all labeled acceptable drafts. Decision coverage
counts only `supported`/`needs_revision`, and labeled agreement counts correct
decisions over **all** observations. An unavailable or uncertain review never
counts as a correct rejection. Missing denominators are `null`, not perfect scores.
The report omits source/draft/revision text and provider error messages. It includes
a SHA-256 of the normalized fixture, labels, timing and enum diagnostics. Output
must be a new file. Only the explicitly selected model is called; an optional
credential comes from `SCONE_ANSWER_REVIEW_API_KEY`.
Use `--progress` for one flushed JSON event on stderr after each completed case,
including failures. Events contain case ID, repeat, completed/total counts,
status, error code and elapsed milliseconds; they omit evidence, drafts, labels,
revisions, credentials and provider error messages. The final report is still
written only when the run completes. SDK callers can pass a synchronous
`on_observation` callback to `evaluate_reviews`; it runs outside the review timer,
and callback failures propagate instead of becoming model failures.
Timeouts rely on cooperative asynchronous providers; a late result earns no
credit even if a provider swallows cancellation. The evaluator cannot forcibly
interrupt blocking synchronous work inside a custom provider.

## Optional extractive answers

For memory questions where exact recorded wording matters, a model can select
evidence instead of composing a free-form answer:

```python
from scone_memory.providers.evidence_selector import SelfHostedEvidenceSelector

conversation = TextConversation(
    memory, "authorized-space", "extractive-session",
    lambda: OpenAICompatibleTextModel(endpoint, model, timeout=45, trust_env=False),
    evidence_selector=SelfHostedEvidenceSelector(endpoint, model, timeout=20),
    evidence_answer_timeout=20,
    turn_timeout=90,
)
```

When memory is prepared, Scone builds bounded source cards and makes at most one
selection call. The model returns up to three known card IDs; Scone renders the
original quotations and source IDs. No free-form generator runs on this path.
Complete supplied paths and connected contradictions stay together in a card;
their constituent passages cannot bypass that grouping. Whole cards are omitted
when they exceed a budget. A path records ordered statements and stored link
directions; it does not assert a new transitive relationship or infer an endpoint.
An exact same-episode passage duplicate is removed only after its standalone
claim card fits the budget. Different sources, text, and claims stay separate;
`deduplicated_card_count` is distinct from budget omissions.

Selection cards also carry typed claim triples, their origins and source IDs,
and ordered path steps. This preserves route identity when several claims quote
the same paragraph. A structural object-to-subject match remains distinct from
a stored relationship and its traversal direction. These fields count toward
the evidence byte budget; they do not change the public quotation rendering or
establish that a quotation entails a stored claim. Existing custom selectors
can still construct cards without this optional metadata.

Sources are checked before and after selection against the original scope,
revision, and records. Preparation, selection, and validation share one deadline.
Stale sources, invalid selections, or provider failures suppress the answer. A
valid empty selection returns a fixed no-support message. Only the final rendered
text enters history, capture, and the text callback.

The `evidence_answer` receipt lists delivered cards and evidence IDs, omissions,
and source status. `verified_accuracy` remains false: exact quotations prevent
new model-authored claims, but the model can still select irrelevant or incomplete
evidence. This is an extractive answer mode, not a guarantee of fluent generation
accuracy. Turns without prepared memory use the normal model and streaming flow.
An evidence selector and answer reviewer cannot be enabled together. Custom
selectors implement the `EvidenceSelector` protocol; their lifecycle belongs to
the caller. This native option is not automatically enabled on HTTP routes.

Add `--evidence-selector-model YOUR_INSTALLED_MODEL --evidence-answer-timeout 20`
to the generation evaluator to enable it for the candidate only. Reports separate
context evidence coverage from `selected_evidence_coverage`, record the actual
answer mode, and count free-form generation calls. Gold labels are used only for
scoring after selection. Quoting sources can inflate lexical scores; compare
selection relevance and evidence coverage separately from generative accuracy.
