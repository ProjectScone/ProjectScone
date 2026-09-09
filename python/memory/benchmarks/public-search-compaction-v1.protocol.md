# Repeated search-result compaction v1 — frozen before inference

Measure whether shortening repeated search results helps the production tool
loop answer questions. This is a development comparison, not a held-out score.
The feature remains disabled by default pending measurements.

## Data and arms

- All 200 original public-qa-v1 development questions: 100 HotpotQA and 100
  SQuAD 1.1, in their original order, with no wording changes or hand selection.
  Earlier development results informed this change. The 200 reserved questions
  remain unrun; do not describe this as an untouched evaluation.
- Same 2,176 original source paragraphs, including distractors, pooled across
  development/reserved sources. Copy the original SQLite and Qdrant stores.
  Verify corpus/question checksums and original/copy logical ledger equality.
  Current production initialization adds two SQLite indexes (`facts_space_id`
  and `facts_source_id`). Permit only those exact additive index statements when
  comparing the original and initialized ledger dumps; all data and remaining
  schema must match. Freeze the initialized dump for all subsequent checks.
  Before freezing vectors, verify original/copy Qdrant file equality and the
  serving image, storage mount and loopback port. Never index questions, gold
  labels, answers or a manually authored fact graph.
- Two arms differ only in `EvidenceToolLoop(compact_search_results=...)`:
  `off` and `on`. Both use `initial_search=True`, the production
  `SelfHostedStructuredToolChat` and computation disabled. Alternate off/on for
  even question indices and on/off for odd indices: 400 planned turns.
- The on arm executes every requested search. Only a nonempty result whose
  entire payload exactly matches an earlier accepted result can be shortened.
  A reference must be smaller than its full payload. Earlier evidence is
  revalidated; fresh receipts remain retained and checked before publication.
  Ranking changes, ordering changes, new sources, empty results and errors are
  not compacted. This measures the reference message and its generic next-step
  guidance together, not a retrieval cache or an isolated byte-size effect.
- Use original `benchmark_messages(question)`, including its existing shortest
  complete answer instruction. No output-schema option, answer requirements,
  question-specific prompting, gold evidence, reviewer, fine-tuning, answer
  repair or forced second search. Model-selected searches and reads are recorded
  unchanged. Do not replace failed or interrupted turns with retries.

## Runtime and budgets

- Python 3.14.7; self-managed Qdrant 1.19.1 server with a copied SQLite ledger;
  cached BGE-small-en-v1.5 and offline embedding initialization. No cloud model,
  storage or browser retrieval.
- Model `gemma4-e4b-ctx8k:latest`, pinned to its prior installed digest and
  verified `num_ctx=8192`. Temperature 0, reasoning disabled, max_tokens=2048.
- Default `ToolLoopLimits`: four tool calls, four tool rounds, 120-second whole
  turn deadline, 256,000 transcript bytes, 128,000 aggregate tool bytes and
  16,000 final-reply bytes. Initial search consumes a call. Scoped tools retain
  their default two-second and 64,000-byte limits. Compacted attempts still use
  a call; only the offered bytes count toward the aggregate tool-byte budget.
- Bytes are not model tokens. Record actual requests and reported token usage;
  the harness does not correct any provider context trimming.
- Run one benchmark turn at a time. Keep the user's memory service and unrelated
  models running. Record model residency at launch, every 20 turns and at
  completion. Latencies include recording, validation and possible contention;
  they are shared-workstation measurements, not isolated throughput.

## Integrity and failures

- Commit and push this protocol before the first inference request. Freeze
  runtime code, runner, scorer, upstream evaluator, original source/question
  files, embedding artifacts, model digest, versions and copied logical stores.
  Check code/input integrity during the run and all frozen material afterward.
- Preserve each planned turn's unattempted/running/completed/failed/interrupted
  state. Save every provider request, received decoded response bytes (including
  partial bodies), parsed decisions, actual tool preparations, exact prepared
  payload strings, offered payload strings, retained packets and raw answer.
  Recording preserves production response-byte limits and cancellation. An
  abrupt process death may lose bytes not yet persisted.
- Resuming a running turn marks it interrupted, with its partial records kept.
  Terminal turns are never rerun. Score only once all 400 turns are terminal and
  frozen checks pass. The inference runner never reads gold; the separate scorer
  verifies original gold checksums only after the integrity gates pass.
- Report initial search equality across paired arms, including mismatches and
  preparation failures. Later changes in model actions are part of the feature's
  effect. Do not silently drop slow, malformed or failed turns.

## Reporting

- Overall and per-dataset answer EM/token F1 under unchanged public-qa-v1 rules,
  scoring raw answers and giving zero credit to failures/interruptions/timeouts.
  Verify Hotpot answer scores against the original upstream evaluator.
  Supporting-fact and joint metrics are not measured here.
- Paired gains/losses/ties, abstentions, errors, model-call distributions and
  actual fresh retrieval counts. Report offered search references separately
  from read reuse; `searched_again: true` means retrieval executed again.
- Report repeated identical search preparations, reference counts, offered
  tool-payload bytes, full prepared payload bytes, serialized provider-request
  bytes and reported prompt/completion tokens with observation denominators.
  Full receipt retention means presentation savings are not storage savings.
- Literal annotation coverage from original passages is a retrieval diagnostic,
  not proof that evidence is sufficient or the generated answer is correct.
- Report turn/provider p50/p95 latency with denominators. Valid receipts, passing
  tests or fewer bytes do not establish improved answer accuracy. Any later
  tuning requires a new frozen experiment; reserved questions remain available
  for subsequent validation.
