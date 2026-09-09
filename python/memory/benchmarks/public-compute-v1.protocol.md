# Public computation-tool comparison v1 — frozen before inference

Measure whether exposing exact quoted-span computation improves the actual
structured tool loop. Arithmetic correctness and model operand selection are
separate requirements. This is a development comparison, not a held-out score.

## Data and arms

- All 200 original public-qa-v1 development questions: 100 HotpotQA and 100
  SQuAD 1.1, in their original order, with no wording changes or hand selection.
  The 200 reserved questions remain unrun. These development results have
  informed earlier changes and must not be called an untouched evaluation.
- Same 2,176 original source paragraphs, including distractors, pooled across
  development/reserved sources. Copy the original SQLite and Qdrant stores;
  verify logical contents and the original corpus/question checksums. Before
  freezing the vector snapshot, require original/copy file equality and verify
  the serving container image, storage mount and loopback port mapping. No
  questions, answers, gold labels or manually authored fact graph are indexed.
- Two arms: `off` and `on`, differing only in
  `ScopedMemoryTools(enable_computation=...)`. Both use the production
  `EvidenceToolLoop(initial_search=True)` and `SelfHostedStructuredToolChat`.
  Use off/on for even question indices and on/off for odd indices: 400 planned
  turns. Never replace an interrupted, timed-out or failed turn with a retry.
- The on arm includes the production calculator schema and its guidance. This
  measures the complete optional feature, not the isolated effect of arithmetic.
  Both arms may perform additional search/read actions with different queries.
  Preserve those actions; do not rewrite the user's original question.
- Use original `benchmark_messages(question)` (including the existing shortest
  complete answer instruction). No question-specific prompts, gold evidence,
  reviewer, fine-tuning, answer repair or forced calculator invocation.

## Runtime and budgets

- Python 3.14.7; self-managed Qdrant 1.19.1 server and copied SQLite ledger;
  cached BGE-small-en-v1.5, offline embedding initialization. No cloud model,
  storage or browser retrieval.
- Model `gemma4-e4b-ctx8k:latest`, pinned to its prior installed digest and
  verified `num_ctx=8192`. Temperature 0, reasoning disabled, max_tokens=2048,
  the production structured adapter's default output limit.
- Production `ToolLoopLimits` defaults: four tool calls, four tool rounds,
  120-second whole-turn deadline, 256,000 transcript bytes, 128,000 aggregate
  tool bytes and 16,000 final-reply bytes. Initial search consumes a call.
  Scoped tool defaults retain their two-second and 64,000-byte limits.
- No claim that a byte budget equals the model's token window. Record provider
  prompt/completion token usage and request sizes to diagnose context pressure.
  Model context trimming, if any, is not corrected by the harness.
- One benchmark turn at a time. The user's memory service and unrelated local
  services remain running. Record all resident models at launch, every 20 turns
  and at completion, and preserve host notes. Do not unload unrelated models.
  Latencies include host recording/source validation and possible contention;
  they are shared-workstation measurements, not isolated model throughput.

## Integrity and failures

- Commit/push this protocol before the first inference request. Freeze runtime
  code, runner, scorer, evaluator, original source/question files, embedding
  artifacts, model digest, runtime versions and copied logical store contents.
  Check code/input integrity during the run and all frozen material afterward.
- Persist each planned turn as unattempted/running/completed/failed/interrupted.
  Save every provider request, received decoded response bytes (including
  partial bodies on transport errors/cancellation), parsed tool
  decisions, tool arguments/results, retained source packets and final output.
  Stream recording preserves the production byte cap; an early close keeps only
  bytes received before that close, marked incomplete. Abrupt process death may
  lose bytes not yet persisted. Raw model answers are scored verbatim. No
  sampling again to replace failures.
- A resumed running turn becomes an interrupted failure. Preserve partial calls
  and tool reads. Terminal turn files are not rerun. Full scoring requires all
  400 planned turns terminal and frozen checks passing.
- Record both arms' initial search packets. Report exact packet equality when
  both preparations succeed; do not silently exclude mismatches or failures.
  Later retrieval differences are part of the feature's behavioral effect.
- The runner does not read gold. Only after terminal completion, the separate
  scorer verifies original gold checksums and scores all 200 questions per arm.

## Reporting

- Overall and per-dataset answer EM/token F1 using existing public-qa-v1 rules,
  unchanged raw answers and zero credit for failures/interruptions/timeouts.
  Verify Hotpot answer scores against the original upstream evaluation script.
  Supporting-fact/joint scores are not measured unless separately implemented.
- Paired gains/losses/ties, abstentions, failures/error categories; exact changes
  in answers and their retained evidence. Report calculator attempts, successes,
  operation types, rejected arguments/quotes and final tool call distributions.
  Attempt counts use successfully parsed model tool calls; malformed protocol
  responses remain provider failures with raw bytes, not parsed attempts.
- Report complete evidence annotation coverage from retained original passages,
  using the same literal matching limitations as prior reports. Separate
  retrieval coverage from answer accuracy and computational validity.
- Turn and provider p50/p95 latency, model-call counts, token usage, first-turn
  and residency observations. Give explicit denominators for missing timings.
- Do not infer improved accuracy from a valid calculation or unit-test result.
  No default change follows automatically. Any subsequent tuning requires a
  new frozen experiment; reserved questions remain available for later validation.
