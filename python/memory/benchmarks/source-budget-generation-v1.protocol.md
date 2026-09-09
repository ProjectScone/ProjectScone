# Retained-source generation comparison v1

Freeze before this experiment's first inference request. This development
comparison follows the [source-budget coverage result](source-budget-v1.results.md).

- Use all 200 original public-QA development questions and the saved five-,
  eight- and ten-source requests from that experiment. Keep the reserved 200
  questions unrun. Verify the complete 600-observation artifact and all original
  question strings before inference. Do not prepare replacement contexts.
- Run all three configurations on every question, rotating order by question
  index modulo three: 5/8/10, 8/10/5, 10/5/8. This creates 600 new planned
  responses, including a contemporaneous five-source control. One attempt per
  arm/question; these are declared new observations, not replacement retries
  for historical responses.
- Use the same installed `gemma4-e4b-ctx8k:latest` digest as public-qa-v1, with
  8192 model context, temperature 0, `think=False`, at most 256 output tokens
  and a 120-second deadline. Verify installed model settings and digest before
  inference and periodically during the run. Use only self-managed Ollama.
- All requests retain the original question and response-format instruction.
  No gold access during inference, fine-tuning, prompt rewrites, answer repair,
  evidence selection by labels, or retries. Larger source limits are chosen
  after development coverage results; this is not untouched held-out testing.
- Save planned requests, model configuration, artifact/code/protocol/runner
  hashes and every public response/status immediately. Persist a running marker
  before each request. On resumption, a previously running request is a terminal
  interrupted failure; only unattempted observations may run. Preserve partial
  output where captured and keep failures in the fixed denominator.
  Adapter-returned failures retain captured deltas; an abrupt process death can
  lose deltas still in memory. Do not report that text as durably captured.
- Record model residency and inference latency, including first-token timing.
  Keep one model loaded and infer sequentially. No deliberate model warm-up;
  preserve cold-start cost in its scheduled observation. Arm rotation reduces
  ordering bias but does not eliminate cache, machine activity or residency
  effects. Temperature zero does not guarantee deterministic responses.
- Score only after all 600 observations are terminal, using unchanged original
  answer EM/F1 rules. Report each arm overall and separately for Hotpot and
  SQuAD, failures, abstentions, status counts, first-token/total p50 and p95,
  exact-match and F1 gains/losses/ties against the new five-source control.
  Report eight versus ten on the same questions as well.
- Keep the historical five-source baseline separate. Report its score and
  response changes versus the new five-source control, whose full requests
  must match it. Do not overwrite earlier responses or claim repeat stability
  without checking it. Freeze historical response and completion-file hashes
  before inference without parsing their answers. Verify Hotpot answer scores against the unchanged
  official evaluator. Supporting-fact and joint generation scores remain
  unmeasured.
- Inspect answer regressions and evidence gains after scoring. No default
  change follows automatically; report quality/latency tradeoffs and uncertainty.
  Preserve the prior context-preparation process's exit-134 shutdown failure
  separately; its complete audited observations are the inputs to this run.
