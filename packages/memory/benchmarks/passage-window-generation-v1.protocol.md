# Passage-window generation comparison v1

Frozen before this experiment's first response, 8 September 2026. This is a
development comparison, following the [context coverage experiment](passage-window-v1.results.md).

- Use all 200 original public-QA development questions in their original order;
  do not select only questions with improved evidence coverage. Keep the 200
  reserved questions unrun.
- Use the already prepared `neighbor_chunks=1` requests from the separate
  context experiment. Verify its artifact hash and freeze all 200 requests
  before inference. Original question text and response-format instruction stay
  unchanged. Five ranked anchors, up to 24 total sources, 8,000 context bytes.
- Run `gemma4-e4b-ctx8k:latest`, the highest exact-match model in the baseline,
  once per question. This model choice is informed by the development baseline;
  this experiment cannot establish a universal model ranking.
- Verify the installed model digest matches the baseline. Keep the same 8192
  model context, temperature 0, `reasoning_effort=none`, maximum 256 output
  tokens and 120-second deadline. Use only the self-managed inference endpoint.
- Sequential inference, no question edits, no answer repairs or replacement
  retries. Preserve public output and completion state immediately. Interrupted
  started requests become terminal failures; resumption attempts only previously
  unattempted questions. Retain every failure in the 200-question denominator.
- Freeze source, protocol, script and request hashes; check them during the run.
  Read no gold labels in generation. Score afterward with the unchanged answer
  EM/F1 functions verified against HotpotQA's evaluator.
- Compare against the 200 original Gemma baseline responses. Report overall
  and dataset-specific EM/F1, abstentions, failures, exact-match gains/losses,
  and inference latency. Preserve both sets of answers. Do not replace the
  original baseline scores with this experiment's scores.
- This is a historical-control development comparison, without a contemporaneous
  rerun of the disabled configuration. Temperature 0 does not guarantee repeat
  determinism. Model residency, cache, order and machine activity confound timing;
  score differences are observations, not proof of causality or generalization.
- Do not enable the option by default based solely on three improved context
  coverage cases. Inspect regressions as well as gains; any later tuning requires
  another separately frozen experiment before evaluating untouched questions.
