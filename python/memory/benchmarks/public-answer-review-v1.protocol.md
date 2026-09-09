# Public answer review v1: frozen protocol

Evaluate the existing optional answer-review pipeline on all 200 unchanged
development questions and Gemma five-source drafts from the public baseline.
The prior source-budget experiment improved evidence coverage but reduced
generation accuracy; this follow-up measures whether review helps with the
evidence already delivered. It does not select questions by success or failure.

## Inputs and execution

- Original 100 HotpotQA and 100 SQuAD questions, in their original order, original
  five-source production context packets and unchanged Gemma draft strings.
- Use the completed source-budget generation run's five-source responses,
  already verified identical to all 200 historical baseline answers.
- Keep the other 200 reserved questions unrun. No new retrieval, source editing,
  reference answers, labels, gold hints or synthetic evidence reach the reviewer.
- Existing `SelfHostedAnswerReviewer` and `review_answer`, with `quote_mode=spans`,
  report policy, at most two review calls and a shared 120-second review deadline.
  A revision is adopted only if the second review supports it. Otherwise the
  original draft remains the output according to the existing production policy.
- Self-managed Ollama at `127.0.0.1:11434`; existing
  `gemma4-e4b-ctx8k:latest`, digest
  `858470a535014f5c6847e388069aa64cb4f892e99309c851b59be288eec784cf`,
  8192-token context, temperature zero, thinking disabled. The existing reviewer
  uses structured output with a 2048-token ceiling per call. No prompt changes.
- Run sequentially. Check installed model identity and residency at launch,
  every 20 cases and completion; only the expected model or an empty cold start
  is permitted. No model warmup or replacement retries.
- Freeze all package Python sources, this protocol, the runner, original query
  and prepared files, and predecessor observations/manifests/completion hashes
  before the first request. Record each outbound body, inbound raw body and
  parsed decision before proceeding. Response bytes after HTTP decompression are
  retained as base64 alongside readable text, including malformed UTF-8 bytes.
  Headers and credentials are not recorded.
- Persist a running marker before each case. Resuming converts previously
  running cases to interrupted failures, never retries them, and proceeds only
  through unattempted cases after verifying the same frozen manifest. Preserve
  recorded partial calls; abrupt process death may lose an unfinished response.

## Scope and scoring

This tests semantic review against immutable saved context, not live storage
authorization or source-retention revalidation. No live source validator is
supplied; review receipts must retain `source_status=unchecked` and
`verified_accuracy=false`. No conversation messages are published or captured.

Read gold files only after all 200 cases are terminal. Report original and final
answer exact match/F1 using the same evaluator as the public baseline, per
dataset and overall. Keep all 200 in the denominator; interrupted or unhandled
case failures score zero. Ordinary unavailable reviews retain the original draft
as dictated by report policy, with their failure receipts separately counted.

Report paired gains/losses, adopted and rejected revisions, first-pass statuses,
final receipts, call count, elapsed time and every changed-answer case. Preserve
all raw answers, including proposed revisions that were not accepted. Verify
Hotpot answer scores against the unchanged official evaluator. Supporting-fact
and joint scores remain unmeasured. A reviewer approving an exact-match failure
is a diagnostic, not proof of hallucination: exact match is not a semantic label.

Separate the quality of revisions from the quality of the approval gate. A
second judgment from the same model is not independent verification. These are
development-set results; enabling review by default requires demonstrated
benefit, and one comparison does not establish significance or general accuracy.
Timings may include cold starts, prefix caching and other shared-host activity.

Raw artifacts stay outside git under
`bench-runs/public-answer-review-dev-2026-09-09/`. Commit and push this protocol
before inference; publish the outcomes regardless of direction.
