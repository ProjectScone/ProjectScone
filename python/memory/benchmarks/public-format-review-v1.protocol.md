# Public format-aware review v1: frozen protocol

Test whether sharing the original answer instructions with the reviewer improves
review outcomes. The preceding review experiment reduced exact match from 64%
to 62.5%; one accepted revision kept the correct entity but expanded the answer
into a paragraph. This follow-up measures a proposed remedy across every original
case, not just known failures. It does not presume that format caused all errors.

## Fixed inputs and treatment

- All 100 HotpotQA and 100 SQuAD development questions, in their original order,
  unchanged five-source context packets and original Gemma draft strings.
- Verify original questions, prepared requests and drafts against the frozen
  public baseline and completed source-budget run. Also freeze the preceding
  review observations, manifest and completion for a paired historical comparison.
- No retrieval, question rewrites, source editing, answer hints, fine-tuning or
  replacement retries. The other 200 reserved questions remain unrun.
- Use the implementation from `3692c25`: `SelfHostedAnswerReviewer`, host-owned
  spans and production `review_answer`, report policy, at most two calls within
  one 120-second deadline. Only a supported second review can adopt a revision.
- The treatment is `AnswerRequirements(instructions=original_instructions)`.
  Read the original short-answer instruction verbatim from the prepared system
  message after its first newline. Assert it is identical for all 200 cases.
  Retain its original insufficient-evidence behavior. Add no task-specific hints.
- Other requirements retain their defaults: text format, 64,000 UTF-8 bytes,
  no line limit. Verify every original draft satisfies those structural limits.
  These limits do not mechanically enforce brevity or factual correctness.
- This introduces host instructions into the reviewer protocol; it is not an
  unchanged-prompt repetition of the prior experiment. Freeze the complete
  candidate package independently of the predecessor package.

## Execution and records

Self-managed Ollama at `127.0.0.1:11434`, `gemma4-e4b-ctx8k:latest`, digest
`858470a535014f5c6847e388069aa64cb4f892e99309c851b59be288eec784cf`.
Use the existing 8192-token context, temperature zero, thinking disabled and
2048-token ceiling per structured call. Run sequentially without warmup.
Check model identity and residency at launch, every 20 cases and completion;
permit only the expected model or an empty cold start. Suspend competing
background extraction during inference, then restore the memory service.

Freeze package sources, protocol, runner and all input artifacts before the
first request. Persist every outbound body, raw response body (lossless base64
plus readable text), parsed decision and original/final answer. No credentials
or headers are saved. Preserve errors, partial attempts and response bytes.
A running case interrupted by process death is scored as a failure on resume;
it is never retried. Resume only untouched cases after frozen checks pass.

This evaluates immutable saved evidence, not live authorization, retention or
conversation publication. Receipts must retain `source_status=unchecked` and
`verified_accuracy=false`. Record structural `format_status` separately from
model support; satisfying the format check does not prove instruction compliance.

## Scoring and decision

Read gold only after all 200 cases are terminal and integrity checks pass.
Use the existing public scorer and unchanged official Hotpot answer evaluator.
Report original draft versus final answer EM/F1, overall and per dataset;
retain failures in the 200-case denominator with zero scores. Ordinary failed
reviews preserve drafts according to report policy, with failures counted.
Also compare final answers with the preceding review run, reporting historical
paired gains, losses and ties. This is not a concurrent randomized comparison.

Report every changed-answer case, all proposed/adopted/rejected revisions,
review statuses, format statuses, errors, provider calls and added latency.
Separate proposal quality from the adoption gate. Neither a same-model second
judgment nor short-answer exact match proves semantic truth. Supporting-fact
and joint scores remain unmeasured. Timings remain subject to cold starts,
prefix caching and shared-host activity.

Publish results regardless of direction. Keep review optional unless evidence
supports a default change; a development-set improvement alone does not establish
general accuracy or significance. Raw artifacts remain ignored in
`bench-runs/public-format-review-dev-2026-09-09/`. Commit and push this protocol
before inference.
