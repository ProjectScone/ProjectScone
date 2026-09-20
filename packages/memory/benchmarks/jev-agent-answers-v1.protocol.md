# Native Jev agent answer experiment v1

Freeze before inference. Run all 200 reused questions (100 HotpotQA, 100 SQuAD)
against the validated 2,176-document Qwen/Qdrant index from the completed Qwen
single-pass run. Use Scone's production EvidenceToolLoop and ScopedMemoryTools,
with Jev on every search, 64 search candidates, and at most 32 reranked passages.
Use the same paid Gemma model, temperature zero, thinking disabled, 256 output
tokens per model call, and unchanged benchmark final-answer instructions.

One host initial search returns five sources. The model may search for missing
bridge facts or read surrounding source chunks. Four total tool calls (including
initial search), four tool rounds, 30 seconds per tool, 120 seconds per turn.
Keep the loop's existing byte limits, scope enforcement and publication-time
source validation. Do not supply gold, aliases, reference answers, or tailored
per-question instructions. Retain every attempt and provider/protocol failure.

This is a follow-up experiment against the prior Qwen + Jev single-pass answers,
not a simultaneous randomized experiment. Retrieval-tool formatting, additional
instructions, inference calls and context lengths differ; the contrast measures
the whole native agent workflow. It cannot isolate any one of these factors.
The prior generation-only timing is not comparable to full agent-turn timing.

Record full model-facing tool transcripts, accepted responses, evidence packets,
tool outcomes, Jev model resolutions, errors, timings, and immutable artifact/code
hashes. Verify corpus content, lexical readiness and Qdrant point count first.
Prewarm trimmed query vectors to match production recall; follow-up searches may
make additional embedding calls. Never read gold until all trials terminate and
integrity checks pass. Failures remain in the 200-question denominator.

Score standard EM/F1 by dataset and overall with the existing Scone scorer. Keep
regressions and inspect why tools helped or hurt. Do not change the official
normalization or accept synonyms just to improve scores. This reused development
set does not establish held-out or official leaderboard performance.

Artifacts: `bench-runs/jev-agent-answers-20260920/run-1/`. Commit this protocol and
tested harness before running inference from an immutable source snapshot.
