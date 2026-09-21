# Fixed-evidence reasoning full-set confirmation v1

Freeze before inference. Extend the completed 40-question pilot to all 200
questions (100 per dataset), regenerating both arms for 400 new answers. Keep
exactly the pilot's model, messages, evidence, limits, order rule and scorer.
The only arm difference remains reasoning_effort none vs medium; both have
2,048 maximum completion tokens and 90/95-second provider/capture deadlines.

Use the original Qwen + Jev prepared requests from
`jev-qwen-answers-20260919/run-2`. Do not reretrieve or reindex; the later
name-initial chunking fix is not being tested here. Verify request hashes and
complete schedules before scoring. No gold is read during inference. Keep
failures and truncations in the denominator; do not retry or select answers.

The 40 pilot questions overlap this run. Report all 200 and additionally the
160 questions outside the pilot, so repeated pilot questions do not masquerade
as independent confirmation. Even those 160 belong to the reused development
set and were seen in previous RAG evaluations; they are not untouched holdout.
No official leaderboard claim or production default change follows from a win.

Run the committed `jev_reasoning_answers.py --per-dataset 100` from an immutable
snapshot. Output: `bench-runs/jev-reasoning-answers-20260920/run-2/`. Record
full answer artifacts, request/code/protocol hashes, timing and all regressions.
