# Fixed-evidence reasoning pilot v1

Question: does enabling reasoning improve evidence use when retrieval is held
exactly constant? Use Scone's OpenAICompatibleTextModel and capture_public_reply
on the original Qwen + Jev prepared messages, with no prompt changes or new
retrieval. OpenRouter documents configurable reasoning for Gemma 4 31B:
https://openrouter.ai/google/gemma-4-31b-it (checked 2026-09-20).

Select 20 questions per dataset by ascending SHA256 of `reasoning-v1:` plus ID,
without reading gold or selecting failures. Sort their union by the same hash.
Both direct and reasoning arms regenerate answers; alternate arm order by
question. Same model, temperature zero, 2,048 maximum completion tokens, 90-second
provider deadline, 95-second capture deadline. Only `think` changes: false maps
to reasoning_effort none; true maps to medium. Record public output, errors and
timings; internal reasoning is not captured. Equal completion limits do not imply
equal final-answer token allowance because providers may include reasoning tokens.

All 80 attempts remain in the denominator, including truncations and timeouts.
Verify full schedule and input/request/artifact hashes before scoring; score
loads gold only after inference finishes. Report EM/F1 and latency by dataset and
paired regressions. This is a development pilot, not proof on a full dataset.
If promising, freeze a full-set confirmation separately. Do not deploy a setting
solely because it wins this small sample.

Commit script and protocol before inference. Immutable snapshot, artifacts at
`bench-runs/jev-reasoning-answers-20260920/run-1/`. No baseline answers are reused.
