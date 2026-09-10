# Public format-aware review v1 results

**Decision: keep answer review optional.** Sharing the original answer instructions did not improve this experiment. Exact match fell from 64% for the original drafts to 62%, compared with 62.5% after the preceding review experiment. All 200 cases completed; there were no provider, validation or timeout errors.

## Frozen comparison

The [protocol](public-format-review-v1.protocol.md) was committed and pushed as `a8d6025` before inference. The candidate uses the answer-requirements implementation from `3692c25`, all original 100 HotpotQA and 100 SQuAD development questions, unchanged five-source contexts and original Gemma drafts. The 200 reserved questions remain unrun. No retrieval, question rewriting, gold hints, fine-tuning or replacement retries were used.

The treatment passes the original short-answer instructions verbatim through `AnswerRequirements`. It includes the candidate reviewer’s generic requirements guidance as well as the instruction field; their individual effects are not isolated. Structural defaults remain text, 64,000 UTF-8 bytes and no line limit. These checks do not enforce brevity, instruction compliance or truth.

The model is the same self-managed `gemma4-e4b-ctx8k:latest`, with 8192-token context, temperature zero, thinking disabled and a 2048-token ceiling per structured call. Production review uses host-owned draft spans, report policy and at most two calls within one 120-second deadline. Only a supported confirmation can replace the original answer. There is no live source validator: every receipt retains `source_status=unchecked` and `verified_accuracy=false`.

Gold was read only after all cases were terminal and frozen-artifact checks passed. The unchanged official Hotpot evaluator matches all 100 original and 100 final answer scores. Supporting-fact and joint metrics remain unmeasured.

An independent audit reproduced all 200 per-case scores and primary/historical aggregates, checked all 221 raw responses and parsed decisions, verified frozen inputs and 12 residency records, and confirmed official Hotpot parity. Every recorded response finished with `finish_reason=stop`.

## Answer quality

| Dataset | Metric | Original drafts | Previous review | Format-aware review |
|---|---|---:|---:|---:|
| All 200 | Exact match | 64% | 62.5% | 62% |
| All 200 | F1 | 0.72949 | 0.71654 | 0.70949 |
| Hotpot, 100 | Exact match | 46% | 43% | 42% |
| Hotpot, 100 | F1 | 0.57335 | 0.54745 | 0.53335 |
| SQuAD, 100 | Exact match | 82% | 82% | 82% |
| SQuAD, 100 | F1 | 0.88563 | 0.88563 | 0.88563 |

Against the original drafts, both EM and F1 show **zero gains, four losses and 196 ties**. All losses occur on Hotpot. Against the preceding review, EM has two gains, three losses and 195 ties; F1 has two gains, four losses and 194 ties. The historical comparison is secondary and was not a concurrent randomized experiment.

## All changed answers

| Hotpot ID | Original draft | Final answer | EM / F1 effect |
|---|---|---|---|
| `5a861ea85542994775f60700` | No | INSUFFICIENT_EVIDENCE | 1 → 0 / 1 → 0 |
| `5abb8aaf5542993f40c73b1c` | Los Angeles | INSUFFICIENT_EVIDENCE | 1 → 0 / 1 → 0 |
| `5ae151985542990adbacf74d` | Manchester Orchestra | INSUFFICIENT_EVIDENCE | 1 → 0 / 1 → 0 |
| `5ab7f1b65542991d322237d3` | None | INSUFFICIENT_EVIDENCE | 0 → 0 / 0 → 0 |
| `5ac3a7f75542993915413890` | Joseph Cotten | INSUFFICIENT_EVIDENCE | 0 → 0 / 0 → 0 |
| `5abb8e2d554299642a094aa4` | John Candy | Eugene Levy | 1 → 0 / 1 → 0 |

Five replacements are abstentions; one replaces John Candy with Eugene Levy. Abstentions increase from 20 to 25. All six replacements received a supported second review, yet none improved the answer score. A same-model confirmation is therefore not sufficient evidence that a revision helps.

## What the saved evidence shows

- **Band-size comparison:** the supplied DC Talk passage calls it a trio, while the Manchester Orchestra passage lists four current members. Review nevertheless replaces the correct group name with an abstention. The relevant counting evidence was delivered.
- **Requested role:** the American Pie passage identifies Eugene Levy as the connecting actor; the film passage lists him alongside John Candy. Review returns the connecting actor instead of the requested co-star. This is a different failure from response verbosity.
- **Joining evidence:** the Chambers passage links Los Angeles to an appearance with Martin Short, and the Ed Grimley passage names Short as its creator. Review still abstains. The saved context contains those connecting statements; this observation does not independently establish every implied relationship in the question.
- **Negative claims:** the satirist question supplies a satirist description for Mencken and a fantasy-author description for Holdstock. The benchmark expects “No,” but those descriptions do not explicitly establish that Holdstock was not known as a satirist. This exposes an evidence-policy boundary as well as a scoring regression.

Two earlier regressions are avoided: the Berkeley and Audioslave drafts remain unchanged. However, the reviewer does not successfully verify them: Berkeley receives the same proposed answer twice with `needs_revision`; Audioslave’s proposed abstention receives an uncertain confirmation, so report policy preserves the draft. These are preserved answers, not demonstrated reasoning repairs. Manchester Orchestra remains a regression, and three new EM losses appear.

## Review behavior and cost

| Outcome | Count |
|---|---:|
| First pass: supported / uncertain / needs revision | 167 / 12 / 21 |
| Final receipt: supported / uncertain / needs revision | 173 / 18 / 9 |
| First-pass non-null proposed answers | 21 |
| First-pass proposals identical to the original draft | 8 |
| Actual changed final answers | 6 |
| First-pass proposals without a changed final answer | 15 |
| Model calls / completed cases | 221 / 200 |
| Structural format status satisfied | 200 |
| Provider, validation or timeout errors | 0 |

Of the 21 first-pass proposals, eight repeat the original draft and 13 differ; six different proposals are adopted and seven are rejected. Nine second-round decisions also contain a proposed answer, but each repeats the answer just reviewed. These do not trigger a third call or an additional adoption. The only first-pass proposal that improves exact match is “American” → “Yes” for the Parker/Kumin question; its second review still returns `needs_revision`, so it is not adopted. The first reviewer approves 53 original drafts that fail exact match. That is a diagnostic disagreement, not a proven hallucination count.

Median added review time was **5,061.97 ms**, p95 **16,409.20 ms**, first case **12,629.13 ms**, and slowest case **29,480.59 ms**. Provider usage totals **344,441 input tokens** and **7,963 output tokens** across 221 calls. Input totals may include cached tokens.

Residency checks permitted only the expected model or an empty cold start, reporting 9,636,843,355 loaded bytes and 8192-token context. Loaded bytes are not peak host RSS. Competing memory extraction was stopped for inference and the memory server was restored afterward. Short tests and git operations for a separate refactor ran on the same host. An additional snapshot showed swap in use; it does not establish the cause of latency variation. Cold starts, prefix caching and shared-host activity limit timing comparisons.

Preserve this negative result. Output instructions did not solve comparison, entity-role or evidence-sufficiency judgments. Further work should measure those capabilities explicitly; changing the questions, forcing known answers or treating format compliance as accuracy would conceal the problem. This development-set experiment establishes neither statistical significance nor performance on other models or corpora. Production defaults remain unchanged.

## Provenance

Untouched requests, raw response bytes after HTTP decompression, parsed decisions, original/final answers and frozen manifests remain in ignored `bench-runs/public-format-review-dev-2026-09-09/`.

- Model digest: `858470a535014f5c6847e388069aa64cb4f892e99309c851b59be288eec784cf`
- `manifest.json` SHA-256: `a0efe101d6c01a17717d1050c27d43c89a9f4b35c8a5a2088e3e3add8e11be70`
- `observations.json` SHA-256: `747a3f7b1c2732265f5bcc67f840c030943f34535975b51a81b3d2095702173a`
- `scores.json` SHA-256: `fa204842db33e0c0afe3c475963c7bd9c301a2e1b49bf612a2ebd8afa7f3d836`
- `score.py` SHA-256: `e8966c1ef74d7817d7ab51ed7dcbfab51867ba2e9743f9a4190afe0798798e08`
- `official-parity.json` SHA-256: `c206f9d1fef870a540d1000c00ff44c976583ae15aa78f88c437e9f0938cc430`
