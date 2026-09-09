# Public answer review v1 results

**Decision: keep answer review optional.** With the tested Gemma model, the
existing review-and-revision policy reduced exact match from 64% to 62.5% and
F1 from 0.72949 to 0.71654. It adopted three revisions; all three scored worse.
All 200 cases completed, with no provider, validation or timeout errors, and
the process exited zero.

## Frozen comparison

The [protocol](public-answer-review-v1.protocol.md) was committed and pushed as
`fec3aae` before inference. It uses all 100 HotpotQA and 100 SQuAD development
questions in their original order, their original five-source context packets,
and the unchanged Gemma drafts. Those drafts match the original public baseline
200/200. No question rewrites, new retrieval, answer hints, fine-tuning or
replacement retries were used. The 200 reserved questions remain unrun.

The existing `SelfHostedAnswerReviewer` used host-owned draft spans, followed
by the production `review_answer` gate: at most two calls within one 120-second
deadline. A proposed revision is adopted only after a second supported review.
Otherwise report policy retains the original draft. No prompt or production
code was changed for this comparison. Each structured call had the existing
2048-token output limit, temperature zero and thinking disabled, using
`gemma4-e4b-ctx8k:latest` with an 8192-token context on self-managed Ollama.

This evaluates review against saved context. No live source validator was
supplied; every receipt remained `source_status=unchecked` and
`verified_accuracy=false`. It does not test conversation publication or prove
that a source remains authorized in a live store. Gold was read only after
all cases were terminal and frozen-artifact checks passed.

## Answer quality

| Dataset | Metric | Original draft | After review |
|---|---|---:|---:|
| All 200 | Exact match | 64% | 62.5% |
| All 200 | F1 | 0.72949 | 0.71654 |
| Hotpot, 100 | Exact match | 46% | 43% |
| Hotpot, 100 | F1 | 0.57335 | 0.54745 |
| SQuAD, 100 | Exact match | 82% | 82% |
| SQuAD, 100 | F1 | 0.88563 | 0.88563 |

For both exact match and F1, the paired outcome is **zero gains, three losses,
197 ties**. Every changed answer is a Hotpot answer. Official Hotpot scoring
matches all 100 original and all 100 final answer scores. Supporting-fact and
joint scores remain unmeasured.

An independent audit reproduced all 200 per-case scores, the three dataset
summaries, paired counts and adoption decisions, and checked all 210 raw
responses, frozen hashes, residency records and official evaluator parity.

| Review outcome | Count |
|---|---:|
| First pass: supported / uncertain / needs revision | 172 / 18 / 10 |
| Final receipt: supported / uncertain / needs revision | 175 / 23 / 2 |
| Proposed revisions | 10 |
| Adopted revisions | 3 |
| Proposals not adopted | 7 |
| Model requests | 210 |
| Completed cases / errors | 200 / 0 |

All ten proposals occurred on Hotpot. Seven were withheld because their second
reviews returned uncertain (five) or still needed revision (two). No proposal
improved exact match. One unaccepted proposal increased token F1 by repeating a
name from the question in a non-answer; that is not counted as a final gain.

The first reviewer approved 48 drafts that fail exact match (33 Hotpot, 15
SQuAD). This is a disagreement diagnostic, not a count of proven hallucinations:
exact-match failures can include aliases, formatting differences and verbosity.
Likewise, a supported label from the model is not independent proof of accuracy.

## What changed

These are all three adopted revisions. Revision descriptions abbreviate the
saved outputs; the untouched raw strings remain in the artifacts.

| Hotpot ID | Original answer | Adopted revision | Effect |
|---|---|---|---|
| `5ae151985542990adbacf74d` | Manchester Orchestra | Says the sources list members but do not compare the groups' sizes | Correct short answer becomes a non-answer; F1 1 → 0.12903 |
| `5a79f04d5542996c55b2dca7` | University of California, Berkeley | Adds a paragraph comparing founding dates and concludes Berkeley is older | Correct entity is retained, but short-answer scoring falls; F1 1 → 0.21622 |
| `5a888c9d5542997e5c09a612` | Audioslave | Gives the bands' locations but says they do not establish which is farther west | Correct short answer becomes a non-answer; F1 1 → 0.06452 |

These failures are not all equivalent. The university case is primarily output
format expansion. The two comparison cases withdraw the answer. Geographic
comparison also raises a policy boundary: city names alone do not explicitly
provide coordinates, while the benchmark expects geographic reasoning. This
experiment does not establish a single cause for every failure.

The other seven proposals were not adopted:

| Hotpot ID / topic | Original draft | Proposed change, abbreviated | Second review |
|---|---|---|---|
| `5ab5c263554299488d4d9a18`, Baltic Cup | INSUFFICIENT_EVIDENCE | Expands the abstention into a sentence | Uncertain |
| `5ac5382c5542996feb3fea43`, game publisher | INSUFFICIENT_EVIDENCE | Says the publisher and founding year are absent | Uncertain |
| `5ade6acf554299728e26c71a`, filmmakers | INSUFFICIENT_EVIDENCE | Says their creative-title counts are absent | Uncertain |
| `5ae0945d55429906c02daadd`, novelist | Lucy Maud Montgomery | Adds discussion of Anne Shirley and a 1939 novel | Needs revision |
| `5ab7f1b65542991d322237d3`, Soviet officer | None | Says the officer is not named in the sources | Uncertain |
| `5ac3a7f75542993915413890`, actor | Joseph Cotten | Replaces the name with an insufficient-information statement | Uncertain |
| `5ae6dd745542996d980e7ca0`, film's novella | INSUFFICIENT_EVIDENCE | Gives a film title but says its source novella is absent | Needs revision |

The gate preserved the correct novelist draft after an unhelpful proposed
expansion. It nevertheless accepted the three regressions above. Two judgments
from the same model can share a mistake; repeating review is not verification
independent of that model.

## Observed cost and limits

Review adds work after draft generation. Median added time was 3,827.11 ms;
p95 was 22,224.90 ms across 200 cases. The first case took 7,049.68 ms and the
slowest 27,900.86 ms. Provider usage reports total 289,860 input tokens and 6,583
output tokens across 210 requests; input totals can include cached tokens.

Residency checks reported only the expected Gemma model or an empty cold start,
with 9,636,843,355 loaded bytes and an 8192-token context. Loaded bytes are not
peak host RSS. During inference, short tests and an isolated worktree operation
ran on the same host. Some calls slowed to roughly 22–25 seconds despite short
outputs; memory compression/swap and other host processes were active. These
observations do not prove a causal explanation. Prefix caching, cold starts and
shared-host activity limit interpretation of latency as intrinsic review cost.

Keep the negative result. The next improvements need to preserve the answer's
output contract and handle comparisons using the available evidence; approving
a more verbose or more cautious answer is not sufficient. This comparison does
not establish statistical significance or generalize to every model, corpus or
review policy. Production defaults remain unchanged.

## Provenance

Raw requests, response bytes after HTTP decompression, parsed decisions and
untouched answers remain outside git in
`bench-runs/public-answer-review-dev-2026-09-09/`.

- Model digest: `858470a535014f5c6847e388069aa64cb4f892e99309c851b59be288eec784cf`
- Manifest SHA256: `71593c971d656cf70be08ef1782e9caedf3d5e0d6e75d3291045925ee52a6558`
- Observations SHA256: `79ae93793715a631e808529358a4fabad1bf2c37a23bfe659fbf49e062a216ba`
- Scores SHA256: `4228e27df81cf6939f4129b4976c3e207876816fa7f90296b3a503cbdd95bb32`
- Scorer SHA256: `217fd93488d17d4538d80d4b9ef4ad3c88e73bc8dee04bec3c500d40955b0366`
- Official parity SHA256: `60397069334257e91ae8a47027306772dcff595d363de1227718e274d64b38e2`
