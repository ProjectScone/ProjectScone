# Live integration checkpoint — September 24, 2026

This is a functional check of the new API on the six-question fictional handbook
fixture, not SearchTome, not a LlamaIndex comparison, and not answer EM/F1.
All six questions were exercised under all four modes: 24 observations.

| Requested mode | Expected fixture evidence present | Effective behavior |
| --- | ---: | --- |
| Broad vector | 6/6 | Broad vector for all questions |
| Automatic | 6/6 | Original procedure section once; scoped vector five times |
| Scoped vector | 6/6 | Scoped vector combined with broad vector for all questions |
| Original sections | 6/6 | Original fetch five times; vector fallback once when whole-book selection exceeded budget |

The procedure question retained all five labeled evidence strings. Automatic
original fetch returned **557 source bytes**, compared with **1,209 bytes** from
broad vector retrieval, under the same five-item / 1,800-byte cap. This is one
integration example, not an established average context-efficiency gain.

Measured API stages for automatic retrieval:

| Stage | Median | Range |
| --- | ---: | ---: |
| Cold Jev address routing | 704 ms | 611–1,305 ms |
| Jev fetch-mode decision | 330 ms | 296–381 ms |
| Automatic total with cached query embedding | 1,028 ms | 938–1,688 ms |

Broad retrieval's median was 685 ms including cold query embedding; its local
vector-search median was 0.62 ms. These totals **cannot be compared directly**:
automatic mode reused the query vector. This implementation adds provider work;
no latency improvement is claimed. Routing and encoding run concurrently on a
cold automatic call, which this reused-vector smoke run did not measure.

Jev reported 11,141 input and 1,123 output tokens across the cold automatic calls.
No dollar cost is inferred from tokens. The pinned Jev model was
`typesafe/jev-1.13-20260917`; embeddings requested
`nvidia/nemotron-3-embed-1b:free`, 2,048 dimensions, with explicit query/document
prefixes. The Nemotron route is not a pinned dated revision.

Earlier probes exposed two integration defects: menus initially hid descendant
topics, causing every route to abstain; then weaker parent selections overrode
stronger child selections. Both were corrected and preserved in regression
checks. The final probe therefore verifies known fixture behavior; it is not
an untouched evaluation set.

Local evidence lives under `bench-runs/structure-routing-2026-09-24/smoke-3/`:

- manifest SHA-256: `becccaeefe2182b2c413a32f215caf24cdf5fa17b34b40849093d63fc22ea072`
- observations SHA-256: `5d403e34da46f96332d55d06924f29f5c5a343040b4c2d873de9b1b3ddb6207c`
- summary SHA-256: `2cefee6e7be4139a68cc8871eef555028cb496a22b2637096625b735b8a7d5e3`

Relevant tests: **73 passed**. Strict mypy: **four modules passed**. Independent
review also reproduced and verified fixes for scoped-search fallback, heading-only
sources, malformed fetch decisions and quadratic ancestor lookup. Constructor-only
cost for 8,000 headings fell from approximately 2,419 ms to 3.6 ms in that check;
this is prototype bookkeeping, not an end-to-end retrieval speedup.

Promotion remains contingent on matched full-corpus evaluation, including ordinary
Scone retrieval and LlamaIndex, answer quality, source coverage, added latency,
cost and document updates. Current results establish working mechanics only.

The live probe used native checkpoint `3b2d5375`. A subsequent bounded-menu
preparation fix skips descendant outline work when a menu is too wide and uses
source offsets for valid menus. At 8,000 headings, rejected-menu preparation fell
from approximately 1,866 ms to 5.4 ms; the functional/type checks above include
that fix. The recorded live API hashes remain those of the original probe.
