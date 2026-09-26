# Counterfactual memory contracts — 2026-09-23

The [prototype and protocol](memory-contracts-v1.protocol.md) test whether
precomputed evidence-withdrawal judgments can make subsequent memory checks
cheap. Two unchanged diagnostic runs completed using direct Jev, resolved model
`jev-1.13.0`. This is a promising latency result, not a demonstrated accuracy gain
or a production-ready memory system.

## Same evidence, same semantic questions

Eight synthetic cases, 60 counterfactual worlds per run; eight empty worlds are
local controls. Each of the 52 nonempty worlds receives a fresh baseline request.
Compilation batches those same questions into eight requests. The two runs reuse
the same fixtures and are repeated measurements, not 120 independent examples.

| Measurement | Run 1 | Run 2 |
| --- | ---: | ---: |
| Compiled exact status classification | 54/60 | 55/60 |
| Fresh-check exact status classification | 55/60 | 54/60 |
| Unsupported claim incorrectly approved, either method | 0 | 0 |
| Compiled/fresh status disagreements | 1 | 1 |
| Compilation latency, median per case | 421.13 ms | 435.90 ms |
| Local contract lookup, median nonempty world | 0.0152 ms | 0.0207 ms |
| Fresh Jev check, median nonempty world | 334.05 ms | 346.77 ms |
| Compilation requests / fresh-check requests | 8 / 52 | 8 / 52 |
| Compilation Noul questions | 104 | 104 |
| Compilation input / output tokens | 26,516 / 2,060 | 26,516 / 2,060 |
| Fresh-check input / output tokens | 39,016 / 2,236 | 39,016 / 2,236 |

The compilation work is paid upfront. Local lookups make no model calls. Under
this unusually favorable workload, which visits every precomputed world, the
observed average latency breaks even after approximately two nonempty lookups
per contract. That is not a billing break-even estimate. A single-use answer or
frequently edited evidence can waste compilation work. An ordinary cache would
also answer repeated identical states locally after their first evaluation.

These timings exclude retrieval, source reads, generation, UI work and any
production consistency checks. They do not imply a sub-millisecond conversation.

## Failures retained

Both methods often returned `uncertain` where the diagnostic expected
`insufficient` or `conflict`. The missing bridge from release captain to owner
sometimes received support around .2–.25; naming Qdrant did not confidently
establish the absence of Pinecone. With explicit mutually conflicting ownership
records, Jev's support probability dropped despite instructions to judge support
and contradiction independently. The contract therefore did not reliably name
those states as `conflict`, though it withheld approval.

The one classification disagreement in each run crossed the fixed .2 boundary.
There was no observed accuracy advantage to compilation. No prompt, threshold,
fixture label or expected answer was changed to improve these numbers. The
all-safe support decisions on this small authored set do not estimate deployment
risk. No real temporal-memory benchmark or adversarial stress set has been run.

## Native storage journey

A separate live journey used a new local SQLite store, Scone MemoryEngine,
`qwen/qwen3-embedding-8b` through the configured API, and one direct Jev
compilation request. All five steps passed:

1. Two independent ownership records: supported.
2. Forget one record through Scone: still supported by the other.
3. Close/reopen the engine and reload the serialized contract: still supported.
4. Forget the remaining source: insufficient; no model call.
5. Ingest a new conflicting owner: recompilation required; old answer not reused.

Two earlier storage-harness attempts failed before reaching the checks: remote
embedding dimensions needed initialization, then the harness used `id` instead
of `episode_id`. Both were corrected. They are not included as successful
journeys. The sandbox-blocked initial provider attempt also produced no result.

The persisted artifact retains original evidence; this synthetic demo checks
answer eligibility, not complete erasure of derived data. Production retirement,
encrypted storage and coherent snapshot integration remain required. The live
webapp was not switched to this experiment.

## Reproducibility and next decision

Fixture SHA256:
`b835a752544350e70c31ec3edd9bcf2f8452cbe010f47ce8cb67c81f16707db3`

Raw reports are retained locally under
`bench-runs/memory-contracts-2026-09-23/`, outside git:

| Artifact | SHA256 |
| --- | --- |
| run-1.json | `491dfc4dc94af8b4bd7a3b7bce66d962678a1e2effd062d18fcc6420135bee51` |
| run-2.json | `026a281af9b6a41f4005ea3dfdf9fbbea2bc0910879bf51ff1b754cf1c60be3d` |
| storage-result.json | `ac884d000d76234c33ee392549c7a7a3d9cfe55a670cb84a1d1b1bc2207b841b` |

Continue as research. Next: evaluate an untouched temporal-memory dataset and
compare exhaustive compilation with a small, adaptively selected intervention
set and ordinary lazy memoization. Measure false reuse, unnecessary invalidation,
unused compilation, total tokens and latency together. Do not promote this into
the answer path based only on these diagnostics.
