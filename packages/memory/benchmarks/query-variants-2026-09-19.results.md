# A third lever closed: asking a question several ways does not find more of the answer

**Verdict: measured on two samples and not shipped.** The gain on the frozen 50
(R@5 0.900 → 0.940 with five variants) does not reproduce: on an independent 100
the same setting wins **nothing** and loses three items at k=5, with every
measure down. The code, its 26 tests and this benchmark are on PR #155's branch
and are not on main.

## What was tried

LlamaIndex's `QueryFusionRetriever` writes several restatements of a question
with a model, retrieves for each, and fuses the rankings. We already had all
three of its fusion modes (`fusion.rrf`, `relative_scores`,
`distribution_scores`) and a single model rewrite (`query_transforms.rewrite`,
which moved R@5 0.82 → 0.86 when it was measured), but not the several-queries
part. `query_variants` adds it, with two rules of its own: the question the
person asked is always searched and argues at full voice while each variant
argues at half, so a model writing nonsense cannot lose the question; and every
variant the rule refuses is recorded with its reason.

## The numbers

LongMemEval-S, hashed-token embedder, chunk 700, `llama3.1-ctx8k` writing the
variants, `benchmarks/query_variants.py`. Deltas carry the items behind them.

### Frozen 50 (`--n 50 --seed 42`)

| setting | R@5 | all@5 | R@10 | R@15 | MRR | k=5 W/L | k=10 W/L |
|---|---|---|---|---|---|---|---|
| question as asked | 0.900 | **0.780** | 0.940 | 1.000 | 0.844 | — | — |
| + 1 variant | 0.900 | 0.700 | 0.960 | 1.000 | **0.870** | 2 / 2 | 2 / 1 |
| + 3 variants | 0.900 | 0.680 | 0.980 | 1.000 | 0.817 | 2 / 2 | 2 / 0 |
| + 5 variants | **0.940** | 0.680 | **1.000** | 1.000 | 0.782 | 5 / 3 | 3 / 0 |

Read on its own this is a clean trade: more variants find more of the answer
deeper in the list (R@10 to 1.000, three items won and none lost) and cost the
top of the ranking (MRR 0.844 → 0.782, all-sessions@5 0.780 → 0.680). It looked
like a recall stage for a pipeline that reranks afterwards.

### Held-out 100 (`--n 100 --seed 7 --holdout-of 42:50`), five variants

| setting | R@5 | all@5 | R@10 | R@15 | MRR |
|---|---|---|---|---|---|
| question as asked | **0.970** | **0.860** | **0.990** | 0.990 | **0.904** |
| + 5 variants | 0.940 | 0.740 | 0.980 | 0.990 | 0.859 |

**0 items won and 3 lost at k=5; 0 won and 1 lost at k=10.** The frozen 50's
5-won-3-lost becomes 0-won-3-lost, and the "no losses at k=10" becomes a loss.
Nothing about the direction survives the second sample.

## What it does establish

The refusal rule earns its place: of 469 variants the model wrote for the
holdout, **26 were refused, every one of them for sharing no word with the
question** — about one in eighteen, on real model output, on both samples. A
version without that rule would have searched them.

And the question's full voice against each variant's half voice did what it was
built to do: on the frozen 50 at one variant, the fused ranking beat the
question alone on MRR (0.870 against 0.844), which is what a quiet second
opinion should look like when it is right. It simply is not right often enough.

## Why it is not offered as a choice

The same test that closed per-query lane trust applies: name a configuration
where a host should turn this on. One variant costs a model call per question to
move MRR by noise; five cost a model call and four extra searches to lose three
items in a hundred. There is no such row, so this is a measurement, not a
feature.
