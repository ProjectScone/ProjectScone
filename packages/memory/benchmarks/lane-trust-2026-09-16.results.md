# Per-query lane trust, measured: it recovers a light weight, and it does not beat the best one

**What was asked.** The scoreboard's retrieval row names one lever: the vector
lane's weight in fusion, one number for every query a space is ever asked. The
setting that wins at rank 5 is not the one that wins on MRR, which says the lane
is worth listening to on some questions and not on others. `lane_trust` decides
that weight per query from the shape of the lane's own scores — the top's
distance above the lane's middle, as a share of the lane's whole range — and
moves the voice inside the band from the configured weight up to the text lane's
1.0. No model, nothing added to the index.

**How.** `benchmarks/northstar_defaults.py --lane-trust off,on`, LongMemEval-S,
the Rust harness's stratified sample of 50 (`--seed 42`), chunk target 700
characters, rank fusion, no diversity, no reranker. The reference is LlamaIndex
0.14.24 on the same embedder, both its BM25+vector hybrid and its vector
default. Sessions folded and scored by `bench.comparative`'s own rule.

## With a real embedder (bge-small-en-v1.5, both sides)

| setting | R@5 | all@5 | R@10 | R@15 | MRR |
|---|---|---|---|---|---|
| vector weight 0.1, trust off | 0.900 | 0.800 | 0.940 | 1.000 | **0.857** |
| vector weight 0.25, trust off | 0.900 | 0.800 | 0.960 | 1.000 | 0.852 |
| vector weight 1.0, trust off | 0.920 | 0.840 | 0.980 | 1.000 | 0.854 |
| **vector weight 0.1, trust on** | **0.920** | **0.840** | **0.980** | 1.000 | 0.854 |
| vector weight 0.25, trust on | 0.920 | 0.840 | 0.980 | 1.000 | 0.853 |
| vector weight 1.0, trust on | 0.920 | 0.840 | 0.980 | 1.000 | 0.854 |
| our text lane alone | 0.900 | 0.780 | 0.940 | 1.000 | 0.844 |
| our vector lane alone | **0.940** | **0.860** | 0.960 | 0.960 | 0.850 |
| llamaindex hybrid (BM25+vector, RRF) | 0.920 | 0.840 | 0.940 | 0.980 | 0.848 |
| llamaindex default (vector) | **0.940** | **0.880** | 0.960 | 0.960 | 0.856 |

A light weight with trust on answers exactly as a full weight does: R@5 0.900 →
0.920, all-sessions@5 0.800 → 0.840, R@10 0.940 → 0.980, MRR a hair down
(0.857 → 0.854). At a full weight it changes nothing at all, which is the design
saying so out loud — the band is empty, the recall event records `no band to
move in`, and the numbers are identical to the row above.

So it removes the penalty for setting the weight too low. It does not beat the
best weight. **It is not a reason to change any default.**

## With the hashed-token embedder (the shipped default path)

| setting | R@5 | all@5 | R@10 | R@15 | MRR |
|---|---|---|---|---|---|
| vector weight 0.01 (the default), trust off | **0.900** | 0.780 | 0.940 | **1.000** | **0.844** |
| vector weight 0.25, trust off | 0.880 | 0.740 | 0.960 | 1.000 | 0.828 |
| vector weight 0.01, trust on | 0.880 | 0.780 | 0.940 | 0.940 | 0.790 |
| vector weight 0.25, trust on | 0.880 | 0.780 | 0.940 | 0.940 | 0.789 |
| our vector lane alone | 0.720 | 0.500 | 0.840 | 0.880 | 0.620 |

Here it loses: R@15 1.000 → 0.940 and MRR 0.844 → 0.790, against the engine's
own default.

The reason is the rule's boundary, and it is worth stating plainly: **separation
measures how decisive a lane is, not how right it is.** A hashed-token lane
scores 0.720 R@5 on its own and still produces sharp-looking gaps, because
hashed tokens collide into confident nonsense. Promoting it from a hundredth of
a voice to nearly a full one on that evidence outvotes a text lane that was
right. Decisiveness is only worth trusting from a lane that is worth trusting at
all, and the shape of one query's scores cannot tell you that.

## What ships

`lane_trust` is **off by default** and stays off, at every embedder. It is a
setting (`SCONE_LANE_TRUST`, `MemoryEngine(lane_trust=True)`) for a host running
a real embedder behind a deliberately light vector weight, and the recall event
records the voice it chose and what it read to choose it — including the reason
whenever it declined to judge, so a voice back at the floor is never mistaken
for a judgment.

## What this run found that does matter

With a real embedder, **our own vector lane alone (R@5 0.940, all@5 0.860) beats
every fused row we produce (0.920 / 0.840)** — and matches LlamaIndex's vector
default, which is the row we are behind on. Fusing the text lane in costs us
0.02 R@5 at every weight we tried, trust or no trust. The lever is not how loudly
each lane speaks; it is that rank fusion is mixing a weaker lane into a top-5
that was already right. That is the next thing to measure: fusion that can
decline to mix, judged on the same 50.
