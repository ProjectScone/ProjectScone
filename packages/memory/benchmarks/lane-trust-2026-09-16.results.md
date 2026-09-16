# A lever closed: deciding the vector lane's voice per query does not move retrieval

**Verdict: measured on two samples and not shipped.** The gain on the frozen 50
is one item and does not reproduce on an independent 100; on the hashed-token
embedder — the shipped default path — it loses. The code was written, tested and
measured, and is not on main: it is PR #143, commit `98260950`, closed unmerged,
and reproducing anything below needs that branch's `--lane-trust off,on`.

This file exists so the next person to reach for this lever finds the
measurement before building it again.

## What was tried

The scoreboard's retrieval row named one lever: the vector lane's weight in
fusion, a single number for every query a space is ever asked. The setting that
wins at rank 5 was not the one that won on MRR, which suggested the lane is
worth listening to on some questions and not others.

`lane_trust` decided that weight per query from the shape of the lane's own
scores — the top's distance above the lane's **middle**, as a share of the
lane's whole range — moving the voice inside the band from the configured weight
up to the text lane's 1.0. No model, nothing added to the index.

## Why it does not work

**Separation measures how decisive a lane is, not how right it is.** Nothing in
one query's score shape distinguishes a lane that found the answer from a lane
that confidently found the wrong thing, and a lane that ranks badly overall
still produces sharp-looking gaps. The hashed-token lane is the clean
demonstration: 0.720 R@5 on its own, and promoting it on the evidence of its own
decisiveness costs the engine real ground.

## The numbers

LongMemEval-S, chunk target 700 characters, rank fusion, no diversity, no
reranker, `benchmarks/northstar_defaults.py --lane-trust off,on`. Deltas carry
the items behind them, because 0.02 on 50 items is one item.

### Frozen 50 (`--n 50 --seed 42`), bge-small-en-v1.5 on both sides

| setting | R@5 | all@5 | R@10 | R@15 | MRR |
|---|---|---|---|---|---|
| weight 0.1, trust off | 0.900 | 0.800 | 0.940 | 1.000 | 0.857 |
| weight 1.0, trust off | 0.920 | 0.840 | 0.980 | 1.000 | 0.854 |
| weight 0.1, trust on | 0.920 | 0.840 | 0.980 | 1.000 | 0.854 |
| weight 1.0, trust on | 0.920 | 0.840 | 0.980 | 1.000 | 0.854 |
| our text lane alone | 0.900 | 0.780 | 0.940 | 1.000 | 0.844 |
| our vector lane alone | 0.940 | 0.860 | 0.960 | 0.960 | 0.850 |
| llamaindex hybrid (BM25+vector, RRF) | 0.920 | 0.840 | 0.940 | 0.980 | 0.848 |
| llamaindex default (vector) | 0.940 | 0.880 | 0.960 | 0.960 | 0.856 |

Trust on at weight 0.1 wins **1 item of 50** at k=5 and loses none. At weight
1.0 it changes nothing at all: the band is empty and the recall event records
`no band to move in`.

### Held-out 100 (`--n 100 --seed 7 --holdout-of 42:50`), same embedder

| setting | R@5 | all@5 | R@10 | R@15 | MRR |
|---|---|---|---|---|---|
| weight 0.1, trust off | 0.970 | 0.890 | 0.990 | 0.990 | 0.923 |
| weight 1.0, trust off | 0.980 | 0.890 | 0.990 | 0.990 | 0.923 |
| weight 0.1, trust on | 0.970 | 0.890 | 0.990 | 0.990 | 0.922 |
| weight 1.0, trust on | 0.980 | 0.890 | 0.990 | 0.990 | 0.923 |
| our text lane alone | 0.970 | 0.860 | 0.990 | 0.990 | 0.904 |
| our vector lane alone | 0.970 | 0.860 | 0.980 | 0.990 | 0.938 |
| llamaindex hybrid (BM25+vector, RRF) | 0.970 | 0.880 | 0.980 | 0.990 | 0.890 |
| llamaindex default (vector) | 0.940 | 0.840 | 0.970 | 1.000 | 0.895 |

Trust on at weight 0.1: **1 item won, 1 item lost** at k=5, and **no item
changed** at k=10. Across both samples that is 2 items gained and 1 lost in 150
— noise, not a direction.

### Hashed-token embedder, frozen 50 (the shipped default path)

| setting | R@5 | all@5 | R@10 | R@15 | MRR |
|---|---|---|---|---|---|
| weight 0.01 (the default), trust off | 0.900 | 0.780 | 0.940 | 1.000 | 0.844 |
| weight 0.01, trust on | 0.880 | 0.780 | 0.940 | 0.940 | 0.790 |
| weight 0.25, trust off | 0.880 | 0.740 | 0.960 | 1.000 | 0.828 |
| weight 0.25, trust on | 0.880 | 0.780 | 0.940 | 0.940 | 0.789 |
| our vector lane alone | 0.720 | 0.500 | 0.840 | 0.880 | 0.620 |

R@15 1.000 → 0.940 (3 items) and MRR 0.844 → 0.790 against the engine's own
default. This is the largest movement in any of the three runs, and it is a
loss.

## Two claims withdrawn

An earlier draft of this file, written from the frozen 50 alone, said fusing the
text lane in "costs us 0.02 R@5 at every weight", and named fusion that can
decline to mix as the next lever. **The holdout refutes it.** Our vector lane
alone beat the fused row by 2 items to 1 on the frozen 50, and on the held-out
100 it is the other way round — 0 items to 1 — with fusion ahead at R@5 (0.980
vs 0.970). There is no lane-versus-fusion effect here, only a sample too small
to tell two settings apart. No lever came out of this run.

## What the run does establish

The sweep **reproduces the 2026-09-15 baseline exactly** at the same setting
(frozen 50, weight 1.0: 0.920 / 0.840 / 0.980 / 1.000 / 0.854), so the harness
and the record agree. The standing scoreboard picture is unchanged: at our
defaults we are ahead of both LlamaIndex rows on the held-out 100, and behind
its vector-only default by one item at k=5 on the frozen 50.
