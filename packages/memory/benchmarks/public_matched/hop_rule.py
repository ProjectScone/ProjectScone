"""Choosing how a second hop joins the first pass, on one half of HotpotQA, and scoring it on the other.

The equal-weight fusion in ``twohop.py`` lost the top ranks. The saved first-pass and hop lists
(``rankings-hotpotqa-twohop.jsonl``) let other rules be scored without searching again. Questions are split by
SHA-256 of their id: even hashes are the development half (3,729), odd the test half (3,676).

- **Rules tried on the development half only:** keep the first pass's top K (1 to 5) whole, then fill the remaining
  places round-robin from the hop lists of the first 1 or 2 seeds and the rest of the first pass, skipping
  documents already placed.
- **Rule chosen:** K = 3 with one seed. It keeps hit@1 and all@2 whole (87.0 / 38.5) and lifts all@5 from 66.2 to
  71.3 and all@10 from 77.4 to 83.4. It also needs one hop search, not two.
- **Test:** the test half was then scored once. Its numbers are in RESULTS.md.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

KEEP = 3
SEEDS = 1


def is_test(question_id: str) -> bool:
    return int(hashlib.sha256(question_id.encode()).hexdigest(), 16) % 2 == 1


def keep_then_fill(first: Sequence[str], hops: Sequence[Sequence[str]], *, keep: int = KEEP, depth: int = 10) -> list[str]:
    """The first pass's top ``keep`` as they are, then one from each hop list and the rest of the first pass in turn."""
    out = list(first[:keep])
    pools = [list(h) for h in hops] + [list(first[keep:])]
    while len(out) < depth and any(pools):
        for pool in pools:
            while pool and pool[0] in out:
                pool.pop(0)
            if pool and len(out) < depth:
                out.append(pool.pop(0))
    return out
