"""Retrieval measurement for the Python engine, under the Rust harness's
definitions so the two products can be compared on the same yardstick
and never confused for each other.

For each LongMemEval item: a fresh space; every haystack session stored
as one conversation episode whose text is the turns joined as
``role: content`` lines, whose source is the session id (the ground
truth) and whose created_at is the haystack date; then one recall of the
question at limit k. Recall@k any: at least one evidence session appears
among the top-k items' sources. Recall@k all: every evidence session
does. Items with no evidence session (abstention items) count toward the
denominator only when ``include_abstention`` is set, matching the Rust
run that reported on all 500; the official evaluator excludes them.
"""

from .runner import BenchItem, ItemResult, RunReport, load_items, run

__all__ = ["BenchItem", "ItemResult", "RunReport", "load_items", "run"]
