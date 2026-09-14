"""The context lane's benchmark measures a lexical miss, and the lane recovers it."""

from __future__ import annotations

from scone_memory.testing.context_lane_benchmark import VERSION, cases, run_under_benchmark


def test_every_case_is_a_lexical_miss_by_construction():
    for case in cases(20):
        body = case["body"].lower()
        assert case["topic"] not in body and case["heading"] not in body, "the body says none of the question's words"
        assert case["tail"] in case["body"] and len(case["document"].encode()) > 200, "the tail is past the first chunk"
        assert case["heading"] in case["document"].lower() and case["topic"] in case["document"].lower()


async def test_the_lane_recovers_bodies_the_text_lane_cannot_see():
    report = await run_under_benchmark(count=20, limit=5)
    assert report.version == VERSION and report.cases == 20 and report.store == "memory"
    # The text lane alone sees a tail only where the topic word is in it,
    # and how many of those reach the window depends on the fusion weights;
    # what the lane adds over that is the claim.
    assert report.body_found_with - report.body_found_without >= 12, report.as_payload()
    assert report.chatter_first_with < report.chatter_first_without, "chatter that repeats the question loses first place"


async def test_the_sqlite_store_measures_the_same(tmp_path):
    report = await run_under_benchmark(count=4, limit=5, sqlite_path=str(tmp_path / "under"))
    assert report.store == "sqlite" and report.body_found_with > report.body_found_without
