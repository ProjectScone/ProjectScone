"""Fusing lanes by their scores, not only by their ranks.

Recall fuses its lanes by reciprocal rank: a chunk's place in each lane,
nothing else. That is robust to lanes whose scores mean different things,
and it throws away how far apart the scores were. A vector lane where the
first hit is 0.99 and the second 0.10 says something rank fusion cannot
hear. Relative-score fusion keeps it: each lane's scores are scaled to
0..1 across that lane's candidates, and the scaled scores are added with
the same weights. Distribution fusion keeps it another way: each lane's
scores are placed by the lane's own mean and spread, so an outlier is an
outlier and not the yardstick. Both are options beside rank fusion, not
replacements, and the result says which one ran.
"""
from __future__ import annotations

import math

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval import fusion


def test_each_lane_is_scaled_to_its_own_range_before_adding():
    scores = fusion.relative_scores([[(1, 10.0), (2, 5.0), (3, 0.0)], [(3, 0.4), (1, 0.2)]])
    assert scores == pytest.approx({1: 1.0 + 0.0, 2: 0.5, 3: 0.0 + 1.0})


def test_weights_apply_after_scaling():
    scores = fusion.relative_scores([[(1, 2.0), (2, 1.0)], [(2, 9.0), (1, 3.0)]], weights=[1.0, 2.0])
    assert scores == pytest.approx({1: 1.0, 2: 2.0})


def test_a_lane_whose_scores_do_not_spread_gives_every_candidate_full_credit():
    assert fusion.relative_scores([[(1, 0.5), (2, 0.5)]]) == pytest.approx({1: 1.0, 2: 1.0})
    assert fusion.relative_scores([[(7, -3.0)]]) == pytest.approx({7: 1.0})


def test_a_lane_that_reports_no_score_contributes_its_order():
    """A bridged index may rank without a cosine and report NaN. Its order
    still counts: first gets 1, last gets 0, evenly between."""
    scores = fusion.relative_scores([[(1, math.nan), (2, math.nan), (3, math.nan)]])
    assert scores == pytest.approx({1: 1.0, 2: 0.5, 3: 0.0})


def test_score_fusion_hears_a_gap_rank_fusion_cannot():
    """A leads the vector lane narrowly and trails the text lane badly; B
    is a close second in both. Rank fusion sees A first and last, B second
    twice, and puts A ahead. Scores say B was nearly as good as A in the
    vector lane and nearly as good as C in the text lane."""
    vector = [(1, 0.99), (2, 0.98), (3, 0.10)]
    text = [(3, 10.0), (2, 9.9), (1, 1.0)]
    by_rank = fusion.rrf([vector, text])
    by_score = fusion.relative_scores([vector, text])
    assert max(by_rank, key=lambda c: (by_rank[c], -c)) != 2
    assert max(by_score, key=by_score.get) == 2


@pytest.fixture
async def memory():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    for text in ("the harbour crane was repainted", "crane repairs took four days", "a note about lunch"):
        await engine.remember("s", text)
    yield engine
    await engine.close()


async def test_recall_says_which_fusion_ran_and_rank_stays_the_default(memory, monkeypatch):
    calls = []
    original = fusion.relative_scores

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fusion, "relative_scores", spy)
    plain = await memory.recall("s", "crane", limit=3)
    assert plain.fusion == "rank" and calls == []
    scored = await memory.recall("s", "crane", limit=3, fusion="score")
    assert scored.fusion == "score" and calls == [1]
    assert {item.chunk_id for item in scored.items} == {item.chunk_id for item in plain.items}


async def test_an_unknown_fusion_is_refused_before_any_lane_runs(memory):
    with pytest.raises(InvalidInput, match="fusion"):
        await memory.recall("s", "crane", fusion="magic")


async def test_the_evidence_event_names_the_fusion():
    from scone_memory.observability.events import InMemoryEventLog

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), events=InMemoryEventLog()).open()
    try:
        await engine.remember("s", "the harbour crane was repainted")
        await engine.recall("s", "crane", fusion="score")
        [event] = await engine.events.query("s", kind="recall", limit=1)
        assert event.payload["fusion"] == "score"
    finally:
        await engine.close()


async def test_over_http_and_on_the_command_line(memory):
    import io
    import json

    from httpx import ASGITransport, AsyncClient

    from scone_memory.api import create_app
    from scone_memory.runtime.cli import build_parser, run

    async with AsyncClient(transport=ASGITransport(app=create_app(memory, {"k": "s"})), base_url="http://fixture") as client:
        auth = {"authorization": "Bearer k"}
        good = await client.get("/v1/recall", params={"q": "crane", "fusion": "score"}, headers=auth)
        assert good.status_code == 200 and good.json()["fusion"] == "score"
        assert (await client.get("/v1/recall", params={"q": "crane"}, headers=auth)).json()["fusion"] == "rank"
        assert (await client.get("/v1/recall", params={"q": "crane", "fusion": "magic"}, headers=auth)).status_code == 422
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--space", "s", "--json", "recall", "crane", "--fusion", "score"]), memory, io.StringIO(""), out)
    assert code == 0 and json.loads(out.getvalue())["fusion"] == "score"


async def test_the_benchmark_can_run_either_fusion(tmp_path):
    from scone_memory.bench import load_items, run

    dataset = [{"question_id": "q1", "question_type": "single-session-user", "question": "which crane was repainted",
                "question_date": "2023/05/20 (Sat) 02:21", "haystack_session_ids": ["s0", "s1"],
                "haystack_dates": ["2023/05/10 (Wed) 09:00", "2023/05/11 (Thu) 09:00"],
                "haystack_sessions": [[{"role": "user", "content": "the harbour crane was repainted"}],
                                      [{"role": "user", "content": "lunch was late"}]],
                "answer_session_ids": ["s0"], "answer": "x"}]
    path = tmp_path / "d.json"
    path.write_text(__import__("json").dumps(dataset))

    asked = []

    class Spying(MemoryEngine):
        async def recall(self, *args, **kwargs):
            asked.append(kwargs.get("fusion", "rank"))
            return await super().recall(*args, **kwargs)

    def make():
        return Spying(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    report = await run(make, load_items(path), ks=(1,), fusion="score")
    assert report.fusion == "score" and report.recall_any[1] == 1.0 and asked == ["score"]
    asked.clear()
    plain = await run(make, load_items(path), ks=(1,))
    assert plain.fusion == "rank" and asked == ["rank"]


def test_distribution_fusion_scales_a_lane_by_its_own_mean_and_spread():
    """Three scores 0, 5, 10: mean 5, deviation ~4.08, range 5 ± 12.25; the
    middle sits at exactly 0.5 and the ends within the range, not at 0 and 1."""
    scores = fusion.distribution_scores([[(1, 10.0), (2, 5.0), (3, 0.0)]])
    assert scores[2] == pytest.approx(0.5)
    assert 0.5 < scores[1] < 1.0 and 0.0 < scores[3] < 0.5 and scores[1] - 0.5 == pytest.approx(0.5 - scores[3])


def test_an_outlier_is_an_outlier_not_the_yardstick():
    """Relative fusion lets one 1000 push eleven close scores to nearly
    zero; distribution fusion keeps the eleven near the middle of their own
    lane, and the clip bites because eleven is enough for one score to sit
    past three deviations (a lane of ten or fewer cannot)."""
    lane = [(0, 1000.0)] + [(i, 10.0 + i * 0.1) for i in range(1, 12)]
    relative = fusion.relative_scores([lane])
    distributed = fusion.distribution_scores([lane])
    assert relative[6] < 0.01 and 0.4 < distributed[6] < 0.6, (relative[6], distributed[6])
    assert distributed[0] == pytest.approx(1.0), "an outlier past three deviations is clipped to the top, not made the yardstick"
    assert distributed[11] > distributed[1] > 0.4, "the close scores keep their order among themselves"


def test_distribution_fusion_keeps_the_conventions_of_the_other_modes():
    assert fusion.distribution_scores([[(1, 0.5), (2, 0.5)]]) == pytest.approx({1: 1.0, 2: 1.0}), "no spread: full credit"
    assert fusion.distribution_scores([[(7, -3.0)]]) == pytest.approx({7: 1.0})
    assert fusion.distribution_scores([[(1, math.nan), (2, math.nan), (3, math.nan)]]) == pytest.approx({1: 1.0, 2: 0.5, 3: 0.0}), "no score: the order"
    two = fusion.distribution_scores([[(1, 2.0), (2, 1.0)], [(2, 9.0), (1, 3.0)]], weights=[1.0, 2.0])
    assert two[2] > two[1], "weights apply after scaling"
    assert fusion.distribution_scores([[], [(1, 1.0)]]) == pytest.approx({1: 1.0}), "an empty lane is silent"
    assert "distribution" in fusion.FUSIONS


async def test_distribution_fusion_is_a_choice_on_recall_and_that_choice_runs_the_distribution_scaling(memory, monkeypatch):
    seen = []
    original = fusion.distribution_scores

    def spy(lanes, weights=None):
        seen.append([len(lane) for lane in lanes])
        return original(lanes, weights)

    monkeypatch.setattr(fusion, "distribution_scores", spy)
    chosen = await memory.recall("s", "crane", limit=3, fusion="distribution")
    assert chosen.fusion == "distribution" and seen and chosen.items, "the result names the mode, and the mode's own scaling ran"
    plain = await memory.recall("s", "crane", limit=3)
    assert plain.fusion == "rank" and len(seen) == 1, "rank fusion does not touch it"
    with pytest.raises(InvalidInput, match="fusion must be one of rank, score, distribution"):
        await memory.recall("s", "crane", fusion="median")


async def test_distribution_fusion_reaches_the_api_and_the_command_line(memory):
    import io
    import json

    from httpx import ASGITransport, AsyncClient

    from scone_memory.api import create_app
    from scone_memory.runtime.cli import build_parser, run

    async with AsyncClient(transport=ASGITransport(app=create_app(memory, {"k": "s"})), base_url="http://fixture") as client:
        answer = await client.get("/v1/recall", params={"q": "crane", "fusion": "distribution"}, headers={"authorization": "Bearer k"})
        assert answer.status_code == 200 and answer.json()["fusion"] == "distribution"
    out = io.StringIO()
    code = await run(build_parser().parse_args(["--space", "s", "--json", "recall", "crane", "--fusion", "distribution"]), memory, io.StringIO(""), out)
    assert code == 0 and json.loads(out.getvalue())["fusion"] == "distribution"
