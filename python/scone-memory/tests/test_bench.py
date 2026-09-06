"""The Python bench runner must use the Rust harness's definitions exactly:
one space per item, sessions as role: content transcripts with the
session id as source, recall any/all over the top-k sources."""

from __future__ import annotations

import json

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench import load_items, run
from scone_memory.bench.runner import ItemResult, iso_date


def item(qid, qtype, question, sessions, answer_ids, dates=None):
    return {
        "question_id": qid, "question_type": qtype, "question": question, "question_date": "2023/06/01 (Thu) 10:00",
        "haystack_sessions": [[{"role": r, "content": c} for r, c in s] for s in sessions],
        "haystack_session_ids": [f"s{i}" for i in range(len(sessions))],
        "haystack_dates": dates or [f"2023/05/{10 + i:02d} (Mon) 09:{i:02d}" for i in range(len(sessions))],
        "answer_session_ids": answer_ids, "answer": "x",
    }


DATASET = [
    item("q1", "single-session-user", "which city did I move to, Lisbon or elsewhere", [
        [("user", "I moved to Lisbon last March"), ("assistant", "Nice")],
        [("user", "my dentist appointment is on the 14th"), ("assistant", "Noted")],
        [("user", "the kubernetes upgrade failed"), ("assistant", "Sorry")],
    ], ["s0"]),
    item("q2", "multi-session", "what happened with the launch and the postmortem", [
        [("user", "launch went live at noon"), ("assistant", "ok")],
        [("user", "postmortem: the launch broke billing"), ("assistant", "ok")],
        [("user", "unrelated grocery list"), ("assistant", "ok")],
    ], ["s0", "s1"]),
    item("q3", "abstention", "what is my cat called", [
        [("user", "I like tea"), ("assistant", "ok")],
    ], []),
]


def test_iso_date_matches_the_rust_harness():
    assert iso_date("2023/05/20 (Sat) 02:21") == "2023-05-20T02:21:00Z"
    assert iso_date("2023/05/20") == "2023-05-20T00:00:00Z"
    assert iso_date("bad") == ""


def test_loader_flattens_turns_as_role_colon_content(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(DATASET))
    items = load_items(path)
    assert len(items) == 3
    assert items[0].sessions[0] == ("user: I moved to Lisbon last March", "assistant: Nice")
    assert items[0].session_ids == ("s0", "s1", "s2") and items[0].answer_session_ids == ("s0",)
    assert items[0].session_dates[0] == "2023-05-10T09:00:00Z"
    assert not items[2].has_evidence


def test_hit_definitions():
    r = ItemResult("q", "t", True, ["s1", "s0", "s9"], 0, 0, 0.0, answer_sessions=["s0", "s1"])
    assert r.any_at(1) and not r.all_at(1)
    assert r.all_at(2) and r.all_at(3)
    none = ItemResult("q", "t", False, ["s1"], 0, 0, 0.0, answer_sessions=[])
    assert not none.any_at(5) and not none.all_at(5), "no evidence can never be a hit"


async def test_run_scores_under_the_stated_definitions(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(DATASET))
    items = load_items(path)

    def make():
        return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    seen: list[tuple[int, int]] = []
    report = await run(make, items, ks=(1, 2, 3), dataset="unit", progress=lambda n, t: seen.append((n, t)))
    assert seen == [(1, 3), (2, 3), (3, 3)]
    assert report.items == 3 and report.scored == 2, "the abstention item is outside the denominator by default"
    assert report.errors == 0 and report.embedder == "hash-256"
    # q1: s0 is the only session mentioning Lisbon; q2: s0 and s1 both mention the launch.
    assert report.recall_any[3] == 1.0
    assert report.recall_all[3] == 1.0
    assert report.recall_any[1] == 1.0, "each question's top item is an evidence session"
    assert report.recall_all[1] == 0.5, "q2 needs two sessions; one cannot hold both"
    assert set(report.by_type) == {"single-session-user", "multi-session"}
    assert report.by_type["multi-session"]["all@1"] == 0.0 and report.by_type["multi-session"]["all@2"] == 1.0
    assert report.context_reduction_median == 0.0, "k=3 returns every one of three sessions: nothing is left behind"
    tight = await run(make, items[:2], ks=(1,), limit=1)
    assert 0 < (tight.context_reduction_median or 0) < 1, "k=1 leaves the other sessions' bytes behind"
    assert report.recall_ms_p50 is not None and report.recall_ms_p95 is not None
    q3 = next(r for r in report.results if r.question_id == "q3")
    assert q3.error is None and q3.retrieved_sessions, "abstention items still run; they are only excluded from scoring"

    with_abst = await run(make, items, ks=(3,), include_abstention=True)
    assert with_abst.scored == 3 and with_abst.recall_any[3] == round(2 / 3, 4)


async def test_each_item_gets_a_fresh_engine(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(DATASET))
    items = load_items(path)
    made = []

    def make():
        e = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())
        made.append(e)
        return e.open()

    await run(make, items, ks=(3,))
    assert len(made) == 3 and len({id(e) for e in made}) == 3
    # The first engine holds only item 1's sessions; nothing leaked forward.
    assert (await made[0].status("item")).episodes == 3 and (await made[2].status("item")).episodes == 1
