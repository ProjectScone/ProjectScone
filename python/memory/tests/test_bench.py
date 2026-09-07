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


class Stub:
    """An engine whose recall reports a chosen top similarity per question,
    so the sweep's expectations can be worked out by hand."""

    class _Meta:
        def __init__(self, name):
            self.name = name
            self.id = name

    def __init__(self, tops: dict[str, float | None], floor=None):
        self.tops, self.similarity_floor = tops, floor
        self.embedder = self._Meta("stub")
        self.documents = self._Meta("stub")
        self.vectors = self._Meta("stub")
        self.events = None

    async def remember_many(self, space, records):
        return []

    async def recall(self, space, question, limit, history=False):
        from scone_memory.models import RecallResult

        top = self.tops[question]
        verdict = None if self.similarity_floor is None else (top is None or top < self.similarity_floor)
        return RecallResult(items=[], facts=[], degraded=[], top_similarity=top, low_confidence=verdict, returned_bytes=0, space_bytes=1)


def sweep_items():
    return [
        item("a", "single-session-user", "a?", [[("user", "x")]], ["s0"]),
        item("b", "multi-session", "b?", [[("user", "x")]], ["s0"]),
        item("c", "abstention", "c?", [[("user", "x")]], []),
        item("d", "abstention", "d?", [[("user", "x")]], []),
    ]


async def test_the_abstention_sweep_counts_caught_and_wrongly_withheld_per_floor(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(sweep_items()))
    tops = {"a?": 0.8, "b?": 0.4, "c?": 0.35, "d?": None}
    report = await run(lambda: Stub(tops), load_items(path), ks=(1,))
    sweep = report.abstention
    assert sweep is not None and (sweep["no_evidence_n"], sweep["evidence_n"]) == (2, 2)
    assert sweep["floors"][0] == 0.3 and sweep["floors"][-1] == 0.9 and len(sweep["floors"]) == 13
    # No-evidence items: d found nothing (flagged at every floor); c's 0.35 clears 0.30 and 0.35, falls at 0.40.
    assert sweep["abstain_rate"][0.3] == 0.5 and sweep["abstain_rate"][0.35] == 0.5 and sweep["abstain_rate"][0.4] == 1.0
    # Evidence items: b's 0.40 is not below 0.40; it is withheld from 0.45; a's 0.80 falls at 0.85.
    assert sweep["false_abstain_rate"][0.4] == 0.0 and sweep["false_abstain_rate"][0.45] == 0.5
    assert sweep["false_abstain_rate"][0.8] == 0.5 and sweep["false_abstain_rate"][0.85] == 1.0
    by_id = {r.question_id: r for r in report.results}
    assert (by_id["a"].top_similarity, by_id["d"].top_similarity) == (0.8, None)
    assert report.similarity_floor is None and report.low_confidence_counts == {}, "no floor: no verdicts to count"

    gated = await run(lambda: Stub(tops, floor=0.5), load_items(path), ks=(1,))
    assert gated.similarity_floor == 0.5
    assert gated.low_confidence_counts == {"evidence_cleared": 1, "evidence_flagged": 1, "no_evidence_flagged": 2}


async def test_the_sweep_is_absent_when_no_abstention_item_ran(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(sweep_items()[:2]))
    report = await run(lambda: Stub({"a?": 0.8, "b?": 0.4}), load_items(path), ks=(1,))
    assert report.abstention is None, "abstention accuracy cannot be measured without an item to abstain on"


async def test_a_degraded_vector_lane_is_outside_the_sweep(tmp_path):
    class Degraded(Stub):
        async def recall(self, space, question, limit, history=False):
            r = await super().recall(space, question, limit)
            if question == "c?":
                r.degraded = ["vectors: RuntimeError: offline"]
                r.top_similarity = None
            return r

    path = tmp_path / "d.json"
    path.write_text(json.dumps(sweep_items()))
    report = await run(lambda: Degraded({"a?": 0.8, "b?": 0.4, "c?": None, "d?": 0.1}), load_items(path), ks=(1,))
    sweep = report.abstention
    assert sweep["no_evidence_n"] == 1, "c saw no similarity because its lane failed, not because the evidence was weak"
    assert sweep["abstain_rate"][0.3] == 1.0


async def test_history_is_passed_through_and_counted_honestly(tmp_path):
    """Experiment 3 needs the runner to ask for history; the report must
    say when there were no facts to show rather than report a zero as a
    result."""
    seen: list[bool] = []

    class Recording(Stub):
        async def recall(self, space, question, limit, history=False):
            seen.append(history)
            r = await super().recall(space, question, limit)
            if history and question == "a?":
                from scone_memory.models import Fact

                f = Fact(fact_id=1, space=space, subject="s", predicate="p", object="o", confidence=1.0, valid_from="2020-01-01T00:00:00.000Z", status="active")
                r.facts = [f]
                r.history = [f.model_copy(update={"fact_id": 2, "status": "closed"})]
            return r

    path = tmp_path / "d.json"
    path.write_text(json.dumps(sweep_items()[:2]))
    plain = await run(lambda: Recording({"a?": 0.8, "b?": 0.4}), load_items(path), ks=(1,))
    assert seen == [False, False] and not plain.history and (plain.items_with_facts, plain.items_with_history) == (0, 0)
    asked = await run(lambda: Recording({"a?": 0.8, "b?": 0.4}), load_items(path), ks=(1,), history=True)
    assert seen[2:] == [True, True] and asked.history
    assert (asked.items_with_facts, asked.items_with_history) == (1, 1)
    by_id = {r.question_id: r for r in asked.results}
    assert (by_id["a"].facts, by_id["a"].history_facts, by_id["b"].facts) == (1, 1, 0)


def synthetic(count, *splits):
    def kind(i):
        return "type-" + "abc"[sum(i >= s for s in splits)]

    return [item(f"q{i}", kind(i), "q", [[("user", "x")]], ["s0"]) for i in range(count)]


def test_the_sample_is_the_rust_harness_sample_item_for_item(tmp_path):
    """Pinned to what crates/scone-bench's stratified_sample returned for
    the same synthetic datasets (100 items, 80 type-a and 20 type-b; the
    harness test's own shape), printed once from a throwaway Rust test at
    f47e17a. Same seed, same n, same ids in the same order. The third
    case has three types of 33, 33 and 34 items, where the per-type
    rounding draws nine of ten and the top-up path picks the last."""
    from scone_memory.bench import stratified_sample

    path = tmp_path / "d.json"
    path.write_text(json.dumps(synthetic(100, 80)))
    items = load_items(path)
    ids = [it.question_id for it in stratified_sample(items, 10, 42)]
    assert ids == ["q74", "q48", "q66", "q38", "q26", "q27", "q71", "q23", "q96", "q88"]
    assert sum(it.question_type == "type-b" for it in stratified_sample(items, 10, 42)) == 2
    ids = [it.question_id for it in stratified_sample(items, 7, 7)]
    assert ids == ["q7", "q31", "q9", "q19", "q62", "q25", "q85"]
    assert [it.question_id for it in stratified_sample(items, 10, 42)] == [it.question_id for it in stratified_sample(items, 10, 42)]
    path.write_text(json.dumps(synthetic(100, 33, 66)))
    ids = [it.question_id for it in stratified_sample(load_items(path), 10, 42)]
    assert ids == ["q25", "q31", "q30", "q49", "q35", "q60", "q87", "q81", "q82", "q84"]


def test_the_cli_runs_the_sample_the_harness_would(tmp_path):
    """--sample 10 --seed 42 on the 80/20 synthetic set runs the ten items
    the Rust harness draws, in its order, and the --out report names them.
    The configured store stays untouched: the bench measures the engine on
    in-process stores, so it must neither create nor migrate the file at
    SCONE_SQLITE_PATH (before 2026-09-06 it opened it, and a run with the
    local embedder against a hash-embedded file failed on the dimension)."""
    import io

    from scone_memory import cli

    dataset = tmp_path / "d.json"
    dataset.write_text(json.dumps(synthetic(100, 80)))
    report = tmp_path / "report.json"
    untouched = tmp_path / "configured.db"
    env = {"SCONE_EMBEDDER": "hash", "SCONE_DOCUMENTS": "sqlite", "SCONE_VECTORS": "sqlite", "SCONE_SQLITE_PATH": str(untouched)}
    code = cli.main(["bench", str(dataset), "--sample", "10", "--out", str(report), "--json"], env=env, stdin=io.StringIO(), out=io.StringIO())
    assert code == 0
    ran = [r["question_id"] for r in json.loads(report.read_text())["results"]]
    assert ran == ["q74", "q48", "q66", "q38", "q26", "q27", "q71", "q23", "q96", "q88"]
    assert not untouched.exists(), "the bench opened the configured store"


def test_the_bench_engine_takes_the_contextual_and_floor_settings(tmp_path):
    """SCONE_CONTEXTUAL_EMBEDDINGS=1 and SCONE_SIMILARITY_FLOOR reach the
    per-item engines the bench builds, and the report states both from the
    engine. Until 2026-09-06 the bench built its engines without either,
    so an experiment 8 leg would have measured the baseline again under
    the other name; the hash embedder makes the prefix visible as a
    changed similarity."""
    import io

    from scone_memory import cli

    dataset = tmp_path / "d.json"
    dataset.write_text(json.dumps(DATASET))

    def bench(env, name):
        report = tmp_path / name
        code = cli.main(["bench", str(dataset), "--out", str(report), "--json"], env=env, stdin=io.StringIO(), out=io.StringIO())
        assert code == 0
        return json.loads(report.read_text())

    plain = bench({"SCONE_EMBEDDER": "hash"}, "plain.json")
    prefixed = bench({"SCONE_EMBEDDER": "hash", "SCONE_CONTEXTUAL_EMBEDDINGS": "1", "SCONE_SIMILARITY_FLOOR": "0.9"}, "prefixed.json")
    assert plain["contextual_embeddings"] is False and plain["similarity_floor"] is None
    assert prefixed["contextual_embeddings"] is True and prefixed["similarity_floor"] == 0.9
    sims = lambda r: [x["top_similarity"] for x in r["results"]]  # noqa: E731
    assert sims(plain) != sims(prefixed), "the prefix never reached the embedded text"


def raw_item(qid, qtype, question, sessions, answer_ids):
    """Like item(), with the session ids given so haystacks can be disjoint."""
    return {
        "question_id": qid, "question_type": qtype, "question": question, "question_date": "2023/06/01 (Thu) 10:00",
        "haystack_sessions": [[{"role": "user", "content": text}] for _, text in sessions],
        "haystack_session_ids": [sid for sid, _ in sessions],
        "haystack_dates": ["2023/05/10 (Wed) 09:00"] * len(sessions),
        "answer_session_ids": answer_ids, "answer": "x",
    }


CROSS_DATASET = [
    raw_item("q1", "single-session-user", "which city did I move to", [("a1", "I moved to Lisbon last March"), ("a2", "dentist on the 14th")], ["a1"]),
    raw_item("q2", "multi-session", "what broke at the launch", [("b1", "launch went live at noon"), ("b2", "the launch broke billing")], ["b1", "b2"]),
    raw_item("q3", "single-session-user", "what did I name the cat", [("c1", "the cat is called Mint"), ("a1", "I moved to Lisbon last March")], ["c1"]),
]


async def test_cross_queries_give_the_sweep_a_no_evidence_population(tmp_path):
    """LongMemEval has no item without evidence (every _abs item's answer
    sessions sit in its haystack), so the sweep was never measurable on
    the real file. With cross_queries each item's store is also asked the
    next item's question whose evidence is absent here: q1 gets q2's
    question (b1, b2 absent), q2 gets q3's, and q3 skips q1 (a1 is in q3's
    haystack) and takes q2's. Those rows feed the sweep and nothing else:
    the recall numbers are the same as a run without them."""
    path = tmp_path / "cross.json"
    path.write_text(json.dumps(CROSS_DATASET))
    items = load_items(path)

    def make():
        return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), similarity_floor=0.5).open()

    plain = await run(make, items, ks=(5,))
    crossed = await run(make, items, ks=(5,), cross_queries=True)
    assert plain.abstention is None
    assert crossed.abstention is not None
    assert crossed.abstention["no_evidence_n"] == 3 and crossed.abstention["cross_item_n"] == 3 and crossed.abstention["evidence_n"] == 3
    assert [r.question_id for r in crossed.cross_results] == ["q2@q1", "q3@q2", "q2@q3"]
    assert all(r.question_type == "cross-item" and not r.has_evidence and r.top_similarity is not None for r in crossed.cross_results)
    assert (crossed.scored, crossed.items, crossed.recall_any, crossed.recall_all) == (plain.scored, plain.items, plain.recall_any, plain.recall_all)
    assert [r.question_id for r in crossed.results] == ["q1", "q2", "q3"]
    assert "cross_results" not in crossed.as_dict(with_items=False) and len(crossed.as_dict()["cross_results"]) == 3
