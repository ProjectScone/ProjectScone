"""Measuring the floor an engine abstains by.

The floor is not a probability and not a constant: it is measured on
questions with an answer in memory and questions without, with the
embedder that will use it, and recorded with what it cost.
"""

from __future__ import annotations

import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench.calibrate import calibrate, write_policy
from scone_memory.bench.runner import BenchItem
from scone_memory.retrieval.abstention import AbstentionPolicy

ITEMS = [
    BenchItem(question_id=f"q{n}", question_type="single-session-user", question=question,
              question_date="2023/04/20 (Thu) 10:12",
              sessions=((f"user: {said}",),), session_ids=(f"s{n}",),
              session_dates=("2023-03-11T09:00:00Z",), answer_session_ids=(f"s{n}",))
    for n, (question, said) in enumerate([
        ("Where did I plant the rosemary?", "I planted the rosemary by the south wall of the garden."),
        ("Which plumber fixed the kitchen tap?", "The plumber Marta fixed the kitchen tap on Tuesday."),
        ("What did the vet say about the cat?", "The vet said the cat needs a softer diet."),
        ("Which bakery has the sourdough?", "The bakery on Rua Nova has the sourdough I like."),
    ], start=1)
]


async def engines():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


async def test_a_floor_is_measured_with_the_embedder_that_will_use_it(tmp_path):
    policy, report = await calibrate(engines, ITEMS, target_false_abstain=0.5, dataset="garden")
    assert policy is not None and policy.embedder_id == report.embedder == HashEmbedder().id
    assert policy.dim == HashEmbedder().dim and -1.0 <= policy.floor <= 1.0
    assert policy.measured["questions"] == 4 and policy.measured["dataset"] == "garden"
    assert policy.measured["false_abstain_rate"] <= 0.5
    assert policy.measured["unanswerable"] and policy.measured["answerable"]

    path = write_policy(policy, tmp_path / "policy.json")
    assert AbstentionPolicy.read(path) == policy
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                abstention=AbstentionPolicy.read(path)).open()
    assert engine.similarity_floor == policy.floor


async def test_no_floor_is_taken_when_none_is_within_the_budget():
    """Withholding no answerable question at all may cost every floor; then
    nothing is recorded, rather than a floor nobody measured."""
    policy, report = await calibrate(engines, ITEMS, target_false_abstain=-0.0 if False else 0.0)
    assert report.abstention is not None
    within = [floor for floor, cost in report.abstention["false_abstain_rate"].items()
              if cost is not None and cost <= 0.0]
    assert (policy is None) == (not within)


async def test_what_was_measured_is_written_down_for_a_reader(tmp_path):
    policy, _ = await calibrate(engines, ITEMS, target_false_abstain=0.5, dataset="garden")
    written = json.loads((write_policy(policy, tmp_path / "policy.json")).read_text())
    assert written["schema_version"] == 1 and written["embedder_id"] == HashEmbedder().id
    assert set(written["measured"]) >= {"questions", "abstain_rate", "false_abstain_rate",
                                        "target_false_abstain", "dataset", "measured_at"}
    assert "floor" in policy.text() and "withholds" in policy.text()


def _dataset(tmp_path):
    """The questions above, written as the file the loader reads."""
    import json as _json

    dataset = tmp_path / "items.json"
    dataset.write_text(_json.dumps([
        {"question_id": item.question_id, "question_type": item.question_type, "question": item.question,
         "question_date": item.question_date, "answer": "",
         # The file's own date style; load_items turns it into an instant.
         "haystack_dates": ["2023/03/11 (Sat) 09:00" for _ in item.session_dates],
         "haystack_session_ids": list(item.session_ids),
         "answer_session_ids": list(item.answer_session_ids),
         "haystack_sessions": [[{"role": "user", "content": said.split(": ", 1)[1]} for said in session]
                               for session in item.sessions]}
        for item in ITEMS]), encoding="utf-8")
    return dataset


def test_the_command_measures_and_writes_the_policy(tmp_path):
    import io

    from scone_memory.runtime import cli

    dataset = _dataset(tmp_path)
    out = io.StringIO()
    written = tmp_path / "policy.json"
    code = cli.main(["calibrate", str(dataset), "--target-false-abstain", "0.5", "--out", str(written)],
                    env={}, stdin=io.StringIO(""), out=out)
    assert code == 0 and "abstention: floor" in out.getvalue() and str(written) in out.getvalue()
    assert AbstentionPolicy.read(written).embedder_id == HashEmbedder().id


def test_the_command_writes_nothing_when_no_floor_is_cheap_enough(tmp_path):
    """Withholding no answer at all may cost every floor. Then nothing is
    written, and the command says so rather than choosing one anyway."""
    import io

    from scone_memory.runtime import cli

    dataset = _dataset(tmp_path)
    out = io.StringIO()
    code = cli.main(["calibrate", str(dataset), "--target-false-abstain", "0.0", "--out", str(tmp_path / "p.json")],
                    env={}, stdin=io.StringIO(""), out=out)
    assert code == 1 and "none written" in out.getvalue()
    assert not (tmp_path / "p.json").exists()


def test_an_engine_built_from_the_environment_reads_the_policy(tmp_path):
    from scone_memory.runtime.config import Settings, build_abstention

    policy = AbstentionPolicy(floor=0.4, embedder_id=HashEmbedder().id, dim=HashEmbedder().dim, measured={})
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy.record()), encoding="utf-8")
    settings = Settings.from_env({"SCONE_ABSTENTION_POLICY": str(path)})
    assert settings.abstention_policy == str(path) and build_abstention(settings) == policy
    assert build_abstention(Settings.from_env({})) is None


def test_a_policy_file_that_cannot_be_read_stops_the_engine_being_built(tmp_path):
    from scone_memory.retrieval.abstention import PolicyError
    from scone_memory.runtime.config import Settings, build_abstention

    path = tmp_path / "policy.json"
    path.write_text("{\"schema_version\": 99}", encoding="utf-8")
    with pytest.raises(PolicyError, match="schema_version"):
        build_abstention(Settings.from_env({"SCONE_ABSTENTION_POLICY": str(path)}))


async def test_a_run_that_embedded_differently_than_it_measured_is_refused():
    """The floor belongs to the embedder that produced the similarities, so
    a run that changed embedder mid-way records nothing."""
    from scone_memory.retrieval.abstention import PolicyError

    made: list[int] = []

    async def changing():
        made.append(1)
        embedder = HashEmbedder() if len(made) == 1 else HashEmbedder(128)
        return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()

    with pytest.raises(PolicyError, match="embedded with"):
        await calibrate(changing, ITEMS, target_false_abstain=0.5)
