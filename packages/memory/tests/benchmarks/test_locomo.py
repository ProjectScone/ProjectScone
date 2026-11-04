"""LoCoMo harness: the data read as written, the original scorer ported
exactly, retrieval scored by evidence turns, and a judge never reported
without its false-accept rate. The fixture is invented; LoCoMo's own data is
CC BY-NC and never ships here."""

from __future__ import annotations

import importlib.util
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench.locomo import (
    Question,
    decoy_for,
    evidence_ids,
    load_locomo,
    multi_f1,
    normalize_answer,
    official_score,
    parse_session_time,
    run_locomo,
    token_f1,
)
from scone_memory.providers.llm import FakeChat

#: The answer score stems with NLTK, as LoCoMo's own scorer does; loading
#: the data and scoring retrieval need nothing beyond the package.
needs_nltk = pytest.mark.skipif(importlib.util.find_spec("nltk") is None, reason="LoCoMo's answer F1 needs nltk")


def conversation(**overrides):
    row = {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Ana", "speaker_b": "Ben",
            "session_2_date_time": "9:05 am on 3 June, 2023",
            "session_2": [
                {"speaker": "Ben", "dia_id": "D2:1", "text": "I adopted a beagle puppy called Biscuit yesterday."},
                {"speaker": "Ana", "dia_id": "D2:2", "text": "Lovely! Look at my garden.", "blip_caption": "a photo of red tulips"},
            ],
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Ana", "dia_id": "D1:1", "text": "I ran the charity marathon in Lisbon last weekend."},
                {"speaker": "Ben", "dia_id": "D1:2", "text": "Impressive. I am learning the cello."},
            ],
        },
        "qa": [
            {"question": "Which city did Ana run the charity marathon in?", "answer": "Lisbon", "evidence": ["D1:1"], "category": 4},
            {"question": "What instrument is Ben learning?", "answer": "cello", "evidence": ["D1:2; D2:1"], "category": 4},
            {"question": "When did Ben adopt the beagle puppy?", "answer": "2 June 2023", "evidence": ["D2:01", "D9:9", "D"], "category": 2},
            {"question": "Which city did Ben run the charity marathon in?", "adversarial_answer": "Lisbon", "evidence": ["D1:1"], "category": 5},
        ],
    }
    row.update(overrides)
    return row


def write(tmp_path, rows):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_sessions_are_read_in_order_with_their_times_and_photos(tmp_path):
    [conv] = load_locomo(write(tmp_path, [conversation()]))
    assert [turn.dia_id for turn in conv.turns] == ["D1:1", "D1:2", "D2:1", "D2:2"]
    assert conv.turns[0].when.isoformat() == "2023-05-08T13:56:00+00:00"
    assert conv.turns[1].when.isoformat() == "2023-05-08T13:57:00+00:00"
    assert conv.turns[3].text == "Ana: Lovely! Look at my garden. [shares a photo of a photo of red tulips]"
    assert conv.speakers == ("Ana", "Ben")


def test_evidence_is_read_as_written_and_what_names_no_turn_is_counted(tmp_path):
    [conv] = load_locomo(write(tmp_path, [conversation()]))
    assert conv.questions[1].evidence == ("D1:2", "D2:1") and conv.questions[1].unreadable_evidence == 0
    assert conv.questions[2].evidence == ("D2:1",) and conv.questions[2].unreadable_evidence == 2
    assert conv.questions[3].answer is None and conv.questions[3].trap == "Lisbon"
    assert evidence_ids(["D:1:2", "D1:2 D1:1"], {"D1:1", "D1:2"}) == (("D1:2", "D1:1"), 0)


def test_a_bad_session_time_or_a_repeated_turn_id_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        parse_session_time("May 8, 2023")
    row = conversation()
    row["conversation"]["session_2"][0]["dia_id"] = "D1:1"
    with pytest.raises(ValueError):
        load_locomo(write(tmp_path, [row]))


@needs_nltk
def test_the_original_scorer_is_ported_exactly():
    assert normalize_answer("The cat, and the dog -- too!") == "cat dog too"
    assert token_f1("7 May 2023", "7 May, 2023") == 1.0
    # "she ran race" against "running race": stems share only "race".
    assert token_f1("She ran the race", "running race") == pytest.approx(0.4)
    assert token_f1("", "anything") == 0.0
    # Multi-hop: each part of the answer by its best part of the prediction.
    assert multi_f1("Paris, Rome", "Rome, Paris, Oslo") == pytest.approx(2 / 3)
    assert official_score(1, "Paris, Rome", "Rome, Paris, Oslo") == pytest.approx(2 / 3)
    assert official_score(3, "likely no", "Likely no; she wants a family") == 1.0
    assert official_score(4, "likely no", "Likely no; she wants a family") < 1.0
    assert official_score(5, "Not mentioned.", None) == 1.0
    assert official_score(5, "No information available about that.", None) == 1.0
    assert official_score(5, "In Lisbon.", None) == 0.0
    with pytest.raises(ValueError):
        official_score(4, "Lisbon", None)


@needs_nltk
def test_a_decoy_is_another_answer_in_the_category_sharing_no_word():
    questions = [Question("q0", 4, "red car", ()), Question("q1", 4, "a red bike", ()),
                 Question("q2", 2, "May 2023", ()), Question("q3", 4, "blue boat", ())]
    assert decoy_for(questions, 0) == "blue boat"
    assert decoy_for(questions, 2) is None
    assert decoy_for([Question("q", 5, None, ())], 0) is None


def engine():
    return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


async def test_retrieval_is_scored_by_evidence_turns_without_a_model(tmp_path):
    conversations = load_locomo(write(tmp_path, [conversation()]))
    report = await run_locomo(engine, conversations, k=2)
    first = report.results[0]
    assert first.evidence == ["D1:1"] and "D1:1" in first.top_sources and first.evidence_recall == 1.0
    assert report.scores["single-hop"]["n"] == 2.0 and report.scores["all"]["n"] == 4.0
    assert report.reader is None and report.scores["all"]["score"] is None and report.judge_false_accept is None
    assert report.evidence_unreadable == 2 and report.no_readable_evidence == 0


@needs_nltk
async def test_answers_are_scored_per_category_and_the_reader_sees_dated_turns(tmp_path):
    conversations = load_locomo(write(tmp_path, [conversation()]))
    reader = FakeChat(["Lisbon", "the cello", "2 June 2023", "Not mentioned."])
    report = await run_locomo(engine, conversations, reader=reader, reader_name="fake", k=4)
    assert [r.score for r in report.results] == [1.0, 1.0, 1.0, 1.0]
    assert report.scores["adversarial"]["score"] == 1.0 and report.scores["all but adversarial"]["answered"] == 3.0
    system, user = reader.calls[2]
    assert "Not mentioned" in system
    assert "[3 June 2023] Ben: I adopted a beagle puppy called Biscuit yesterday." in user
    # Turns reach the reader oldest first, whatever their rank.
    assert user.index("[8 May 2023]") < user.index("[3 June 2023]")


@needs_nltk
async def test_a_judge_is_reported_with_its_false_accept_rate(tmp_path):
    rows = conversation()
    rows["qa"].append({"question": "What did Ana grow?", "answer": "red tulips", "evidence": ["D2:2"], "category": 4})
    conversations = load_locomo(write(tmp_path, [rows]))
    reader = FakeChat(["Lisbon", "cello", "2 June 2023", "Not mentioned.", "tulips"])
    # Per question with an answer: the reader's answer, then its decoy if it
    # has one (calls: q0, q0 decoy, q1, q1 decoy, q2, q4, q4 decoy). This
    # judge accepts every answer it can be read on, decoys included; its
    # reply on the last decoy cannot be read.
    judge = FakeChat(['{"verdict": "CORRECT"}'] * 6 + ["no verdict here"])
    report = await run_locomo(engine, conversations, reader=reader, reader_name="fake", judge=judge, judge_name="lenient", k=4)
    single = [r for r in report.results if r.category == 4]
    assert [r.decoy for r in single] == ["cello", "red tulips", "Lisbon"]
    assert report.results[2].decoy is None  # the only temporal question has no decoy
    assert report.judge_audited == 2 and report.judge_false_accept == 1.0
    assert report.judge_unreadable == 1
    assert all(r.judged_correct is None for r in report.results if r.category == 5)
    assert report.as_dict(with_items=False)["judge"] == "lenient" and "results" not in report.as_dict(with_items=False)


async def test_a_judge_needs_a_reader(tmp_path):
    conversations = load_locomo(write(tmp_path, [conversation()]))
    with pytest.raises(ValueError):
        await run_locomo(engine, conversations, judge=FakeChat([]))


def test_the_command_scores_retrieval_and_leaves_the_configured_store_alone(tmp_path):
    import io

    from scone_memory.runtime import cli

    path = write(tmp_path, [conversation()])
    report = tmp_path / "report.json"
    untouched = tmp_path / "configured.db"
    env = {"SCONE_EMBEDDER": "hash", "SCONE_DOCUMENTS": "sqlite", "SCONE_VECTORS": "sqlite", "SCONE_SQLITE_PATH": str(untouched)}
    out = io.StringIO()
    code = cli.main(["bench-locomo", str(path), "--k", "2", "--categories", "4,5", "--out", str(report)],
                    env=env, stdin=io.StringIO(), out=out)
    assert code == 0, out.getvalue()
    saved = json.loads(report.read_text())
    assert saved["k"] == 2 and [r["category"] for r in saved["results"]] == [4, 4, 5]
    # The unreadable evidence is the temporal question's, which was not asked.
    assert saved["questions"] == 3 and saved["evidence_unreadable"] == 0
    assert not untouched.exists()
    for bad in (["--reader"], ["--categories", "4,6"], ["--categories", "x"], ["--conversations", "conv-1,conv-9"],
                ["--k", "0"], ["--questions", "0"]):
        assert cli.main(["bench-locomo", str(path), *bad], env=env, stdin=io.StringIO(), out=io.StringIO()) == 2, bad
