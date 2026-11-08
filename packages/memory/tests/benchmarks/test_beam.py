"""BEAM harness: the data read as written, retrieval scored by source
messages, rubric scores from a judge reported beside the credit it gives
known-wrong answers, and event ordering by Kendall tau-b. The fixtures are
invented; BEAM's own data is CC BY-SA and never ships here."""

from __future__ import annotations

import asyncio
import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench.beam import (
    _stale_only,
    decoy_for,
    event_positions,
    kendall_tau_b,
    load_beam,
    nugget_marks,
    parse_time_anchor,
    run_beam,
    source_ids,
)
from scone_memory.memory.directives import DEFAULT_BLOCK_BYTES, MIN_BLOCK_BYTES
from scone_memory.providers.llm import FakeChat


def trip(**overrides):
    questions = {
        "information_extraction": [
            {"question": "Which city am I planning to visit?", "answer": "Kyoto",
             "rubric": ["LLM response should state: Kyoto"], "source_chat_ids": [0]},
            {"question": "What season will it be when I arrive?", "answer": "Cherry blossom season",
             "rubric": ["LLM response should state: cherry blossom season"], "source_chat_ids": [[1], [99]]},
        ],
        "knowledge_update": [
            {"question": "What is my trip budget?", "answer": "3500 dollars", "rubric": ["LLM response should state: 3500 dollars"],
             "source_chat_ids": {"original_info": [2], "updated_info": [4]}},
        ],
        "abstention": [
            {"question": "Which airline did I book?", "ideal_response": "There is no information about an airline.",
             "rubric": ["Says there is no information about the airline"], "source_chat_ids": None},
        ],
        "event_ordering": [
            {"question": "In what order did I tell you about my trip plans?", "answer": "Kyoto, the budget, then the raise.",
             "rubric": ["Trip to Kyoto", "Budget of 2000 dollars", "Budget raised to 3500 dollars"], "source_chat_ids": [0, 2, 4]},
        ],
        "instruction_following": [
            {"question": "Summarize my trip plans.", "expected_compliance": "Answers in bullet points",
             "rubric": ["Uses bullet points"], "source_chat_ids": [6]},
        ],
    }
    row = {
        "conversation_id": "c1",
        "chat": [
            [
                {"id": 0, "role": "user", "content": "I am planning a trip to Kyoto in April. ->-> 1,1", "time_anchor": "March-15-2024"},
                {"id": 1, "role": "assistant", "content": "Kyoto in April is cherry blossom season.", "time_anchor": None},
                {"id": 2, "role": "user", "content": "My budget for the trip is 2000 dollars. ->-> 1,2"},
                {"id": 3, "role": "assistant", "content": "That budget covers a week of hostels. ->-> 2,N/A"},
            ],
            [
                {"id": 4, "role": "user", "content": "Update: my budget is now 3500 dollars after my bonus.", "time_anchor": "April-2-2024"},
                {"id": 5, "role": "assistant", "content": "That opens up a ryokan stay. ->-> 5,14)"},
                {"id": 6, "role": "user", "content": "Please always answer me in bullet points. ->-> 2,22, 24"},
                {"id": 7, "role": "assistant", "content": "Understood, bullet points from now on."},
            ],
        ],
        # BEAM stores the questions as a Python literal.
        "probing_questions": repr(questions),
    }
    row.update(overrides)
    return row


def cello():
    return {
        "conversation_id": "c2",
        "chat": [[
            {"id": 0, "role": "user", "content": "I started learning the cello last week.", "time_anchor": "May-1-2024"},
            {"id": 1, "role": "assistant", "content": "The cello is a lovely instrument."},
            {"id": 2, "role": "user", "content": "I practise forty minutes a day."},
            {"id": 3, "role": "user", "content": "Now I practise an hour a day."},
        ]],
        "probing_questions": {
            "information_extraction": [{"question": "Which instrument am I learning?", "answer": "The cello",
                                        "rubric": ["States the cello"], "source_chat_ids": [0]}],
            "knowledge_update": [{"question": "How long do I practise each day?", "answer": "An hour",
                                  "rubric": ["States an hour"], "source_chat_ids": {"original_info": [2], "updated_info": [3]}}],
        },
    }


def write(tmp_path, rows):
    path = tmp_path / "beam.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_messages_inherit_their_session_anchor_and_lose_the_index_marker(tmp_path):
    first, second = load_beam(write(tmp_path, [trip(), cello()]))
    assert [m.id for m in first.messages] == list(range(8))
    assert first.messages[0].text == "user: I am planning a trip to Kyoto in April."
    assert first.messages[2].text == "user: My budget for the trip is 2000 dollars."
    # The marker in every form the data writes it.
    assert first.messages[3].text == "assistant: That budget covers a week of hostels."
    assert first.messages[5].text == "assistant: That opens up a ryokan stay."
    assert first.messages[6].text == "user: Please always answer me in bullet points."
    assert first.messages[0].when.isoformat() == "2024-03-15T00:00:00+00:00"
    assert first.messages[3].when.isoformat() == "2024-03-15T00:00:03+00:00"
    assert first.messages[4].when.isoformat() == "2024-04-02T00:00:00+00:00"
    # The questions keep the data's order of abilities; a mapping is read as one.
    assert [p.ability for p in first.probes] == ["information_extraction", "information_extraction", "knowledge_update",
                                                 "abstention", "event_ordering", "instruction_following"]
    assert [p.ability for p in second.probes] == ["information_extraction", "knowledge_update"]


def test_each_ability_keeps_its_reference_and_its_source_messages(tmp_path):
    [conv] = load_beam(write(tmp_path, [trip()]))
    season, update, abstain, order, instruct = conv.probes[1:]
    assert season.evidence == (1,) and season.unreadable_evidence == 1
    assert update.roles == (("original_info", (2,)), ("updated_info", (4,))) and update.evidence == (2, 4)
    assert abstain.evidence == () and abstain.reference == "There is no information about an airline." and not abstain.factual
    assert order.rubric == ("Trip to Kyoto", "Budget of 2000 dollars", "Budget raised to 3500 dollars") and order.factual
    assert instruct.reference == "Answers in bullet points" and not instruct.factual


def test_source_ids_of_every_shape_are_flattened_and_what_names_no_message_is_counted():
    known = {3, 4}
    assert source_ids(None, known) == ((), (), 0)
    # True equals 1 to Python, but names no message.
    assert source_ids([True, 1], {1}) == ((1,), (), 1)
    assert source_ids([3, [4, 3], "4", 99, True], known) == ((3, 4), (), 3)
    assert source_ids({"first_event": [4], "second_event": [3, 7]}, known) == (
        (4, 3), (("first_event", (4,)), ("second_event", (3,))), 1)


def test_a_session_without_an_anchor_or_restating_one_continues_the_count(tmp_path):
    row = trip()
    row["chat"][1][0]["time_anchor"] = None
    row["chat"][1][2]["time_anchor"] = "April-2-2024"
    row["chat"][1][3]["time_anchor"] = "April-2-2024"
    [conv] = load_beam(write(tmp_path, [row]))
    # 4 and 5 inherit March 15; 6 starts April 2; 7 restates it and continues.
    assert [m.when.isoformat()[11:19] for m in conv.messages] == [f"00:00:0{i}" for i in range(6)] + ["00:00:00", "00:00:01"]
    assert conv.messages[5].when.date().isoformat() == "2024-03-15" and conv.messages[6].when.date().isoformat() == "2024-04-02"
    times = [m.when for m in conv.messages[:6]]
    assert times == sorted(times) and len(set(times)) == 6
    restated = trip()
    restated["chat"][1][1]["time_anchor"] = "April-2-2024"
    [conv] = load_beam(write(tmp_path, [restated]))
    assert [m.when.isoformat()[11:19] for m in conv.messages[4:]] == ["00:00:00", "00:00:01", "00:00:02", "00:00:03"]


def test_a_malformed_row_is_an_error_and_the_questions_are_never_run_as_code(tmp_path):
    with pytest.raises(ValueError):
        parse_time_anchor("2024-03-15")
    no_anchor = trip()
    no_anchor["chat"][0][0]["time_anchor"] = None
    repeated = trip()
    repeated["chat"][1][0]["id"] = 3
    unknown = trip(probing_questions={"mind_reading": []})
    no_rubric = trip(probing_questions={"abstention": [{"question": "Why?", "rubric": []}]})
    # Valid questions if evaluated as code; literal_eval refuses the call.
    code = trip(probing_questions="dict(abstention=[])")
    for row in (no_anchor, repeated, unknown, no_rubric, code):
        with pytest.raises(ValueError):
            load_beam(write(tmp_path, [row]))
    with pytest.raises(ValueError):
        load_beam(write(tmp_path, [trip(), trip()]))


def test_kendall_tau_b_matches_the_textbook_values():
    assert kendall_tau_b([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)
    # One tie in the second ranking: 5 concordant of 6 pairs, 5 untied.
    assert kendall_tau_b([1, 2, 3, 4], [1, 1, 2, 3]) == pytest.approx(5 / 30 ** 0.5)
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    assert kendall_tau_b([1, 2], [5, 5]) is None
    with pytest.raises(ValueError):
        kendall_tau_b([1, 2], [1])


def test_the_stale_only_state_is_the_original_recalled_without_its_update(tmp_path):
    [conv] = load_beam(write(tmp_path, [trip()]))
    update = conv.probes[2]
    assert _stale_only(update, [2, 0]) is True
    assert _stale_only(update, [4, 2]) is False and _stale_only(update, [4]) is False and _stale_only(update, []) is False
    assert _stale_only(conv.probes[0], [0]) is None


def test_a_decoy_is_a_same_ability_reference_from_another_conversation(tmp_path):
    conversations = load_beam(write(tmp_path, [trip(), cello()]))
    pool = [(c.conversation_id, p) for c in conversations for p in c.probes]
    assert decoy_for(pool, 0) == "The cello" and decoy_for(pool, 1) == "The cello"
    assert decoy_for(pool, 2) == "An hour" and decoy_for(pool, 6) == "Kyoto"
    # No factual reference, or no other conversation with one: no decoy.
    assert decoy_for(pool, 3) is None and decoy_for(pool, 5) is None and decoy_for(pool, 4) is None
    alone = [("c1", p) for p in conversations[0].probes]
    assert decoy_for(alone, 0) is None


async def test_a_judge_reply_is_read_only_when_it_marks_every_item_once():
    rubric = ["a", "b"]

    async def marks(reply):
        return await nugget_marks(FakeChat([reply]), "q", rubric, "r")

    assert await marks('Sure: {"scores": [{"index": 1, "score": 0.5}, {"index": 0, "score": 1}]}') == [1.0, 0.5]
    for bad in ('{"scores": [{"index": 0, "score": 1}]}', '{"scores": [{"index": 0, "score": 0.7}, {"index": 1, "score": 1}]}',
                '{"scores": [{"index": 0, "score": true}, {"index": 1, "score": 1}]}',
                '{"scores": [{"index": 0, "score": 1}, {"index": 2, "score": 1}]}',
                '{"scores": [{"index": 0, "score": 1}, {"index": 1, "score": 1}, {"index": 1, "score": 0}]}', "no JSON"):
        assert await marks(bad) is None, bad

    async def positions(reply):
        return await event_positions(FakeChat([reply]), rubric, "r")

    assert await positions('{"positions": [{"index": 0, "position": 2}, {"index": 1, "position": null}]}') == [2, None]
    for bad in ('{"positions": [{"index": 0, "position": 0}, {"index": 1, "position": 1}]}',
                '{"positions": [{"index": 0, "position": 1}]}',
                '{"positions": [{"index": 0, "position": 1}, {"index": 0, "position": 2}, {"index": 1, "position": 3}]}'):
        assert await positions(bad) is None, bad


def engine():
    return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


async def test_retrieval_is_scored_by_source_messages_without_a_model(tmp_path):
    conversations = load_beam(write(tmp_path, [trip(), cello()]))
    report = await run_beam(engine, conversations, k=8)
    first = report.results[0]
    assert first.evidence == [0] and 0 in first.top_sources and first.evidence_recall == 1.0 and first.all_evidence
    assert report.scores["information_extraction"]["n"] == 3.0 and report.scores["all"]["n"] == 8.0
    assert report.scores["abstention"]["evidence_recall"] is None and report.scores["abstention"]["evidenced"] == 0.0
    assert report.scores["all"]["evidenced"] == 7.0
    # Every message is recalled at k=8, so no update is missed.
    assert report.scores["knowledge_update"]["stale_only"] == 0.0 and report.scores["all"]["stale_only"] == 0.0
    assert report.evidence_unreadable == 1 and report.reader is None and report.scores["all"]["score"] is None
    assert report.judge_decoy_score is None and report.judge_audited == 0


async def test_answers_are_scored_on_rubrics_beside_the_credit_the_judge_gives_decoys(tmp_path):
    conversations = load_beam(write(tmp_path, [trip(), cello()]))
    reader = FakeChat(["Kyoto.", "Cherry blossom season.", "3500 dollars.", "There is no information about an airline.",
                       "First Kyoto, then the raise to 3500, then the 2000 budget.", "- Kyoto in April\n- 3500 dollars"])
    one = '{{"scores": [{{"index": 0, "score": {}}}]}}'.format
    # Per question: its answer, then its decoy if it has one. Event ordering
    # is aligned instead; abstention and instructions have no factual decoy.
    judge = FakeChat([one(1), one(0),                  # Kyoto; decoy "The cello" refused
                      one(0.5), one(1),                # season half right; decoy "The cello" accepted
                      one(1), "I cannot grade this",   # budget; its decoy's grading unreadable
                      one(1),                          # abstention
                      '{"positions": [{"index": 0, "position": 1}, {"index": 1, "position": 3}, {"index": 2, "position": 2}]}',
                      one(1)])                         # bullet points
    report = await run_beam(engine, conversations[:1], reader=reader, reader_name="fake", judge=judge, judge_name="lenient",
                            k=8, decoys_from=conversations)
    assert [r.score for r in report.results] == [1.0, 0.5, 1.0, 1.0, None, 1.0]
    assert [r.decoy for r in report.results] == ["The cello", "The cello", "An hour", None, None, None]
    assert [r.decoy_score for r in report.results] == [0.0, 1.0, None, None, None, None]
    order = report.results[4]
    assert order.tau_b == pytest.approx(0.3333) and order.events_mentioned == 1.0 and order.marks is None
    assert report.judge_decoy_score == 0.5 and report.judge_audited == 2 and report.judge_unreadable == 1
    assert report.scores["event_ordering"]["tau_b"] == pytest.approx(0.3333) and report.scores["event_ordering"]["score"] is None
    assert report.scores["all"]["score"] == pytest.approx(0.9) and report.scores["all"]["scored"] == 5.0
    assert len(judge.calls) == 9 and report.errors == 0
    system, user = reader.calls[0]
    assert "no information" in system
    assert "[15 March 2024] user: I am planning a trip to Kyoto in April." in user and "->->" not in user
    assert user.index("[15 March 2024]") < user.index("[2 April 2024]")
    # Messages reach the reader oldest first, whatever their rank; ids here
    # are in time order, and some ranks are not, or this would prove nothing.
    texts = {m.id: m.text for m in conversations[0].messages}
    assert any(r.top_sources != sorted(r.top_sources) for r in report.results)
    for result, (_, prompt) in zip(report.results, reader.calls):
        assert sorted(result.top_sources, key=lambda i: prompt.index(texts[i])) == sorted(result.top_sources)
    assert report.as_dict(with_items=False)["judge"] == "lenient" and "results" not in report.as_dict(with_items=False)


async def test_a_failed_model_call_is_counted_against_its_question_only(tmp_path):
    conversations = load_beam(write(tmp_path, [trip()]))
    reader = FakeChat([RuntimeError("timeout"), "Cherry blossom season."])
    report = await run_beam(engine, conversations, reader=reader, reader_name="fake", k=8, questions=2)
    assert report.errors == 1 and report.results[0].error == "RuntimeError: timeout"
    assert report.results[1].prediction == "Cherry blossom season." and report.results[1].error is None


async def test_questions_run_concurrently_up_to_the_limit_and_keep_their_order(tmp_path):
    conversations = load_beam(write(tmp_path, [trip()]))
    asked = [p.question for p in conversations[0].probes]

    class Slow:
        active = peak = 0

        async def complete(self, system, user):
            Slow.active += 1
            Slow.peak = max(Slow.peak, Slow.active)
            question = user.rsplit("User: ", 1)[1]
            # Earlier questions take longer, so they finish in reverse order.
            await asyncio.sleep(0.01 * (len(asked) - asked.index(question)))
            Slow.active -= 1
            return f"answer to {question}"

    report = await run_beam(engine, conversations, reader=Slow(), reader_name="slow", k=8, concurrency=3)
    assert Slow.peak == 3
    assert [r.question for r in report.results] == asked
    assert [r.prediction for r in report.results] == [f"answer to {q}" for q in asked]


async def test_directives_reach_the_reader_whatever_recall_found(tmp_path):
    from scone_memory.memory.directives import DIRECTIVE_PREFIX

    row = trip()
    # An assistant turn worded like a directive: it is the assistant's, and
    # is never read as the user's request.
    row["chat"][1][1]["content"] = "Always book ryokans early. ->-> 5,14)"
    conversations = load_beam(write(tmp_path, [row]))
    reader = FakeChat(["answer"] * 6)
    report = await run_beam(engine, conversations, reader=reader, reader_name="fake", k=1, directives=True)
    assert report.directives["c1"] == {"read": 4, "added": 1, "restated": 0, "cut": 0, "clipped": 0, "text_cut": 0,
                                       "held": 1, "shown": 1, "omitted": 0,
                                       "bytes": report.directives["c1"]["bytes"]}
    assert report.directive_bytes == DEFAULT_BLOCK_BYTES
    assert 0 < report.directives["c1"]["bytes"] <= DEFAULT_BLOCK_BYTES
    bullets = next(r for r in report.results if r.ability == "instruction_following")
    # Recall at k=1 misses the instruction; the block carries it anyway.
    assert 6 not in bullets.top_sources and bullets.evidence_recall == 0.0 and bullets.in_context == 1.0
    assert report.scores["instruction_following"]["in_context"] == 1.0
    assert report.scores["instruction_following"]["evidence_recall"] == 0.0
    for _, prompt in reader.calls:
        assert prompt.startswith(DIRECTIVE_PREFIX) and "Please always answer me in bullet points." in prompt
        assert "book ryokans" not in prompt.split("Recalled messages:")[0]
    without = await run_beam(engine, conversations, reader=FakeChat(["answer"] * 6), reader_name="fake", k=1)
    assert without.directives == {} and without.directive_bytes is None
    assert all(r.in_context is None for r in without.results)
    assert without.scores["instruction_following"]["in_context"] is None


async def test_arguments_that_cannot_be_honoured_are_refused(tmp_path):
    conversations = load_beam(write(tmp_path, [trip()]))
    with pytest.raises(ValueError):
        await run_beam(engine, conversations, judge=FakeChat([]))
    with pytest.raises(ValueError):
        await run_beam(engine, conversations, concurrency=0)
    with pytest.raises(ValueError):
        await run_beam(engine, conversations, abilities={"mind_reading"})
    with pytest.raises(ValueError):
        await run_beam(engine, conversations, directives=True, directive_bytes=MIN_BLOCK_BYTES - 1)


def test_the_command_scores_retrieval_and_leaves_the_configured_store_alone(tmp_path):
    from scone_memory.runtime import cli

    path = write(tmp_path, [trip(), cello()])
    report = tmp_path / "report.json"
    untouched = tmp_path / "configured.db"
    env = {"SCONE_EMBEDDER": "hash", "SCONE_DOCUMENTS": "sqlite", "SCONE_VECTORS": "sqlite", "SCONE_SQLITE_PATH": str(untouched)}
    out = io.StringIO()
    code = cli.main(["bench-beam", str(path), "--k", "3", "--conversations", "c1", "--abilities", "abstention,knowledge_update",
                     "--out", str(report)], env=env, stdin=io.StringIO(), out=out)
    assert code == 0, out.getvalue()
    saved = json.loads(report.read_text())
    assert saved["k"] == 3 and [r["ability"] for r in saved["results"]] == ["knowledge_update", "abstention"]
    # The unreadable source id is the season question's, which was not asked.
    assert saved["questions"] == 2 and saved["conversations"] == 1 and saved["evidence_unreadable"] == 0
    assert "knowledge_update" in out.getvalue() and not untouched.exists()
    out = io.StringIO()
    assert cli.main(["bench-beam", str(path), "--k", "1", "--conversations", "c1", "--abilities", "instruction_following",
                     "--directives"], env=env, stdin=io.StringIO(), out=out) == 0
    assert "in-ctx" in out.getvalue() and "directives: 1 recorded" in out.getvalue()
    assert cli.build_parser().parse_args(["bench-beam", str(path)]).directive_bytes == DEFAULT_BLOCK_BYTES
    for bad in (["--reader"], ["--judge"], ["--abilities", "abstention,mind_reading"], ["--conversations", "c1,c9"],
                ["--k", "0"], ["--questions", "0"], ["--concurrency", "0"],
                ["--directive-bytes", str(MIN_BLOCK_BYTES - 1)]):
        assert cli.main(["bench-beam", str(path), *bad], env=env, stdin=io.StringIO(), out=io.StringIO()) == 2, bad


def test_parquet_rows_read_like_json_rows(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    row = trip()
    path = tmp_path / "beam.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    [from_parquet] = load_beam(path)
    [from_json] = load_beam(write(tmp_path, [row]))
    assert from_parquet == from_json
