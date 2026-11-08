"""The turns around a recalled turn, read from its own session.

The fixture's search finds the one turn that names the harbour launch; the
turns either side of it, which hold why and when, name nothing the question
asks for. The first test proves a search cut at one passage misses them, so
the rest test a mechanism that has something to do.
"""

from __future__ import annotations

from typing import Optional

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, Record
from scone_memory.core.errors import InvalidInput
from scone_memory.core.models import RecallResult
from scone_memory.core.ports import TextFilter
from scone_memory.retrieval import session_turns
from scone_memory.retrieval.session_turns import MAX_RADIUS, MAX_REACH, MAX_SEEDS, add_session_turns

QUESTION = "When is the harbour launch?"
FIRST = [
    "Morning! Did you sleep well after the trip?",
    "Not really, the audit kept me up most of the night.",
    "Oh no. What happened with it?",
    "We had to move the harbour launch to the fourteenth of March.",
    "Because the auditors found two open tickets.",
    "That sounds stressful, take it easy today.",
]
SECOND = [
    "Hey, back from the coast now.",
    "Bought a kite for my nephew on the way home.",
]


def turn(session: str, n: int, text: str, **metadata: str) -> Record:
    return Record(content=text, kind="conversation", source=f"{session}:{n}",
                  metadata={"session_id": session, **metadata})


async def engine_with(records: list[Record]) -> MemoryEngine:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    await engine.remember_many("s", records)
    return engine


async def conversation() -> MemoryEngine:
    """Session a, then session b, each written in order."""
    return await engine_with([turn("a", n, text) for n, text in enumerate(FIRST)]
                             + [turn("b", n, text) for n, text in enumerate(SECOND)])


def sources(result: RecallResult) -> list[Optional[str]]:
    return [item.source for item in result.items]


async def bring(engine: MemoryEngine, limit: int = 1, **options: object
                ) -> tuple[RecallResult, session_turns.SessionTurns]:
    found = await engine.recall("s", QUESTION, limit=limit)
    return await add_session_turns(found, engine.documents, "s", **options)  # type: ignore[arg-type]


async def test_a_recalled_turn_brings_the_turns_either_side_of_it_in_its_session():
    engine = await conversation()
    ordinary = await engine.recall("s", QUESTION, limit=1)
    assert sources(ordinary) == ["a:3"], "the fixture must rank the launch turn first and alone"
    found, trace = await bring(engine, radius=1)
    assert sources(found) == ["a:3", "a:2", "a:4"]
    assert "auditors found two open tickets" in found.items[2].text
    assert [item.score for item in found.items[1:]] == [0.0, 0.0]
    assert trace.record() == {"seeds": 1, "found": 2, "added": 2, "moved": 0, "not_looked": 0, "no_session": 0,
                              "short": 0, "shortened": 0, "skipped": None}
    assert trace.brought_chunk_ids == tuple(item.chunk_id for item in found.items[1:])


async def test_a_wider_radius_brings_more_turns_in_conversation_order():
    engine = await conversation()
    found, trace = await bring(engine, radius=2)
    assert sources(found) == ["a:3", "a:1", "a:2", "a:4", "a:5"] and trace.added == 4


async def only(engine: MemoryEngine, source: str) -> RecallResult:
    """A result holding just the turn ``source``, as if the search found it alone."""
    every = await engine.recall("s", QUESTION, limit=20)
    return every.model_copy(update={"items": [item for item in every.items if item.source == source]})


async def test_turns_of_another_session_are_never_brought_however_close():
    engine = await conversation()
    widened, trace = await add_session_turns(await only(engine, "a:5"), engine.documents, "s", radius=2)
    # a:5 is the last turn of session a; b:0 and b:1 follow it in the store.
    assert sources(widened) == ["a:5", "a:3", "a:4"]
    assert trace.short == 1, "one side found fewer turns than the radius"


async def test_turns_written_between_other_sessions_turns_are_found_within_reach():
    records = []
    for n, text in enumerate(FIRST):
        records.append(turn("a", n, text))
        records.append(turn("noise", n, f"Unrelated chatter number {n} about gardening."))
    engine = await engine_with(records)
    found, trace = await bring(engine, radius=1)
    assert sources(found) == ["a:3", "a:2", "a:4"] and trace.short == 0


async def test_reach_that_bites_is_counted_not_hidden():
    records = []
    for n, text in enumerate(FIRST):
        records.append(turn("a", n, text))
        records.extend(turn("noise", 10 * n + k, f"Unrelated chatter {n}.{k} about gardening.") for k in range(3))
    engine = await engine_with(records)
    found, trace = await bring(engine, radius=1, reach=3)
    # The neighbours are four ids away; three ids either side see only noise.
    assert sources(found) == ["a:3"]
    assert (trace.found, trace.short) == (0, 2)
    found, trace = await bring(engine, radius=1, reach=4)
    assert sources(found) == ["a:3", "a:2", "a:4"] and trace.short == 0


async def test_reach_counts_ids_so_records_of_another_kind_between_turns_use_it_up():
    records = []
    for n, text in enumerate(FIRST):
        records.append(turn("a", n, text))
        records.extend(Record(content=f"Note {n}.{k} on gardening.", kind="note", source=f"note:{n}.{k}",
                              metadata={"session_id": "a"}) for k in range(3))
    engine = await engine_with(records)
    found, trace = await bring(engine, radius=1, reach=3)
    # Paging by kind skips the notes, but reach counts ids: four ids away is past it.
    assert sources(found) == ["a:3"] and trace.short == 2
    found, trace = await bring(engine, radius=1, reach=4)
    assert sources(found) == ["a:3", "a:2", "a:4"]


async def test_a_turn_without_a_session_brings_nothing_and_is_counted():
    engine = await engine_with([Record(content=text, kind="conversation", source=str(n))
                                for n, text in enumerate(FIRST)])
    found, trace = await bring(engine, radius=1)
    assert sources(found) == ["3"]
    assert (trace.seeds, trace.no_session, trace.added) == (0, 1, 0)


async def test_a_neighbour_ranked_below_its_turn_moves_up_to_follow_it_with_its_own_score():
    engine = await conversation()
    ordinary = await engine.recall("s", QUESTION, limit=8)
    assert sources(ordinary)[0] == "a:3" and sources(ordinary).index("a:4") > 1
    widened, trace = await add_session_turns(ordinary, engine.documents, "s", radius=1, seeds=1)
    assert sources(widened)[:3] == ["a:3", "a:2", "a:4"]
    assert sorted(sources(widened), key=str) == sorted(sources(ordinary), key=str), "each passage once"
    assert {item.chunk_id: item.score for item in widened.items} == {item.chunk_id: item.score for item in ordinary.items}
    assert trace.moved == 2 and trace.added == 0


async def test_a_neighbour_ranked_above_its_turn_stays_where_it_ranked():
    engine = await conversation()
    ordinary = await engine.recall("s", QUESTION, limit=8)
    four = next(item for item in ordinary.items if item.source == "a:4")
    three = next(item for item in ordinary.items if item.source == "a:3")
    ranked = ordinary.model_copy(update={"items": [four, three]})
    widened, trace = await add_session_turns(ranked, engine.documents, "s", radius=1, seeds=2)
    # a:4 brings a:3 up behind it and adds a:5; a:3, already placed, keeps a:4
    # where it ranked and adds a:2 at the end. Each passage once.
    assert sources(widened) == ["a:4", "a:3", "a:5", "a:2"]
    assert (trace.seeds, trace.found, trace.added, trace.moved) == (2, 4, 2, 1)


async def test_seeds_past_the_limit_are_not_looked_at_and_say_so():
    engine = await conversation()
    found, trace = await bring(engine, limit=3, radius=1, seeds=1)
    assert (trace.seeds, trace.not_looked) == (1, 2)


async def test_a_neighbour_outside_the_callers_scope_is_not_brought():
    records = [turn("a", n, text, user="bob" if n in (2, 4) else "ann") for n, text in enumerate(FIRST)]
    engine = await engine_with(records)
    found, _ = await bring(engine, radius=1, scope=TextFilter(where={"user": "ann"}))
    # Both neighbours are bob's: neither is brought, and the turns beyond them
    # (a:1, a:5) are not brought in their stead.
    assert sources(found) == ["a:3"]
    found, _ = await bring(engine, radius=2, scope=TextFilter(where={"user": "ann"}))
    assert sources(found) == ["a:3", "a:1", "a:5"]
    found, _ = await bring(engine, radius=1, excluded_session="a")
    assert sources(found) == ["a:3"], "an excluded session's turns are not brought"


async def test_a_neighbour_dated_after_now_is_not_brought():
    engine = await engine_with([
        Record(content=text, kind="conversation", source=f"a:{n}", metadata={"session_id": "a"},
               created_at=f"2023-05-01T10:0{n}:00Z") for n, text in enumerate(FIRST)])
    found = await engine.recall("s", QUESTION, limit=1)
    assert sources(found) == ["a:3"]
    widened, _ = await add_session_turns(found, engine.documents, "s", radius=1, now="2023-05-01T10:03:30Z")
    assert sources(widened) == ["a:3", "a:2"]


async def test_a_long_neighbour_brings_only_its_first_chunks_and_says_so():
    long_reply = " ".join(f"Sentence {n} of a very long reply about the gardening club and its plans." for n in range(400))
    engine = await engine_with([turn("a", 0, long_reply), turn("a", 1, FIRST[3])])
    found, trace = await bring(engine, radius=1)
    brought = [item for item in found.items if item.source == "a:0"]
    assert len(brought) == session_turns.MAX_TURN_CHUNKS
    assert trace.shortened == 1


async def test_a_store_that_cannot_page_episodes_leaves_the_result_and_says_why():
    engine = await conversation()
    found = await engine.recall("s", QUESTION, limit=1)

    class Plain:
        async def get_episode(self, space, episode_id):
            return await engine.documents.get_episode(space, episode_id)

    widened, trace = await add_session_turns(found, Plain(), "s", radius=1)  # type: ignore[arg-type]
    assert widened == found and trace.skipped == "the store cannot page its episodes"


async def test_a_failing_store_leaves_the_result_whole_and_degraded():
    engine = await conversation()
    found = await engine.recall("s", QUESTION, limit=1)

    class Broken:
        def __getattr__(self, name):
            return getattr(engine.documents, name)

        async def page_episodes(self, space, before, limit, kind):
            raise RuntimeError("disk gone")

    widened, trace = await add_session_turns(found, Broken(), "s", radius=1)  # type: ignore[arg-type]
    assert widened.items == found.items
    assert any("session_turns: RuntimeError" in d for d in widened.degraded)
    assert trace.skipped == "the store failed: RuntimeError"


async def test_radius_zero_does_nothing():
    engine = await conversation()
    found, trace = await bring(engine, radius=0)
    assert sources(found) == ["a:3"] and trace.skipped == "no turns either side were asked for"


@pytest.mark.parametrize("options", [
    {"radius": -1}, {"radius": MAX_RADIUS + 1}, {"radius": 1.0}, {"radius": True},
    {"radius": 1, "seeds": -1}, {"radius": 1, "seeds": MAX_SEEDS + 1},
    {"radius": 1, "reach": 0}, {"radius": 1, "reach": MAX_REACH + 1},
    {"radius": 1, "session_field": ""},
], ids=lambda o: ",".join(f"{k}={v!r}" for k, v in o.items()))
async def test_arguments_out_of_range_are_refused(options):
    engine = await conversation()
    with pytest.raises(InvalidInput):
        await bring(engine, **options)
