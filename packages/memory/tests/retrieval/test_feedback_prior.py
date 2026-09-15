"""Recorded feedback as a ranking prior: a bounded term fused beside recency.

Lessons (``retrieval/lessons.py``) say what people said about a passage and
leave the order alone. This turns the same judgements into a small additive
term on the fused score, off unless ``feedback_weight`` is set: each
judgement is the latest of its question and halves every half-life; one useful
judgement gives nothing and two questions' do; only the newest two each way
count; a judgement against the passage outweighs every useful one older than
it; a judgement made of a passage whose content has since changed counts for
nothing; and the term is cut at ``MAX_FEEDBACK_BOOST``, with the result
saying each of those bounds bit.
"""

from __future__ import annotations

import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.core.ports import Event
from scone_memory.memory.engine import Record
from scone_memory.retrieval import feedback_prior
from scone_memory.retrieval.feedback_prior import MAX_FEEDBACK_BOOST, prior_terms
from scone_memory.testing import Clock

NOW = "2026-06-01T00:00:00Z"
PRINT = "fingerprint-of-seven"


def judged(event_id: int, ts: str, chunk: int, useful: bool, recall: int, fingerprint: str | None = PRINT,
           question: str | None = "") -> Event:
    """A judgement of ``recall``, which asked its own question unless ``question`` names one (None: unrecorded)."""
    payload: dict[str, object] = {"recall_event_id": recall, "chunk_id": chunk, "useful": useful, "note": None}
    if fingerprint is not None:
        payload["fingerprint"] = fingerprint
    if question is not None:
        payload["question"] = question or f"question-of-recall-{recall}"
    return Event(event_id=event_id, space="default", kind="feedback", ts=ts, schema_version=1, payload=payload)


def terms(events: list[Event], **settings: object) -> feedback_prior.PriorTerms:
    options: dict[str, object] = {"now": NOW, "weight": 0.001, "half_life_days": 30, "min_corroboration": 2,
                                  "max_boost": 1.0}
    options.update(settings)
    return prior_terms(events, {7: PRINT}, **options)  # type: ignore[arg-type]


def test_one_useful_judgement_gives_no_boost_and_two_do():
    one = terms([judged(1, NOW, 7, True, recall=1)])
    assert one.terms == {} and one.tentative == 1, "one person's word is not yet a preference"
    two = terms([judged(1, NOW, 7, True, recall=1), judged(2, NOW, 7, True, recall=2)])
    assert two.terms == {7: pytest.approx(0.002)} and two.tentative == 0
    again = terms([judged(1, NOW, 7, True, recall=1), judged(2, NOW, 7, True, recall=1)])
    assert again.terms == {}, "one recall judged twice is one judgement"


def test_one_question_asked_again_and_judged_again_is_one_judgement():
    asked_twice = terms([judged(1, NOW, 7, True, recall=1, question="same"), judged(2, NOW, 7, True, recall=2, question="same")])
    assert asked_twice.terms == {} and asked_twice.tentative == 1, "two recalls of one question do not corroborate"
    turned = terms([judged(1, "2026-05-30T00:00:00Z", 7, False, recall=1, question="same"),
                    judged(2, NOW, 7, True, recall=2, question="same"), judged(3, NOW, 7, True, recall=3)])
    assert turned.terms == {7: pytest.approx(0.002)}, "the latest judgement of a question replaces its earlier one"
    days = [f"2026-05-{day:02d}T00:00:00Z" for day in (28, 29, 30, 31)]
    rejudged = terms([judged(1, days[0], 7, True, recall=1, question="a"), judged(2, days[1], 7, False, recall=2, question="b"),
                      judged(3, days[2], 7, True, recall=3, question="a"), judged(4, days[3], 7, True, recall=4, question="c")])
    assert rejudged.terms == {7: pytest.approx(0.001 * (-(0.5 ** (3 / 30)) + 0.5 ** (2 / 30) + 0.5 ** (1 / 30)))}, \
        "a question judged again counts from when it was judged again: after the judgement against"
    unasked = terms([judged(1, NOW, 7, True, recall=1, question=None), judged(2, NOW, 7, True, recall=2)])
    assert unasked.terms == {} and unasked.unverified == 1, "a judgement whose question is not recorded is not trusted"


def test_a_pile_of_judgements_weighs_no_more_than_the_corroboration_that_lets_it_count():
    piled = [judged(index, NOW, 7, True, recall=index) for index in range(1, 7)]
    held = terms(piled)
    assert held.terms == {7: pytest.approx(0.002)} and held.held == 1 and held.capped == 0
    assert terms(piled[:2]).held == 0, "two fresh judgements are exactly what corroboration asks"
    sunk = terms([judged(index, NOW, 7, False, recall=index) for index in range(1, 7)])
    assert sunk.terms == {7: pytest.approx(-0.002)} and sunk.held == 1
    aged_against = terms([judged(1, "2026-04-02T00:00:00Z", 7, False, recall=1), *(judged(index, NOW, 7, False, recall=index)
                                                                               for index in (2, 3))])
    assert aged_against.terms == {7: pytest.approx(-0.002)}, "the newest two against, not the oldest"
    assert terms(piled, min_corroboration=3).terms == {7: pytest.approx(0.003)}
    old = [judged(index, "2026-04-02T00:00:00Z", 7, True, recall=index) for index in range(1, 7)]
    assert terms(old).terms == {7: pytest.approx(0.001 * 2 * 0.25)}, "held to its newest two, so age still tells"
    mixed = [judged(1, "2026-05-02T00:00:00Z", 7, True, recall=1), *piled[1:3],
             judged(9, "2026-04-02T00:00:00Z", 7, True, recall=9)]
    assert terms(mixed).terms == {7: pytest.approx(0.002)}, "the newest two, not the oldest"
    assert feedback_prior.PriorTerms({7: 0.002}, held=1).record([7])["held"] == 1


def test_judgements_weigh_less_with_age():
    old = terms([judged(1, "2026-04-02T00:00:00Z", 7, True, recall=1), judged(2, "2026-04-02T00:00:00Z", 7, True, recall=2)])
    assert old.terms == {7: pytest.approx(0.001 * 2 * 0.5 ** (60 / 30))}


def test_a_judgement_against_outweighs_every_useful_one_older_than_it():
    older_useful = [judged(1, "2026-05-20T00:00:00Z", 7, True, recall=1), judged(2, "2026-05-21T00:00:00Z", 7, True, recall=2),
                    judged(3, "2026-05-22T00:00:00Z", 7, True, recall=3)]
    against = terms([*older_useful, judged(4, "2026-05-01T00:00:00Z", 7, False, recall=4)])
    assert against.terms[7] > 0, "an older judgement against is outweighed by newer useful ones"
    turned = terms([*older_useful, judged(4, "2026-05-30T00:00:00Z", 7, False, recall=4)])
    assert turned.terms == {7: pytest.approx(-0.001 * 0.5 ** (2 / 30))}, "the useful judgements before it no longer count"
    recovered = terms([*older_useful, judged(4, "2026-05-30T00:00:00Z", 7, False, recall=4),
                       judged(5, NOW, 7, True, recall=5)])
    assert recovered.terms == {7: pytest.approx(-0.001 * 0.5 ** (2 / 30))}, \
        "one useful judgement after it is not yet corroborated, so it offsets nothing"
    corroborated = terms([*older_useful, judged(4, "2026-05-30T00:00:00Z", 7, False, recall=4),
                          judged(5, NOW, 7, True, recall=5), judged(6, NOW, 7, True, recall=6)])
    assert corroborated.terms[7] == pytest.approx(0.001 * (2 - 0.5 ** (2 / 30)))


def test_a_judgement_of_content_that_has_changed_counts_for_nothing():
    changed = [judged(1, NOW, 7, True, recall=1, fingerprint="what-it-said-then"), judged(2, NOW, 7, True, recall=2)]
    found = terms(changed)
    assert found.terms == {} and found.stale == 1 and found.tentative == 1
    unmarked = terms([judged(1, NOW, 7, True, recall=1, fingerprint=None), judged(2, NOW, 7, True, recall=2)])
    assert unmarked.terms == {} and unmarked.unverified == 1, "a judgement that cannot be checked is not trusted"


def test_the_boost_is_cut_at_its_bound_and_says_so():
    many = [judged(index, NOW, 7, True, recall=index) for index in range(1, 3)]
    within = terms(many, weight=0.001)
    assert within.terms == {7: pytest.approx(0.002)} and within.capped == 0
    cut = terms(many, weight=0.001, max_boost=0.0015)
    assert cut.terms == {7: pytest.approx(0.0015)} and cut.capped == 1
    sunk = terms([judged(index, NOW, 7, False, recall=index) for index in range(1, 3)], weight=0.001, max_boost=0.0015)
    assert sunk.terms == {7: pytest.approx(-0.0015)} and sunk.capped == 1
    assert MAX_FEEDBACK_BOOST == pytest.approx(2 * (1 / 61 - 1 / 62)), "first place over second when both lanes agree at full voice"


def test_only_the_candidates_asked_about_are_folded():
    other = terms([judged(1, NOW, 8, True, recall=1, fingerprint="x"), judged(2, NOW, 8, True, recall=2, fingerprint="x")])
    assert other.terms == {} and other.stale == 0


def test_a_recall_judged_again_counts_at_its_latest_time():
    """Its first judgement came before the one against, its latest after: it counts."""
    events = [judged(1, "2026-05-01T00:00:00Z", 7, True, recall=1), judged(2, "2026-05-02T00:00:00Z", 7, False, recall=2),
              judged(3, "2026-05-03T00:00:00Z", 7, True, recall=1), judged(4, "2026-05-04T00:00:00Z", 7, True, recall=3)]
    found = terms(events)
    ages = (30, 29, 28)
    assert found.terms == {7: pytest.approx(0.001 * (-(0.5 ** (ages[0] / 30)) + 0.5 ** (ages[1] / 30) + 0.5 ** (ages[2] / 30)))}


def test_events_other_than_feedback_are_not_judgements():
    recall = Event(event_id=9, space="default", kind="recall", ts=NOW, schema_version=1,
                   payload={"items": [{"chunk_id": 7}], "useful": True, "chunk_id": 7})
    assert terms([recall, judged(1, NOW, 7, True, recall=1), judged(2, NOW, 7, True, recall=2)]).terms == \
        {7: pytest.approx(0.002)}


def test_the_record_names_the_terms_of_returned_passages_only():
    record = feedback_prior.PriorTerms({1: 0.0002, 2: -0.0001}, weight=0.0001).record([1, 3])
    assert record["returned_terms"] == {"1": 0.0002} and (record["boosted"], record["demoted"]) == (1, 1)
    folded = terms([judged(1, NOW, 7, True, recall=1)], half_life_days=7, min_corroboration=3).record([7])
    assert (folded["half_life_days"], folded["min_corroboration"]) == (7.0, 3), "the settings it was folded with"


async def test_the_read_skips_what_it_cannot_check_and_what_recall_did_not_find():
    from scone_memory.core.models import Chunk

    class NoEpisodes:
        async def get_episode(self, space: str, episode_id: int) -> None:
            return None

    log = InMemoryEventLog()
    from scone_memory.core.ports import NewEvent

    for recall, chunk in ((1, 7), (2, 7), (3, 8), (4, 8)):
        await log.append(NewEvent(ts=NOW, space="default", kind="feedback",
                                  payload={"recall_event_id": recall, "chunk_id": chunk, "useful": True, "fingerprint": "x",
                                           "question": f"q{recall}"}))
    candidate = Chunk(chunk_id=7, episode_id=70, space="default", ordinal=0, start=0, end=4, text="text", created_at=NOW)
    found = await feedback_prior.read_prior(log, NoEpisodes(), "default", {7: candidate},  # type: ignore[arg-type]
                                            now=NOW, weight=0.0001)
    assert found.terms == {} and found.events_read == 4 and found.stale == 0
    assert found.unverified == 2, "a candidate whose episode cannot be read cannot be checked: its judgements say so"


def test_bad_weights_are_refused():
    for weight in (-0.1, float("nan"), float("inf"), True, "0.1", feedback_prior.MAX_FEEDBACK_WEIGHT + 1):
        with pytest.raises(InvalidInput):
            feedback_prior.validate_feedback_weight(weight)
    feedback_prior.validate_feedback_weight(0)


async def engine_with(clock: Clock, *, weight: float, events: InMemoryEventLog | None = None,
                      documents: InMemoryDocumentStore | None = None, passages: tuple[str, ...] = ()) -> MemoryEngine:
    engine = await MemoryEngine(documents or InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock,
                                events=events if events is not None else InMemoryEventLog(),
                                feedback_weight=weight).open()
    for passage in passages or ("The harbour crane survey is booked for May.", "The harbour crane was repainted in May."):
        await engine.remember("default", passage)
    return engine


async def judge_twice(engine: MemoryEngine, query: str, chunk: int, useful: bool = True) -> None:
    """Two judgements, from two questions: the query and the same words in another case."""
    for asked in (query, query.upper()):
        result = await engine.recall("default", asked, lanes=TEXT)
        await engine.feedback("default", result.event_id, chunk, useful)


#: One lane, so first place is ahead of second by one rank step (1/61 - 1/62), which a
#: corroborated term below the bound can cross; two agreeing lanes put it a full bound ahead.
TEXT = ("text",)
QUERY = "harbour crane survey"


async def test_feedback_records_what_the_passage_said_when_it_was_judged():
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with(clock, weight=0.0)
    try:
        result = await engine.recall("default", QUERY)
        item = result.items[0]
        event = await engine.feedback("default", result.event_id, item.chunk_id, True)
        episode = await engine.documents.get_episode("default", item.episode_id)
    finally:
        await engine.close()
    assert episode is not None
    assert event.payload["fingerprint"] == feedback_prior.fingerprint(episode.content, item.text)


async def test_recall_with_the_weight_off_is_as_it_was_and_on_moves_a_corroborated_passage():
    clock = Clock("2026-05-01T00:00:00.000Z")
    events = InMemoryEventLog()
    engine = await engine_with(clock, weight=0.0, events=events)
    try:
        before = await engine.recall("default", QUERY, lanes=TEXT)
        second = before.items[1].chunk_id
        await judge_twice(engine, QUERY, second)
        after_off = await engine.recall("default", QUERY, lanes=TEXT)
        engine.feedback_weight = 0.0002
        moved = await engine.recall("default", QUERY, lanes=TEXT)
        emitted = (await events.query("default", kind="recall", limit=1))[0]
    finally:
        await engine.close()
    assert [item.chunk_id for item in after_off.items] == [item.chunk_id for item in before.items]
    assert after_off.feedback_prior is None and "feedback_prior" not in json.dumps(after_off.model_dump(mode="json"))
    assert moved.items[0].chunk_id == second
    record = moved.feedback_prior
    assert record is not None and record["weight"] == 0.0002 and record["boosted"] == 1 and record["demoted"] == 0
    assert record["events_read"] == 2 and record["events_cut"] is False and record["capped"] == 0
    assert record["returned_terms"] == {str(second): pytest.approx(0.0004)}
    assert emitted.payload["feedback_prior"]["boosted"] == 1


async def test_one_caller_asking_the_same_question_again_corroborates_nothing():
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with(clock, weight=0.0002)
    try:
        engine.record_queries = True  # even a query kept in the clear is only a hash on the judgement
        before = await engine.recall("default", QUERY, lanes=TEXT)
        second = before.items[1].chunk_id
        marks = []
        for _ in range(3):
            shown = await engine.recall("default", QUERY, lanes=TEXT)
            marks.append(await engine.feedback("default", shown.event_id, second, True))
        after = await engine.recall("default", QUERY, lanes=TEXT)
    finally:
        await engine.close()
    assert len({mark.payload["question"] for mark in marks}) == 1 and marks[0].payload["question"] != QUERY
    assert [item.chunk_id for item in after.items] == [item.chunk_id for item in before.items]
    assert after.feedback_prior is not None and after.feedback_prior["boosted"] == 0 and after.feedback_prior["tentative"] == 1


async def test_the_same_words_are_one_question_whether_their_query_was_kept_in_the_clear_or_hashed():
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with(clock, weight=0.0002)
    try:
        second = (await engine.recall("default", QUERY, lanes=TEXT)).items[1].chunk_id
        marks = []
        for kept_in_the_clear in (True, False):
            engine.record_queries = kept_in_the_clear
            shown = await engine.recall("default", QUERY, lanes=TEXT)
            marks.append(await engine.feedback("default", shown.event_id, second, True))
        after = await engine.recall("default", QUERY, lanes=TEXT)
    finally:
        await engine.close()
    assert marks[0].payload["question"] == marks[1].payload["question"], "turning query recording on corroborates nothing"
    assert after.feedback_prior is not None and after.feedback_prior["boosted"] == 0 and after.feedback_prior["tentative"] == 1


async def test_the_bound_limits_each_term_not_who_a_passage_can_pass():
    """Both lanes put the leader first; a leader sunk and a follower lifted close twice the bound."""
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with(clock, weight=0.0, passages=(
        "harbour crane survey booked", "harbour crane survey notes about the old paint and the rust"))
    engine.recency_weight = 0.0
    query = "harbour crane survey booked"
    try:
        lanes = [[item.chunk_id for item in (await engine.recall("default", query, lanes=(lane,))).items]
                 for lane in ("text", "vector")]
        lead, follow = lanes[0][:2]
        first, second = (await engine.recall("default", query)), (await engine.recall("default", query.upper()))
        await engine.feedback("default", first.event_id, lead, False)
        await engine.feedback("default", first.event_id, follow, True)
        await engine.feedback("default", second.event_id, follow, True)
        engine.feedback_weight = 1.0
        after = await engine.recall("default", query)
    finally:
        await engine.close()
    assert lanes[1][:2] == [lead, follow], "both lanes agree"
    assert after.feedback_prior is not None
    assert after.feedback_prior["returned_terms"] == {str(follow): round(MAX_FEEDBACK_BOOST, 6),
                                                      str(lead): round(-MAX_FEEDBACK_BOOST, 6)}
    assert [item.chunk_id for item in after.items][:2] == [follow, lead]


def test_the_environment_template_states_the_bound_the_code_cuts_at():
    from ..paths import REPO_ROOT

    template = (REPO_ROOT / ".env.example").read_text()
    block = template[:template.index("# SCONE_FEEDBACK_WEIGHT=")].rsplit("\n\n", 1)[-1].rsplit("SCONE_RECENCY_HALF_LIFE_DAYS", 1)[-1]
    assert f"{MAX_FEEDBACK_BOOST:.6f}" in block and "0.01" not in block, block


async def test_the_bound_cuts_the_term_on_the_recall_path_and_says_so(monkeypatch):
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with(clock, weight=0.0002)
    try:
        before = await engine.recall("default", QUERY, lanes=TEXT)
        second = before.items[1].chunk_id
        await judge_twice(engine, QUERY, second)
        monkeypatch.setattr(feedback_prior, "MAX_FEEDBACK_BOOST", 0.0001)
        held = await engine.recall("default", QUERY, lanes=TEXT)
    finally:
        await engine.close()
    assert held.items[0].chunk_id != second, "a term cut below one rank step moves nothing"
    assert held.feedback_prior is not None and held.feedback_prior["capped"] == 1
    assert held.feedback_prior["max_boost"] == 0.0001 and held.feedback_prior["returned_terms"] == {str(second): 0.0001}


async def test_a_judgement_against_sinks_a_passage():
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with(clock, weight=0.0003)
    try:
        before = await engine.recall("default", QUERY, lanes=TEXT)
        first = before.items[0].chunk_id
        await engine.feedback("default", before.event_id, first, False)
        after = await engine.recall("default", QUERY, lanes=TEXT)
    finally:
        await engine.close()
    assert after.items[0].chunk_id != first and after.feedback_prior is not None and after.feedback_prior["demoted"] == 1


async def test_a_prior_is_dropped_when_the_passage_under_its_id_now_says_something_else():
    """A document store rebuilt against a kept event log hands the same ids to other text."""
    clock = Clock("2026-05-01T00:00:00.000Z")
    events = InMemoryEventLog()
    first = await engine_with(clock, weight=0.0002, events=events)
    try:
        judged_result = await first.recall("default", QUERY, lanes=TEXT)
        second = judged_result.items[1].chunk_id
        await judge_twice(first, QUERY, second)
    finally:
        await first.close()
    rebuilt = await engine_with(clock, weight=0.0002, events=events, passages=(
        "The harbour crane survey is booked for June.", "The harbour crane was sold in June."))
    try:
        after = await rebuilt.recall("default", QUERY, lanes=TEXT)
    finally:
        await rebuilt.close()
    assert [item.chunk_id for item in after.items][1] == second, "the id judged now names other text"
    assert after.feedback_prior is not None
    assert after.feedback_prior["stale"] == 2 and after.feedback_prior["boosted"] == 0
    assert after.items[0].text == "The harbour crane survey is booked for June."


async def test_a_prior_is_dropped_when_the_rest_of_its_episode_changed_though_its_own_text_did_not():
    clock = Clock("2026-05-01T00:00:00.000Z")
    events = InMemoryEventLog()

    async def split(ending: str) -> MemoryEngine:
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock,
                                    events=events, feedback_weight=0.0002, chunk_target=44).open()
        # Keyed, as a synced file is: its content hash is its key, the same for both contents.
        await engine.remember_many("default", [Record("The harbour crane survey is booked for May.\n\n" + ending,
                                                      dedup_key="notes/crane.md")])
        return engine

    hashes = []
    first = await split("The harbour crane was painted blue in the spring.")
    try:
        hashes.append(await first.documents.get_episode("default", 1))
        judged_result = await first.recall("default", QUERY, lanes=TEXT)
        judged_item = judged_result.items[0]
        await judge_twice(first, QUERY, judged_item.chunk_id)
        kept = await first.recall("default", QUERY, lanes=TEXT)
    finally:
        await first.close()
    rebuilt = await split("The harbour crane was sold for scrap in autumn.")
    try:
        hashes.append(await rebuilt.documents.get_episode("default", 1))
        after = await rebuilt.recall("default", QUERY, lanes=TEXT)
    finally:
        await rebuilt.close()
    assert kept.feedback_prior is not None and kept.feedback_prior["boosted"] == 1
    same = next(item for item in after.items if item.chunk_id == judged_item.chunk_id)
    assert (same.episode_id, same.text) == (judged_item.episode_id, judged_item.text), "only the episode around it changed"
    assert len({episode.content_hash for episode in hashes}) == 1, "and its content hash did not"
    assert after.feedback_prior is not None
    assert after.feedback_prior["stale"] == 2 and after.feedback_prior["boosted"] == 0


async def test_a_prior_is_dropped_when_the_same_episode_was_chunked_differently():
    clock = Clock("2026-05-01T00:00:00.000Z")
    events = InMemoryEventLog()
    content = "The harbour crane survey is booked for May.\n\nThe harbour crane was sold for scrap in autumn."

    async def chunked(target: int) -> MemoryEngine:
        engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock,
                                    events=events, feedback_weight=0.0002, chunk_target=target).open()
        await engine.remember("default", content)
        return engine

    first = await chunked(44)
    try:
        judged_result = await first.recall("default", QUERY, lanes=TEXT)
        judged_item = judged_result.items[0]
        await judge_twice(first, QUERY, judged_item.chunk_id)
    finally:
        await first.close()
    rechunked = await chunked(24)
    try:
        after = await rechunked.recall("default", QUERY, lanes=TEXT)
    finally:
        await rechunked.close()
    same = next(item for item in after.items if item.chunk_id == judged_item.chunk_id)
    assert same.episode_id == judged_item.episode_id and same.text != judged_item.text
    assert after.feedback_prior is not None
    assert after.feedback_prior["stale"] == 2 and after.feedback_prior["boosted"] == 0


async def test_the_read_says_when_its_bound_cut_it(monkeypatch):
    import scone_memory.retrieval.lessons as lessons

    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with(clock, weight=0.0002)
    try:
        before = await engine.recall("default", QUERY, lanes=TEXT)
        await judge_twice(engine, QUERY, before.items[1].chunk_id)
        whole = await engine.recall("default", QUERY, lanes=TEXT)
        monkeypatch.setattr(lessons, "MAX_FEEDBACK_EVENTS", 1)
        cut = await engine.recall("default", QUERY, lanes=TEXT)
    finally:
        await engine.close()
    assert whole.feedback_prior is not None and whole.feedback_prior["events_cut"] is False
    assert whole.feedback_prior["boosted"] == 1
    assert cut.feedback_prior is not None and cut.feedback_prior["events_cut"] is True
    assert cut.feedback_prior["events_read"] == 1 and cut.feedback_prior["boosted"] == 0


async def test_judgements_older_than_the_window_are_not_read(monkeypatch):
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with(clock, weight=0.0002)
    try:
        before = await engine.recall("default", QUERY, lanes=TEXT)
        await judge_twice(engine, QUERY, before.items[1].chunk_id)
        clock.now = "2026-05-03T00:00:00.000Z"
        with monkeypatch.context() as patched:
            patched.setattr(feedback_prior, "FEEDBACK_WINDOW_DAYS", 1)
            narrow = await engine.recall("default", QUERY, lanes=TEXT)
        clock.now = "2026-07-29T00:00:00.000Z"
        inside = await engine.recall("default", QUERY, lanes=TEXT)
        clock.now = "2026-07-31T00:00:00.000Z"
        outside = await engine.recall("default", QUERY, lanes=TEXT)
    finally:
        await engine.close()
    assert narrow.feedback_prior is not None and (narrow.feedback_prior["events_read"], narrow.feedback_prior["window_days"]) == (0, 1)
    assert inside.feedback_prior is not None and inside.feedback_prior["events_read"] == 2
    assert outside.feedback_prior is not None and outside.feedback_prior["events_read"] == 0
    # A read of none says what it looked over, so it does not read as "never judged".
    assert (outside.feedback_prior["window_days"], outside.feedback_prior["half_life_days"],
            outside.feedback_prior["min_corroboration"]) == (90, 30.0, 2)


async def test_feedback_on_a_passage_that_cannot_be_read_records_no_fingerprint(monkeypatch):
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await engine_with(clock, weight=0.0)
    try:
        result = await engine.recall("default", QUERY)
        forgotten, kept = result.items[0], result.items[1]
        await engine.forget("default", forgotten.episode_id)
        gone = await engine.feedback("default", result.event_id, forgotten.chunk_id, True)

        async def no_episode(space: str, episode_id: int) -> None:
            return None

        monkeypatch.setattr(engine.documents, "get_episode", no_episode)
        orphan = await engine.feedback("default", result.event_id, kept.chunk_id, True)
    finally:
        await engine.close()
    assert "fingerprint" not in gone.payload and "fingerprint" not in orphan.payload


async def test_a_weight_with_no_event_log_says_there_was_nothing_to_read():
    clock = Clock("2026-05-01T00:00:00.000Z")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock,
                                feedback_weight=0.0002).open()
    try:
        await engine.remember("default", "The harbour crane survey is booked for May.")
        result = await engine.recall("default", "harbour crane survey")
    finally:
        await engine.close()
    assert result.feedback_prior is not None and result.feedback_prior["read"] is False


async def test_the_engine_refuses_a_bad_weight():
    with pytest.raises(InvalidInput):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), feedback_weight=-1)
