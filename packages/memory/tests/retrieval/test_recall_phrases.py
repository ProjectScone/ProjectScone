"""Passages that must hold a phrase, or must not, chosen before the limit.

The reference's keyword postprocessor drops retrieved nodes that lack a
required keyword or hold an excluded one, after retrieval, so a filter
that drops three of five returns two and says nothing. Here the phrases
are checked across the fused candidates before the limit, so a passage
ranked sixth that holds the phrase fills the place of one that does not;
every required phrase must appear and any excluded one removes a
passage; and the answer says how many candidates were read, how many
each rule dropped, and whether it came back short with passages beyond
the window never checked.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import create_app
from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.phrases import MAX_PHRASE_CHARS, MAX_PHRASES, holds
from scone_memory.runtime.cli import build_parser, run

pytestmark = pytest.mark.asyncio

NOTES = [
    "The crane survey found rust on the jib and grease missing from the slew ring.",
    "The crane survey was booked for May; the jib was repainted in June.",
    "The crane survey report went to the harbour board, who asked about the jib.",
    "The crane survey cost more than planned because the jib needed a new hook.",
    "The crane survey team noted the jib hoist rope was frayed near the drum.",
    "The canteen menu changed in April and the crane survey was not discussed.",
]


async def stored(**options):
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=InMemoryEventLog(), **options).open()
    for note in NOTES:
        await engine.remember("default", note)
    return engine


def test_a_phrase_matches_whole_words_whatever_the_case_and_punctuation_between_them():
    assert holds("Invoice INV-20931 was paid.", "inv 20931")
    assert holds("The SLEW-RING needed grease", "slew ring")
    assert not holds("The party was late", "art")
    assert not holds("slew, then ring", "slew ring"), "words apart are not the phrase"
    assert holds("港口起重机已检修", "起重机"), "scripts written without spaces match inside a run"
    assert holds("ㄅㄆㄇㄈ", "ㄆㄇ"), "every script the search tokenizer reads as unspaced, bopomofo included"


async def test_passages_without_a_required_phrase_give_their_places_to_ones_ranked_below_the_limit():
    engine = await stored()
    try:
        plain = await engine.recall("default", "crane survey jib", limit=3)
        needed = await engine.recall("default", "crane survey jib", limit=3, require=["hoist rope"])
    finally:
        await engine.close()
    assert not any("hoist rope" in item.text for item in plain.items), "the fixture must rank it below the limit"
    assert [item.text for item in needed.items] == [NOTES[4]]
    trace = needed.phrases
    assert trace.required == ["hoist rope"] and trace.dropped_required == trace.checked - 1
    # Every passage in the space was a candidate, so nothing beyond the window went unchecked.
    assert trace.short is False and "window" not in trace.why


async def test_every_required_phrase_must_appear_and_any_excluded_one_removes_a_passage():
    engine = await stored()
    try:
        both = await engine.recall("default", "crane survey jib", limit=6, require=["crane survey", "jib"])
        without = await engine.recall("default", "crane survey jib", limit=6, require=["crane survey"],
                                      exclude=["canteen", "new hook"])
    finally:
        await engine.close()
    assert NOTES[5] not in [item.text for item in both.items] and len(both.items) == 5
    assert {item.text for item in without.items} == set(NOTES[:3] + NOTES[4:5])
    assert without.phrases.dropped_excluded == 2 and without.phrases.dropped_required == 0


async def test_an_answer_filled_to_its_limit_is_not_short():
    engine = await stored()
    try:
        full = await engine.recall("default", "crane survey jib", limit=2, exclude=["canteen"])
        plain = await engine.recall("default", "crane survey jib", limit=2)
    finally:
        await engine.close()
    assert len(full.items) == 2 and full.phrases.short is False
    assert plain.phrases is None, "without phrases nothing is reported"


async def test_the_recall_event_records_what_the_phrases_dropped():
    engine = await stored()
    try:
        await engine.recall("default", "crane survey jib", limit=3, exclude=["canteen"])
        [event] = await engine.events.query("default", kind="recall")
    finally:
        await engine.close()
    assert event.payload["phrases"]["excluded"] == ["canteen"] and "dropped_excluded" in event.payload["phrases"]


@pytest.mark.parametrize("options", [{"require": [" "]}, {"require": ["jib"], "exclude": ["JIB"]},
                                     {"require": ["x"] * (MAX_PHRASES + 1)}, {"exclude": ["y" * (MAX_PHRASE_CHARS + 1)]},
                                     {"require": "jib"}, {"exclude": ["!!!"]}])
async def test_phrases_that_cannot_be_matched_or_contradict_are_refused(options):
    engine = await stored()
    try:
        with pytest.raises(InvalidInput):
            await engine.recall("default", "crane", **options)
    finally:
        await engine.close()


async def test_phrases_are_asked_for_over_http_and_the_command_line():
    engine = await stored()
    with TestClient(create_app(engine, {"key-a": "default"})) as client:
        headers = {"authorization": "Bearer key-a"}
        made = client.get("/v1/recall", params=[("q", "crane survey jib"), ("limit", "3"), ("require", "hoist rope"),
                                                ("exclude", "canteen")], headers=headers)
        bad = client.get("/v1/recall", params={"q": "crane", "require": "jib", "exclude": "jib"}, headers=headers)
        features = client.get("/v1/capabilities", headers=headers).json()["features"]
    out = io.StringIO()
    code = await run(build_parser().parse_args(["recall", "crane survey jib", "--limit", "3", "--require", "hoist rope",
                                                "--json"]), engine, io.StringIO(""), out)
    await engine.close()
    assert made.status_code == 200, made.text
    assert [item["text"] for item in made.json()["items"]] == [NOTES[4]]
    assert made.json()["phrases"]["required"] == ["hoist rope"] and made.json()["phrases"]["excluded"] == ["canteen"]
    assert bad.status_code == 422 and features["recall.phrases"] is True
    said = json.loads(out.getvalue())["phrases"]
    assert code == 0 and said["dropped_required"] == said["checked"] - 1 and said["short"] is False


async def test_an_answer_short_only_because_the_space_is_small_is_not_blamed_on_the_phrases():
    engine = await stored()
    try:
        small = await engine.recall("default", "crane survey jib", limit=10, exclude=["telescope"])
    finally:
        await engine.close()
    assert len(small.items) < 10 and small.phrases.dropped_excluded == 0 and small.phrases.short is False


async def test_an_answer_short_with_a_full_window_is_not_blamed_on_phrases_that_dropped_nothing():
    engine = await stored()
    try:
        full = await engine.recall("default", "crane survey jib", limit=5, candidate_limit=1, exclude=["telescope"])
    finally:
        await engine.close()
    assert len(full.items) < 5 and full.phrases.dropped_excluded == 0 and full.phrases.short is False


async def test_an_answer_short_after_drops_is_short_only_when_a_lane_filled_its_window():
    engine = await stored()
    try:
        # Six passages, a limit of ten: one dropped, five back, and every passage was a candidate.
        roomy = await engine.recall("default", "crane survey jib", limit=10, exclude=["canteen"])
        # Two candidates a lane: the lanes stop at their window, and passages past it went unread.
        narrow = await engine.recall("default", "crane survey jib", limit=3, candidate_limit=2,
                                     require=["hoist rope"])
    finally:
        await engine.close()
    assert len(roomy.items) == 5 and roomy.phrases.dropped_excluded == 1 and roomy.phrases.short is False
    assert narrow.phrases.short is True and "window" in narrow.phrases.why
    engine = await stored()
    try:
        vector_only = await engine.recall("default", "crane survey jib", limit=3, candidate_limit=2,
                                          lanes=("vector",), require=["hoist rope"])
    finally:
        await engine.close()
    assert vector_only.phrases.short is True, "the vector lane filling its window counts as much as the text lane"
    engine = await stored()
    try:
        text_only = await engine.recall("default", "crane survey jib", limit=3, candidate_limit=2,
                                        lanes=("text",), require=["hoist rope"])
    finally:
        await engine.close()
    assert text_only.phrases.short is True
    assert len(narrow.items) < 3 and narrow.phrases.dropped_required > 0
