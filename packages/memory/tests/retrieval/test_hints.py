"""A question's own words name its scope: dates, a kind, tags, a place -- read by rule, said back, never over the caller's."""
from __future__ import annotations

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.hints import InferredScope, Reading, apply_scope, infer_scope

NOW = "2026-09-14T12:00:00Z"  # a Monday


def readings(question):
    return [(r.filter, r.value, r.words) for r in infer_scope(question, now=NOW).readings]


def test_a_date_the_question_says_becomes_a_window_with_its_words():
    scope = infer_scope("what did we decide about the launch last week?", now=NOW)
    assert scope.since == "2026-09-07T00:00:00Z" and scope.until == "2026-09-13T23:59:59Z"
    assert [(r.filter, r.words) for r in scope.readings] == [("since", "last week"), ("until", "last week")]
    assert infer_scope("what did we decide about the launch?", now=NOW).empty


def test_a_kind_a_tag_and_a_place_are_read_with_their_words():
    assert readings("the budget figures in my notes") == [("kind", "note", "in my notes")]
    assert readings("anything from the PDFs about pilots") == [("kind", "file", "from the PDFs")]
    scope = infer_scope("billing in our conversations tagged urgent #finance", now=NOW)
    assert scope.kind == "conversation" and scope.tags == ("urgent", "finance")
    assert [(r.filter, r.words) for r in scope.readings] == [("kind", "in our conversations"), ("tag", "tagged urgent"), ("tag", "#finance")]
    assert readings("the retry policy under docs/agents/") == [("source_prefix", "docs/agents/", "under docs/agents/")]
    assert readings("the steps in README.md for install") == [("source_prefix", "README.md", "in README.md")]
    assert not any(r[0] == "tag" for r in readings("issue #12 in the tracker")), "a tag begins with a letter"
    assert readings("see https://docs.example.com/setup/#install for the steps") == [], "a URL fragment is not a tag"
    assert readings("what does #include do") == [("tag", "include", "#include")], "read by shape; the space says whether it is a tag"
    flagship = infer_scope("what did we decide about the launch in last week's notes?", now=NOW)
    assert (flagship.kind, flagship.since, flagship.until) == ("note", "2026-09-07T00:00:00Z", "2026-09-13T23:59:59Z")
    assert [(r.filter, r.words) for r in flagship.readings] == [("since", "last week"), ("until", "last week"), ("kind", "in last week's notes")]
    assert readings("in order to read notes") == [], "only a determiner, a dating word or a possessive stands between the placing word and the kind"
    assert readings("notes about the harbour") == [], "a kind is read only after a word that places the question in it"
    assert readings("what our conversations say about billing") == [], "and 'our conversations' with no placing word is left alone"


def test_the_callers_filters_are_kept_over_the_questions():
    scope = infer_scope("the budget in my notes last week tagged urgent under docs/", now=NOW)
    searched = apply_scope(scope, since=None, until=None, kind=None, tags=[], source_prefix=None)
    assert (searched.since, searched.until, searched.kind, searched.tags, searched.source_prefix) == (
        "2026-09-07T00:00:00Z", "2026-09-13T23:59:59Z", "note", ("urgent",), "docs/")
    assert [r.filter for r in searched.applied] == ["since", "until", "kind", "tag", "source_prefix"]
    assert searched.record(scope)["applied"][2] == {"filter": "kind", "value": "note", "words": "in my notes"}
    kept = apply_scope(scope, since="2026-01-01T00:00:00Z", until=None, kind="file", tags=["x"], source_prefix="src/")
    assert (kept.since, kept.until, kept.kind, kept.tags, kept.source_prefix) == ("2026-01-01T00:00:00Z", None, "file", ("x",), "src/")
    assert kept.applied == (), "a caller's window, kind, tags and place stand; the question's readings are said, not applied"
    assert [(r.filter, why) for r, why in kept.withheld] == [
        ("since", "the caller set the window"), ("until", "the caller set the window"), ("kind", "the caller set the kind"),
        ("tag", "the caller set tags"), ("source_prefix", "the caller set the place")]
    assert kept.record(scope)["withheld"][3] == {"filter": "tag", "value": "urgent", "words": "tagged urgent", "because": "the caller set tags"}


def test_a_tag_the_space_does_not_hold_is_read_and_withheld():
    scope = infer_scope("the harbour #Urgent #finance", now=NOW)
    searched = apply_scope(scope, since=None, until=None, kind=None, tags=[], source_prefix=None, known_tags={"URGENT", "billing"})
    assert searched.tags == ("URGENT",), "applied as the space spells it, so the filter matches what is stored"
    assert [r.record() for r in searched.applied] == [{"filter": "tag", "value": "URGENT", "words": "#Urgent"}]
    assert [(r.value, why) for r, why in searched.withheld] == [("finance", "no memory in this space carries the tag")]
    unknown = apply_scope(scope, since=None, until=None, kind=None, tags=[], source_prefix=None)
    assert unknown.tags == ("urgent", "finance"), "with no word on what the space holds, every tag read is applied"


def test_records_say_what_was_read_and_bad_questions_are_refused():
    scope = infer_scope("in my notes", now=NOW)
    assert scope.record()["readings"] == [{"filter": "kind", "value": "note", "words": "in my notes"}]
    assert "never replaced" in scope.record()["notice"]
    assert InferredScope().record()["readings"] == [] and Reading("kind", "note", "in notes").record()["words"] == "in notes"
    with pytest.raises(InvalidInput):
        infer_scope("x" * 4_001, now=NOW)
    with pytest.raises(InvalidInput):
        infer_scope(None, now=NOW)  # type: ignore[arg-type]


async def test_the_route_and_the_command_search_with_what_the_question_says_and_say_so(tmp_path):
    import io

    from fastapi.testclient import TestClient

    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.api.app import create_app
    from scone_memory.runtime.cli import build_parser, run

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        await engine.remember("s", "The harbour closes to sailing boats every November.", kind="note", tags=["tides"])
        await engine.remember("s", "The harbour closes to sailing boats every November, the file says.", kind="file", source="docs/harbour.md")
        with TestClient(create_app(engine, {"key-a": "s"})) as client:
            plain = client.get("/v1/recall", params={"q": "when does the harbour close, in my notes"}, headers={"Authorization": "Bearer key-a"}).json()
            assert {item["episode_id"] for item in plain["items"]} == {1, 2} and "inferred" not in plain
            scoped = client.get("/v1/recall", params={"q": "when does the harbour close, in my notes", "infer": "true"},
                                headers={"Authorization": "Bearer key-a"}).json()
            assert [item["episode_id"] for item in scoped["items"]] == [1], "the question's own words narrow it to notes"
            assert scoped["inferred"]["applied"] == [{"filter": "kind", "value": "note", "words": "in my notes"}]
            kept = client.get("/v1/recall", params={"q": "when does the harbour close, in my notes", "infer": "true", "kind": "file"},
                              headers={"Authorization": "Bearer key-a"}).json()
            assert [item["episode_id"] for item in kept["items"]] == [2] and kept["inferred"]["applied"] == [], "the caller's kind stands"
            assert kept["inferred"]["withheld"] == [{"filter": "kind", "value": "note", "words": "in my notes", "because": "the caller set the kind"}]
            shaped = client.get("/v1/recall", params={"q": "what does #include say about the harbour #tides", "infer": "true"},
                                headers={"Authorization": "Bearer key-a"}).json()
            assert [item["episode_id"] for item in shaped["items"]] == [1], "the tag the space holds narrows; the one it does not is withheld"
            assert shaped["inferred"]["applied"] == [{"filter": "tag", "value": "tides", "words": "#tides"}]
            assert shaped["inferred"]["withheld"] == [{"filter": "tag", "value": "include", "words": "#include",
                                                       "because": "no memory in this space carries the tag"}]
        out = io.StringIO()
        code = await run(build_parser().parse_args(["--json", "--space", "s", "recall", "when does the harbour close, under docs/", "--infer"]),
                         engine, io.StringIO(""), out)
        import json

        payload = json.loads(out.getvalue())
        assert code == 0 and [item["episode_id"] for item in payload["items"]] == [2]
        assert payload["inferred"]["applied"][0]["filter"] == "source_prefix"
        out = io.StringIO()
        await run(build_parser().parse_args(["--space", "s", "recall", "when does the harbour close, in my notes", "--infer"]), engine, io.StringIO(""), out)
        assert "inferred: kind=note (in my notes)" in out.getvalue(), out.getvalue()
        out = io.StringIO()
        await run(build_parser().parse_args(["--space", "s", "recall", "the harbour #nothing", "--infer"]), engine, io.StringIO(""), out)
        assert "inferred: nothing applied\nwithheld: tag=nothing (#nothing): no memory in this space carries the tag" in out.getvalue(), out.getvalue()
    finally:
        await engine.close()
