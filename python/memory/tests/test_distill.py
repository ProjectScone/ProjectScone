"""Consolidation: episodes go through a (fake) model and come out as
dated facts, with the engine's supersession rules doing the rest."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.distill import EXTRACTION_PROMPT, DistillError, Distiller, Extracted, parse_triples
from scone_memory.llm import ChatError, FakeChat, OpenAICompatibleChat
from scone_memory.testing import Clock

SPACE = "default"
AUSTIN = json.dumps([{"subject": "Ana", "predicate": "lives_in", "object": "Austin", "confidence": 0.9}])
LISBON = json.dumps([{"subject": "Ana", "predicate": "lives_in", "object": "Lisbon", "confidence": 0.9}])


def grounded(subject, predicate, object, quote, statement_type="observation", confidence=0.9):
    return {
        "subject": subject,
        "predicate": predicate,
        "object": object,
        "confidence": confidence,
        "statement_type": statement_type,
        "quote": quote,
    }


@pytest.fixture
async def engine():
    clock = Clock("2025-01-01T00:00:00.000Z")
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=clock).open()


# -- distill_episode --------------------------------------------------------


async def test_facts_inherit_the_episode_date_and_name_their_source(engine):
    added = await engine.remember(SPACE, "Ana moved to Lisbon and started at Farfetch.", created_at="2024-03-02")
    chat = FakeChat(
        [
            json.dumps(
                [
                    {"subject": "Ana", "predicate": "lives_in", "object": "Lisbon", "confidence": 0.9},
                    {"subject": "Ana", "predicate": "works_at", "object": "Farfetch", "confidence": 0.8},
                ]
            )
        ]
    )

    outcome = await Distiller(engine, chat, accept_at=0.0, require_grounding=False).distill_episode(
        SPACE, added.episode_id
    )

    assert (len(outcome.added), outcome.closed, outcome.skipped, outcome.failed) == (2, 0, 0, False)
    facts = await engine.facts(SPACE)
    assert {(f.subject, f.predicate, f.object) for f in facts} == {
        ("ana", "lives_in", "Lisbon"),
        ("ana", "works_at", "Farfetch"),
    }
    # Dated to when it happened, not to the clock's 2025 "now".
    assert {f.valid_from for f in facts} == {"2024-03-02T00:00:00.000Z"}
    assert {f.source_episode_id for f in facts} == {added.episode_id}
    assert {f.confidence for f in facts} == {0.9, 0.8}
    assert chat.calls == [(EXTRACTION_PROMPT, "Ana moved to Lisbon and started at Farfetch.")]


async def test_a_restated_fact_is_skipped_not_duplicated(engine):
    added = await engine.remember(SPACE, "Ana still lives in Lisbon.", created_at="2024-06-01")
    await engine.assert_fact(SPACE, "ana", "lives_in", "Lisbon", valid_from="2024-03-02")

    outcome = await Distiller(
        engine, FakeChat([LISBON]), accept_at=0.0, require_grounding=False
    ).distill_episode(SPACE, added.episode_id)

    assert (outcome.added, outcome.closed, outcome.skipped) == ([], 0, 1)
    assert len(await engine.facts(SPACE, include_closed=True)) == 1


async def test_a_missing_episode_is_a_not_found_error(engine):
    from scone_memory import NotFound

    with pytest.raises(NotFound):
        await Distiller(engine, FakeChat(["[]"]), accept_at=0.0).distill_episode(SPACE, 999)


# -- supersession through distill_pending -------------------------------------


@pytest.mark.parametrize("remember_newer_first", [False, True])
async def test_pending_runs_oldest_first_so_the_newer_fact_supersedes(engine, remember_newer_first):
    older = ("Ana lives in Austin now.", "2023-01-01")
    newer = ("Ana moved to Lisbon.", "2024-03-02")
    for content, when in (newer, older) if remember_newer_first else (older, newer):
        await engine.remember(SPACE, content, created_at=when)
    chat = FakeChat([AUSTIN, LISBON])

    outcomes = await Distiller(engine, chat, accept_at=0.0, require_grounding=False).distill_pending(SPACE)

    assert [user for _, user in chat.calls] == ["Ana lives in Austin now.", "Ana moved to Lisbon."]
    assert [(len(o.added), o.closed) for o in outcomes] == [(1, 0), (1, 1)]
    active = await engine.facts(SPACE)
    assert [f.object for f in active] == ["Lisbon"]
    [austin] = [f for f in await engine.facts(SPACE, include_closed=True) if f.object == "Austin"]
    assert austin.status == "closed"
    assert austin.valid_until == "2024-03-02T00:00:00.000Z"
    assert austin.closed_reason == f"superseded by fact {active[0].fact_id}"


async def test_a_stale_episode_distilled_late_does_not_overwrite_the_fresher_fact(engine):
    fresh = await engine.remember(SPACE, "Ana moved to Lisbon.", created_at="2024-03-02")
    await Distiller(
        engine, FakeChat([LISBON]), accept_at=0.0, require_grounding=False
    ).distill_episode(SPACE, fresh.episode_id)
    stale = await engine.remember(SPACE, "Ana lives in Austin now.", created_at="2023-01-01")

    outcome = await Distiller(
        engine, FakeChat([AUSTIN]), accept_at=0.0, require_grounding=False
    ).distill_episode(SPACE, stale.episode_id)

    assert [f.object for f in await engine.facts(SPACE)] == ["Lisbon"]
    [austin] = outcome.added
    assert (austin.status, austin.valid_until) == ("closed", "2024-03-02T00:00:00.000Z")
    assert outcome.closed == 0


# -- failures ------------------------------------------------------------------


async def test_a_reply_that_is_not_json_is_a_loud_typed_error(engine):
    added = await engine.remember(SPACE, "nothing structured here", created_at="2024-01-01")

    with pytest.raises(DistillError) as raised:
        await Distiller(engine, FakeChat(["I could not find any facts, sorry."]), accept_at=0.0).distill_episode(SPACE, added.episode_id)

    assert "JSON array" in str(raised.value)
    assert await engine.facts(SPACE, include_closed=True) == []


async def test_pending_parks_an_episode_after_max_attempts_and_stops_calling_the_model(engine):
    added = await engine.remember(SPACE, "garbage in", created_at="2024-01-01")
    chat = FakeChat(["not json", "still not json", "nope", "[]"])
    distiller = Distiller(engine, chat, max_attempts=3, accept_at=0.0)

    for attempt in range(3):
        with pytest.raises(DistillError) as raised:
            await distiller.distill_pending(SPACE)
        assert raised.value.failed == 1
        assert [o.failed for o in raised.value.outcomes] == [True]
        assert len(chat.calls) == attempt + 1

    # Parked: reported as failed, the model is left alone, no error raised.
    outcomes = await distiller.distill_pending(SPACE)
    assert len(chat.calls) == 3
    assert [(o.episode_id, o.failed) for o in outcomes] == [(added.episode_id, True)]
    assert outcomes[0].error.startswith("parked: DistillError")
    assert list(distiller.parked(SPACE)) == [added.episode_id]

    # A fresh instance has no memory of the failures and asks again.
    retried = await Distiller(engine, chat, accept_at=0.0).distill_pending(SPACE)
    assert [(o.episode_id, o.failed) for o in retried] == [(added.episode_id, False)]
    assert len(chat.calls) == 4


async def test_a_transport_failure_counts_as_an_attempt_and_the_others_still_run(engine):
    await engine.remember(SPACE, "Ana lives in Austin now.", created_at="2023-01-01")
    await engine.remember(SPACE, "Ana moved to Lisbon.", created_at="2024-03-02")
    chat = FakeChat([ChatError("connection refused"), LISBON])

    with pytest.raises(DistillError) as raised:
        await Distiller(engine, chat, accept_at=0.0, require_grounding=False).distill_pending(SPACE)

    assert raised.value.failed == 1
    assert [o.failed for o in raised.value.outcomes] == [True, False]
    assert "connection refused" in str(raised.value)
    assert [f.object for f in await engine.facts(SPACE)] == ["Lisbon"]


# -- parse_triples ---------------------------------------------------------------


def test_parse_triples_reads_fenced_json():
    text = '```json\n[{"subject": "Ana", "predicate": "lives_in", "object": "Lisbon", "confidence": 0.9}]\n```'
    assert parse_triples(text) == [Extracted("Ana", "lives_in", "Lisbon", 0.9)]


def test_parse_triples_reads_json_surrounded_by_prose():
    text = (
        "Here are the facts [as requested]:\n"
        '[{"subject": "Ana", "predicate": "lives_in", "object": "Lisbon", "confidence": 1}]\n'
        "Let me know if you need more."
    )
    assert parse_triples(text) == [Extracted("Ana", "lives_in", "Lisbon", 1.0)]


def test_parse_triples_drops_malformed_entries_and_keeps_the_rest():
    text = json.dumps(
        [
            {"subject": "Ana", "predicate": "lives_in", "object": "Lisbon", "confidence": 0.9},
            {"subject": "Ana", "predicate": "lives_in"},
            {"subject": "", "predicate": "lives_in", "object": "Porto"},
            {"subject": 7, "predicate": "lives_in", "object": "Porto"},
            "not an object",
            None,
        ]
    )
    assert parse_triples(text) == [Extracted("Ana", "lives_in", "Lisbon", 0.9)]


def test_parse_triples_clamps_confidence_and_defaults_junk():
    text = json.dumps(
        [
            {"subject": "a", "predicate": "p", "object": "high", "confidence": 1.7},
            {"subject": "a", "predicate": "q", "object": "low", "confidence": -0.2},
            {"subject": "a", "predicate": "r", "object": "missing"},
            {"subject": "a", "predicate": "s", "object": "text", "confidence": "high"},
            {"subject": "a", "predicate": "t", "object": "bool", "confidence": True},
        ]
    )
    assert [t.confidence for t in parse_triples(text)] == [1.0, 0.0, 0.5, 0.5, 0.5]


def test_parse_triples_collapses_whitespace():
    text = json.dumps([{"subject": "  Ana\n Silva ", "predicate": " lives_in ", "object": "Lisbon,\tPortugal "}])
    assert parse_triples(text) == [Extracted("Ana Silva", "lives_in", "Lisbon, Portugal", 0.5)]


def test_parse_triples_preserves_optional_grounding_fields_without_changing_legacy_callers():
    text = json.dumps(
        [grounded("Ana", "lives_in", "Lisbon", "Ana lives in Lisbon.")]
    )

    assert parse_triples(text) == [
        Extracted(
            "Ana",
            "lives_in",
            "Lisbon",
            0.9,
            quote="Ana lives in Lisbon.",
            statement_type="observation",
        )
    ]


@pytest.mark.parametrize("text", ["", "no facts", '{"subject": "a"}', "[not json"])
def test_parse_triples_without_an_array_is_an_error(text):
    with pytest.raises(DistillError):
        parse_triples(text)


# -- empty replies and already-distilled episodes ------------------------------


async def test_an_empty_array_adds_nothing_and_is_not_an_error(engine):
    added = await engine.remember(SPACE, "just chatter", created_at="2024-01-01")

    outcome = await Distiller(engine, FakeChat(["[]"]), accept_at=0.0).distill_episode(SPACE, added.episode_id)

    assert (outcome.added, outcome.closed, outcome.skipped, outcome.failed) == ([], 0, 0, False)
    assert await engine.facts(SPACE, include_closed=True) == []


async def test_pending_remembers_an_episode_that_stated_no_facts(engine):
    await engine.remember(SPACE, "just chatter", created_at="2024-01-01")
    chat = FakeChat(["[]", "[]"])
    distiller = Distiller(engine, chat, accept_at=0.0)

    first = await distiller.distill_pending(SPACE)
    second = await distiller.distill_pending(SPACE)

    assert [o.failed for o in first] == [False]
    assert second == []
    assert len(chat.calls) == 1


async def test_pending_skips_episodes_that_already_have_facts(engine):
    done = await engine.remember(SPACE, "Ana lives in Austin now.", created_at="2023-01-01")
    await engine.assert_fact(SPACE, "ana", "lives_in", "Austin", valid_from="2023-01-01", source_episode_id=done.episode_id)
    todo = await engine.remember(SPACE, "Ana moved to Lisbon.", created_at="2024-03-02")
    chat = FakeChat([LISBON])

    outcomes = await Distiller(engine, chat, accept_at=0.0, require_grounding=False).distill_pending(SPACE)

    assert [o.episode_id for o in outcomes] == [todo.episode_id]
    assert [user for _, user in chat.calls] == ["Ana moved to Lisbon."]
    assert [f.object for f in await engine.facts(SPACE)] == ["Lisbon"]


async def test_pending_honours_the_limit_oldest_first(engine):
    for day in ("03", "01", "02"):
        await engine.remember(SPACE, f"note from day {day}", created_at=f"2024-01-{day}")
    chat = FakeChat(["[]", "[]"])

    outcomes = await Distiller(engine, chat, accept_at=0.0).distill_pending(SPACE, limit=2)

    assert len(outcomes) == 2
    assert [user for _, user in chat.calls] == ["note from day 01", "note from day 02"]


# -- distill_text -------------------------------------------------------------------


async def test_distill_text_dates_facts_without_an_episode(engine):
    facts = await Distiller(
        engine, FakeChat([LISBON]), accept_at=0.0, require_grounding=False
    ).distill_text(SPACE, "Ana moved to Lisbon.", created_at="2024-03-02")

    [fact] = facts
    assert (fact.subject, fact.object, fact.valid_from, fact.source_episode_id) == (
        "ana",
        "Lisbon",
        "2024-03-02T00:00:00.000Z",
        None,
    )


# -- the HTTP client ------------------------------------------------------------------


async def test_openai_compatible_chat_posts_messages_and_reads_the_reply():
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "[]"}}]})

    chat = OpenAICompatibleChat(
        "http://llm.local/v1/", "qwen3", api_key="k", think=False, transport=httpx.MockTransport(handle)
    )
    assert await chat.complete("sys", "hello") == "[]"

    [request] = seen
    assert str(request.url) == "http://llm.local/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k"
    body = json.loads(request.content)
    assert body["model"] == "qwen3"
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    assert body["temperature"] == 0.0
    assert body["think"] is False


async def test_openai_compatible_chat_leaves_think_out_unless_set():
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    chat = OpenAICompatibleChat("http://llm.local/v1", "gpt", transport=httpx.MockTransport(handle))
    await chat.complete("sys", "hello")

    body = json.loads(seen[0].content)
    assert "think" not in body
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(httpx.Response(500, json={"choices": [{"message": {"content": "[]"}}]}), id="status-500"),
        pytest.param(httpx.Response(200, json={"choices": []}), id="no-choices"),
        pytest.param(httpx.Response(200, json={"choices": [{"message": {"content": None}}]}), id="null-content"),
        pytest.param(httpx.Response(200, text="not json"), id="not-json"),
    ],
)
async def test_openai_compatible_chat_turns_bad_replies_into_chat_errors(response):
    chat = OpenAICompatibleChat("http://llm.local/v1", "gpt", transport=httpx.MockTransport(lambda _: response))
    with pytest.raises(ChatError):
        await chat.complete("sys", "hello")


async def test_openai_compatible_chat_turns_transport_failures_into_chat_errors():
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    chat = OpenAICompatibleChat("http://llm.local/v1", "gpt", transport=httpx.MockTransport(handle))
    with pytest.raises(ChatError, match="unreachable"):
        await chat.complete("sys", "hello")



async def new_engine():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock("2025-01-01T00:00:00.000Z")).open()


async def test_extractions_are_proposed_for_review_by_default():
    """A grounded model reading is never presented as established."""
    engine = await new_engine()
    source = "Ana moved to Lisbon in March 2024."
    added = await engine.remember(SPACE, source, created_at="2024-03-02")
    reply = json.dumps([grounded("Ana", "moved_to", "Lisbon", source)])
    outcome = await Distiller(engine, FakeChat([reply])).distill_episode(SPACE, added.episode_id)
    [fact] = outcome.added
    assert (fact.status, fact.origin) == ("proposed", "extracted")
    assert await engine.facts(SPACE) == []
    assert [f.fact_id for f in await engine.facts(SPACE, status="proposed")] == [fact.fact_id]

    accepted_engine = await new_engine()
    added = await accepted_engine.remember(SPACE, "Ana moved to Lisbon in March 2024.", created_at="2024-03-02")
    outcome = await Distiller(
        accepted_engine,
        FakeChat([LISBON]),
        accept_at=0.8,
        require_grounding=False,
    ).distill_episode(SPACE, added.episode_id)
    [fact] = outcome.added
    assert (fact.status, fact.origin) == ("active", "extracted")


# -- source-grounded extraction -----------------------------------------------


async def test_grounded_observation_is_proposed_even_when_accept_threshold_would_activate_it(engine):
    source = "Ana lives in Lisbon."
    added = await engine.remember(SPACE, source, created_at="2024-03-02")
    reply = json.dumps([grounded("Ana", "lives_in", "Lisbon", source, confidence=1.0)])

    outcome = await Distiller(engine, FakeChat([reply]), accept_at=0.0).distill_episode(SPACE, added.episode_id)

    [proposal] = outcome.added
    assert (proposal.status, proposal.origin, proposal.source_episode_id) == (
        "proposed",
        "extracted",
        added.episode_id,
    )
    assert (proposal.quote, proposal.grounded) == (source, True)
    assert outcome.rejected == []
    assert await engine.facts(SPACE) == []


@pytest.mark.parametrize(
    ("source", "candidate", "reason"),
    [
        (
            "Ana lives in Lisbon.",
            grounded("Ana", "lives_in", "Lisbon", "Ana lives in Porto."),
            "quote_not_in_source",
        ),
        (
            "Ana lives in Lisbon.",
            {"subject": "Ana", "predicate": "lives_in", "object": "Lisbon", "confidence": 0.9},
            "missing_quote",
        ),
        (
            "Ana moved recently.",
            grounded("Ana", "lives_in", "Lisbon", "Ana moved recently."),
            "object_not_in_quote",
        ),
        (
            "She works at Farfetch.",
            grounded("Ana", "works_at", "Farfetch", "She works at Farfetch."),
            "subject_not_in_quote",
        ),
    ],
)
async def test_grounding_rejects_missing_or_non_supporting_literal_quotes(engine, source, candidate, reason):
    added = await engine.remember(SPACE, source, created_at="2024-03-02")

    outcome = await Distiller(engine, FakeChat([json.dumps([candidate])])).distill_episode(SPACE, added.episode_id)

    assert outcome.added == []
    assert [item.reason for item in outcome.rejected] == [reason]
    assert await engine.facts(SPACE, include_closed=True) == []


@pytest.mark.parametrize("predicate", ["dislikes", "does_not_like"])
async def test_literal_subject_and_object_do_not_make_an_opposite_predicate_supported(engine, predicate):
    source = "Ana likes Lisbon."
    added = await engine.remember(SPACE, source, created_at="2024-03-02")
    candidate = grounded("Ana", predicate, "Lisbon", source)

    outcome = await Distiller(engine, FakeChat([json.dumps([candidate])])).distill_episode(
        SPACE, added.episode_id
    )

    assert outcome.added == []
    assert [item.reason for item in outcome.rejected] == ["predicate_not_in_quote"]


@pytest.mark.parametrize("extra", [" ", "\n"])
async def test_grounding_does_not_trim_model_added_whitespace_into_a_literal_quote(engine, extra):
    source = "Zoë uses\nRust\tfor Scone."
    added = await engine.remember(SPACE, source, created_at="2024-03-02")
    candidate = grounded("Zoë", "uses", "Rust for Scone", extra + source)

    outcome = await Distiller(engine, FakeChat([json.dumps([candidate])])).distill_episode(
        SPACE, added.episode_id
    )

    assert outcome.added == []
    assert [item.reason for item in outcome.rejected] == ["quote_not_in_source"]


async def test_grounding_accepts_an_exact_unicode_quote_with_internal_source_whitespace(engine):
    source = "Zoë uses\nRust\tfor Scone."
    added = await engine.remember(SPACE, source, created_at="2024-03-02")
    candidate = grounded("Zoë", "uses", "Rust for Scone", source)

    outcome = await Distiller(engine, FakeChat([json.dumps([candidate])])).distill_episode(
        SPACE, added.episode_id
    )

    [proposal] = outcome.added
    assert proposal.object == "Rust for Scone"
    assert outcome.rejected == []


async def test_grounding_withholds_a_quote_too_long_for_fact_persistence(engine):
    source = "Ana " + ("x" * 1990) + " Lisbon"
    assert len(source) > 2000
    added = await engine.remember(SPACE, source, created_at="2024-03-02")
    candidate = grounded("Ana", "lives_in", "Lisbon", source)

    outcome = await Distiller(engine, FakeChat([json.dumps([candidate])])).distill_episode(
        SPACE, added.episode_id
    )

    assert outcome.added == []
    assert [item.reason for item in outcome.rejected] == ["quote_too_long"]


@pytest.mark.parametrize(
    ("source", "candidate", "reason"),
    [
        (
            "If the hook is installed, it fires on prompt events.",
            grounded("hook", "fires_on", "prompt events", "the hook is installed, it fires on prompt events"),
            "context_not_asserted",
        ),
        (
            "The hook might fire on prompt events.",
            grounded("hook", "fires_on", "prompt events", "The hook might fire on prompt events."),
            "context_not_asserted",
        ),
        (
            "Verify the hook fires on prompt events.",
            grounded("hook", "fires_on", "prompt events", "the hook fires on prompt events", "instruction"),
            "not_an_observation",
        ),
        (
            "Does the hook fire on prompt events?",
            grounded("hook", "fires_on", "prompt events", "the hook fire on prompt events", "question"),
            "not_an_observation",
        ),
        (
            "The hook does not fire on prompt events.",
            grounded("hook", "fires_on", "prompt events", "The hook does not fire on prompt events."),
            "context_not_asserted",
        ),
    ],
)
async def test_grounding_preserves_non_asserted_context_instead_of_turning_it_positive(
    engine, source, candidate, reason
):
    added = await engine.remember(SPACE, source, created_at="2024-03-02")

    outcome = await Distiller(engine, FakeChat([json.dumps([candidate])])).distill_episode(SPACE, added.episode_id)

    assert outcome.added == []
    assert [item.reason for item in outcome.rejected] == [reason]


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        (
            "Ana doesn't live in Lisbon.",
            grounded("Ana", "lives_in", "Lisbon", "Ana doesn't live in Lisbon."),
        ),
        (
            "Ana doesn’t live in Lisbon.",
            grounded("Ana", "lives_in", "Lisbon", "Ana doesn’t live in Lisbon."),
        ),
        (
            "Ana isn't living in Lisbon.",
            grounded("Ana", "living_in", "Lisbon", "Ana isn't living in Lisbon."),
        ),
        (
            "Ana isn’t living in Lisbon.",
            grounded("Ana", "living_in", "Lisbon", "Ana isn’t living in Lisbon."),
        ),
        (
            "Ana is unlikely to live in Lisbon.",
            grounded("Ana", "lives_in", "Lisbon", "Ana is unlikely to live in Lisbon."),
        ),
        (
            "I doubt Ana lives in Lisbon.",
            grounded("Ana", "lives_in", "Lisbon", "I doubt Ana lives in Lisbon."),
        ),
        (
            "Scone plans to use Rust.",
            grounded("Scone", "uses", "Rust", "Scone plans to use Rust."),
        ),
        (
            "Scone intends to use Rust.",
            grounded("Scone", "uses", "Rust", "Scone intends to use Rust."),
        ),
        (
            "Scone won't use Rust.",
            grounded("Scone", "uses", "Rust", "Scone won't use Rust."),
        ),
        (
            "Scone won’t use Rust.",
            grounded("Scone", "uses", "Rust", "Scone won’t use Rust."),
        ),
        (
            "Scone cannot use Rust.",
            grounded("Scone", "uses", "Rust", "Scone cannot use Rust."),
        ),
    ],
)
async def test_mislabeled_contractions_doubt_likelihood_and_plans_are_not_observations(
    engine, source, candidate
):
    added = await engine.remember(SPACE, source, created_at="2024-03-02")

    outcome = await Distiller(engine, FakeChat([json.dumps([candidate])])).distill_episode(
        SPACE, added.episode_id
    )

    assert outcome.added == []
    assert [item.reason for item in outcome.rejected] == ["context_not_asserted"]


async def test_strict_distillation_accounts_for_every_malformed_array_candidate(engine):
    source = "Ana lives in Lisbon."
    await engine.remember(SPACE, source, created_at="2024-03-02")
    malformed = {"subject": "Ana", "predicate": "lives_in", "confidence": 0.9}
    reply = json.dumps([malformed, "junk", None])
    chat = FakeChat([reply])
    distiller = Distiller(engine, chat)

    [outcome] = await distiller.distill_pending(SPACE)

    assert outcome.added == []
    assert [item.reason for item in outcome.rejected] == ["malformed_candidate"] * 3
    assert outcome.rejected[0].extraction is None
    assert outcome.rejected[0].raw == malformed
    assert await engine.facts(SPACE, include_closed=True) == []
    assert await distiller.distill_pending(SPACE) == []
    assert len(chat.calls) == 1


async def test_repeated_quote_cannot_bypass_a_non_asserted_occurrence(engine):
    quote = "the hook fires on prompt events"
    source = f"Observed: {quote}. If configured, {quote}."
    added = await engine.remember(SPACE, source, created_at="2024-03-02")
    candidate = grounded("hook", "fires_on", "prompt events", quote)

    outcome = await Distiller(engine, FakeChat([json.dumps([candidate])])).distill_episode(
        SPACE, added.episode_id
    )

    assert outcome.added == []
    assert [item.reason for item in outcome.rejected] == ["context_not_asserted"]


async def test_an_asserted_clause_is_retained_beside_a_separate_hypothetical_clause(engine):
    source = "The hook fires on prompt events; if the network fails, it may miss responses."
    added = await engine.remember(SPACE, source, created_at="2024-03-02")
    candidate = grounded("hook", "fires_on", "prompt events", "The hook fires on prompt events")

    outcome = await Distiller(engine, FakeChat([json.dumps([candidate])])).distill_episode(
        SPACE, added.episode_id
    )

    [proposal] = outcome.added
    assert proposal.object == "prompt events"
    assert outcome.rejected == []


async def test_known_capture_verification_text_produces_no_facts_from_literal_but_non_entailed_quotes(engine):
    fixture = json.loads((Path(__file__).parent / "fixtures" / "grounding_failure.json").read_text())
    added = await engine.remember(SPACE, fixture["source"], created_at="2026-09-05")

    outcome = await Distiller(engine, FakeChat([json.dumps(fixture["reply"])])).distill_episode(
        SPACE, added.episode_id
    )

    assert outcome.added == []
    assert [item.reason for item in outcome.rejected] == [
        "context_not_asserted",
        "context_not_asserted",
        "context_not_asserted",
    ]
    assert await engine.facts(SPACE, include_closed=True) == []


async def test_same_source_conflicts_are_all_rejected_without_ordering_them_as_updates(engine):
    source = "Ana lives in Austin. Ana lives in Lisbon."
    added = await engine.remember(SPACE, source, created_at="2024-03-02")
    reply = json.dumps(
        [
            grounded("Ana", "lives_in", "Austin", "Ana lives in Austin."),
            grounded("Ana", "lives_in", "Lisbon", "Ana lives in Lisbon."),
        ]
    )

    outcome = await Distiller(engine, FakeChat([reply])).distill_episode(SPACE, added.episode_id)

    assert outcome.added == []
    assert [item.reason for item in outcome.rejected] == ["same_source_conflict", "same_source_conflict"]
    assert await engine.facts(SPACE, include_closed=True) == []


async def test_grounded_correction_is_a_proposal_and_does_not_rewrite_approved_history(engine):
    approved = await engine.assert_fact(SPACE, "Ana", "lives_in", "Austin", valid_from="2024-01-01")
    source = "Ana lives in Lisbon."
    added = await engine.remember(SPACE, source, created_at="2024-03-02")
    reply = json.dumps([grounded("Ana", "lives_in", "Lisbon", source)])

    outcome = await Distiller(engine, FakeChat([reply]), accept_at=0.0).distill_episode(SPACE, added.episode_id)

    [proposal] = outcome.added
    assert proposal.status == "proposed"
    assert [(f.fact_id, f.object, f.status, f.valid_until) for f in await engine.facts(SPACE)] == [
        (approved.fact_id, "Austin", "active", None)
    ]


async def test_grounded_distill_text_refuses_a_claim_when_its_quote_cannot_be_persisted(engine):
    source = "Ana lives in Lisbon."
    reply = json.dumps([grounded("Ana", "lives_in", "Lisbon", source)])

    with pytest.raises(DistillError, match="stored episode"):
        await Distiller(engine, FakeChat([reply])).distill_text(
            SPACE, source, created_at="2024-03-02"
        )

    assert await engine.facts(SPACE, include_closed=True) == []


async def test_legacy_unquoted_replies_require_an_explicit_compatibility_opt_out(engine):
    added = await engine.remember(SPACE, "Ana lives in Lisbon.", created_at="2024-03-02")

    outcome = await Distiller(
        engine, FakeChat([LISBON]), accept_at=0.0, require_grounding=False
    ).distill_episode(SPACE, added.episode_id)

    [fact] = outcome.added
    assert (fact.object, fact.status) == ("Lisbon", "active")
