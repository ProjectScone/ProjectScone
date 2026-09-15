"""Dynamic-schema extraction: a model proposes triples and the types they use, and a person decides.

A model reads one chunk with a suggested vocabulary of entity kinds and
predicates, and proposes (subject, predicate, object) triples, each with
the kinds of its two ends and a span quoted from the chunk. A triple
without a verbatim quote is rejected and counted, and so is one the
source-grounding gate of model extraction refuses. What passes is stored
as a proposal, never as a held claim. A kind or predicate outside the
suggested vocabulary is proposed vocabulary, recorded with the quotes of
the proposals that use it; with new types switched off it is a rejection.
Calls, triples per chunk and new types are bounded, and each bound says
when it cut.
"""

from __future__ import annotations

import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex, MemoryEngine
from scone_memory.backends import SqliteDocumentStore
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import distill
from scone_memory.ingestion import dynamic_schema as module
from scone_memory.ingestion.distill import Distiller
from scone_memory.ingestion.dynamic_schema import extract_dynamic_schema, schema_term
from scone_memory.providers.llm import ChatError, FakeChat
from scone_memory.runtime.cli import build_parser, run

SPACE = "s"
LEDGER = "Scone stores its ledger in SQLite. The worker uses Ollama for extraction. Scone does not use Redis."
STORES = "Scone stores its ledger in SQLite."
USES = "The worker uses Ollama for extraction."
NEGATED = "Scone does not use Redis."
HARBOUR = "The harbour at Vellmar closes to sailing boats every November."
ORCHARD = "The orchard on the ridge grows only Bramley apples."
KINDS = ("project", "product")
PREDICATES = ("uses",)


def triple(subject: str, subject_kind: str, predicate: str, object: str, object_kind: str, quote: str | None,
           statement_type: str = "observation") -> dict[str, object]:
    entry: dict[str, object] = {"subject": subject, "subject_kind": subject_kind, "predicate": predicate,
                                "object": object, "object_kind": object_kind, "statement_type": statement_type}
    if quote is not None:
        entry["quote"] = quote
    return entry


def reply(*triples: object) -> str:
    return json.dumps({"triples": list(triples)})


GOOD_USES = triple("worker", "product", "uses", "Ollama", "product", USES)
GOOD_STORES = triple("Scone", "project", "stores", "ledger", "database", STORES)


@pytest.fixture(params=["memory", "sqlite"])
async def documents(request, tmp_path):
    store = InMemoryDocumentStore() if request.param == "memory" else SqliteDocumentStore(str(tmp_path / "s.db"))
    yield store
    close = getattr(store, "close", None)
    if close is not None:
        result = close()
        if hasattr(result, "__await__"):
            await result


async def opened(documents=None) -> MemoryEngine:
    return await MemoryEngine(documents or InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              chunk_target=4000, events=InMemoryEventLog()).open()


async def test_a_quoted_triple_is_stored_as_a_proposal_with_its_quote_and_holds_nothing_until_approved(documents):
    engine = await opened(documents)
    added = await engine.remember(SPACE, LEDGER, created_at="2026-03-02")
    chat = FakeChat([reply(GOOD_USES)])

    report = await extract_dynamic_schema(engine, SPACE, chat, entity_kinds=KINDS, predicates=PREDICATES,
                                          model_name="fake-3b")

    assert (report.chunks_total, report.chunks_asked, report.model_calls, report.model) == (1, 1, 1, "fake-3b")
    assert [p.fact_id for p in report.proposed] and report.rejected_reasons == {} and report.quotes_settled == 0
    fact = await engine.documents.get_fact(SPACE, report.proposed[0].fact_id)
    assert (fact.subject, fact.predicate, fact.object) == ("worker", "uses", "Ollama")
    assert (fact.status, fact.origin, fact.quote, fact.source_episode_id) == ("proposed", "extracted", USES,
                                                                              added.episode_id)
    assert fact.valid_from == "2026-03-02T00:00:00.000Z", "dated to the episode, not to when it was read"
    assert await engine.facts(SPACE) == [], "a proposal answers nothing"
    system, user = chat.calls[0]
    assert system == module.SYSTEM and LEDGER in user
    await engine.approve(SPACE, fact.fact_id)
    assert [(f.subject, f.object) for f in await engine.facts(SPACE)] == [("worker", "Ollama")]


async def test_a_triple_without_a_verbatim_quote_is_rejected_and_counted():
    engine = await opened()
    await engine.remember(SPACE, LEDGER)
    chat = FakeChat([reply(GOOD_USES,
                           triple("worker", "product", "uses", "Ollama", "product", None),
                           triple("worker", "product", "uses", "Ollama", "product", "The worker uses Ollama daily."),
                           "not a triple")])

    report = await extract_dynamic_schema(engine, SPACE, chat, entity_kinds=KINDS, predicates=PREDICATES)

    assert report.rejected_reasons == {"missing_quote": 1, "quote_not_in_source": 1, "malformed": 1}
    assert (report.triples_read, report.unquoted, len(report.proposed)) == (4, 2, 1)
    assert report.quoted_share == pytest.approx(1 / 3), "of the three whole triples, one quoted the chunk"
    assert len(await engine.documents.list_facts(SPACE, include_closed=True)) == 1


async def test_a_quote_that_changes_only_whitespace_is_stored_as_the_chunk_writes_it(documents):
    # A model copying a hard-wrapped line joins it with a space. The words are
    # the chunk's, so the chunk's own span is the quote; a change of case is not.
    wrapped = "Scone stores its ledger\nin SQLite. The worker uses Ollama for extraction."
    engine = await opened(documents)
    await engine.remember(SPACE, wrapped)
    chat = FakeChat([reply(
        {**GOOD_STORES, "quote": "Scone stores its ledger in SQLite."},
        {**GOOD_USES, "quote": "  The worker  uses Ollama for extraction. "},
        {**GOOD_STORES, "object": "SQLite", "quote": "scone stores its ledger in sqlite."},
    )])

    report = await extract_dynamic_schema(engine, SPACE, chat)

    assert (report.quotes_settled, report.rejected_reasons, report.unquoted) == (2, {"quote_not_in_source": 1}, 1)
    assert [p.quote for p in report.proposed] == ["Scone stores its ledger\nin SQLite.", USES]
    stored = await engine.documents.list_facts(SPACE, include_closed=True)
    assert sorted(f.quote for f in stored) == sorted(["Scone stores its ledger\nin SQLite.", USES])
    assert report.record()["quotes_settled"] == 2 and "2 quote(s) settled to the chunk's whitespace" in report.text()


async def test_a_settled_quote_is_refused_when_its_words_stand_hedged_elsewhere_in_the_chunk():
    # Settling takes the first span of the quote's words; another span of
    # them, wrapped at another word in a hedged clause, is read too.
    wrapped = "The worker\nuses Ollama today. We may decide that the worker uses\nOllama later."
    engine = await opened_with(wrapped)

    report = await extract_dynamic_schema(engine, SPACE, FakeChat([reply({**GOOD_USES, "quote": "worker uses Ollama"})]))

    assert (report.quotes_settled, report.rejected_reasons, report.proposed) == (1, {"context_not_asserted": 1}, ())


async def test_a_quote_hedged_in_another_chunk_of_its_episode_is_refused_as_the_distiller_refuses_it():
    # The proposal rests on the episode, so every place in the episode that
    # holds the quote's words is read, not only the chunk the model saw.
    filler = " ".join(f"Sentence number {i} talks about harbours and orchards on the ridge." for i in range(12))
    text = (filler + " The worker uses Ollama for extraction.\n\n"
            + filler.replace("harbours", "boats") + " We may decide that the worker uses Ollama for extraction.")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=300,
                                events=InMemoryEventLog()).open()
    added = await engine.remember(SPACE, text)
    chunks = await engine.documents.chunks_of(SPACE, added.episode_id)
    candidate = {**GOOD_USES, "quote": "worker uses Ollama"}
    asserted = [chunk for chunk in chunks if "The worker uses Ollama" in chunk.text]
    assert len(chunks) > 1 and len(asserted) == 1 and "may decide" not in asserted[0].text

    report = await extract_dynamic_schema(engine, SPACE, FakeChat(
        [reply(candidate) if chunk in asserted else reply() for chunk in chunks]))

    assert (report.rejected_reasons, report.proposed) == ({"context_not_asserted": 1}, ())
    distilled = await Distiller(engine, FakeChat([json.dumps([candidate])])).distill_episode(SPACE, added.episode_id)
    assert [item.reason for item in distilled.rejected] == ["context_not_asserted"]


async def test_a_quote_too_long_to_store_is_counted_as_unquoted(monkeypatch):
    engine = await opened()
    await engine.remember(SPACE, LEDGER)
    monkeypatch.setattr(distill, "MAX_QUOTE", 10)

    report = await extract_dynamic_schema(engine, SPACE, FakeChat([reply(GOOD_USES)]), entity_kinds=KINDS,
                                          predicates=PREDICATES)

    assert (report.rejected_reasons, report.unquoted, report.proposed) == ({"quote_too_long": 1}, 1, ())


async def test_the_grounding_gate_of_model_extraction_applies_to_every_triple():
    engine = await opened()
    await engine.remember(SPACE, LEDGER)
    chat = FakeChat([reply(
        triple("Scone", "project", "uses", "Redis", "product", NEGATED),
        triple("Scone", "project", "uses", "Ollama", "product", USES),
        triple("worker", "product", "uses", "Ollama", "product", USES, statement_type="instruction"),
        triple("worker", "product", "runs", "Ollama", "product", USES),
        triple("worker", "product", "uses", "extraction", "concept", "The worker uses Ollama"),
    )])

    report = await extract_dynamic_schema(engine, SPACE, chat, entity_kinds=KINDS, predicates=PREDICATES)

    assert report.rejected_reasons == {"context_not_asserted": 1, "subject_not_in_quote": 1, "not_an_observation": 1,
                                       "predicate_not_in_quote": 1, "object_not_in_quote": 1}
    assert report.unquoted == 0 and report.quoted_share == 1.0, "each of these quoted the chunk verbatim"
    assert report.proposed == () and await engine.documents.list_facts(SPACE, include_closed=True) == []


async def test_a_triple_whose_two_ends_are_one_name_is_rejected_and_counted():
    # "Relation quotes participate in the coverage audit" came back from a
    # small model as (coverage audit, participate_in, coverage audit).
    engine = await opened_with(LEDGER)
    chat = FakeChat([reply({**GOOD_STORES, "subject": "SQLite", "object": "sqlite"}, GOOD_STORES,
                           {**GOOD_STORES, "subject": "Redis", "object": "Redis"})])

    report = await extract_dynamic_schema(engine, SPACE, chat)

    assert report.rejected_reasons == {"self_reference": 1, "subject_not_in_quote": 1}, \
        "an ungrounded triple is counted by what the gate refused it for"
    assert [p.object for p in report.proposed] == ["ledger"]


async def test_an_entry_that_is_not_a_whole_triple_is_malformed_and_counted():
    engine = await opened()
    await engine.remember(SPACE, LEDGER)
    no_kind = dict(GOOD_USES)
    del no_kind["object_kind"]
    chat = FakeChat([reply("worker uses Ollama", no_kind, {**GOOD_USES, "predicate": " -- "},
                           {**GOOD_USES, "subject": 7}, {**GOOD_USES, "subject_kind": "x" * 65},
                           {**GOOD_USES, "object": "  "}, {**GOOD_USES, "predicate": 7}, GOOD_USES)])

    report = await extract_dynamic_schema(engine, SPACE, chat, entity_kinds=KINDS, predicates=PREDICATES)

    assert (report.rejected_reasons, report.triples_read, len(report.proposed)) == ({"malformed": 7}, 8, 1)
    assert report.quoted_share == 1.0, "the quoted share is of whole triples"


async def test_kinds_and_predicates_are_compared_and_stored_as_snake_case_terms():
    assert [schema_term(t) for t in ("Depends On", "USED-BY", "  part_of ", "Software Component")] == \
        ["depends_on", "used_by", "part_of", "software_component"]
    assert schema_term(" -- ") is None and schema_term("x" * 65) is None and schema_term("x" * 64) == "x" * 64
    assert schema_term(None) is None and schema_term(7) is None
    engine = await opened()
    await engine.remember(SPACE, LEDGER)
    chat = FakeChat([reply({**GOOD_USES, "predicate": "USES", "subject_kind": "Product", "object_kind": "PRODUCT"})])

    report = await extract_dynamic_schema(engine, SPACE, chat, entity_kinds=("Project", "product", "PRODUCT"),
                                          predicates=("Uses",))

    assert (report.new_predicates, report.new_kinds, report.proposed_outside_schema) == ((), (), 0)
    assert report.proposed[0].predicate == "uses" and report.proposed[0].subject_kind == "product"
    assert (report.entity_kinds, report.predicates) == (("project", "product"), ("uses",))
    assert (await engine.documents.get_fact(SPACE, report.proposed[0].fact_id)).predicate == "uses"


async def test_a_new_predicate_and_kind_are_proposed_vocabulary_with_the_quotes_that_use_them(monkeypatch):
    engine = await opened()
    await engine.remember(SPACE, LEDGER)
    other = "Scone stores its ledger in SQLite"
    monkeypatch.setattr(module, "MAX_EXAMPLES", 1)
    chat = FakeChat([reply(GOOD_USES, GOOD_STORES, {**GOOD_STORES, "object": "SQLite", "quote": other})])

    report = await extract_dynamic_schema(engine, SPACE, chat, entity_kinds=KINDS, predicates=PREDICATES)

    assert (len(report.proposed), report.proposed_outside_schema) == (3, 2), "a strict fixed schema misses two"
    [stores] = report.new_predicates
    assert (stores.term, stores.uses, stores.examples_cut) == ("stores", 2, 1)
    [example] = stores.examples
    assert (example.quote, example.fact_id) == (STORES, report.proposed[1].fact_id)
    assert example.chunk_id == report.proposed[1].chunk_id
    [database] = report.new_kinds
    assert (database.term, database.uses, database.examples_cut, database.examples[0].quote) == ("database", 2, 1, STORES)
    assert report.record()["new_predicates"][0]["examples"][0]["quote"] == STORES


async def test_with_new_types_off_a_triple_outside_the_vocabulary_is_rejected_and_counted():
    engine = await opened()
    await engine.remember(SPACE, LEDGER)
    # The kind-only case names another object: the same triple again would be
    # on record already, and restated whatever its kinds.
    chat = FakeChat([reply(GOOD_USES, GOOD_STORES, {**GOOD_USES, "object": "extraction", "object_kind": "service"})])

    report = await extract_dynamic_schema(engine, SPACE, chat, entity_kinds=KINDS, predicates=PREDICATES,
                                          allow_new_types=False)

    assert (report.rejected_reasons, report.restated) == ({"new_type_not_allowed": 2}, 0)
    assert (len(report.proposed), report.new_predicates, report.new_kinds) == (1, (), ())
    assert "not allowed" in chat.calls[0][1] and "uses" in chat.calls[0][1] and "project, product" in chat.calls[0][1]
    opened_chat = FakeChat([reply(GOOD_USES)])
    await extract_dynamic_schema(await opened_with(LEDGER), SPACE, opened_chat, entity_kinds=KINDS,
                                 predicates=PREDICATES)
    assert "not allowed" not in opened_chat.calls[0][1]


async def opened_with(text: str) -> MemoryEngine:
    engine = await opened()
    await engine.remember(SPACE, text)
    return engine


async def test_an_empty_suggested_vocabulary_tells_the_model_to_name_its_own_types():
    chat = FakeChat([reply(GOOD_STORES, GOOD_USES)])
    report = await extract_dynamic_schema(await opened_with(LEDGER), SPACE, chat)
    assert "Suggested entity kinds: none suggested" in chat.calls[0][1]
    assert "Suggested predicates: none suggested" in chat.calls[0][1]
    assert [t.term for t in report.new_predicates] == ["stores", "uses"]
    assert [(t.term, t.uses) for t in report.new_kinds] == [("project", 1), ("database", 1), ("product", 1)], \
        "a kind at both ends of one proposal is one use"


async def test_new_types_past_the_budget_are_cut_and_counted_and_a_term_already_admitted_costs_nothing():
    engine = await opened_with(LEDGER)
    chat = FakeChat([reply(
        GOOD_STORES,
        {**GOOD_STORES, "object": "SQLite", "quote": "Scone stores its ledger in SQLite"},
        triple("worker", "product", "uses for", "extraction", "product", USES),
        {**GOOD_USES, "object_kind": "service"},
    )])

    report = await extract_dynamic_schema(engine, SPACE, chat, entity_kinds=KINDS, predicates=PREDICATES,
                                          max_new_predicates=1, max_new_kinds=1)

    assert report.rejected_reasons == {"new_type_cut": 2}
    assert len(report.proposed) == 2, "the second 'stores' triple uses a predicate and kind already admitted"
    assert ([t.term for t in report.new_predicates], [t.term for t in report.new_kinds]) == (["stores"], ["database"])
    assert report.cut_by == ("new_types",)


async def test_triples_past_the_per_chunk_limit_are_dropped_and_counted():
    engine = await opened_with(LEDGER)
    report = await extract_dynamic_schema(engine, SPACE, FakeChat([reply(GOOD_USES, GOOD_STORES, GOOD_USES)]),
                                          max_triples_per_chunk=1)
    assert (report.triples_read, report.dropped_extra, len(report.proposed)) == (1, 2, 1)
    assert report.cut_by == ("triples_per_chunk",)


async def test_calls_are_bounded_and_the_report_names_where_to_resume(documents):
    engine = await opened(documents)
    for text in (LEDGER, HARBOUR, ORCHARD):
        await engine.remember(SPACE, text)
    chat = FakeChat(["[]", "[]"])

    first = await extract_dynamic_schema(engine, SPACE, chat, max_calls=2)

    assert (first.chunks_total, first.chunks_asked, first.model_calls, first.chunks_cut) == (3, 2, 2, 1)
    assert first.cut_by == ("calls",) and "resume after chunk" in first.text()
    assert first.quoted_share is None, "no triple was read, so there is no share to state"
    assert LEDGER in chat.calls[0][1] and HARBOUR in chat.calls[1][1]
    chat.replies.append("[]")
    rest = await extract_dynamic_schema(engine, SPACE, chat, after_chunk=first.resume_after)
    assert (rest.chunks_total, rest.chunks_asked, rest.chunks_cut, rest.resume_after, rest.cut_by) == (1, 1, 0, None, ())
    assert ORCHARD in chat.calls[2][1]


async def test_a_chunk_too_long_to_show_is_not_sent_and_is_counted(monkeypatch):
    engine = await opened_with(LEDGER)
    await engine.remember(SPACE, "é" * 10)
    await engine.remember(SPACE, "é" * 11)
    monkeypatch.setattr(module, "MAX_CHUNK_BYTES", 20)
    chat = FakeChat(["[]", "[]"])

    report = await extract_dynamic_schema(engine, SPACE, chat, max_calls=2)

    assert (report.skipped_long, report.chunks_asked, report.chunks_cut, report.model_calls) == (2, 1, 0, 1), \
        "twenty bytes is shown; twenty-two bytes in eleven characters is not"
    assert "é" * 10 in chat.calls[0][1] and report.cut_by == ("chunk_bytes",)


async def test_a_failed_call_or_an_unreadable_reply_costs_that_chunk_only():
    engine = await opened_with(HARBOUR)
    await engine.remember(SPACE, ORCHARD)
    await engine.remember(SPACE, LEDGER)
    chat = FakeChat([ChatError("timed out"), "I cannot help with that.", reply(GOOD_USES)])

    report = await extract_dynamic_schema(engine, SPACE, chat)

    assert (report.calls_failed, report.replies_unparsed, report.model_calls, len(report.proposed)) == (1, 1, 3, 1)


@pytest.mark.parametrize("text", [json.dumps([GOOD_USES]), "Here you are:\n" + json.dumps([GOOD_USES]),
                                  reply(GOOD_USES)], ids=["array", "array-in-prose", "object"])
async def test_a_bare_array_and_an_object_holding_triples_both_read(text):
    report = await extract_dynamic_schema(await opened_with(LEDGER), SPACE, FakeChat([text]))
    assert (len(report.proposed), report.replies_unparsed) == (1, 0)


@pytest.mark.parametrize("text", [json.dumps({"facts": [GOOD_USES]}), json.dumps({"triples": "none"})],
                         ids=["no-triples", "triples-not-a-list"])
async def test_an_object_without_a_triples_list_is_unreadable(text):
    report = await extract_dynamic_schema(await opened_with(LEDGER), SPACE, FakeChat([text]))
    assert (report.replies_unparsed, report.triples_read, report.proposed) == (1, 0, ())


async def test_the_same_reading_again_is_restated_and_a_declined_proposal_does_not_come_back(documents):
    engine = await opened(documents)
    await engine.remember(SPACE, LEDGER)
    first = await extract_dynamic_schema(engine, SPACE, FakeChat([reply(GOOD_USES, GOOD_STORES)]))
    assert len(first.proposed) == 2
    await engine.decline(SPACE, first.proposed[1].fact_id, "not a fact about the ledger")

    again = await extract_dynamic_schema(engine, SPACE, FakeChat([reply(GOOD_USES, GOOD_STORES, GOOD_USES)]))

    assert (again.restated, again.proposed, again.new_predicates) == (3, (), ())
    assert len(await engine.documents.list_facts(SPACE, include_closed=True)) == 2
    elsewhere = await engine.remember(SPACE, USES + " It runs every night.")
    other = await extract_dynamic_schema(engine, SPACE, FakeChat([reply(GOOD_USES, {**GOOD_USES, "object": "extraction"})]),
                                         episode_ids=[elsewhere.episode_id])
    assert (other.chunks_total, other.restated, [p.object for p in other.proposed]) == (1, 0, ["Ollama", "extraction"]), \
        "the same triple from another episode, and another object from the same one, are proposals of their own"


async def test_the_new_type_bounds_do_not_count_a_triple_already_on_record():
    # A triple already proposed is restated whatever the budget or switch:
    # nothing new would be proposed, so no bound cut anything.
    engine = await opened_with(USES)
    first = await extract_dynamic_schema(engine, SPACE, FakeChat([reply(GOOD_USES)]))
    assert [t.term for t in first.new_predicates] == ["uses"]

    tight = await extract_dynamic_schema(engine, SPACE, FakeChat([reply(GOOD_USES)]), max_new_predicates=0,
                                         max_new_kinds=0)
    closed = await extract_dynamic_schema(engine, SPACE, FakeChat([reply(GOOD_USES)]), allow_new_types=False,
                                          entity_kinds=("project",), predicates=("stores",))

    for again in (tight, closed):
        assert (again.restated, again.rejected_reasons, again.cut_by, again.proposed) == (1, {}, (), ())


async def test_the_chunks_of_an_episode_forgotten_during_the_pass_are_not_sent():
    filler = " ".join(f"Sentence number {i} talks about harbours and orchards on the ridge." for i in range(14))
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=300,
                                events=InMemoryEventLog()).open()
    added = await engine.remember(SPACE, filler)
    await engine.remember(SPACE, HARBOUR)
    chunks = await engine.documents.chunks_of(SPACE, added.episode_id)
    assert len(chunks) > 2
    calls: list[str] = []

    class Forgets:
        async def complete(self, system: str, user: str) -> str:
            calls.append(user)
            if len(calls) == 1:
                await engine.forget(SPACE, added.episode_id)
            return reply()

    report = await extract_dynamic_schema(engine, SPACE, Forgets())

    assert (len(calls), report.model_calls, report.chunks_gone) == (2, 2, len(chunks) - 1), \
        "the first chunk was already sent; the rest are not, and the other episode still is"
    assert HARBOUR in calls[1] and "forgotten before or while the model answered" in report.text()


async def test_an_episode_forgotten_while_the_model_answers_gets_nothing_written():
    engine = await opened_with(LEDGER)
    [episode] = await engine.documents.recent_episodes(SPACE, 1)

    class Forgets:
        async def complete(self, system: str, user: str) -> str:
            await engine.forget(SPACE, episode.episode_id)
            return reply(GOOD_USES, GOOD_STORES)

    report = await extract_dynamic_schema(engine, SPACE, Forgets())

    assert (report.chunks_gone, report.proposed) == (1, ())
    assert await engine.documents.list_facts(SPACE, include_closed=True) == []


async def test_a_model_that_can_answer_to_a_schema_is_asked_with_one():
    engine = await opened_with(LEDGER)

    class Structured:
        def __init__(self) -> None:
            self.schemas: list[dict[str, object]] = []

        async def complete(self, system: str, user: str) -> str:
            raise AssertionError("a structured model is asked with the schema")

        async def complete_structured(self, system: str, user: str, schema: dict[str, object]) -> str:
            self.schemas.append(schema)
            return reply(GOOD_USES)

    chat = Structured()
    report = await extract_dynamic_schema(engine, SPACE, chat)

    assert len(report.proposed) == 1 and chat.schemas == [module.REPLY_SCHEMA] and report.model == "Structured"


@pytest.mark.parametrize("options", [
    {"max_calls": 0}, {"max_calls": module.MAX_CALLS + 1}, {"max_calls": True},
    {"max_triples_per_chunk": 0}, {"max_triples_per_chunk": module.MAX_TRIPLES_PER_CHUNK + 1},
    {"max_new_predicates": -1}, {"max_new_predicates": module.MAX_NEW_TYPES + 1},
    {"max_new_kinds": -1}, {"max_new_kinds": module.MAX_NEW_TYPES + 1},
    {"predicates": ("uses", " -- ")}, {"entity_kinds": ("x" * 65,)}, {"predicates": "uses"},
    {"predicates": tuple(f"p{i}" for i in range(module.MAX_SUGGESTED + 1))},
    {"entity_kinds": tuple(f"k{i}" for i in range(module.MAX_SUGGESTED + 1))},
    {"allow_new_types": False, "entity_kinds": KINDS}, {"allow_new_types": False, "predicates": PREDICATES},
], ids=["calls-0", "calls-over", "calls-bool", "triples-0", "triples-over", "new-predicates-negative",
        "new-predicates-over", "new-kinds-negative", "new-kinds-over", "predicate-empty", "kind-long",
        "predicates-string", "predicates-many", "kinds-many", "closed-no-predicates", "closed-no-kinds"])
async def test_a_bound_or_vocabulary_out_of_range_is_refused_before_any_call(options):
    chat = FakeChat([])
    with pytest.raises(InvalidInput):
        await extract_dynamic_schema(await opened_with(LEDGER), SPACE, chat, **options)
    assert chat.calls == []


async def test_the_bounds_at_their_edges_are_accepted():
    chat = FakeChat(["[]"])
    report = await extract_dynamic_schema(await opened_with(LEDGER), SPACE, chat, max_calls=module.MAX_CALLS,
                                          max_triples_per_chunk=module.MAX_TRIPLES_PER_CHUNK, max_new_predicates=0,
                                          max_new_kinds=module.MAX_NEW_TYPES,
                                          predicates=tuple(f"p{i}" for i in range(module.MAX_SUGGESTED)))
    assert report.chunks_asked == 1


async def test_the_pass_leaves_an_event_with_its_counts_and_proposed_vocabulary():
    engine = await opened_with(LEDGER)
    report = await extract_dynamic_schema(engine, SPACE, FakeChat([reply(GOOD_STORES, triple(
        "worker", "product", "uses", "Ollama", "product", None))]), entity_kinds=KINDS, predicates=PREDICATES)

    [event] = await engine.events.query(SPACE, kind="dynamic_schema")

    assert event.payload["proposed"] == [report.proposed[0].fact_id]
    assert (event.payload["unquoted"], event.payload["proposed_outside_schema"]) == (1, 1)
    assert event.payload["new_predicates"][0]["examples"][0]["quote"] == STORES


async def test_the_command_runs_one_pass_by_hand(monkeypatch):
    from scone_memory.runtime import config
    from scone_memory.runtime.config import Settings

    engine = await opened_with(LEDGER)
    await engine.remember(SPACE, HARBOUR)
    chat = FakeChat([reply(GOOD_USES, GOOD_STORES), "[]"])
    monkeypatch.setattr(config, "build_chat", lambda settings: chat)
    settings = Settings.from_env({"SCONE_CHAT_MODEL": "fake-3b"})
    out = io.StringIO()
    args = build_parser().parse_args(["--json", "--space", SPACE, "dynamic-schema", "--kind", "project", "--kind",
                                      "product", "--predicate", "uses", "--no-new-types", "--max-calls", "1",
                                      "--max-triples", "5", "--max-new-predicates", "3", "--max-new-kinds", "2"])

    assert await run(args, engine, io.StringIO(""), out, settings) == 0

    record = json.loads(out.getvalue())
    assert (record["model"], record["allow_new_types"], record["chunks_cut"]) == ("fake-3b", False, 1)
    assert (record["max_triples_per_chunk"], record["max_new_predicates"], record["max_new_kinds"]) == (5, 3, 2)
    assert record["rejected_reasons"] == {"new_type_not_allowed": 1} and len(record["proposed"]) == 1
    text = io.StringIO()
    rest = build_parser().parse_args(["--space", SPACE, "dynamic-schema", "--after-chunk", str(record["resume_after"])])
    assert await run(rest, engine, io.StringIO(""), text, settings) == 0
    assert text.getvalue().startswith("0 proposal(s) from 1 of 1 chunk(s) asked")
    monkeypatch.setattr(config, "build_chat", lambda settings: None)
    with pytest.raises(InvalidInput):
        await run(build_parser().parse_args(["dynamic-schema"]), engine, io.StringIO(""), io.StringIO(), settings)
