"""Named-entity mentions read into the entity graph.

A recognizer reads each stored record; every named thing it finds becomes
a claim that the record names it, quoted from its sentence, extracted
rather than stated, with the kind the recognizer's label gave inferred on
the entity and never forced over a hint that disagrees. Most of these
tests use a recognizer that finds fixed strings, so they need no spaCy;
one reads a real pipeline when one is installed.
"""

from __future__ import annotations

import sys
import types
from typing import Sequence

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.entities.mentions import MENTIONS, is_mention, mention_predicate
from scone_memory.entities.read import load_projection
from scone_memory.ingestion.entity_mentions import (Mention, MentionRecorder, SpacyRecognizer,
                                                    ontonotes_kind, plan_mentions, quote_for)
from scone_memory.ingestion.worker import ConsolidationWorker
from scone_memory.memory.catalog import pending_distillation
from scone_memory.retrieval.fact_recall import scan_facts_for_query
from scone_memory.runtime.mcp import pending_episodes


WHEN = "2024-03-01T00:00:00Z"
NOTE = ("Alice Chen joined Acme Robotics in Lisbon on 3 May 2021. "
        "The French team met at Web Summit. Acme Robotics paid $3 million.")
LABELS = {"Alice Chen": "PERSON", "Acme Robotics": "ORG", "Lisbon": "GPE", "3 May 2021": "DATE",
          "French": "NORP", "Web Summit": "EVENT", "$3 million": "MONEY"}


class FixedRecognizer:
    """Finds every occurrence of fixed strings, labelled as given."""

    name = "fixed/test"

    def __init__(self, labels: dict[str, str]) -> None:
        self.labels = labels
        self.calls = 0

    def recognize(self, texts: Sequence[str]) -> list[list[Mention]]:
        self.calls += 1
        found = []
        for text in texts:
            mentions = []
            for surface, label in self.labels.items():
                at = text.find(surface)
                while at >= 0:
                    mentions.append(Mention(at, at + len(surface), surface, label, ontonotes_kind(label)))
                    at = text.find(surface, at + 1)
            found.append(sorted(mentions, key=lambda m: m.start))
        return found


async def engine() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


def entity(projection, key):
    return next(item for item in projection.entities if item.key == key)


# -- the mapping and the plan --------------------------------------------------


@pytest.mark.parametrize(("label", "kind"), [
    ("ORG", "organisation"), ("PERSON", "person"), ("GPE", "place"), ("LOC", "place"), ("FAC", "place"),
    ("PRODUCT", "product"), ("EVENT", "event"), ("NORP", "nationality"),
    ("WORK_OF_ART", None), ("LAW", None), ("LANGUAGE", None), ("DATE", None), ("MISC", None)])
def test_ontonotes_labels_map_onto_scone_kinds(label, kind):
    assert ontonotes_kind(label) == kind


def test_values_are_counted_and_left_out():
    found = FixedRecognizer(LABELS).recognize([NOTE])[0]
    plan = plan_mentions(NOTE, found)
    assert [(item.name, item.kind) for item in plan.planned] == [
        ("Alice Chen", "person"), ("Acme Robotics", "organisation"), ("Lisbon", "place"),
        ("French", "nationality"), ("Web Summit", "event")]
    assert plan.values == 2, "the date and the amount"
    assert plan.repeated == 1, "Acme Robotics twice is one mention"
    assert plan.cut == 0


def test_an_unkinded_label_is_still_a_thing():
    text = "She read War and Peace."
    plan = plan_mentions(text, [Mention(7, 22, "War and Peace", "WORK_OF_ART", None)])
    assert [(item.name, item.kind) for item in plan.planned] == [("War and Peace", None)]
    assert mention_predicate(None) == MENTIONS


def test_the_cap_counts_every_distinct_pair_it_leaves_out():
    names = [f"Org{i}" for i in range(6)]
    text = " ".join(names) + " Org0 Org5."
    mentions = FixedRecognizer({name: "ORG" for name in names}).recognize([text])[0]
    plan = plan_mentions(text, mentions, limit=2)
    assert [item.name for item in plan.planned] == ["Org0", "Org1"]
    assert plan.cut == 4, "Org2..Org5, each once"
    assert plan.repeated == 2, "the second Org0 and the second Org5"


def test_the_quote_is_the_sentence_and_narrows_when_it_is_too_long():
    text = "First one. Then Acme Robotics hired Bob! Last."
    start = text.index("Acme")
    assert quote_for(text, start, start + 13) == "Then Acme Robotics hired Bob!"
    plan = plan_mentions(text, [Mention(start, start + 13, "Acme Robotics", "ORG", "organisation")], quote_limit=20)
    assert plan.planned[0].quote == "Acme Robotics" and plan.narrowed == 1


def test_a_mention_whose_span_and_text_disagree_is_unplaced_not_misquoted():
    plan = plan_mentions("Acme hired Bob.", [Mention(0, 4, "Globex", "ORG", "organisation")])
    assert plan.planned == () and plan.unplaced == 1


# -- the ledger and the projection ---------------------------------------------


async def test_mentions_become_quoted_extracted_facts_and_typed_entities():
    memory = await engine()
    try:
        added = await memory.remember("s", NOTE, source="notes/meeting.md", created_at=WHEN)
        outcomes = await MentionRecorder(memory, FixedRecognizer(LABELS)).record_pending("s")
        assert [(o.episode_id, o.added, o.values, o.repeated, o.cut) for o in outcomes] == [
            (added.episode_id, 5, 2, 1, 0)]
        facts = {fact.object: fact for fact in await memory.facts("s")}
        acme = facts["Acme Robotics"]
        assert (acme.subject, acme.predicate, acme.origin, acme.status) == (
            "notes/meeting.md", "scone:mentions organisation", "extracted", "active")
        assert acme.source_episode_id == added.episode_id and acme.valid_from == "2024-03-01T00:00:00.000Z"
        assert acme.quote == "Alice Chen joined Acme Robotics in Lisbon on 3 May 2021."
        assert facts["French"].quote == "The French team met at Web Summit."

        projection, _ = await load_projection(memory, "s", mode="current")
        found = entity(projection, "acme robotics")
        assert (found.kind, found.kind_status, found.kind_basis) == ("organisation", "inferred", (acme.fact_id,))
        assert entity(projection, "french").kind == "nationality"
        assert entity(projection, "lisbon").kind == "place"
        relation = next(r for r in projection.relations if r.object_id == found.entity_id)
        assert relation.support.quoted == 1 and relation.support.extracted == 1
    finally:
        await memory.close()


async def test_a_recognized_name_shaped_like_a_value_is_still_an_entity():
    """`3M` reads as a quantity to the classifier's shapes; the recognizer
    already said it is an organisation."""
    memory = await engine()
    try:
        await memory.remember("s", "Bob joined 3M last year.", source="cv.md", created_at=WHEN)
        await MentionRecorder(memory, FixedRecognizer({"3M": "ORG"})).record_pending("s")
        projection, _ = await load_projection(memory, "s", mode="current")
        assert entity(projection, "3m").kind == "organisation"
        assert projection.attributes == ()
    finally:
        await memory.close()


async def test_two_names_of_one_kind_hold_side_by_side():
    """Naming Globex is no change of mind about Acme: neither mention
    closes the other, from one record or from two of the same source."""
    memory = await engine()
    try:
        recognizer = FixedRecognizer({"Acme": "ORG", "Globex": "ORG", "Initech": "ORG"})
        await memory.remember("s", "Acme and Globex merged.", source="news.md", created_at=WHEN)
        await memory.remember("s", "Initech followed.", source="news.md", created_at="2024-04-01T00:00:00Z")
        await MentionRecorder(memory, recognizer).record_pending("s")
        assert await memory.facts("s", status="closed") == []
        assert sorted(fact.object for fact in await memory.facts("s")) == ["Acme", "Globex", "Initech"]
    finally:
        await memory.close()


async def test_a_record_without_a_source_is_named_by_its_episode():
    memory = await engine()
    try:
        added = await memory.remember("s", "Bob moved to Lisbon.", created_at=WHEN)
        await MentionRecorder(memory, FixedRecognizer({"Lisbon": "GPE"})).record_pending("s")
        [fact] = await memory.facts("s")
        assert fact.subject == f"episode {added.episode_id}"
    finally:
        await memory.close()


async def test_running_again_or_storing_again_adds_nothing():
    memory = await engine()
    try:
        recognizer = FixedRecognizer(LABELS)
        recorder = MentionRecorder(memory, recognizer)
        added = await memory.remember("s", NOTE, source="notes/meeting.md", created_at=WHEN)
        await recorder.record_pending("s")
        before = await memory.facts("s", include_closed=True)
        revision = await memory.revision("s")

        assert await recorder.record_pending("s") == [], "nothing is pending once read"
        again = await memory.remember("s", NOTE, source="notes/meeting.md", created_at=WHEN)
        assert again.episode_id == added.episode_id
        assert await MentionRecorder(memory, recognizer).record_pending("s") == [], "a fresh recorder too"
        forced = await recorder.record_episode("s", added.episode_id)
        assert (forced.added, forced.restated) == (0, 5)
        assert await memory.facts("s", include_closed=True) == before
        assert await memory.revision("s") == revision, "a restatement writes nothing"
    finally:
        await memory.close()


async def test_the_cap_bites_and_is_counted_on_the_worker_pass():
    memory = await engine()
    try:
        await memory.remember("s", NOTE, source="notes/meeting.md", created_at=WHEN)
        recorder = MentionRecorder(memory, FixedRecognizer(LABELS), max_mentions=2)
        worker = ConsolidationWorker(memory, None, ["s"], recorder=recorder)
        report = await worker.run_once("s")
        assert report.error is None and report.recognizer == "fixed/test"
        assert (report.mentions_read, report.mentions_added, report.mentions_cut) == (1, 2, 3)
        assert report.mentions_left_out == {"values": 2, "repeated": 1}
        assert len(await memory.facts("s")) == 2
    finally:
        await memory.close()


async def test_a_long_record_is_read_in_part_and_says_so():
    memory = await engine()
    try:
        await memory.remember("s", "Acme hired Bob. " + "x" * 50 + " Globex too.", source="a.md", created_at=WHEN)
        recorder = MentionRecorder(memory, FixedRecognizer({"Acme": "ORG", "Globex": "ORG"}), max_chars=20)
        [outcome] = await recorder.record_pending("s")
        assert outcome.text_cut == len("Acme hired Bob. " + "x" * 50 + " Globex too.") - 20
        assert [fact.object for fact in await memory.facts("s")] == ["Acme"]
        report = await ConsolidationWorker(memory, None, ["s"], recorder=MentionRecorder(
            memory, FixedRecognizer({"Globex": "ORG"}), max_chars=20)).run_once("s")
        assert report.mentions_read == 0, "the record already has its mentions"
    finally:
        await memory.close()


async def test_a_recognizer_kind_beside_a_disagreeing_predicate_is_a_conflict():
    memory = await engine()
    try:
        works = await memory.assert_fact("s", "Alice Chen", "works_at", "Apple", valid_from=WHEN)
        await memory.remember("s", "The Apple launch drew crowds.", source="news.md", created_at=WHEN)
        await MentionRecorder(memory, FixedRecognizer({"Apple": "PRODUCT"})).record_pending("s")
        mention = next(fact for fact in await memory.facts("s") if is_mention(fact))
        projection, _ = await load_projection(memory, "s", mode="current")
        apple = entity(projection, "apple")
        assert (apple.kind, apple.kind_status) == (None, "conflict")
        assert set(apple.kind_basis) == {works.fact_id, mention.fact_id}
    finally:
        await memory.close()


async def test_code_sources_are_skipped_and_counted():
    memory = await engine()
    try:
        await memory.remember("s", "class Acme:\n    pass\n", kind="file", source="pkg/acme.py", created_at=WHEN)
        recognizer = FixedRecognizer({"Acme": "ORG"})
        [outcome] = await MentionRecorder(memory, recognizer).record_pending("s")
        assert outcome.skipped == "code" and recognizer.calls == 0
        assert not [fact for fact in await memory.facts("s") if is_mention(fact)]
    finally:
        await memory.close()


async def test_mentions_stay_out_of_fact_recall_and_the_distillers_queue():
    memory = await engine()
    try:
        await memory.remember("s", NOTE, source="notes/meeting.md", created_at=WHEN)
        await MentionRecorder(memory, FixedRecognizer(LABELS)).record_pending("s")
        assert await pending_distillation(memory.documents, "s") == 1, "still unread by the distiller"
        assert len(await pending_episodes(memory, "s", 5)) == 1, "still pending for an MCP agent"
        assert await scan_facts_for_query(memory.documents, "s", "Acme Robotics", memory.clock()) == []
    finally:
        await memory.close()


async def test_a_sources_page_shows_recognized_names_as_mentions_not_claims():
    from scone_memory.entities.sources import sources_view

    memory = await engine()
    try:
        added = await memory.remember("s", NOTE, source="notes/meeting.md", created_at=WHEN)
        await MentionRecorder(memory, FixedRecognizer(LABELS)).record_pending("s")
        page = await sources_view(memory, "s", added.episode_id)
        assert page["claims"] == []
        assert {item["label"] for item in page["mentions"]} >= {"Acme Robotics", "Lisbon", "Alice Chen"}
        assert "claim_limit" not in page["coverage"]["reasons"]
        capped = await sources_view(memory, "s", added.episode_id, max_claims=2)
        assert "claim_limit" in capped["coverage"]["reasons"], "a read filled by mentions may hide claims"
    finally:
        await memory.close()


async def test_a_replaced_record_closes_its_old_mentions_and_is_read_again():
    from scone_memory.ingestion.batch import Record

    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_graph=True).open()
    try:
        await memory.remember("s", "Acme hired Bob.", kind="file", source="notes.md", dedup_key="notes.md",
                              created_at=WHEN)
        recorder = MentionRecorder(memory, FixedRecognizer({"Acme": "ORG", "Globex": "ORG"}))
        await recorder.record_pending("s")
        await memory.replace("s", Record("Globex hired Bob.", kind="file", source="notes.md",
                                         dedup_key="notes.md", created_at="2024-04-01T00:00:00Z"))
        [outcome] = await recorder.record_pending("s")
        assert outcome.added == 1
        held = {fact.object for fact in await memory.facts("s") if is_mention(fact)}
        assert held == {"Globex"}, "the old record's mention closed with it, the new one read"
    finally:
        await memory.close()


async def test_nobody_else_can_state_a_mention():
    memory = await engine()
    try:
        with pytest.raises(InvalidInput, match="reserved for the entity recognizer"):
            await memory.assert_fact("s", "notes.md", "scone:mentions person", "Bob")
    finally:
        await memory.close()


# -- configuration ---------------------------------------------------------------


async def test_recognition_is_off_unless_configured():
    from scone_memory.runtime.config import Settings, build_recognizer, build_worker

    settings = Settings.from_env({})
    assert settings.entity_recognizer is None and settings.entity_model == "en_core_web_trf"
    assert build_recognizer(settings) is None
    memory = await engine()
    try:
        assert build_worker(memory, settings, ["s"]) is None
        kept = build_worker(memory, Settings.from_env({"SCONE_RETAIN": "observation=30"}), ["s"])
        assert kept is not None and kept.recorder is None
        await memory.remember("s", NOTE, source="notes/meeting.md", created_at=WHEN)
        report = await kept.run_once("s")
        assert report.recognizer is None and report.mentions_read == 0
        assert not [fact for fact in await memory.facts("s") if is_mention(fact)]
    finally:
        await memory.close()


def test_an_unknown_recognizer_is_refused():
    from scone_memory.runtime.config import Settings

    with pytest.raises(InvalidInput, match="SCONE_ENTITY_RECOGNIZER"):
        Settings.from_env({"SCONE_ENTITY_RECOGNIZER": "flair"})


def test_enabled_without_spacy_is_refused_with_the_install_command(monkeypatch):
    from scone_memory.runtime.config import Settings, build_recognizer

    monkeypatch.setitem(sys.modules, "spacy", None)
    settings = Settings.from_env({"SCONE_ENTITY_RECOGNIZER": "spacy"})
    with pytest.raises(InvalidInput, match=r"pip install 'scone-memory\[entities\]'"):
        build_recognizer(settings)


def test_enabled_without_the_model_is_refused_with_the_download_command(monkeypatch):
    fake = types.ModuleType("spacy")

    def load(name: str) -> object:
        raise OSError(f"[E050] Can't find model '{name}'")

    fake.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", fake)
    with pytest.raises(InvalidInput, match="python -m spacy download en_core_web_lg"):
        SpacyRecognizer("en_core_web_lg")


# -- exposure ------------------------------------------------------------------------


async def test_the_entity_list_filters_by_kind():
    from scone_memory.api import create_app

    memory = await engine()
    await memory.remember("alpha", NOTE, source="notes/meeting.md", created_at=WHEN)
    await MentionRecorder(memory, FixedRecognizer(LABELS)).record_pending("alpha")
    with TestClient(create_app(memory, {"key-a": "alpha"})) as client:
        headers = {"Authorization": "Bearer key-a"}
        listed = client.get("/v1/entities", params={"kind": "organisation"}, headers=headers).json()
        assert [item["label"] for item in listed["entities"]] == ["Acme Robotics"]
        assert listed["entities"][0]["kind_status"] == "inferred" and listed["filters"]["kind"] == "organisation"
        everything = client.get("/v1/entities", headers=headers).json()
        assert len(everything["entities"]) == 6, "five mentioned, and the record that mentions them"
        assert client.get("/v1/entities", params={"kind": "galaxy"}, headers=headers).status_code == 422


# -- a real pipeline ---------------------------------------------------------------


def _english_pipeline() -> str | None:
    try:
        import spacy
    except ImportError:
        return None
    for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"):
        if spacy.util.is_package(name):
            return name
    return None


@pytest.mark.skipif(_english_pipeline() is None, reason="needs spaCy and an English pipeline "
                    "(pip install 'scone-memory[entities]' && python -m spacy download en_core_web_sm)")
async def test_a_real_pipeline_records_typed_mentions():
    model = _english_pipeline()
    assert model is not None
    memory = await engine()
    try:
        await memory.remember("s", "Tim Cook announced that Apple will open an office in Berlin.",
                              source="news.md", created_at=WHEN)
        [outcome] = await MentionRecorder(memory, SpacyRecognizer(model)).record_pending("s")
        assert outcome.error is None and outcome.added >= 2
        projection, _ = await load_projection(memory, "s", mode="current")
        kinds = {item.key: item.kind for item in projection.entities}
        assert kinds.get("berlin") == "place"
    finally:
        await memory.close()


# -- failures are retried, isolated and parked ------------------------------------


def test_a_name_too_long_to_quote_is_counted_not_planned():
    text = "Then Acme Robotics hired Bob!"
    start = text.index("Acme")
    plan = plan_mentions(text, [Mention(start, start + 13, "Acme Robotics", "ORG", "organisation")], quote_limit=5)
    assert plan.planned == () and plan.too_long == 1 and plan.narrowed == 0


def fail_writes(memory: MemoryEngine, monkeypatch, *, on: str, times: int) -> list[int]:
    """Make the ledger refuse writing the mention named ``on``, ``times`` times."""
    real = memory._assert_placed
    refused = [0]

    async def flaky(space, subject, predicate, obj, **options):
        if obj == on and refused[0] < times:
            refused[0] += 1
            raise InvalidInput("the store refused this write")
        return await real(space, subject, predicate, obj, **options)

    monkeypatch.setattr(memory, "_assert_placed", flaky)
    return refused


async def test_a_record_whose_writes_stop_partway_is_completed_on_the_next_pass(monkeypatch):
    memory = await engine()
    try:
        await memory.remember("s", NOTE, source="notes/meeting.md", created_at=WHEN)
        recorder = MentionRecorder(memory, FixedRecognizer(LABELS))
        fail_writes(memory, monkeypatch, on="Lisbon", times=1)
        [first] = await recorder.record_pending("s")
        assert first.error is not None and first.added == 2, "two written before the refusal"
        [second] = await recorder.record_pending("s")
        assert second.error is None and (second.added, second.restated) == (3, 2)
        assert len([fact for fact in await memory.facts("s") if is_mention(fact)]) == 5
        assert await recorder.record_pending("s") == []
    finally:
        await memory.close()


async def test_a_record_that_always_fails_is_parked_after_its_attempts_and_counted(monkeypatch):
    memory = await engine()
    try:
        await memory.remember("s", NOTE, source="notes/meeting.md", created_at=WHEN)
        recognizer = FixedRecognizer(LABELS)
        recorder = MentionRecorder(memory, recognizer, max_attempts=2)
        refused = fail_writes(memory, monkeypatch, on="Lisbon", times=99)
        worker = ConsolidationWorker(memory, None, ["s"], recorder=recorder)
        reports = [await worker.run_once("s") for _ in range(4)]
        assert [len(report.mention_errors) for report in reports] == [1, 1, 0, 0]
        assert reports[2].mentions_left_out == {"skipped_parked": 1} == reports[3].mentions_left_out
        assert refused[0] == 2 and recognizer.calls == 2, "a parked record is not read again"
        assert recorder.parked("s") == [1]
    finally:
        await memory.close()


class Poisoned(FixedRecognizer):
    """Raises on any batch holding a poisoned text, and on that text alone;
    ``transient`` raises on the first call only, whatever it holds."""

    def __init__(self, labels: dict[str, str], *, transient: bool = False) -> None:
        super().__init__(labels)
        self.transient = transient

    def recognize(self, texts: Sequence[str]) -> list[list[Mention]]:
        if self.transient and self.calls == 0:
            self.calls += 1
            raise RuntimeError("the model was busy")
        if not self.transient and any("poison" in text for text in texts):
            self.calls += 1
            raise RuntimeError("the model cannot read this")
        return super().recognize(texts)


async def test_a_text_the_recognizer_cannot_read_fails_alone_and_is_parked():
    memory = await engine()
    try:
        await memory.remember("s", "Acme hired Bob.", source="a.md", created_at="2024-01-01T00:00:00Z")
        await memory.remember("s", "poison Globex", source="b.md", created_at="2024-01-02T00:00:00Z")
        await memory.remember("s", "Initech hired Bob.", source="c.md", created_at="2024-01-03T00:00:00Z")
        labels = {"Acme": "ORG", "Globex": "ORG", "Initech": "ORG"}
        recorder = MentionRecorder(memory, Poisoned(labels))
        worker = ConsolidationWorker(memory, None, ["s"], recorder=recorder)
        first = await worker.run_once("s")
        assert first.mentions_added == 2 and list(first.mention_errors) == ["2"]
        assert sorted(fact.object for fact in await memory.facts("s")) == ["Acme", "Initech"]
        await memory.remember("s", "Umbrella hired Bob.", source="d.md", created_at="2024-01-04T00:00:00Z")
        labels["Umbrella"] = "ORG"
        reports = [await worker.run_once("s") for _ in range(3)]
        assert reports[0].mentions_added == 1, "a newer record is not held back"
        assert [len(report.mention_errors) for report in reports] == [1, 1, 0]
        assert reports[2].mentions_left_out == {"skipped_parked": 1}
    finally:
        await memory.close()


async def test_a_recognizer_that_fails_once_succeeds_on_the_retry():
    memory = await engine()
    try:
        await memory.remember("s", "Acme hired Bob.", source="a.md", created_at=WHEN)
        await memory.remember("s", "Globex hired Bob.", source="b.md", created_at=WHEN)
        recorder = MentionRecorder(memory, Poisoned({"Acme": "ORG", "Globex": "ORG"}, transient=True))
        outcomes = await recorder.record_pending("s")
        assert [(outcome.error, outcome.added) for outcome in outcomes] == [(None, 1), (None, 1)]
    finally:
        await memory.close()
