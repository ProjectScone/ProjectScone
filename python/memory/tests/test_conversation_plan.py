import pytest

from scone_memory.core.models import RecallItem
from scone_memory.retrieval.conversation_plan import overview_evidence, plan_conversation_retrieval


@pytest.mark.parametrize("query", ["Hello! What have we been working on?", "Catch me up", "What's new?",
    "Give me an overview of our projects", "What decisions have we made recently?", "What do you know about me?",
    "What have we been working on today?", "Where did we leave off?", "What have we accomplished?"])
def test_general_questions_request_an_overview(query):
    assert plan_conversation_retrieval(query).mode == "overview"


@pytest.mark.parametrize("query", ["How is Juniper calibrated?", "What did we decide about SQLite?",
    "Hello! What do you remember about Polaris?", "What changed in authentication?", "天文台如何校准？"])
def test_named_topics_keep_specific_search(query):
    assert plan_conversation_retrieval(query).mode == "search"


def test_greeting_is_removed_from_search_without_rewriting_the_question():
    assert plan_conversation_retrieval("Hello! What have we been working on?").query == "What have we been working on?"


def test_overview_keeps_distinct_substantive_reports_over_question_echoes():
    def item(number, text, source):
        return RecallItem(chunk_id=number, episode_id=number, score=1, text=text,
                          source=source, created_at="2026-09-07T00:00:00Z")
    report = "We completed the local transcription service and connected the microphone pipeline. " * 5
    updates = [item(1, "What have we been working on?", "chat"), item(2, report, "voice"),
               item(3, report, "voice"), item(4, "The SQLite source filter now runs before the search limit. "
                    "The test retrieved the correct document among ten thousand distractors. " * 4, "retrieval")]
    selected = overview_evidence(updates, 2)
    assert {record.episode_id for record in selected} == {2, 4}
    assert selected[0].text in {record.text for record in updates}, "selection never rewrites source text"


def test_overview_does_not_let_reproduced_or_generated_answers_crowd_out_evidence():
    def item(number, text, **metadata):
        return RecallItem(chunk_id=number, episode_id=number, score=1, text=text,
                          source="same-session", metadata=metadata, created_at="2026-09-07T00:00:00Z")
    bad_answer = "Unfortunately I have no information about the project. Please provide more context. " * 12
    dump = "Latest messages\nYou↗ Source 31\nWhat have we been working on?\nAssistant↗ Source 32\n" + bad_answer
    updates = [item(1, dump), item(2, bad_answer, integration="scone-text", role="assistant"),
               item(3, "We completed telescope alignment and are preparing the microphone interface for local testing."),
               item(4, "We moved the database filters ahead of ranking and verified isolation across separate customer spaces.")]
    assert {record.episode_id for record in overview_evidence(updates, 2)} == {3, 4}
    assert {record.episode_id for record in overview_evidence(updates, 5)} == {3, 4}
