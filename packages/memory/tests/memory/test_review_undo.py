"""Taking back a review decision.

Excluding a claim could always be undone; declining one could not, and
neither could closing one by hand. A review is a person's judgement, and
people are wrong sometimes — a system that records judgements without a
way to revise them teaches its users not to judge.

Nothing is erased by an undo. The decision that was taken back stays in
the event log with its reason, and the one that takes it back is recorded
beside it, so the history of what people decided reads in full.
"""

from __future__ import annotations

import pytest

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryEventLog, InMemoryVectorIndex,
                          MemoryEngine)
from scone_memory.core.errors import InvalidInput
from scone_memory.testing import Clock

pytestmark = pytest.mark.asyncio
SPACE = "alpha"
DAY = "2024-01-01T00:00:00Z"


async def memory() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              clock=Clock("2025-06-01T00:00:00.000Z"), events=InMemoryEventLog()).open()


async def decisions(engine) -> list[str]:
    return [str(event.payload.get("decision"))
            for event in reversed(await engine.events.query(SPACE, kind="fact_review", limit=20))]


async def test_a_declined_claim_can_be_put_back_for_review():
    engine = await memory()
    fact = await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=DAY, proposed=True)
    await engine.decline(SPACE, fact.fact_id, "looked wrong")
    back = await engine.reconsider(SPACE, fact.fact_id, "the source says otherwise")
    assert back.status == "proposed" and back.closed_reason is None
    assert await decisions(engine) == ["declined", "reconsidered"], "both decisions are kept"


async def test_a_claim_that_was_not_declined_cannot_be_reconsidered():
    engine = await memory()
    fact = await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=DAY, proposed=True)
    with pytest.raises(InvalidInput, match="declined"):
        await engine.reconsider(SPACE, fact.fact_id, "no")


async def test_a_reconsidered_claim_can_be_approved_as_any_proposal_can():
    engine = await memory()
    fact = await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=DAY, proposed=True)
    await engine.decline(SPACE, fact.fact_id, "looked wrong")
    await engine.reconsider(SPACE, fact.fact_id, "the source says otherwise")
    held = await engine.approve(SPACE, fact.fact_id)
    assert held.status == "active"


async def test_a_claim_closed_by_hand_can_be_reopened():
    engine = await memory()
    fact = await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=DAY)
    await engine.close_fact(SPACE, fact.fact_id, "she left")
    again = await engine.reopen(SPACE, fact.fact_id, "she had not left")
    assert again.status == "active" and again.valid_until is None and again.closed_reason is None
    closes = await engine.events.query(SPACE, kind="fact_close", limit=10)
    assert [event.payload.get("action") or event.payload.get("reason_kind") for event in reversed(closes)] == [
        "manual", "reopen"], "the close it took back is still recorded"


async def test_a_claim_another_claim_superseded_is_not_reopened_behind_its_back():
    """Reopening it would make two claims hold at once, one of which says
    the other is wrong. The claim that superseded it is named, so a person
    can decide what they actually meant."""
    engine = await memory()
    first = await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=DAY)
    second = await engine.assert_fact(SPACE, "alice", "works_at", "Globex",
                                      valid_from="2024-06-01T00:00:00Z")
    with pytest.raises(InvalidInput, match=f"fact {second.fact_id}"):
        await engine.reopen(SPACE, first.fact_id, "no")


async def test_a_claim_that_holds_is_not_reopened():
    engine = await memory()
    fact = await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=DAY)
    with pytest.raises(InvalidInput, match="holds"):
        await engine.reopen(SPACE, fact.fact_id, "no")


async def test_an_undo_needs_a_reason_like_every_other_decision():
    engine = await memory()
    fact = await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=DAY, proposed=True)
    await engine.decline(SPACE, fact.fact_id, "looked wrong")
    with pytest.raises(InvalidInput, match="reason"):
        await engine.reconsider(SPACE, fact.fact_id, "   ")


async def test_an_undo_is_a_review_decision_over_http_and_needs_the_role_for_one():
    from fastapi.testclient import TestClient

    from scone_memory.api import create_app

    engine = await memory()
    fact = await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=DAY, proposed=True)
    await engine.decline(SPACE, fact.fact_id, "looked wrong")
    app = create_app(engine, {"key-r": SPACE, "key-w": SPACE}, roles={"key-r": "review", "key-w": "write"})
    with TestClient(app) as client:
        refused = client.post(f"/v1/facts/{fact.fact_id}/reconsider", json={"reason": "the source says otherwise"},
                              headers={"Authorization": "Bearer key-w"})
        assert refused.status_code == 403, "a write key may write, not decide"
        said = client.post(f"/v1/facts/{fact.fact_id}/reconsider", json={"reason": "the source says otherwise"},
                           headers={"Authorization": "Bearer key-r"})
        assert said.status_code == 200, said.text
        assert said.json()["status"] == "proposed"


async def test_reopening_over_http_refuses_what_the_engine_refuses():
    from fastapi.testclient import TestClient

    from scone_memory.api import create_app

    engine = await memory()
    first = await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=DAY)
    second = await engine.assert_fact(SPACE, "alice", "works_at", "Globex", valid_from="2024-06-01T00:00:00Z")
    with TestClient(create_app(engine, {"key-a": SPACE})) as client:
        said = client.post(f"/v1/facts/{first.fact_id}/reopen", json={"reason": "no"},
                           headers={"Authorization": "Bearer key-a"})
        assert said.status_code == 422 and f"fact {second.fact_id}" in said.json()["error"]


async def test_both_undos_are_at_the_command_line_too():
    import io

    from scone_memory.runtime.cli import build_parser, run

    engine = await memory()
    proposal = await engine.assert_fact(SPACE, "alice", "works_at", "Acme", valid_from=DAY, proposed=True)
    await engine.decline(SPACE, proposal.fact_id, "looked wrong")
    held = await engine.assert_fact(SPACE, "bob", "lives_in", "Lisbon", valid_from=DAY)
    await engine.close_fact(SPACE, held.fact_id, "he moved")

    out = io.StringIO()
    assert await run(build_parser().parse_args(
        ["--space", SPACE, "reconsider", str(proposal.fact_id), "--reason", "the source says otherwise"]),
        engine, io.StringIO(""), out) == 0
    assert (await engine.documents.get_fact(SPACE, proposal.fact_id)).status == "proposed"

    out = io.StringIO()
    assert await run(build_parser().parse_args(
        ["--space", SPACE, "reopen", str(held.fact_id), "--reason", "he had not moved"]),
        engine, io.StringIO(""), out) == 0
    assert (await engine.documents.get_fact(SPACE, held.fact_id)).status == "active"
