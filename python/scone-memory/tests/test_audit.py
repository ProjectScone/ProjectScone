"""Re-run today's grounding checks over a ledger written before them.

The live store holds claims extracted when the distiller had no source
checks: "claude code / is_installed / nothing" is active there, taken from
a sentence that says the opposite. The audit finds those without touching
them, so a person can decide what to do about each one.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.audit import audit_grounding

SPACE = "default"
#: The live episode that produced the two malformed claims, verbatim.
DENIAL = (
    "An authenticated memory API only proves the server accepts writes from a caller "
    "holding a valid token; it says nothing about whether the Claude Code or Codex "
    "hooks are actually installed, fire on the relevant session events, and "
    "successfully post to that endpoint."
)


@pytest.fixture
async def engine():
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


@pytest.fixture
def client(engine):
    from fastapi.testclient import TestClient

    from scone_memory.api import create_app

    with TestClient(create_app(engine, {"key-a": "alpha"})) as c:
        yield c


async def test_a_claim_taken_from_a_denial_is_flagged_with_the_clause_that_denies_it(engine):
    added = await engine.remember(SPACE, DENIAL)
    await engine.assert_fact(
        SPACE, "claude code", "is_installed", "nothing",
        source_episode_id=added.episode_id, origin="extracted", confidence=0.5,
    )

    found = await audit_grounding(engine, SPACE)

    assert [f.verdict for f in found] == ["object_only_in_non_asserted_context"]
    assert found[0].flagged
    assert "says nothing about whether" in found[0].evidence


async def test_a_claim_whose_object_is_absent_from_its_source_is_flagged(engine):
    added = await engine.remember(SPACE, "Ana moved to Lisbon in March.")
    await engine.assert_fact(
        SPACE, "ana", "works_at", "Farfetch",
        source_episode_id=added.episode_id, origin="extracted",
    )

    found = await audit_grounding(engine, SPACE)

    assert [(f.verdict, f.flagged) for f in found] == [("object_not_in_source", True)]


async def test_a_claim_carrying_a_quote_that_still_holds_is_reported_grounded(engine):
    added = await engine.remember(SPACE, "Ana moved to Lisbon in March.")
    await engine.assert_fact(
        SPACE, "ana", "moved_to", "Lisbon",
        source_episode_id=added.episode_id, origin="extracted",
        quote="Ana moved to Lisbon in March.",
    )

    found = await audit_grounding(engine, SPACE)

    assert [(f.verdict, f.flagged) for f in found] == [("grounded", False)]


async def test_an_imported_quote_that_is_not_in_its_source_is_flagged(engine):
    """A claim that stored evidence is judged on that evidence. assert_fact
    refuses a quote its episode does not contain, but an import carries
    whatever the other store wrote, so the audit checks it again here."""
    await engine.import_records(SPACE, [
        {"type": "episode", "episode_id": 1, "content": "Ana moved to Lisbon in March."},
        {"type": "fact", "subject": "ana", "predicate": "moved_to", "object": "Porto",
         "valid_from": "2024-03-02", "status": "active", "origin": "extracted",
         "source_episode_id": 1, "quote": "Ana moved to Porto in March."},
    ])

    found = await audit_grounding(engine, SPACE)

    assert [(f.verdict, f.flagged) for f in found] == [("quote_not_in_source", True)]


async def test_a_claim_the_audit_cannot_settle_is_not_flagged(engine):
    """The checks are necessary, not sufficient: an object sitting in an
    ordinary asserted clause proves nothing either way, and saying so is
    the honest answer. Flagging it would send a person to re-read good
    claims."""
    added = await engine.remember(SPACE, "Ana moved to Lisbon in March.")
    await engine.assert_fact(
        SPACE, "ana", "moved_to", "Lisbon",
        source_episode_id=added.episode_id, origin="extracted",
    )

    found = await audit_grounding(engine, SPACE)

    assert [(f.verdict, f.flagged) for f in found] == [("unverifiable_without_a_quote", False)]


async def test_what_a_person_stated_is_not_audited_for_grounding(engine):
    """Grounding is a question about extraction. A claim a person typed
    answers to that person, not to a source text."""
    added = await engine.remember(SPACE, "Ana moved to Lisbon in March.")
    await engine.assert_fact(SPACE, "ana", "works_at", "Farfetch", source_episode_id=added.episode_id)

    assert await audit_grounding(engine, SPACE) == []


async def test_an_extracted_claim_whose_source_is_gone_is_reported_not_judged(engine):
    added = await engine.remember(SPACE, "Ana moved to Lisbon in March.")
    await engine.assert_fact(
        SPACE, "ana", "moved_to", "Lisbon",
        source_episode_id=added.episode_id, origin="extracted",
    )
    await engine.forget(SPACE, added.episode_id)

    found = await audit_grounding(engine, SPACE)

    assert [(f.verdict, f.flagged) for f in found] == [("source_missing", False)]


def test_the_cli_reports_a_flagged_claim_with_its_evidence(tmp_path):
    """A person needs to run this over a store that already exists, so it
    is a command, and it says which claim, which verdict and on what
    evidence, without opening the store's episodes by hand."""
    import io
    import json

    from scone_memory import cli

    env = {"SCONE_SQLITE_PATH": str(tmp_path / "audit.db")}
    dump = "\n".join([
        json.dumps({"type": "episode", "episode_id": 1, "content": DENIAL, "kind": "note",
                    "created_at": "2026-09-06T04:46:32.687Z"}),
        json.dumps({
            "type": "fact", "fact_id": 1, "subject": "claude code", "predicate": "is_installed",
            "object": "nothing", "confidence": 0.5, "valid_from": "2026-09-06T04:46:32.687Z",
            "status": "active", "origin": "extracted", "source_episode_id": 1,
        }),
    ])

    def run(*argv, stdin=""):
        out = io.StringIO()
        code = cli.main(list(argv), env=env, stdin=io.StringIO(stdin), out=out)
        return code, out.getvalue()

    assert run("import", stdin=dump)[0] == 0
    code, text = run("audit-grounding", "--json")
    assert code == 0
    reported = json.loads(text)
    assert reported["fact_id"] == 1
    assert reported["verdict"] == "object_only_in_non_asserted_context"
    assert "says nothing about whether" in reported["evidence"]

    code, human = run("audit-grounding")
    assert code == 0
    assert "fact 1" in human and "object_only_in_non_asserted_context" in human
    assert "says nothing about whether" in human  # the evidence, not only the verdict
    assert "1 of 1 claim needs a person" in human

    second = json.dumps({
        "type": "fact", "fact_id": 2, "subject": "codex hooks", "predicate": "are_installed",
        "object": "nothing", "confidence": 0.5, "valid_from": "2026-09-06T04:46:32.687Z",
        "status": "active", "origin": "extracted", "source_episode_id": 1,
    })
    assert run("import", stdin="\n".join([dump, second]))[0] == 0
    assert "2 of 2 claims need a person" in run("audit-grounding")[1]


def test_the_audit_is_readable_over_http(client):
    """The repair path is a review screen, so the verdicts have to reach a
    page: which claims, on what evidence, and the revision the list was
    read at, so a batch built from it can be refused if the space moves."""
    head = {"authorization": "Bearer key-a"}
    episode = client.post("/v1/episodes", json={"content": DENIAL}, headers=head).json()["episode_id"]
    for subject, object in (("claude code", "nothing"), ("ana", "Lisbon")):
        client.post("/v1/facts", json={
            "subject": subject, "predicate": "is_installed", "object": object,
            "source_episode_id": episode, "origin": "extracted",
        }, headers=head)

    # A third claim the audit cannot settle, so the filter has something to
    # leave out and the counts have something to count that is not a problem.
    plain = client.post("/v1/episodes", json={"content": "Ana moved to Lisbon in March."},
                        headers=head).json()["episode_id"]
    client.post("/v1/facts", json={"subject": "ana", "predicate": "moved_to", "object": "Lisbon",
                                   "source_episode_id": plain, "origin": "extracted"}, headers=head)

    body = client.get("/v1/facts/audit", headers=head).json()
    assert body["counts"] == {"object_only_in_non_asserted_context": 1, "object_not_in_source": 1,
                              "unverifiable_without_a_quote": 1}
    assert body["revision"] == client.get("/v1/facts", headers=head).json()["revision"]
    assert len(body["findings"]) == 3
    denied = next(f for f in body["findings"] if f["subject"] == "claude code")
    assert "says nothing about whether" in denied["evidence"]

    only = client.get("/v1/facts/audit", params={"flagged": "true"}, headers=head).json()
    assert [f["verdict"] for f in only["findings"]] == [
        "object_only_in_non_asserted_context", "object_not_in_source",
    ]
    assert only["counts"] == body["counts"]  # what is shown is filtered, what is counted is not
    assert client.get("/v1/facts/audit", params={"status": "proposed"}, headers=head).json()["findings"] == []
