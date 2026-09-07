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


async def test_a_claim_whose_object_is_the_denial_itself_is_flagged(engine):
    """The live store's "claude code / is_installed / nothing" took the
    denial's own word as the object. An object made only of denial words
    is an extraction that swallowed the negation, and there is no reading
    of the source under which the claim holds."""
    added = await engine.remember(SPACE, DENIAL)
    await engine.assert_fact(
        SPACE, "claude code", "is_installed", "nothing",
        source_episode_id=added.episode_id, origin="extracted", confidence=0.5,
    )

    found = await audit_grounding(engine, SPACE)

    assert [f.verdict for f in found] == ["object_is_a_denial"]
    assert found[0].flagged
    assert "says nothing about whether" in found[0].evidence


async def test_a_real_object_inside_a_denied_clause_is_reported_not_condemned(engine):
    """"the hook / posts_to_endpoint / successfully" comes from the same
    denial, and may well be wrong, but the object is real text and only a
    person reading the sentence can say. It is reported with its clause
    and left unflagged: a queue, not a verdict."""
    added = await engine.remember(SPACE, DENIAL)
    await engine.assert_fact(
        SPACE, "the hook", "posts_to_endpoint", "successfully",
        source_episode_id=added.episode_id, origin="extracted",
    )

    found = await audit_grounding(engine, SPACE)

    assert [(f.verdict, f.flagged) for f in found] == [("denial_in_the_same_clause", False)]
    assert "says nothing about whether" in found[0].evidence


async def test_a_clause_that_only_says_would_is_not_a_finding(engine):
    """The distiller refuses "would" at extraction time, where a false
    positive costs one missed claim. Used against a stored ledger the same
    rule condemns "every extraction would land as a proposal for Review",
    which the source plainly asserts. Precision at write time is not
    precision at audit time."""
    added = await engine.remember(
        SPACE, "With consolidation on, every extraction would land as a proposal for Review."
    )
    await engine.assert_fact(
        SPACE, "extraction", "would_land_as", "a proposal for Review",
        source_episode_id=added.episode_id, origin="extracted",
    )

    found = await audit_grounding(engine, SPACE)

    assert [(f.verdict, f.flagged) for f in found] == [("unverifiable_without_a_quote", False)]


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
    assert reported["verdict"] == "object_is_a_denial"
    assert "says nothing about whether" in reported["evidence"]

    code, human = run("audit-grounding")
    assert code == 0
    assert "fact 1" in human and "object_is_a_denial" in human
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
    assert body["counts"] == {"object_is_a_denial": 1, "object_not_in_source": 1,
                              "unverifiable_without_a_quote": 1}
    assert body["revision"] == client.get("/v1/facts", headers=head).json()["revision"]
    assert len(body["findings"]) == 3
    denied = next(f for f in body["findings"] if f["subject"] == "claude code")
    assert "says nothing about whether" in denied["evidence"]

    only = client.get("/v1/facts/audit", params={"flagged": "true"}, headers=head).json()
    assert [f["verdict"] for f in only["findings"]] == ["object_is_a_denial", "object_not_in_source"]
    assert only["counts"] == body["counts"]  # what is shown is filtered, what is counted is not
    assert client.get("/v1/facts/audit", params={"status": "proposed"}, headers=head).json()["findings"] == []


async def test_a_claim_that_says_its_source_in_other_words_is_not_flagged(engine):
    """The live store's "would_be_corrupted / if synthetic fixtures were
    mixed in" comes from "mixing in synthetic fixtures would let seeded
    data masquerade as captured evidence, corrupting retrieval". Every
    word is there, in another order. An exact-phrase check calls that
    fabricated and sends a person to reject a good claim, so a claim whose
    words are mostly present is unproven, not wrong."""
    added = await engine.remember(SPACE, (
        "Demo records must stay separate: mixing in synthetic fixtures would let seeded "
        "data masquerade as captured evidence, corrupting retrieval."
    ))
    await engine.assert_fact(
        SPACE, "benchmarks", "would_be_corrupted", "if synthetic fixtures were mixed in",
        source_episode_id=added.episode_id, origin="extracted",
    )

    found = await audit_grounding(engine, SPACE)

    assert [(f.verdict, f.flagged) for f in found] == [("unverifiable_without_a_quote", False)]


async def test_a_value_with_no_words_to_match_is_not_flagged(engine):
    """"true" cannot be grounded by looking for it in a sentence. The
    audit says so instead of calling a boolean a fabrication."""
    added = await engine.remember(SPACE, "The doc comment joins the embedding; it is off by default.")
    await engine.assert_fact(
        SPACE, "contextual code", "is_off_by_default", "true",
        source_episode_id=added.episode_id, origin="extracted",
    )

    found = await audit_grounding(engine, SPACE)

    assert [(f.verdict, f.flagged) for f in found] == [("unverifiable_without_a_quote", False)]


async def test_a_claim_whose_words_are_absent_is_still_flagged(engine):
    """The class has to keep its meaning: an object that shares almost
    nothing with its source is the one worth a person's time."""
    added = await engine.remember(SPACE, "Ana moved to Lisbon in March.")
    await engine.assert_fact(
        SPACE, "ana", "works_at", "Farfetch and Kinsta",
        source_episode_id=added.episode_id, origin="extracted",
    )

    found = await audit_grounding(engine, SPACE)

    assert [(f.verdict, f.flagged) for f in found] == [("object_not_in_source", True)]


async def test_an_unprovable_claim_is_given_the_sentence_to_check_it_against(engine):
    """184 claims in the live store carry no quote, so the audit can only
    say it cannot tell. It can do better than that: find the sentence in
    the source that best covers the claim and hand it over, so a person
    reads one line instead of a whole episode. The audit still asserts
    nothing; it just stops making the reader do the searching."""
    added = await engine.remember(SPACE, (
        "Lisbon was warm that spring. Ana moved to Lisbon in March and started at "
        "Farfetch the same week. Nobody has reviewed the backlog since."
    ))
    await engine.assert_fact(
        SPACE, "ana", "moved_to", "Lisbon",
        source_episode_id=added.episode_id, origin="extracted",
    )

    found = await audit_grounding(engine, SPACE)

    assert found[0].verdict == "unverifiable_without_a_quote"
    assert found[0].candidate_quote == "Ana moved to Lisbon in March and started at Farfetch the same week."


async def test_a_claim_with_a_real_quote_needs_no_candidate(engine):
    added = await engine.remember(SPACE, "Ana moved to Lisbon in March.")
    await engine.assert_fact(
        SPACE, "ana", "moved_to", "Lisbon", source_episode_id=added.episode_id,
        origin="extracted", quote="Ana moved to Lisbon in March.",
    )

    found = await audit_grounding(engine, SPACE)

    assert (found[0].verdict, found[0].candidate_quote) == ("grounded", None)


async def test_no_sentence_is_offered_when_none_covers_the_claim(engine):
    """Offering the least bad sentence would be inventing evidence.

    Every word of this claim is somewhere in the source, so the audit
    cannot call it absent, but no single sentence carries enough of it to
    be worth a person's eye. Handing over the closest one would dress up
    a scattered coincidence as a lead."""
    added = await engine.remember(SPACE, (
        "Ana signed the lease. Lisbon appeared first. Porto came later. "
        "Madrid and Faro followed."
    ))
    await engine.assert_fact(
        SPACE, "ana", "lived_in", "Lisbon Porto Madrid Faro",
        source_episode_id=added.episode_id, origin="extracted",
    )

    found = await audit_grounding(engine, SPACE)

    assert found[0].verdict == "unverifiable_without_a_quote"
    assert found[0].candidate_quote is None
