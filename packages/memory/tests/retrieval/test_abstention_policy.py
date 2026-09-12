"""A floor for abstaining, measured rather than guessed, and tied to the
embedder it was measured with.

A similarity is a number one embedder produces under one set of settings.
It is not a probability, and a floor that abstains well for one embedder
says nothing about another. So a floor is measured on questions with an
answer in memory and questions without, recorded with what it cost, and
refused by an engine that embeds differently.
"""

from __future__ import annotations

import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.abstention import AbstentionPolicy, PolicyError, choose_floor

SWEEP = {
    "floors": [0.3, 0.4, 0.5, 0.6],
    "no_evidence_n": 20, "evidence_n": 20, "cross_item_n": 20,
    "abstain_rate": {0.3: 0.4, 0.4: 0.7, 0.5: 0.9, 0.6: 1.0},
    "false_abstain_rate": {0.3: 0.0, 0.4: 0.05, 0.5: 0.2, 0.6: 0.6},
}


def test_the_floor_chosen_catches_the_most_while_withholding_few():
    """Raising the floor abstains more and withholds more: the floor taken
    is the highest one still inside the budget for withheld answers."""
    assert choose_floor(SWEEP, target_false_abstain=0.05) == 0.4
    assert choose_floor(SWEEP, target_false_abstain=0.0) == 0.3
    assert choose_floor(SWEEP, target_false_abstain=0.5) == 0.5
    assert choose_floor({**SWEEP, "false_abstain_rate": {0.3: 0.9, 0.4: 0.9, 0.5: 0.9, 0.6: 0.9}},
                        target_false_abstain=0.05) is None, "no floor is cheap enough to take"


def test_a_policy_says_what_it_was_measured_with_and_what_it_cost(tmp_path):
    policy = AbstentionPolicy(floor=0.4, embedder_id=HashEmbedder().id, dim=HashEmbedder().dim,
                              measured={"questions": 40, "abstain_rate": 0.7, "false_abstain_rate": 0.05,
                                        "dataset": "items.json"})
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy.record()), encoding="utf-8")
    read = AbstentionPolicy.read(path)
    assert read == policy and read.record()["schema_version"] == 1
    assert "0.05" in read.text() and "measured" in read.text()


@pytest.mark.parametrize("change, message", [
    ({"schema_version": 2}, "schema_version"),
    ({"floor": 3.0}, "floor"),
    ({"embedder_id": ""}, "embedder"),
    ({"dim": 0}, "dim"),
])
def test_a_policy_that_cannot_be_trusted_is_refused(tmp_path, change, message):
    written = {"schema_version": 1, "floor": 0.4, "embedder_id": "hash-256", "dim": 256, "measured": {}}
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({**written, **change}), encoding="utf-8")
    with pytest.raises(PolicyError, match=message):
        AbstentionPolicy.read(path)


async def test_an_engine_uses_a_policy_measured_with_its_own_embedder():
    embedder = HashEmbedder()
    policy = AbstentionPolicy(floor=0.42, embedder_id=embedder.id, dim=embedder.dim, measured={"questions": 40})
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder,
                                abstention=policy).open()
    assert engine.similarity_floor == 0.42 and engine.abstention == policy
    status = await engine.status("alpha")
    assert status.abstention == policy.record()


async def test_a_policy_measured_with_another_embedder_is_refused():
    """A cosine from one embedder means nothing for another, so a floor
    from one is never applied to the other."""
    policy = AbstentionPolicy(floor=0.42, embedder_id="somebody-else-v1", dim=256, measured={})
    with pytest.raises(InvalidInput, match="somebody-else-v1"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), abstention=policy)


async def test_a_recall_below_the_measured_floor_is_flagged_for_the_reader():
    embedder = HashEmbedder()
    policy = AbstentionPolicy(floor=0.99, embedder_id=embedder.id, dim=embedder.dim, measured={"questions": 40})
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder,
                                abstention=policy).open()
    await engine.remember("alpha", "The kitchen tap was replaced on Tuesday by the plumber.")
    assert (await engine.recall("alpha", "what colour is the roof")).low_confidence is True


def test_a_policy_file_too_large_to_be_one_is_refused(tmp_path):
    """A policy is a few hundred bytes; anything else is not one, and is
    refused before it is parsed."""
    from scone_memory.retrieval.abstention import MAX_POLICY_BYTES

    path = tmp_path / "policy.json"
    path.write_text("{\"schema_version\": 1, \"pad\": \"" + "x" * (MAX_POLICY_BYTES + 1) + "\"}", encoding="utf-8")
    with pytest.raises(PolicyError, match="too large"):
        AbstentionPolicy.read(path)


@pytest.mark.parametrize("floor", [float("nan"), 2.0, float("inf")])
def test_a_floor_that_is_not_a_similarity_is_refused_wherever_it_is_built(floor):
    """A policy read from a file was checked and one built in code was not,
    and the engine took the policy's floor over its own checked one, so a
    floor of NaN made nothing ever read as low confidence."""
    with pytest.raises(PolicyError, match="floor"):
        AbstentionPolicy(floor=floor, embedder_id="hash-256", dim=256)


@pytest.mark.parametrize("embedder_id, dim", [("", 256), ("   ", 256), ("hash-256", 0), ("hash-256", -1)])
def test_a_policy_must_name_the_embedder_and_its_width(embedder_id, dim):
    with pytest.raises(PolicyError, match="embedder|width"):
        AbstentionPolicy(floor=0.4, embedder_id=embedder_id, dim=dim)


async def test_an_embedder_whose_width_is_not_known_yet_is_not_refused():
    """A remote embedder learns its width from its first answer. Refusing a
    policy before it has spoken rejects a matching one for a width nobody
    asked it for yet."""
    class NotYet(HashEmbedder):
        """Embeds as the hash embedder does, but says nothing of its width
        until it has embedded once, as a remote one does."""

        def __init__(self) -> None:
            super().__init__(256)
            self.id, self.dim = "remote:some-model", 0

        async def embed(self, texts):
            vectors = await super().embed(texts)
            self.dim = 256
            return vectors

    policy = AbstentionPolicy(floor=0.4, embedder_id="remote:some-model", dim=256, measured={})
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), NotYet(),
                                abstention=policy).open()
    assert engine.similarity_floor == 0.4


async def test_a_policy_whose_width_is_known_to_differ_is_still_refused():
    """Unknown is not the same as different: an embedder that says 256
    cannot use a floor measured at 384."""
    policy = AbstentionPolicy(floor=0.4, embedder_id=HashEmbedder().id, dim=384, measured={})
    with pytest.raises(InvalidInput, match="384-d"):
        MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), abstention=policy)
