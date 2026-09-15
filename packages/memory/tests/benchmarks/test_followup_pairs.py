"""Follow-up queries measured on two-turn pairs: the second turn gains, the first is untouched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scone_memory.bench.followup import REPLY, load_pairs, measure, report
from scone_memory.core.errors import InvalidInput
from scone_memory.retrieval.followup import carry

PAIRS = Path(__file__).resolve().parents[2] / "benchmarks" / "followup-pairs-v1.json"


async def test_carrying_raises_second_turn_recall_and_leaves_the_first_turn_alone():
    tally = await measure(PAIRS)
    assert tally.pairs >= 20
    for surface in ("context", "chat"):
        off, carry = tally.hits[surface]["off"], tally.hits[surface]["carry"]
        assert carry["first"] == off["first"], surface
        assert carry["second"] > off["second"], surface
    assert tally.lost == [] and tally.gained
    assert set(tally.gained) <= set(tally.carried) and len(tally.carried) < tally.pairs
    assert "second R@5" in report(tally)


def test_every_first_turn_asked_after_another_pairs_first_turn_is_left_alone():
    firsts = [pair["turns"][0]["question"] for pair in load_pairs(PAIRS)["pairs"]]
    carried = [question for before, question in zip([firsts[-1], *firsts], firsts)
               if carry([{"role": "user", "content": before}, {"role": "assistant", "content": REPLY},
                         {"role": "user", "content": question}]).applied]
    assert len(firsts) >= 20 and carried == []


async def test_a_split_measures_only_its_pairs():
    development = await measure(PAIRS, split="development")
    held_out = await measure(PAIRS, split="held_out")
    assert development.pairs + held_out.pairs == (await measure(PAIRS)).pairs
    assert development.pairs and held_out.pairs


def test_a_pairs_file_of_another_schema_is_refused(tmp_path):
    other = tmp_path / "pairs.json"
    other.write_text(json.dumps({**json.loads(PAIRS.read_text()), "schema_version": 2}))
    with pytest.raises(InvalidInput):
        load_pairs(other)


def test_a_pair_whose_gold_is_not_stored_is_refused(tmp_path):
    data = json.loads(PAIRS.read_text())
    data["passages"].remove(data["pairs"][0]["turns"][1]["gold"])
    broken = tmp_path / "pairs.json"
    broken.write_text(json.dumps(data))
    with pytest.raises(InvalidInput):
        load_pairs(broken)
