"""When a later statement replaces an earlier one, retrieval must not put
the earlier one first. The two differ in one word, so the lanes score them
almost equally and insertion order decides; MemoryAgentBench's Conflict
Resolution split is built on exactly that, and E34 measured Scone ranking
the superseded statement above its replacement in 72 of 74 cases."""

from __future__ import annotations

from scone_memory.fusion import Fused, demote_restated, restates, words_of

FINLAND = "pesäpallo was created in the country of Finland."
PHILIPPINES = "pesäpallo was created in the country of Philippines."
GOALTENDER = "goaltender is associated with the sport of pesäpallo."


def restate(a, b):
    return restates(words_of(a), words_of(b))


def test_a_restatement_is_the_same_statement_with_a_different_ending():
    assert restate(FINLAND, PHILIPPINES)
    assert not restate(GOALTENDER, FINLAND), "different statements about the same subject"
    assert not restate(FINLAND, FINLAND), "a text does not restate itself"
    # A value of several words is still caught (the case the first cut of
    # this rule missed: it keyed on all but the last word, so these two
    # went on ranking oldest first).
    assert restate("The author of The Marriage of Figaro is Pierre Beaumarchais.",
                   "The author of The Marriage of Figaro is Thomas Kyd.")
    assert restate("goaltender is associated with the sport of ice hockey.",
                   "goaltender is associated with the sport of pesäpallo.")
    # A shared opening is not enough: four words and 60% of the shorter text.
    assert not restate("the meeting is on Tuesday", "the meeting is cancelled")
    assert not restate("pesäpallo was created in the country of Finland.",
                       "pesäpallo was created in a hurry by a committee of schoolteachers.")
    assert not restate("the cat sat", "the cat left"), "short prose is never grouped"
    assert restate("  Pesäpallo WAS created in the COUNTRY of Philippines.  ", FINLAND), "case and padding are not evidence here"


def test_the_newest_restatement_takes_the_groups_best_rank():
    items = [Fused(1, 1.0, 0.99), Fused(2, 0.98, 0.98), Fused(3, 0.9, 0.9)]
    text = {1: FINLAND, 2: PHILIPPINES, 3: GOALTENDER}
    time = {1: "2024-01-01T00:00:00Z", 2: "2024-06-01T00:00:00Z", 3: "2024-03-01T00:00:00Z"}
    out = demote_restated(items, text, time)
    assert [f.chunk_id for f in out] == [2, 1, 3], "the newer statement moves to the older one's rank"
    assert [f.score for f in out] == [0.98, 1.0, 0.9], "scores travel with their own chunk, so nothing is invented"
    # An unrelated chunk between them keeps its place.
    items = [Fused(1, 1.0, 0.99), Fused(3, 0.95, 0.95), Fused(2, 0.9, 0.9)]
    out = demote_restated(items, text, time)
    assert [f.chunk_id for f in out] == [2, 3, 1]


def test_nothing_moves_without_a_restatement_and_nothing_is_dropped():
    items = [Fused(1, 1.0, None), Fused(3, 0.9, None)]
    text = {1: FINLAND, 3: GOALTENDER}
    time = {1: "2024-01-01T00:00:00Z", 3: "2024-06-01T00:00:00Z"}
    assert demote_restated(items, text, time) == items
    both = [Fused(1, 1.0, None), Fused(2, 0.9, None)]
    out = demote_restated(both, {1: FINLAND, 2: PHILIPPINES}, {1: "2024-01-01T00:00:00Z", 2: "2024-06-01T00:00:00Z"})
    assert {f.chunk_id for f in out} == {1, 2}, "the superseded statement is still there for a question about the past"


async def test_the_engine_puts_the_replacement_first_only_when_asked():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.engine import Record

    async def engine_with(**kw):
        e = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), **kw).open()
        await e.remember_many("s", [
            Record(content=FINLAND, created_at="2024-01-01T00:00:00Z"),
            Record(content=GOALTENDER, created_at="2024-02-01T00:00:00Z"),
            Record(content=PHILIPPINES, created_at="2024-03-01T00:00:00Z"),
        ])
        return e

    plain = await engine_with()
    pack = await plain.recall("s", "In which country was pesäpallo created?", limit=5)
    texts = [i.text for i in pack.items]
    assert texts.index(FINLAND) < texts.index(PHILIPPINES), "the superseded statement leads without the setting"

    demoting = await engine_with(demote_restated=True)
    pack = await demoting.recall("s", "In which country was pesäpallo created?", limit=5)
    texts = [i.text for i in pack.items]
    assert texts.index(PHILIPPINES) < texts.index(FINLAND), "the replacement leads with it"
    assert set(texts) == {FINLAND, PHILIPPINES, GOALTENDER}, "nothing is dropped"
    assert pack.items[0].score == 1.0, "normalisation still runs after the reorder"


def test_the_setting_comes_from_the_environment():
    from scone_memory.config import Settings

    assert Settings.from_env({"SCONE_DEMOTE_RESTATED": "1"}).demote_restated is True
    assert Settings.from_env({}).demote_restated is False


def test_both_benches_build_their_engines_with_every_setting():
    """A bench that builds its engines by hand goes on measuring the
    default under a new setting's name. That happened with contextual
    embeddings (fixed 2026-09-06) and again with this one, so the engine
    both benches use is built in one place and every setting is carried."""
    import inspect

    from scone_memory import cli
    from scone_memory.config import ENGINE_SETTINGS, Settings, build_in_process_engine

    for command in (cli.bench_command, cli.conflicts_command):
        assert "build_in_process_engine" in inspect.getsource(command), command.__name__
    source = inspect.getsource(build_in_process_engine)
    for setting in ENGINE_SETTINGS:
        assert f"{setting}=settings.{setting}" in source, setting
        assert hasattr(Settings(), setting), setting
