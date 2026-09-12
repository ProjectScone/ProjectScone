"""A space's vocabulary belongs to the space, not to whoever is asking.

What a predicate means to another predicate decides which edges a reader
is shown: configure `works_at` as the inverse of `employs` and the graph
answers "who works here" from claims that never said it. Today that
configuration is read from the environment at engine construction, so the
CLI and the server can hold **different** vocabularies for the **same**
space and hand two readers different graphs. That is not a preference; it
is one space with two meanings.

Saving it in the space fixes that, but only if a saved vocabulary is
actually *used* -- a save that leaves every reader on its old
configuration would be a silent no-op, which is worse than not offering
it.
"""

from __future__ import annotations

import pathlib

import pytest

from scone_memory import (HashEmbedder, InMemoryDocumentStore, InMemoryEventLog,
                          InMemoryVectorIndex, MemoryEngine)
from scone_memory.entities.meanings import RelationMeanings
from scone_memory.entities.vocabulary import (NoEventLog, clear_meanings, read_vocabulary,
                                              save_meanings)

pytestmark = pytest.mark.asyncio

EMPLOYS = RelationMeanings(inverse={"works_at": "employs"})
LIVES = RelationMeanings(symmetric=["married_to"])


#: A durable sink for these tests, not the in-memory ring. A vocabulary is
#: configuration a reader depends on, so it may only be kept somewhere that
#: promises to keep it -- which makes the real SQLite log the honest thing
#: to test against.
def _kept(tmp):
    from scone_memory.observability.events import SqliteEventLog

    return SqliteEventLog(tmp / "events.db")


async def memory(meanings=None, events=True, tmp=None):
    import tempfile

    where = tmp or pathlib.Path(tempfile.mkdtemp())
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              events=_kept(where) if events else None,
                              relation_meanings=meanings).open()


async def test_a_space_with_nothing_saved_uses_the_process_and_says_so():
    engine = await memory(meanings=EMPLOYS)
    try:
        held = await read_vocabulary(engine, "default")
        assert held.source == "process" and held.meanings is not None
        assert held.meanings.opposite("works_at") == "employs"
        assert "this process" in held.why, held.why
    finally:
        await engine.close()


async def test_a_space_with_nothing_saved_and_nothing_configured_has_none():
    engine = await memory()
    try:
        held = await read_vocabulary(engine, "default")
        assert held.source == "none" and held.meanings is None
        assert "only what was said" in held.why, held.why
    finally:
        await engine.close()


async def test_what_the_space_holds_beats_what_the_process_was_told():
    """The whole point: two processes reading one space must agree, and the
    space is the thing they share."""
    engine = await memory(meanings=EMPLOYS)
    try:
        await save_meanings(engine, "default", LIVES)
        held = await read_vocabulary(engine, "default")
        assert held.source == "space", held.why
        assert held.meanings is not None and held.meanings.reads_both_ways("married_to")
        assert held.meanings.opposite("works_at") is None, \
            "the process's configuration must not leak into the space's vocabulary"
    finally:
        await engine.close()


async def test_a_reader_with_no_configuration_of_its_own_sees_the_space_s(tmp_path):
    """The case that is broken without this: a second process, configured
    with nothing, must read the graph the same way as the first.

    Two engines over the same durable log on disk, which is what two
    processes actually are -- not one log object passed between them.
    """
    saved = await memory(meanings=EMPLOYS, tmp=tmp_path)
    try:
        await save_meanings(saved, "default", EMPLOYS)
    finally:
        await saved.close()
    other = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                               events=_kept(tmp_path)).open()
    try:
        held = await read_vocabulary(other, "default")
        assert held.source == "space" and held.meanings is not None
        assert held.meanings.opposite("works_at") == "employs", \
            "a process told nothing still reads the vocabulary the space holds"
    finally:
        await other.close()


async def test_clearing_says_the_space_has_none_and_falls_back():
    engine = await memory(meanings=EMPLOYS)
    try:
        await save_meanings(engine, "default", LIVES)
        assert (await read_vocabulary(engine, "default")).source == "space"
        await clear_meanings(engine, "default")
        held = await read_vocabulary(engine, "default")
        assert held.source == "process", held.why
        assert "cleared" in held.why, held.why
    finally:
        await engine.close()


async def test_the_latest_save_is_the_one_that_holds():
    engine = await memory()
    try:
        await save_meanings(engine, "default", EMPLOYS)
        await save_meanings(engine, "default", LIVES)
        held = await read_vocabulary(engine, "default")
        assert held.meanings is not None
        assert held.meanings.reads_both_ways("married_to")
        assert held.meanings.opposite("works_at") is None, "the older save no longer applies"
    finally:
        await engine.close()


async def test_one_space_s_vocabulary_is_not_another_s():
    engine = await memory()
    try:
        await save_meanings(engine, "alpha", EMPLOYS)
        assert (await read_vocabulary(engine, "alpha")).source == "space"
        assert (await read_vocabulary(engine, "beta")).source == "none"
    finally:
        await engine.close()


async def test_a_store_that_keeps_no_events_cannot_hold_a_vocabulary():
    """It must refuse rather than accept a save that goes nowhere."""
    engine = await memory(meanings=EMPLOYS, events=False)
    try:
        with pytest.raises(NoEventLog):
            await save_meanings(engine, "default", EMPLOYS)
        held = await read_vocabulary(engine, "default")
        assert held.source == "process", "reading still works; only saving is impossible"
        assert "keeps no events" in held.why, held.why
    finally:
        await engine.close()


async def test_the_vocabulary_has_an_identity_that_changes_when_it_does():
    """A cache keyed on the space's revision would not notice a vocabulary
    change, because saving one is not a write to the ledger. The identity
    is what a cache key needs."""
    engine = await memory()
    try:
        first = await read_vocabulary(engine, "default")
        await save_meanings(engine, "default", EMPLOYS)
        second = await read_vocabulary(engine, "default")
        await save_meanings(engine, "default", LIVES)
        third = await read_vocabulary(engine, "default")
        assert len({first.identity, second.identity, third.identity}) == 3, \
            (first.identity, second.identity, third.identity)
    finally:
        await engine.close()


async def test_saving_a_vocabulary_changes_what_the_graph_returns_at_once():
    """The test that makes this feature real rather than a receipt. A saved
    vocabulary that left every reader on its old configuration would be a
    silent no-op -- and saving one is not a write to the ledger, so a cache
    keyed on the space's revision would never notice it."""
    engine = await memory()
    try:
        said = await engine.remember("default", "Ana Alves works at Meridian Health.")
        await engine.assert_fact("default", "ana alves", "works_at", "Meridian Health",
                                 valid_from="2024-01-01T00:00:00Z",
                                 source_episode_id=said.episode_id,
                                 quote="Ana Alves works at Meridian Health.")
        from scone_memory.entities.read import load_projection

        before, _ = await load_projection(engine, "default", mode="current")
        assert not before.implied, "nothing is implied before a vocabulary says anything"

        await save_meanings(engine, "default", EMPLOYS)
        after, _ = await load_projection(engine, "default", mode="current")
        assert after.implied, "the saved vocabulary has to reach the projection"
        assert any(one.predicate == "employs" for one in after.implied), \
            [one.predicate for one in after.implied]
    finally:
        await engine.close()


async def test_a_space_can_hold_an_explicitly_empty_vocabulary():
    """"This space has no relation meanings" is a thing to say, and it is
    not the same as saying nothing. A shared empty vocabulary is how you
    stop two differently-configured processes inventing different edges.

    `RelationMeanings()` is **falsy**, so testing it for truth collapsed an
    explicit empty into a clear and handed the reader its own process
    configuration back -- the opposite of what was asked for.
    """
    engine = await memory(meanings=EMPLOYS)
    try:
        await save_meanings(engine, "default", RelationMeanings())
        held = await read_vocabulary(engine, "default")
        assert held.source == "space", held.why
        assert held.meanings is not None
        assert held.meanings.opposite("works_at") is None, \
            "the space says there are no meanings; the process's must not come back"
    finally:
        await engine.close()


async def test_an_empty_vocabulary_and_a_cleared_one_are_different_identities():
    engine = await memory(meanings=EMPLOYS)
    try:
        await save_meanings(engine, "default", RelationMeanings())
        empty = await read_vocabulary(engine, "default")
        await clear_meanings(engine, "default")
        cleared = await read_vocabulary(engine, "default")
        assert empty.source == "space" and cleared.source == "process"
        assert empty.identity != cleared.identity, (empty.identity, cleared.identity)
    finally:
        await engine.close()


async def test_a_deleted_space_holds_no_vocabulary():
    """Writing into a space that has been deleted leaves a record nothing
    else will ever honour, and a save that reports success for one is a
    lie about durable state."""
    from scone_memory.core.errors import NotFound

    engine = await memory()
    try:
        await engine.remember("doomed", "something")
        await engine.delete_space("doomed")
        # NotFound, the same refusal every other write path gives for a
        # deleted space -- consistency matters more here than a prettier
        # error of this module's own.
        with pytest.raises(NotFound):
            await save_meanings(engine, "doomed", EMPLOYS)
        with pytest.raises(NotFound):
            await clear_meanings(engine, "doomed")
    finally:
        await engine.close()


async def test_an_explicitly_empty_vocabulary_serialises_as_empty_not_as_nothing():
    """The same truthiness fault in two more places: `record()` and the
    identity. A reader handed `meanings: null` cannot tell "the space says
    there are none" from "the space says nothing"."""
    engine = await memory(meanings=EMPLOYS)
    try:
        await save_meanings(engine, "default", RelationMeanings())
        held = await read_vocabulary(engine, "default")
        said = held.record()
        assert said["source"] == "space"
        assert isinstance(said["meanings"], dict), said["meanings"]
        assert said["meanings"]["inverse"] == {}, said["meanings"]
    finally:
        await engine.close()


async def test_a_log_that_may_evict_its_events_cannot_hold_a_vocabulary():
    """The durability flaw. An event log is allowed to be a ring buffer or
    to expire by age, so unrelated traffic in another space could evict the
    one record that held this space's configuration -- and the reader would
    then be told "the space holds no vocabulary", which by then is false.

    Configuration a reader depends on cannot live somewhere that may
    forget it, so a log that declares a bound refuses the save outright
    rather than accepting one it might lose.
    """
    from scone_memory.observability.events import InMemoryEventLog as Bounded

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=Bounded(max_events=2),
                                relation_meanings=EMPLOYS).open()
    try:
        with pytest.raises(NoEventLog) as refused:
            await save_meanings(engine, "alpha", RelationMeanings())
        assert "does not keep what it is given" in str(refused.value), str(refused.value)
        held = await read_vocabulary(engine, "alpha")
        assert held.source == "process"
        assert "cannot hold" in held.why, held.why
    finally:
        await engine.close()


async def test_a_zero_day_retention_is_the_most_aggressive_setting_not_the_absence_of_one(tmp_path):
    """`max_age_days=0` expires everything immediately. Testing `> 0`
    accepted it -- the guard failed open at exactly the setting that
    destroys configuration fastest."""
    from scone_memory.observability.events import SqliteEventLog

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=SqliteEventLog(tmp_path / "e.db", max_age_days=0),
                                relation_meanings=EMPLOYS).open()
    try:
        with pytest.raises(NoEventLog):
            await save_meanings(engine, "default", RelationMeanings())
        assert (await read_vocabulary(engine, "default")).source == "process"
    finally:
        await engine.close()


async def test_a_log_that_makes_no_promise_is_not_assumed_to_keep_anything():
    """Duck typing is not a promise. A log with no retention attributes
    might keep everything, or might be a wrapper that drops whatever it
    likes -- and reading "no bound declared" as "keeps forever" is absence
    of evidence used as evidence of absence.

    The promise has to be made, not inferred.
    """
    class Anonymous:
        """An event log that says nothing about what it keeps."""

        name = "anonymous"

        async def append(self, new):
            raise AssertionError("not reached: the save must be refused first")

        async def query(self, space, kind=None, since=None, limit=100, after_id=None):
            return []

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                events=Anonymous(), relation_meanings=EMPLOYS).open()
    try:
        with pytest.raises(NoEventLog) as refused:
            await save_meanings(engine, "default", EMPLOYS)
        assert "does not promise" in str(refused.value), str(refused.value)
        held = await read_vocabulary(engine, "default")
        assert held.source == "process" and "does not promise" in held.why, held.why
    finally:
        await engine.close()
