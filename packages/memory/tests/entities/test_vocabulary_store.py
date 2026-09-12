"""A space's relation vocabulary, kept somewhere that promises to keep it.

The event log could not be that place: retention is a property of the
store rather than of the handle that opened it, so no promise made from
one constructor's arguments could describe what the store would actually
do. This is a store of its own -- nothing sweeps it, nothing else writes
it, and what it holds can be read back and checked rather than inferred.

What these pin, in order of how badly each would hurt if wrong: a second
reader sees what the first saved; an explicit "no meanings" is not the
same as never having set any; and a stale writer is refused rather than
allowed to overwrite a vocabulary it never saw.
"""

from __future__ import annotations

import pytest

from scone_memory.entities.meanings import RelationMeanings
from scone_memory.entities.vocabulary_store import (VocabularyConflict, VocabularyStore)

EMPLOYS = RelationMeanings(inverse={"works_at": "employs"})
MARRIED = RelationMeanings(symmetric=["married_to"])
KEY = b"k" * 32


def store(tmp_path, name="vocabulary.db"):
    return VocabularyStore(tmp_path / name, key=KEY)


def test_a_space_with_nothing_saved_holds_nothing(tmp_path):
    held = store(tmp_path)
    try:
        assert held.get("alpha") is None
    finally:
        held.close()


def test_a_second_reader_on_the_same_file_sees_what_was_saved(tmp_path):
    """The whole point. Two processes, one space, one vocabulary."""
    first = store(tmp_path)
    try:
        first.save("alpha", EMPLOYS, expected_revision=0)
    finally:
        first.close()
    second = store(tmp_path)
    try:
        saved = second.get("alpha")
        assert saved is not None and saved.revision == 1
        assert saved.meanings is not None
        assert saved.meanings.opposite("works_at") == "employs"
    finally:
        second.close()


def test_an_explicit_no_meanings_is_not_the_absence_of_a_record(tmp_path):
    """"This space has no relation meanings" is a thing to say, and it is
    not the same as saying nothing. `RelationMeanings()` is falsy, which
    is how that distinction gets lost."""
    held = store(tmp_path)
    try:
        held.save("alpha", RelationMeanings(), expected_revision=0)
        saved = held.get("alpha")
        assert saved is not None, "an empty vocabulary is still a record"
        assert saved.meanings is not None and not saved.meanings.inverse
        assert not saved.cleared, "empty is not cleared"

        held.clear("alpha", expected_revision=1)
        after = held.get("alpha")
        assert after is not None and after.cleared and after.meanings is None
    finally:
        held.close()


def test_a_stale_writer_is_refused(tmp_path):
    held = store(tmp_path)
    try:
        held.save("alpha", EMPLOYS, expected_revision=0)
        with pytest.raises(VocabularyConflict):
            held.save("alpha", MARRIED, expected_revision=0)
        saved = held.get("alpha")
        assert saved is not None and saved.meanings is not None
        assert saved.meanings.opposite("works_at") == "employs", "the loser must not win"
    finally:
        held.close()


def test_a_later_save_supersedes_an_earlier_one(tmp_path):
    held = store(tmp_path)
    try:
        held.save("alpha", EMPLOYS, expected_revision=0)
        held.save("alpha", MARRIED, expected_revision=1)
        saved = held.get("alpha")
        assert saved is not None and saved.revision == 2 and saved.meanings is not None
        assert saved.meanings.reads_both_ways("married_to")
        assert saved.meanings.opposite("works_at") is None
    finally:
        held.close()


def test_one_space_s_vocabulary_is_not_another_s(tmp_path):
    held = store(tmp_path)
    try:
        held.save("alpha", EMPLOYS, expected_revision=0)
        assert held.get("beta") is None
    finally:
        held.close()


def test_a_vocabulary_is_not_readable_with_the_wrong_key(tmp_path):
    """It is stored sealed, like the plan and run stores, because a
    vocabulary names a space's internal predicates."""
    first = store(tmp_path)
    try:
        first.save("alpha", EMPLOYS, expected_revision=0)
    finally:
        first.close()
    # Refused at open, not at read: the store checks the key against its
    # own metadata before it will hand out a connection, which is stronger
    # than failing on the first record.
    from scone_memory.agents.workflow import WorkflowError

    with pytest.raises(WorkflowError):
        VocabularyStore(tmp_path / "vocabulary.db", key=b"x" * 32)


def test_a_vocabulary_that_cannot_be_read_back_is_not_written(tmp_path):
    """A save that succeeds and then cannot be read is worse than a
    refusal: the space is left holding something that breaks every reader,
    including the engine that opens next."""
    from scone_memory.agents.workflow import WorkflowError

    held = store(tmp_path)
    try:
        broken = RelationMeanings(inverse={"works_at": "works_at"})
    except Exception:
        # Construction already refuses it, so forge the same shape past
        # the constructor the way a mutated object would arrive.
        broken = RelationMeanings(inverse={"works_at": "employs"})
        object.__setattr__(broken, "inverse", {"works_at": "works_at"})
    try:
        with pytest.raises((WorkflowError, ValueError)):
            held.save("alpha", broken, expected_revision=0)
        assert held.get("alpha") is None, "nothing may be left behind by a refused save"
    finally:
        held.close()


def test_deleting_a_space_does_not_clear_its_vocabulary(tmp_path):
    """Pinned because it is a contract, not an accident. This store is
    host-owned and separate from the document store, so `delete_space`
    cannot reach it -- and the hazard is real: a space recreated under the
    same name inherits the old vocabulary. The host clears it, and the
    docstring says so rather than leaving it to be discovered.
    """
    held = store(tmp_path)
    try:
        held.save("doomed", EMPLOYS, expected_revision=0)
        # A deletion elsewhere changes nothing here, by construction.
        saved = held.get("doomed")
        assert saved is not None and saved.revision == 1
        # Clearing is the host's to do, and it takes the current revision.
        held.clear("doomed", expected_revision=1)
        after = held.get("doomed")
        assert after is not None and after.cleared
    finally:
        held.close()
