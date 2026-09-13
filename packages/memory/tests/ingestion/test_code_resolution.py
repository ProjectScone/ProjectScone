"""Binding a call across files, when exactly one declaration can answer.

A file on its own cannot say what `thing.method()` refers to: the type of
`thing` is stated nowhere in it. Measured over this package, that is
59.3% of the distinct call names the graph cannot place, and no rule read
off one file's syntax reaches them.

A **corpus** can answer a useful share of it. If exactly one declaration
anywhere in what was read is named `method`, that is what the call meant
-- and if two are, nothing is. The single-definition rule is the whole
safety property, and it is the reference's bar too: an ambiguous name
fabricates nothing.

Measured before building: of 831 distinct unplaced names in this package,
**31.9% have exactly one declaration**, 32.3% have several and are
refused, and 35.9% have none because they belong to somebody else's
library.

These edges are **inferred**, not read. A call bound here was matched by
name against the corpus, not quoted from the source that made it, and the
ledger records that difference so a reader can weigh it.
"""

from __future__ import annotations

from scone_memory.ingestion.code_resolution import resolve_across_files


def test_a_name_with_one_declaration_anywhere_is_bound():
    found = resolve_across_files(
        [("app/a.py:use", "thing.rank")],
        {"rank": ("app/shelf.py:Shelf.rank",)},
    )
    assert found.edges == (("app/a.py:use", "app/shelf.py:Shelf.rank"),)
    assert (found.ambiguous, found.unknown) == (0, 0)


def test_a_name_with_two_declarations_binds_to_neither():
    """The rule that keeps this honest. Two plausible answers is not half
    an edge; it is no edge, and it is counted so the reader can see how
    much was refused rather than missed."""
    found = resolve_across_files(
        [("app/a.py:use", "thing.rank")],
        {"rank": ("app/shelf.py:Shelf.rank", "app/pile.py:Pile.rank")},
    )
    assert found.edges == ()
    assert (found.ambiguous, found.unknown) == (1, 0)


def test_a_name_declared_nowhere_here_is_counted_apart():
    """Somebody else's library. Different from ambiguous, and a reader
    who sees the two added together learns nothing from either."""
    found = resolve_across_files([("app/a.py:use", "session.commit")], {})
    assert found.edges == ()
    assert (found.ambiguous, found.unknown) == (0, 1)


def test_only_the_last_segment_names_the_declaration():
    """`thing.rank` is a call to something named `rank`; `thing` is a
    value, not a place to look."""
    found = resolve_across_files(
        [("app/a.py:use", "self.inner.rank")],
        {"rank": ("app/shelf.py:Shelf.rank",)},
    )
    assert found.edges == (("app/a.py:use", "app/shelf.py:Shelf.rank"),)


def test_a_call_is_never_bound_to_its_own_caller():
    """An edge from a thing to itself says nothing, and the line reader
    refuses it for the same reason."""
    found = resolve_across_files(
        [("app/shelf.py:Shelf.rank", "other.rank")],
        {"rank": ("app/shelf.py:Shelf.rank",)},
    )
    assert found.edges == ()
    assert (found.ambiguous, found.unknown) == (0, 0)


def test_the_same_edge_found_twice_is_one_edge():
    """A method called in a loop is one relationship, not twenty."""
    found = resolve_across_files(
        [("app/a.py:use", "thing.rank"), ("app/a.py:use", "other.rank")],
        {"rank": ("app/shelf.py:Shelf.rank",)},
    )
    assert found.edges == (("app/a.py:use", "app/shelf.py:Shelf.rank"),)


def test_edges_come_back_in_a_settled_order():
    """A receipt that reorders between runs cannot be compared."""
    found = resolve_across_files(
        [("app/z.py:z", "thing.rank"), ("app/a.py:a", "thing.hold")],
        {"rank": ("app/shelf.py:Shelf.rank",), "hold": ("app/shelf.py:Shelf.hold",)},
    )
    assert found.edges == (("app/a.py:a", "app/shelf.py:Shelf.hold"),
                           ("app/z.py:z", "app/shelf.py:Shelf.rank"))


def test_nothing_in_gives_nothing_out():
    found = resolve_across_files([], {"rank": ("app/shelf.py:Shelf.rank",)})
    assert found.edges == () and found.ambiguous == 0 and found.unknown == 0
