"""What a codebase says about itself beyond who calls whom.

Two things a code graph needs that call edges cannot give it. **Which
types are which** -- a class hierarchy is how people navigate a codebase,
and `calls` says nothing about it. And **why the code is the way it is**:
the rationale sits in the comments, and the decision it came from sits in
an ADR or an RFC. Read without a model, attached to the declaration it
explains, and dated and quoted like every other claim.

Nothing here guesses. A base class the file can see is named by its path;
one it cannot is recorded as the source wrote it, exactly as an import of
an external module is.
"""

from __future__ import annotations

import pytest

from scone_memory.ingestion.code_graph import CITES, FLAGS, INHERITS, NOTES, code_claims

MODULE = '''"""Storage shelves."""

from __future__ import annotations

from pkg.base import Store
from .local import Cache


class Shelf(Store, Cache):
    """A shelf of things."""

    # WHY: the index is rebuilt on open because a half-written index is
    # worse than none, see ADR-0007.
    def open(self) -> None:
        pass


class Paper(Shelf):
    # TODO: the paper shelf cannot hold two of the same thing yet
    pass


def loose() -> None:
    # NOTE: RFC 7231 says a conditional request without a validator is
    # not conditional.
    pass
'''


def claims(content=MODULE, path="pkg/shelf.py", language="python", resolve=None):
    return code_claims(content, path, language=language, resolve=resolve)


def said(predicate, **kw):
    return [(c.subject, c.object) for c in claims(**kw) if c.predicate == predicate]


def test_a_class_says_what_it_is_built_on():
    assert ("pkg/shelf.py:Shelf", "pkg.base.Store") in said(INHERITS)
    assert ("pkg/shelf.py:Paper", "pkg/shelf.py:Shelf") in said(INHERITS), \
        "a base this file defines is named by its path, not by a bare word"


def test_every_base_is_recorded_not_just_the_first():
    bases = [obj for subject, obj in said(INHERITS) if subject.endswith(":Shelf")]
    assert len(bases) == 2, bases


def test_a_base_from_a_relative_import_is_resolved_when_somebody_knows_the_tree():
    def resolve(path, level, module):
        return "pkg/local.py" if module == "local" else None

    found = said(INHERITS, resolve=resolve)
    assert ("pkg/shelf.py:Shelf", "pkg/local.py:Cache") in found, found


def test_a_base_from_an_unresolvable_relative_import_is_left_as_written():
    """The same rule imports follow: say what the file said, and do not
    invent a location for it."""
    assert ("pkg/shelf.py:Shelf", "Cache") in said(INHERITS)


def test_the_reason_the_code_is_this_way_is_attached_to_the_code():
    notes = said(NOTES)
    assert any(subject == "pkg/shelf.py:Shelf.open" and "half-written index" in obj
               for subject, obj in notes), notes


def test_a_rationale_and_a_known_problem_are_not_the_same_claim():
    """"Why this exists" and "what is wrong with it" are different
    questions, and one predicate for both answers neither."""
    assert any(subject == "pkg/shelf.py:Paper" and "two of the same thing" in obj
               for subject, obj in said(FLAGS)), said(FLAGS)
    assert all("TODO" not in obj for _, obj in said(NOTES)), said(NOTES)


def test_a_decision_record_becomes_something_the_graph_can_reach():
    assert ("pkg/shelf.py:Shelf.open", "ADR-0007") in said(CITES), said(CITES)
    assert ("pkg/shelf.py:loose", "RFC-7231") in said(CITES), said(CITES)


def test_a_note_outside_any_declaration_belongs_to_the_file():
    found = said(NOTES, content="# WHY: this module exists for the shelves\nx = 1\n")
    assert found == [("pkg/shelf.py", "this module exists for the shelves")], found


def test_a_comment_that_says_nothing_special_is_not_a_claim():
    plain = "# just a comment\ndef f():\n    pass\n"
    assert said(NOTES, content=plain) == [] and said(FLAGS, content=plain) == []


def test_a_class_in_a_brace_language_says_what_it_extends():
    source = """// WHY: the cache wraps the store, see ADR-12
class Shelf extends Store {
  open() {}
}
"""
    found = code_claims(source, "web/shelf.ts", language="braces")
    pairs = [(c.subject, c.predicate, c.object) for c in found]
    assert ("web/shelf.ts:Shelf", INHERITS, "Store") in pairs, pairs
    assert any(p == CITES and o == "ADR-12" for _, p, o in pairs), pairs
