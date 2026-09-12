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
    assert ("pkg/shelf.py:Shelf.open", "ADR-7") in said(CITES), said(CITES)
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
    assert any(p == CITES and o == "ADR-12" for _, p, o in pairs), pairs  # already canonical


# The counterexamples below are hand-built rather than drawn from a real
# corpus. A graph whose whole claim is that it does not guess has to be
# tested on the shapes that tempt it into guessing.

def test_a_comment_inside_a_string_is_not_a_comment():
    """The most damaging failure available to this feature: text that
    happens to look like a comment, quoted as data, asserted as a claim
    the code never made."""
    source = '''
TEMPLATE = "# WHY: this is a template, not a reason"
HELP = "see ADR-0007 for the rationale"
CODE = "class Fake extends Invented {}"


def real() -> None:
    pass
'''
    found = claims(content=source)
    assert said(NOTES, content=source) == [], said(NOTES, content=source)
    assert said(CITES, content=source) == [], said(CITES, content=source)
    assert [c.object for c in found if c.predicate == INHERITS] == []


def test_a_citation_in_a_docstring_is_still_read():
    """A docstring is documentation, not data. This is the line the
    string rule must not cross."""
    source = '''"""The shelves, per ADR-0007."""


def keep() -> None:
    """Kept because RFC 7231 says so."""
    pass
'''
    assert ("pkg/shelf.py", "ADR-7") in said(CITES, content=source), said(CITES, content=source)
    assert ("pkg/shelf.py:keep", "RFC-7231") in said(CITES, content=source)


def test_two_spellings_of_one_decision_record_are_one_node():
    """The docs claim this normalises. It has to be true."""
    source = "# WHY: see ADR-0007 and ADR 7 and adr#7\nx = 1\n"
    assert {obj for _, obj in said(CITES, content=source)} == {"ADR-7"}


def test_an_aliased_import_does_not_rename_another_one():
    """`import a, b` must not make every alias point at the last module."""
    source = '''import json, csv
from pkg.base import Store as Shelf


class Reader(Shelf):
    pass
'''
    found = said(INHERITS, content=source)
    assert found == [("pkg/shelf.py:Reader", "pkg.base.Store")], found


def test_typescript_extends_and_implements_are_separate_targets():
    source = "class Shelf extends Base implements Face, Other {\n}\n"
    found = code_claims(source, "web/shelf.ts", language="braces")
    bases = sorted(c.object for c in found if c.predicate == INHERITS)
    assert bases == ["Base", "Face", "Other"], bases


def test_a_brace_comment_inside_a_string_is_not_a_comment():
    source = 'const t = "// WHY: not a reason, see ADR-0007";\nclass Real {}\n'
    found = code_claims(source, "web/shelf.ts", language="braces")
    assert [c.predicate for c in found if c.predicate in (NOTES, CITES)] == []


# Three regressions the corrections above introduced. Hand-built again,
# and each one is a position or a target that has to be exactly right:
# a claim citing the wrong line is a citation nobody can check.

def test_a_citation_on_the_third_line_of_a_docstring_points_at_that_line():
    """`ast.get_docstring` returns cleaned, escape-decoded text, so
    counting its lines drifts from the file. The position has to come from
    the source."""
    source = '''def keep() -> None:
    """Kept.

    Because ADR-0007 says so.
    """
    pass
'''
    [cite] = [c for c in claims(content=source) if c.predicate == CITES]
    assert cite.first_line == 4, cite
    assert "ADR-0007" in cite.quote, cite.quote


def test_an_escaped_newline_in_a_docstring_does_not_move_a_citation():
    source = 'def keep() -> None:\n    """One line\\nwith an escape, per ADR-7."""\n    pass\n'
    [cite] = [c for c in claims(content=source) if c.predicate == CITES]
    assert cite.first_line == 2, cite
    assert "pass" not in cite.quote, cite.quote


def test_each_name_of_a_from_dot_import_keeps_its_own_module():
    """`from . import alpha, beta` names two modules, and pairing every
    alias with the first one sends a base class to the wrong file."""
    def resolve(path, level, module):
        return {"alpha": "pkg/alpha.py", "beta": "pkg/beta.py"}.get(module)

    source = '''from . import alpha, beta


class Child(beta.Parent):
    pass
'''
    found = said(INHERITS, content=source, resolve=resolve)
    assert found == [("pkg/shelf.py:Child", "pkg/beta.py:Parent")], found


def test_colon_inheritance_is_still_read():
    """The clause rewrite dropped it. C++ and C# write a base after a
    colon, and the previous version read them."""
    for source, base in (("class Child : Base {\n}\n", "Base"),
                         ("class Child : public Base {\n}\n", "Base"),
                         ("struct Pair : First, Second {\n}\n", "First")):
        found = code_claims(source, "web/a.cpp", language="braces")
        bases = [c.object for c in found if c.predicate == INHERITS]
        assert base in bases, (source, bases)


def test_a_colon_that_is_not_inheritance_is_not_read():
    """A type annotation in a body is not a base class."""
    source = "class Shelf {\n  size: number = 3;\n}\n"
    found = code_claims(source, "web/a.ts", language="braces")
    assert [c.object for c in found if c.predicate == INHERITS] == []
