"""Code as claims: what a file defines, imports and calls.

A codebase is already a graph — this reads it without a model and without
leaving the machine, and says it in the only language this framework has:
claims, each carrying the line it was read from. Once they are claims they
are everything else too: cited, dated, recalled, traversed, and subject to
the same vocabulary as anything else a space knows.

What it will not do is guess. A call to something this file cannot see is
not recorded as a call to a name that might mean anything; it is left out
and counted, because a graph with edges nobody can check is worse than a
smaller graph.
"""

from __future__ import annotations

import pytest

from scone_memory.ingestion.code_graph import code_claims

MODULE = '''"""The planner."""

from __future__ import annotations

import json
from pkg.store import Shelf, keep


def plan(question: str) -> str:
    """Work out what is being asked."""
    return tidy(question)


def tidy(text: str) -> str:
    return text.strip()


class Engine:
    def recall(self, query: str) -> list[str]:
        found = self.rank(query)
        return json.dumps(found)

    def rank(self, query: str) -> list[str]:
        return keep(query)
'''


def claims(source: str = MODULE, path: str = "app/planner.py"):
    return code_claims(source, path, language="python")


def triples(found) -> set[tuple[str, str, str]]:
    return {(claim.subject, claim.predicate, claim.object) for claim in found}


def test_a_file_defines_what_it_holds():
    found = triples(claims())
    assert ("app/planner.py", "defines", "app/planner.py:plan") in found
    assert ("app/planner.py", "defines", "app/planner.py:Engine") in found
    assert ("app/planner.py:Engine", "defines", "app/planner.py:Engine.recall") in found, \
        "a class defines its methods, rather than the file defining them directly"


def test_a_file_imports_what_it_names():
    found = triples(claims())
    assert ("app/planner.py", "imports", "json") in found
    assert ("app/planner.py", "imports", "pkg.store") in found


def test_a_call_this_file_can_see_is_recorded():
    found = triples(claims())
    assert ("app/planner.py:plan", "calls", "app/planner.py:tidy") in found
    assert ("app/planner.py:Engine.recall", "calls", "app/planner.py:Engine.rank") in found, \
        "self.rank is the method of the class it is written in"


def test_a_call_this_file_cannot_see_is_left_out_and_counted():
    """keep() comes from another module and json.dumps from a package. A
    graph with edges nobody can check is worse than a smaller graph."""
    found = claims()
    assert not [claim for claim in found if claim.predicate == "calls" and "keep" in claim.object]
    assert not [claim for claim in found if claim.predicate == "calls" and "dumps" in claim.object]


def test_every_claim_carries_the_line_it_was_read_from():
    for claim in claims():
        assert claim.quote and "\n" not in claim.quote
        assert claim.first_line >= 1
        assert MODULE.encode()[claim.start:claim.end].decode().strip() == claim.quote


def test_a_file_that_does_not_parse_says_nothing_rather_than_guessing():
    assert code_claims("def broken(:\n", "app/broken.py", language="python") == ()


def test_a_language_this_cannot_read_yet_says_nothing():
    assert code_claims("function f() {}", "app/x.ts", language="braces") == ()
