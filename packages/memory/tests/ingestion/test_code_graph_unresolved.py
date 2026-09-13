"""A call the graph could not bind is disclosed, not silently dropped.

`_calls` records an edge only when it can name the target, and refuses to
point at a name that might mean anything. That decision is right. The
silence around it is not: "what calls this?" then answers with everything
it *could* see, and nothing in the answer distinguishes a function
genuinely called by nobody from one whose callers the reader could not
resolve.

This is the same fault `entities/affected.py` carries `unresolved_imports`
for, one level down, and the reference names it too -- `graphify`'s
cross-repo pass exists because single-repo resolvers "hold the receiver
type and drop the call anyway".

Names only, never edges. An edge nobody can check is worse than no edge;
a name is a disclosure, not a claim.
"""

from __future__ import annotations

from scone_memory.ingestion.code_graph import unresolved_calls

BOUND_AND_NOT = '''\
import other

def local():
    return 1

class Shelf:
    def rank(self):
        return 2

    def use(self, thing):
        self.rank()
        local()
        other.far()
        thing.method()
        return len(str(self))
'''


def test_it_names_the_calls_it_could_not_bind():
    """`self.rank` and `local` resolve to declarations in this file.
    `other.far` is another module's, and `thing.method` is a method on a
    value whose type nobody stated -- neither can become an edge, and
    both are exactly what a reader needs told."""
    assert unresolved_calls(BOUND_AND_NOT, "a.py", language="python") == (
        "other.far", "thing.method")


def test_builtins_are_not_a_disclosure():
    """`len` and `str` resolve to nothing in this file and never will.
    Reporting them would bury the two names that matter under noise every
    Python file produces."""
    found = unresolved_calls(BOUND_AND_NOT, "a.py", language="python")
    assert "len" not in found and "str" not in found


def test_a_file_that_binds_everything_discloses_nothing():
    """The other half: this must be able to return empty, or it is not a
    disclosure but a constant."""
    source = "def one():\n    return 2\n\ndef two():\n    return one()\n"
    assert unresolved_calls(source, "a.py", language="python") == ()


def test_a_recursive_call_is_bound_not_unresolved():
    """`_calls` emits no edge for a self-call, because an edge from a
    thing to itself says nothing. That is a resolved call it chose not to
    record, not a call it could not name."""
    source = "def countdown(n):\n    return countdown(n - 1)\n"
    assert unresolved_calls(source, "a.py", language="python") == ()


def test_each_name_is_reported_once_and_in_order():
    """A name called twenty times is one thing the reader cannot see, not
    twenty, and a stable order keeps a receipt comparable between runs."""
    source = ("def caller(thing):\n"
              "    thing.zulu()\n"
              "    thing.alpha()\n"
              "    thing.zulu()\n")
    assert unresolved_calls(source, "a.py", language="python") == (
        "thing.alpha", "thing.zulu")


def test_a_language_without_a_call_reader_discloses_nothing_rather_than_guessing():
    """Brace files claim no calls at all -- the scanner cannot bind one --
    so there is no subset it failed to resolve. Reporting every call in
    the file would read as a graph defect rather than an absent reader."""
    source = "function caller(thing) { thing.method(); other.far(); }\n"
    assert unresolved_calls(source, "a.js", language="braces") == ()
