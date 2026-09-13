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
    """`self.rank` and `local` resolve to declarations in this file, and
    `other.far` resolves through the import that brought `other` in.

    `thing.method` is a method on a value whose type nobody stated, and
    nothing here can reach it -- that is what a reader needs told.

    `other.far` was in this list when the disclosure was written, and
    moved out of it when imports began to be followed. Both halves are
    asserted so the two cannot drift: a name that became an edge must
    stop being reported as unseen.
    """
    from scone_memory.ingestion.code_graph import CALLS, code_claims

    assert unresolved_calls(BOUND_AND_NOT, "a.py", language="python") == ("thing.method",)
    edges = {(c.subject, c.object) for c in
             code_claims(BOUND_AND_NOT, "a.py", language="python") if c.predicate == CALLS}
    assert ("a.py:Shelf.use", "other.far") in edges, edges


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


# --- Binding a call to a name the file imported ---------------------------

def _calls_in(source: str, path: str = "app/a.py", resolve=None):
    from scone_memory.ingestion.code_graph import CALLS, code_claims

    return {(c.subject, c.object) for c in
            code_claims(source, path, language="python", resolve=resolve)
            if c.predicate == CALLS}


def test_a_call_to_an_imported_name_is_bound_to_where_it_came_from():
    """`_base` has always resolved an inherited name through the import
    that brought it in. `_target` never consulted the same table, so the
    identical name in a call position was dropped."""
    source = ("from app.core import Base\n"
              "\n"
              "def use():\n"
              "    return Base()\n")
    assert ("app/a.py:use", "app.core.Base") in _calls_in(source)


def test_an_alias_binds_to_the_name_it_had_where_it_was_declared():
    """`Store as Shelf` is Store -- the same rule inheritance follows,
    because the declaration is what it is called at home."""
    source = ("from app.core import Store as Shelf\n"
              "\n"
              "def use():\n"
              "    return Shelf()\n")
    assert ("app/a.py:use", "app.core.Store") in _calls_in(source)


def test_a_call_on_an_imported_module_is_bound_to_that_module():
    source = ("import json\n"
              "\n"
              "def use():\n"
              "    return json.dumps({})\n")
    assert ("app/a.py:use", "json.dumps") in _calls_in(source)


def test_an_attribute_of_an_imported_THING_is_never_bound():
    """The false edge this must not make.

    `from app.core import Base` then `Base.make()` -- `make` is an
    attribute of `Base`, **not** a declaration in `app.core`. Binding it
    to `app.core.make` would name something that need not exist. The
    import table already distinguishes the two cases: a module alias
    carries no inner name, a thing taken out of a module carries the name
    it had there.
    """
    source = ("from app.core import Base\n"
              "\n"
              "def use():\n"
              "    return Base.make()\n")
    assert not any("make" in obj for _, obj in _calls_in(source)), _calls_in(source)


def test_a_dotted_module_path_is_not_walked():
    """`os.path.join()` -- only the head is a name this file imported, and
    `os.path` is not it. One level, or nothing."""
    source = ("import os\n"
              "\n"
              "def use():\n"
              "    return os.path.join('a', 'b')\n")
    assert not any("join" in obj for _, obj in _calls_in(source)), _calls_in(source)


def test_binding_an_import_removes_it_from_the_disclosure():
    """The two halves must agree: a call that became an edge is no longer
    a call the graph could not see."""
    source = ("from app.core import Base\n"
              "\n"
              "def use(thing):\n"
              "    Base()\n"
              "    return thing.run()\n")
    assert unresolved_calls(source, "app/a.py", language="python") == ("thing.run",)
