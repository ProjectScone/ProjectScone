"""A call through a class this file declares binds to the method it names.

``self.keep()`` inside a class was bound, and ``Shelf.keep()`` was not,
even with ``class Shelf`` declared a few lines up: the call graph could
see both the call and the declaration and still reported the call as
one it could not place. A dotted call whose whole spelling names a
declaration of this file -- a method of a class here, a class nested in
one -- is now bound, and so is ``cls.keep()`` in a class method. A class
brought in by an import is not: a file cannot tell an imported class
from an imported object, and an edge to a method that need not exist is
the false edge this graph refuses.
"""

from __future__ import annotations

from scone_memory.ingestion.code_graph import code_claims, unresolved_calls

SOURCE = '''
from pkg.store import Vault

class Shelf:
    def keep(self, paper):
        return paper

    @classmethod
    def stocked(cls):
        return cls.keep(None, "sample")

    class Label:
        def print(self):
            return "label"

    def put(self, paper):
        Shelf.Label.print(None)
        return Shelf.keep(self, paper)


def restock(shelf):
    Shelf.keep(shelf, "paper")
    shelf.keep("paper")
    Vault.open()
    Missing.keep()
'''

PATH = "pkg/shelf.py"


def calls() -> set[tuple[str, str]]:
    return {(claim.subject, claim.object) for claim in code_claims(SOURCE, PATH, language="python")
            if claim.predicate == "calls"}


def test_a_call_through_a_class_this_file_declares_binds_to_the_method():
    made = calls()
    assert (f"{PATH}:Shelf.put", f"{PATH}:Shelf.keep") in made
    assert (f"{PATH}:restock", f"{PATH}:Shelf.keep") in made


def test_a_class_nested_in_one_here_binds_through_its_whole_spelling():
    assert (f"{PATH}:Shelf.put", f"{PATH}:Shelf.Label.print") in calls()


def test_cls_in_a_class_method_is_that_class():
    assert (f"{PATH}:Shelf.stocked", f"{PATH}:Shelf.keep") in calls()


def test_a_method_on_a_value_an_imported_class_or_an_unknown_name_stays_unbound():
    made = calls()
    targets = {target for _, target in made}
    assert not any(target.endswith(":Vault.open") or "Missing" in target for target in targets)
    unbound = set(unresolved_calls(SOURCE, PATH, language="python"))
    assert {"shelf.keep", "Vault.open", "Missing.keep"} <= unbound
    assert "Shelf.keep" not in unbound and "cls.keep" not in unbound and "Shelf.Label.print" not in unbound


def test_a_name_that_is_a_function_not_a_class_does_not_invent_a_method():
    source = "def helper():\n    return 1\n\ndef main():\n    return helper.attr()\n"
    made = {(c.subject, c.object) for c in code_claims(source, "m.py", language="python") if c.predicate == "calls"}
    assert made == set() and "helper.attr" in unresolved_calls(source, "m.py", language="python")


def test_a_function_nested_in_a_function_or_a_method_is_not_reached_through_its_holder():
    source = ("class Shelf:\n    def put(self):\n        def inner():\n            return 1\n        return inner()\n\n"
              "def outer():\n    def helper():\n        return 2\n    return helper()\n\n"
              "def main():\n    outer.helper()\n    Shelf.put.inner()\n")
    made = {(c.subject, c.object) for c in code_claims(source, "m.py", language="python") if c.predicate == "calls"}
    # From main, through the holder's name; each holder's own call to its nested function is a real edge.
    assert not any(target in ("m.py:outer.helper", "m.py:Shelf.put.inner") for caller, target in made
                   if caller == "m.py:main"), made
    assert ("m.py:outer", "m.py:outer.helper") in made and ("m.py:Shelf.put", "m.py:Shelf.put.inner") in made
    assert {"outer.helper", "Shelf.put.inner"} <= set(unresolved_calls(source, "m.py", language="python"))


def test_a_class_that_is_also_imported_is_not_followed_through_the_local_one():
    """``try: from fast import Shelf`` with ``class Shelf`` as the fallback:
    which one runs is decided when the file is imported, so ``Shelf.keep()``
    names a method of either and is left unbound."""
    source = ("try:\n    from fast import Shelf\nexcept ImportError:\n    class Shelf:\n        def keep(self):\n"
              "            return 1\n\n        class Label:\n            def print(self):\n                return 2\n\n\n"
              "def main():\n    Shelf.Label.print(None)\n    return Shelf.keep(None)\n")
    made = {(c.subject, c.object) for c in code_claims(source, "m.py", language="python") if c.predicate == "calls"}
    assert not {("m.py:main", "m.py:Shelf.keep"), ("m.py:main", "m.py:Shelf.Label.print")} & made, made
    assert {"Shelf.keep", "Shelf.Label.print"} <= set(unresolved_calls(source, "m.py", language="python"))
