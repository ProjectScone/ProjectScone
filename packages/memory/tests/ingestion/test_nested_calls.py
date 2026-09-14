"""Calls made inside nested functions and nested classes, named as the graph defines them.

``defines`` names a function written inside another ``outer.inner``, and a
method of a class inside a class ``Shelf.Label.render``. The call pass
named callers differently: a call inside ``inner`` was recorded as made by
``path:inner``, an entity nothing defines, and credited to ``outer`` as
well; ``outer``'s own call to ``inner()`` went unbound; and ``self`` inside
``Shelf.Label`` was read as ``Shelf``. Here each call belongs to the
function whose own body holds it, a bare name is looked up from the
innermost enclosing function outwards, and ``self`` is the class the
method is written in.
"""

from __future__ import annotations

from scone_memory.ingestion.code_graph import code_claims

SOURCE = '''
def helper():
    return 1


def outer():
    def inner():
        def deepest():
            return helper()
        return deepest()
    return inner()


class Shelf:
    def keep(self):
        return 1

    class Label:
        def text(self):
            return "label"

        def render(self):
            return self.text()
'''

PATH = "m.py"


def claims(predicate: str) -> set[tuple[str, str]]:
    return {(c.subject, c.object) for c in code_claims(SOURCE, PATH, language="python") if c.predicate == predicate}


def test_a_call_inside_a_nested_function_is_made_by_the_name_it_is_defined_by():
    calls = claims("calls")
    assert ("m.py:outer.inner.deepest", "m.py:helper") in calls
    assert {subject for subject, _ in calls} <= {obj for _, obj in claims("defines")}, calls


def test_the_enclosing_function_is_not_credited_with_its_inner_functions_calls():
    assert ("m.py:outer", "m.py:helper") not in claims("calls")


def test_a_bare_name_is_found_in_the_enclosing_functions_first():
    calls = claims("calls")
    assert ("m.py:outer", "m.py:outer.inner") in calls
    assert ("m.py:outer.inner", "m.py:outer.inner.deepest") in calls


def test_self_in_a_nested_class_is_that_class():
    assert ("m.py:Shelf.Label.render", "m.py:Shelf.Label.text") in claims("calls")


def test_a_bare_name_in_a_method_skips_the_class_body_as_python_does():
    source = ("def helper():\n    return 1\n\nclass Shelf:\n    def helper(self):\n        return 2\n\n"
              "    def keep(self):\n        return helper()\n")
    calls = {(c.subject, c.object) for c in code_claims(source, "m.py", language="python") if c.predicate == "calls"}
    assert ("m.py:Shelf.keep", "m.py:helper") in calls and ("m.py:Shelf.keep", "m.py:Shelf.helper") not in calls
