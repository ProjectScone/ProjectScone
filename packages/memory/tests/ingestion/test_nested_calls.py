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


def calls_in(source: str) -> set[tuple[str, str]]:
    return {(c.subject, c.object) for c in code_claims(source, PATH, language="python") if c.predicate == "calls"}


def test_a_parameter_or_local_of_the_same_name_is_not_the_declaration():
    source = '''
def load():
    return 1


def other():
    return 2


def run(load):
    return load()


def outer():
    def load():
        return 1

    def by_parameter(load):
        return load()

    def by_assignment():
        load = other
        return load()

    def by_import():
        from pkg import load
        return load()

    def plain():
        return load()
    return plain
'''
    calls = calls_in(source)
    assert ("m.py:run", "m.py:load") not in calls
    assert not {pair for pair in calls if pair[0] in ("m.py:outer.by_parameter", "m.py:outer.by_assignment")}, calls
    # An import inside the function is still what the import names.
    assert ("m.py:outer.by_import", "pkg.load") in calls and ("m.py:outer.by_import", "m.py:outer.load") not in calls
    assert ("m.py:outer.plain", "m.py:outer.load") in calls


def test_what_a_def_statement_evaluates_is_called_by_the_function_it_is_written_in():
    # Decorators, default values and a nested class's own body run when the enclosing
    # function runs the statement, not when the function they belong to is called.
    source = '''
def make_default():
    return 1


def register():
    return lambda function: function


def keyword_default():
    return 2


def size():
    return 3


def seal():
    return lambda kind: kind


def outer():
    @register()
    def inner(y=make_default(), *, z=keyword_default()):
        return None

    @seal()
    class Box:
        size = size()

        @register()
        def open(self):
            return None

    def measure():
        # `size` above is Box's, not a local of outer's hiding the module's function.
        return size()
    return inner


def top(y=make_default()):
    return None
'''
    calls = calls_in(source)
    assert {("m.py:outer", "m.py:make_default"), ("m.py:outer", "m.py:keyword_default"), ("m.py:outer", "m.py:register"),
            ("m.py:outer", "m.py:seal"), ("m.py:outer", "m.py:size"), ("m.py:outer.measure", "m.py:size")} <= calls
    assert not {pair for pair in calls if pair[0] in ("m.py:outer.inner", "m.py:outer.Box.open", "m.py:top")}, calls


def test_self_is_the_class_only_where_the_method_takes_it():
    source = '''
class Shelf:
    def text(self):
        return 1

    def render(self):
        def closure():
            return self.text()

        def takes_its_own(self):
            return self.text()
        return closure

    @staticmethod
    def still():
        return self.text()
'''
    calls = calls_in(source)
    assert ("m.py:Shelf.render.closure", "m.py:Shelf.text") in calls
    assert ("m.py:Shelf.render.takes_its_own", "m.py:Shelf.text") not in calls
    assert ("m.py:Shelf.still", "m.py:Shelf.text") not in calls
