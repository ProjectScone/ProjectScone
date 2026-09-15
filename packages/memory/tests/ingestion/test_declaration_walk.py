"""Python declarations are found wherever a statement can stand, and nowhere else.

A function or class definition is a statement, so it can only sit in a body:
a module's, a definition's, a branch's, a handler's or a case's. The reader
looks only there. These tests hold it to a reader that visits every node of
the tree, on a fixture with a definition in each kind of body and on every
sixth Python file in this package, and hold the line table to one built a
character at a time.
"""

from __future__ import annotations

import ast
from pathlib import Path

from scone_memory.ingestion import code
from scone_memory.ingestion.code import MAX_DEPTH, Declaration, declarations

from ..paths import PACKAGE_ROOT

EVERY_BODY = '''
import contextlib

if True:
    def in_if(): pass
elif False:
    def in_elif(): pass
else:
    def in_else(): pass

for item in ():
    def in_for(): pass
else:
    def in_for_else(): pass

while False:
    def in_while(): pass
else:
    def in_while_else(): pass

with contextlib.nullcontext():
    def in_with(): pass

try:
    def in_try(): pass
except ValueError:
    def in_except(): pass
else:
    def in_try_else(): pass
finally:
    def in_finally(): pass

try:
    pass
except* OSError:
    def in_except_star(): pass

match item:
    case 1:
        def in_case(): pass
    case _:
        class InCase:
            def method(self): pass

async def outer():
    async with contextlib.nullcontext():
        async def in_async_with(): pass
    async for _ in ():
        def in_async_for(): pass
    square = lambda value: value * value

class Outer:
    # A comment directly above belongs to the method.
    @staticmethod
    def decorated(): pass

    class Inner:
        def deepest(self):
            def local(): pass
'''


def every_node(content: str) -> tuple[Declaration, ...]:
    """The reader as it was: every node of the tree is visited."""
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, RecursionError):
        return ()
    starts = line_starts_one_character_at_a_time(content)
    found: list[Declaration] = []
    work: list[tuple[ast.AST, str, bool, int]] = [(tree, "", False, 0)]
    while work:
        node, prefix, in_class, depth = work.pop()
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                work.append((child, prefix, in_class, depth))
                continue
            if depth >= MAX_DEPTH:
                continue
            name = prefix + child.name
            opens = min([child.lineno, *(d.lineno for d in child.decorator_list)])
            first = code._with_comments(content, starts, opens)
            last = min(child.end_lineno or child.lineno, len(starts))
            kind = "class" if isinstance(child, ast.ClassDef) else "method" if in_class else "function"
            found.append(Declaration(name, kind, starts[first - 1], code._line_end(content, starts, last), first, last))
            work.append((child, f"{name}.", isinstance(child, ast.ClassDef), depth + 1))
    return tuple(sorted(found, key=lambda d: (d.start, -d.end)))


def line_starts_one_character_at_a_time(content: str) -> list[int]:
    starts = [0]
    index = 0
    while index < len(content):
        if content[index] == "\r":
            index += 2 if content[index + 1:index + 2] == "\n" else 1
            starts.append(index)
        elif content[index] == "\n":
            index += 1
            starts.append(index)
        else:
            index += 1
    return starts


def found(content: str) -> tuple[Declaration, ...]:
    return code._python(content)


def package_files() -> list[Path]:
    """Every sixth Python file of this package, in path order: real code in
    every shape the package writes, at a cost a unit test can carry."""
    files = sorted((PACKAGE_ROOT / "src").rglob("*.py"))
    assert len(files) > 300
    return files[::6]


def test_a_definition_is_found_in_every_kind_of_body():
    names = {declaration.name for declaration in found(EVERY_BODY)}
    assert names == {
        "in_if", "in_elif", "in_else", "in_for", "in_for_else", "in_while", "in_while_else", "in_with",
        "in_try", "in_except", "in_try_else", "in_finally", "in_except_star", "in_case", "InCase",
        "InCase.method", "outer", "outer.in_async_with", "outer.in_async_for", "Outer", "Outer.decorated",
        "Outer.Inner", "Outer.Inner.deepest", "Outer.Inner.deepest.local",
    }
    assert found(EVERY_BODY) == every_node(EVERY_BODY)


def test_this_packages_own_python_files_read_as_they_did_when_every_node_was_visited():
    for path in package_files():
        content = path.read_text(encoding="utf-8")
        assert found(content) == every_node(content), path.relative_to(PACKAGE_ROOT)


def test_definitions_nested_past_the_depth_bound_are_left_out_as_before():
    nested = "".join(f"{'    ' * level}def level{level}():\n" for level in range(MAX_DEPTH + 3))
    nested += "    " * (MAX_DEPTH + 3) + "pass\n"
    assert len(found(nested)) == MAX_DEPTH
    assert found(nested) == every_node(nested)
    assert declarations(nested, language="python") == every_node(nested)


def test_line_starts_count_each_ending_as_python_does():
    for content in ("", "a", "a\n", "\n\n", "a\r\nb", "a\rb", "a\r", "\r\n", "\r\r\n\n", "x\ry\r\nz\n",
                    "caf\u00e9\r\n\u2028line\n"):
        assert code._line_starts(content) == line_starts_one_character_at_a_time(content), repr(content)


def test_this_packages_own_python_files_have_the_line_starts_they_had():
    for path in package_files():
        content = path.read_text(encoding="utf-8")
        assert code._line_starts(content) == line_starts_one_character_at_a_time(content), path
        crlf = content.replace("\n", "\r\n")
        assert code._line_starts(crlf) == line_starts_one_character_at_a_time(crlf), path


def test_the_walk_never_goes_into_an_expression(monkeypatch):
    visited: list[type] = []
    children = ast.iter_child_nodes

    def recording(node):
        visited.append(type(node))
        return children(node)

    monkeypatch.setattr(ast, "iter_child_nodes", recording)
    assert len(found(EVERY_BODY)) == 24
    assert visited and not [kind for kind in visited if issubclass(kind, ast.expr)]
