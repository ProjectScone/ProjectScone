"""Code as claims: what a file defines, imports and calls.

A codebase is already a graph. This reads it without a model and without
anything leaving the machine, and says what it read in the only language
this framework has: claims, each carrying the line it came from. Once they
are claims they are everything else too — cited, dated, recalled,
traversed, and under the same vocabulary as anything else a space knows,
so "what calls this?" is the question the graph already answers.

What it will not do is guess. A call to something the file cannot see is
not recorded as a call to a name that might mean anything; it is left out,
because a graph with edges nobody can check is worse than a smaller graph.
Reading a file is exact where Python's own parser is exact and silent
everywhere else: a file that does not parse says nothing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional

from .code import Language, MAX_LINES, _line_starts

#: Claims from one file, past which a generated file is not worth reading.
MAX_CLAIMS = 20_000
#: What a claim can say. Each is a predicate in the ledger like any other.
DEFINES = "defines"
IMPORTS = "imports"
CALLS = "calls"


@dataclass(frozen=True)
class CodeClaim:
    """One thing a file says, and the line it says it on."""

    subject: str
    predicate: str
    object: str
    quote: str
    first_line: int
    #: The UTF-8 byte span of the line, so the claim can be checked
    #: against the source the way every other quote is.
    start: int
    end: int


def code_claims(content: str, path: str, *, language: Optional[Language]) -> tuple[CodeClaim, ...]:
    """What a source file defines, imports and calls, as claims about it.

    Names are paths: a module is its path, and a declaration is its path
    and its qualified name, so two files with a function of the same name
    stay two things."""
    if language != "python" or not content or content.count("\n") > MAX_LINES:
        return ()
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, RecursionError):
        return ()
    starts = _line_starts(content)
    lines = content.split("\n")
    found: list[CodeClaim] = []
    named: dict[str, str] = {}

    def at(line: int) -> tuple[str, int, int]:
        """The line as a quote, and the bytes it occupies."""
        text = lines[line - 1] if 0 <= line - 1 < len(lines) else ""
        begins = len(content[: starts[line - 1]].encode()) if line - 1 < len(starts) else 0
        return text.strip(), begins, begins + len(text.encode())

    def say(subject: str, predicate: str, obj: str, line: int) -> None:
        if len(found) >= MAX_CLAIMS:
            return
        quote, begins, ends = at(line)
        found.append(CodeClaim(subject, predicate, obj, quote, line, begins, ends))

    # What the file holds, and what holds what: a class defines its
    # methods, and the file defines the class, so the graph has the
    # nesting people read.
    def walk(node: ast.AST, owner: str, inside: Optional[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{inside}.{child.name}" if inside else child.name
                whole = f"{path}:{name}"
                named[name] = whole
                say(owner, DEFINES, whole, child.lineno)
                walk(child, whole, name)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for module in _imported(child):
                    say(path, IMPORTS, module, child.lineno)
            else:
                walk(child, owner, inside)

    walk(tree, path, None)
    _calls(tree, path, named, say)
    return tuple(found)


def _imported(node: ast.AST) -> list[str]:
    """The modules an import names, as written."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module] if node.module and not node.level else []
    return []


def _calls(tree: ast.AST, path: str, named: dict[str, str], say) -> None:
    """Calls between things this file can see, and no others.

    A bare name is a call to this file's own declaration when it has one.
    ``self.rank`` inside a class is that class's method. Anything else —
    another module's function, a method on a value whose type nobody
    stated — is left out rather than pointed at a name that might mean
    anything."""
    for holder, inside in _holders(tree, None):
        whole = f"{path}:{_qualified(holder, inside)}"
        for call in ast.walk(holder):
            if not isinstance(call, ast.Call):
                continue
            target = _target(call.func, inside, named)
            if target is not None and target != whole:
                say(whole, CALLS, target, call.lineno)


def _holders(node: ast.AST, inside: Optional[str]):
    """Every function in the file, with the class it is written in."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child, inside
            yield from _holders(child, inside)
        elif isinstance(child, ast.ClassDef):
            within = f"{inside}.{child.name}" if inside else child.name
            yield from _holders(child, within)
        else:
            yield from _holders(child, inside)


def _qualified(holder: ast.AST, inside: Optional[str]) -> str:
    name = getattr(holder, "name", "")
    return f"{inside}.{name}" if inside else name


def _target(func: ast.AST, inside: Optional[str], named: dict[str, str]) -> Optional[str]:
    if isinstance(func, ast.Name):
        return named.get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
        within = inside.split(".")[0] if inside else None
        return named.get(f"{within}.{func.attr}") if within else None
    return None
