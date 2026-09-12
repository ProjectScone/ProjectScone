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
from typing import Callable, Optional

#: How a relative import is turned into something nameable. It takes the
#: importing file's path, how many dots the import had, and the module it
#: named, and answers with the thing imported or None. A file alone cannot
#: do this — it does not know where the package root is or what else
#: exists — so whoever walked the tree decides, and a name nothing
#: resolves to is left out rather than guessed at.
Resolve = Callable[[str, int, str], Optional[str]]

import re

from .code import Language, MAX_LINES, _line_starts, declarations

#: Claims from one file, past which a generated file is not worth reading.
MAX_CLAIMS = 20_000
#: What a claim can say. Each is a predicate in the ledger like any other.
DEFINES = "defines"
IMPORTS = "imports"
CALLS = "calls"

#: How the brace languages write an import. A header line is written
#: down, so it can be read; what a name in the body refers to is not, so
#: it is not guessed at. Nothing here claims a call for these languages.
_FROM = re.compile(r"""\bfrom\s+["']([^"']+)["']""")
_BARE = re.compile(r"""^\s*import\s+["']([^"']+)["']""")
_REQUIRE = re.compile(r"""\brequire\s*\(\s*["']([^"']+)["']""")
_QUOTED = re.compile(r"""^\s*(?:_\s+|\w+\s+)?["']([^"']+)["']\s*$""")
_USE = re.compile(r"^\s*(?:pub\s+)?use\s+([A-Za-z_][\w:]*)")
#: Where a language writes its imports in a block rather than a line.
_OPENS = re.compile(r"^\s*import\s*\($")


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


def code_claims(content: str, path: str, *, language: Optional[Language],
                resolve: Optional["Resolve"] = None) -> tuple[CodeClaim, ...]:
    """What a source file defines, imports and calls, as claims about it.

    Names are paths: a module is its path, and a declaration is its path
    and its qualified name, so two files with a function of the same name
    stay two things. ``resolve`` turns a relative import into something
    nameable; without one, relative imports are left out, because a file
    on its own cannot tell where its package root is."""
    if not content or content.count("\n") > MAX_LINES:
        return ()
    if language == "braces":
        return _brace_claims(content, path, resolve)
    if language != "python":
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
                for module in _imported(child, path, resolve):
                    say(path, IMPORTS, module, child.lineno)
            else:
                walk(child, owner, inside)

    walk(tree, path, None)
    _calls(tree, path, named, say)
    return tuple(found)


def _imported(node: ast.AST, path: str, resolve: Optional["Resolve"]) -> list[str]:
    """What an import names: an absolute module as written, and a relative
    one only when somebody who knows the tree can say what it is."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if not node.level:
        return [node.module] if node.module else []
    if resolve is None:
        return []
    # "from .code import x" names the module in module; "from . import
    # code" names it in the aliases. Either way what is imported is a
    # module, and that is what the resolver is asked for.
    wanted = [node.module] if node.module else [alias.name for alias in node.names]
    found = [resolve(path, node.level, one) for one in wanted]
    return [one for one in found if one]


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


async def record_claims(engine, space: str, *, episode_id: int, content: str, path: str,
                        when: str, resolve: Optional["Resolve"] = None) -> int:
    """Record what a file says about itself, and say how many claims that
    was. One place decides how these are written — quoted from the line,
    cited to the episode, extracted rather than stated — so the engine and
    the command line cannot come to differ about it."""
    from .code import code_language

    said = 0
    for claim in code_claims(content, path, language=code_language(path), resolve=resolve):
        await engine.assert_fact(space, claim.subject, claim.predicate, claim.object,
                                 valid_from=when, source_episode_id=episode_id,
                                 quote=claim.quote, origin="extracted")
        said += 1
    return said


def _brace_claims(content: str, path: str, resolve: Optional["Resolve"]) -> tuple[CodeClaim, ...]:
    """What a brace-language file declares and imports, and nothing else.

    Declarations come from the same scanner that cuts these files into
    chunks. Imports are read from the lines that write them, including a
    block of them where the language does that. No call is claimed: a call
    means knowing what a name refers to, which needs a parser this does
    not have, and an edge nobody can check is worse than no edge."""
    starts = _line_starts(content)
    lines = content.split("\n")
    found: list[CodeClaim] = []

    def say(subject: str, predicate: str, obj: str, line: int) -> None:
        if len(found) >= MAX_CLAIMS:
            return
        text = lines[line - 1] if 0 <= line - 1 < len(lines) else ""
        begins = len(content[: starts[line - 1]].encode()) if line - 1 < len(starts) else 0
        found.append(CodeClaim(subject, predicate, obj, text.strip(), line, begins,
                               begins + len(text.encode())))

    for item in declarations(content, language="braces"):
        owner = path if "." not in item.name else f"{path}:{item.name.rsplit('.', 1)[0]}"
        say(owner, DEFINES, f"{path}:{item.name}", item.first_line)

    inside = False
    for number, line in enumerate(lines, start=1):
        if _OPENS.match(line):
            inside = True
            continue
        if inside:
            if line.strip().startswith(")"):
                inside = False
                continue
            named = _QUOTED.match(line)
            if named:
                say(path, IMPORTS, named.group(1), number)
            continue
        for pattern in (_FROM, _BARE, _REQUIRE):
            match = pattern.search(line)
            if match:
                where = _named(match.group(1), path, resolve)
                if where:
                    say(path, IMPORTS, where, number)
                break
        else:
            used = _USE.match(line)
            if used:
                say(path, IMPORTS, used.group(1).split("::")[0], number)
    return tuple(found)


def _named(module: str, path: str, resolve: Optional["Resolve"]) -> Optional[str]:
    """An import as it can be named. One written relative to this file is
    left to whoever knows the tree, because this file cannot say what it
    points at; anything else is a package, and names itself."""
    if not module.startswith("."):
        return module
    return resolve(path, 1, module) if resolve is not None else None
