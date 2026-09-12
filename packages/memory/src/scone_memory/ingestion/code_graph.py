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
import io
import tokenize
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
#: What a class is built on. A hierarchy is how people navigate a
#: codebase and no call edge says anything about it.
INHERITS = "inherits"
#: What a type satisfies without extending it: an interface, a protocol,
#: a Rust trait, a Scala mixin. "What implements this interface?" is a
#: different question from "what extends this class?", and flattening
#: both into one predicate means the graph answers neither precisely.
#: Rust has no class inheritance at all, so every Rust edge is one of
#: these -- calling them inheritance described a relation the language
#: does not have.
MIXES_IN = "mixes_in"
#: Why the code is the way it is, in the words of whoever wrote it.
NOTES = "notes"
#: What is known to be wrong with it. A different question from why it
#: exists, so a different predicate: one for both answers neither.
FLAGS = "flags"
#: A decision record or standard the code says it follows. Unlike a note,
#: this names something the graph can reach from more than one place.
CITES = "cites"

#: A comment that says why. The tag is kept out of the claim's object,
#: which is the sentence a person wrote.
_WHY = re.compile(r"(?:#|//|/\*|\*)\s*(WHY|NOTE|RATIONALE)\s*:\s*(.+?)\s*(?:\*/)?$", re.IGNORECASE)
#: A comment that says what is wrong.
_FLAG = re.compile(r"(?:#|//|/\*|\*)\s*(TODO|FIXME|HACK|XXX)\s*:?\s*(.+?)\s*(?:\*/)?$", re.IGNORECASE)
#: A decision record or a standard, however it is spelled. Normalised to
#: one node name so two spellings of the same document are one thing.
_CITED = re.compile(r"\b(ADR|RFC)[\s\-_#]*([0-9]{1,6})\b", re.IGNORECASE)
#: Where a brace language names a class, and the clauses saying what it is
#: built on. Read separately: `extends Base implements Face` is two
#: relationships, and one capture for both invented a single target named
#: after neither.
_DECLARES = re.compile(r"\b(?:class|interface|struct|enum|record|protocol)\s+([A-Za-z_]\w*)")
#: Go writes a type with `type Name struct` / `type Name interface`, which
#: the class-first pattern misses entirely -- so Go types were invisible
#: while Go was on the supported list.
_GO_TYPE = re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+(?:struct|interface)\b")
#: Rust says what a type implements with `impl Trait for Type`, which is
#: the language's most important relation and produced no edge at all.
#: `impl Type` alone is an inherent block: the same type, not a second
#: definition of it and not an inheritance.
_RUST_IMPL = re.compile(r"^\s*impl(?:\s*<[^>]*>)?\s+([A-Za-z_][\w:]*)"
                        r"(?:\s*<[^>]*>)?(?:\s+for\s+([A-Za-z_][\w:]*))?")
#: Scala joins its mixins with `with`, and Kotlin and Scala write a base
#: as a constructor call -- `extends Base(3) with Store`. A capture class
#: without parentheses matched nothing at all there, so Scala produced no
#: edges while it was on the supported list.
_BUILT_ON = re.compile(r"\b(extends|implements|with)\s+([A-Za-z_][\w.:<>,()\s]*?)"
                       r"(?=\b(?:extends|implements|with)\b|[{;]|$)")
#: Which relation a clause keyword names. `extends` is the only one that
#: extends; `implements` and Scala's `with` satisfy without extending.
_CLAUSE_MEANS = {"extends": INHERITS, "implements": MIXES_IN, "with": MIXES_IN}
#: C++ and C# write the bases after a colon, immediately following the
#: declared name. Anchored there on purpose: a colon anywhere else is a
#: type annotation or a label, and reading those would invent bases.
_AFTER_COLON = re.compile(r"\s*:\s*([^{;]+)")
#: Specifiers a C++ base list carries, which name no type.
_SPECIFIERS = frozenset("public private protected internal virtual override sealed".split())

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


def _cited(text: str) -> list[str]:
    """The decision records a line names, one node name per document.

    The number is normalised, so ``ADR-0007``, ``ADR 7`` and ``adr#7`` are
    one node rather than three. Without it the graph holds a document per
    spelling, and reaching every declaration that cites one -- the whole
    reason to put a citation in a graph -- stops working.
    """
    return [f"{tag.upper()}-{int(number)}" for tag, number in _CITED.findall(text)]


def masked(content: str, *, prose: bool = True) -> str:
    """The source with text that is not code blanked, in place.

    A regex over raw source cannot tell code from data, and there are two
    different things to hide depending on what is being looked for:

    - ``prose=True`` blanks **string literals** and keeps comments. That
      is what rationale and citations are read from: they live in
      comments, while a string holding ``# WHY: ...`` is data.
    - ``prose=False`` blanks **strings and comments both**. That is what
      declarations and inheritance are read from, because a commented-out
      ``// class Fake extends Invented {}`` is not a class, and an edge
      invented from one is the fabrication this module exists to avoid.

    Either way the blanking is in place, so every line and column stays
    where it was and line numbers, offsets and quotes are unaffected.
    Comments are recognised before strings, which is what makes
    ``// "unclosed`` safe.
    """
    out: list[str] = []
    quote = comment = ""
    at = 0
    while at < len(content):
        here, pair = content[at], content[at:at + 2]
        if comment:
            if comment == "//" and here == "\n":
                comment = ""
                out.append(here)
            elif comment == "/*" and pair == "*/":
                comment = ""
                out.append(pair if prose else "  ")
                at += 2
                continue
            else:
                out.append(here if prose or here == "\n" else " ")
        elif quote:
            if here == "\\" and at + 1 < len(content):
                out.append("  " if content[at + 1] != "\n" else " \n")
                at += 2
                continue
            out.append(here if here == quote or here == "\n" else " ")
            if here == quote:
                quote = ""
        elif pair in ("//", "/*"):
            comment = pair
            out.append(pair if prose else "  ")
            at += 2
            continue
        else:
            if here in "\"'`":
                quote = here
            out.append(here)
        at += 1
    return "".join(out)


def _holder(declared: tuple, path: str, line: int) -> str:
    """The innermost declaration a line sits in, or the file itself.

    Rationale belongs to the thing it explains. A note above a method is
    about that method, not about the file that happens to contain it.
    """
    inside = [d for d in declared if d.first_line <= line <= d.last_line]
    if not inside:
        return path
    return f"{path}:{max(inside, key=lambda d: d.first_line).name}"


def _tagged(text: str, path: str, line: int, declared: tuple,
            say: "Callable[[str, str, str, int], None]") -> None:
    """Read one piece of prose for rationale, problems and citations."""
    note = _WHY.search(text)
    if note:
        say(_holder(declared, path, line), NOTES, note.group(2), line)
    flagged = _FLAG.search(text)
    if flagged:
        say(_holder(declared, path, line), FLAGS, flagged.group(2), line)
    for document in _cited(text):
        say(_holder(declared, path, line), CITES, document, line)


def _python_prose(content: str) -> list[tuple[int, str]]:
    """Every comment and docstring, with its line. Nothing else.

    Comments come from ``tokenize`` rather than a regex over the source,
    because a regex cannot tell a comment from a string that looks like
    one, and asserting a claim the code never made is the one thing this
    must not do. Docstrings come from the tree: a docstring is
    documentation and belongs here, an arbitrary string is data and does
    not.
    """
    found: list[tuple[int, str]] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(content).readline):
            if token.type == tokenize.COMMENT:
                found.append((token.start[0], token.string))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return []
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, RecursionError):
        return found
    # Read from the file's own lines, never from `ast.get_docstring`. That
    # returns cleaned, escape-decoded text, so counting its lines drifts
    # from the source: a citation on the third line of a docstring pointed
    # at the opening quotes, and one after an escaped newline pointed at
    # the statement below. A claim citing the wrong line is a citation
    # nobody can check.
    lines = content.split("\n")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            continue
        last = getattr(first, "end_lineno", first.lineno) or first.lineno
        for number in range(first.lineno, min(last, len(lines)) + 1):
            found.append((number, lines[number - 1]))
    return found


def _meaning(content: str, path: str, language: str,
             say: "Callable[[str, str, str, int], None]") -> None:
    """Why the code is the way it is, and what it says it follows.

    Only tagged comments become claims -- an untagged line is a remark,
    not a statement about the code -- and a citation is read from a
    comment or a docstring, which is where people put them. Never from a
    string literal: that is data, and reading it would have the graph
    assert something the source never said.
    """
    from .code import declarations

    declared = declarations(content, language=language)  # type: ignore[arg-type]
    if language == "python":
        for at, prose in _python_prose(content):
            _tagged(prose, path, at, declared, say)
        return
    for number, line in enumerate(masked(content).split("\n"), start=1):
        _tagged(line, path, number, declared, say)


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
    imported: dict[str, tuple[str, Optional[str]]] = {}

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
                if isinstance(child, ast.ClassDef):
                    for base in child.bases:
                        target = _base(base, path, named, imported)
                        if target:
                            say(whole, INHERITS, target, child.lineno)
                walk(child, whole, name)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for module in _imported(child, path, resolve):
                    say(path, IMPORTS, module, child.lineno)
                # One entry per alias, keeping the name it had in its
                # module. Pairing every alias with every module of a
                # multi-name import made `import json, csv` claim both
                # names came from the last one, and using the local alias
                # made `Store as Shelf` resolve to Shelf.
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        imported[alias.asname or alias.name] = (alias.name, None)
                elif child.module:
                    # One module, several names taken out of it.
                    came = _imported(child, path, resolve)
                    for alias in child.names:
                        if came:
                            imported[alias.asname or alias.name] = (came[0], alias.name)
                elif resolve is not None and child.level:
                    # "from . import alpha, beta" names a module per alias.
                    # Reusing the first resolved module for all of them sent
                    # `beta.Parent` to alpha's file.
                    for alias in child.names:
                        where = resolve(path, child.level, alias.name)
                        if where:
                            imported[alias.asname or alias.name] = (where, None)
            else:
                walk(child, owner, inside)

    walk(tree, path, None)
    _calls(tree, path, named, say)
    _meaning(content, path, "python", say)
    return tuple(found)


def _base(node: ast.AST, path: str, named: dict[str, str],
          imported: dict[str, tuple[str, Optional[str]]]) -> Optional[str]:
    """What a base class names, and where it lives when that is knowable.

    A base defined in this file is named by its path, like any other
    declaration. One that arrived through an import is named by the module
    it came from and **the name it had there**, not by whatever this file
    calls it locally: `Store as Shelf` is Store. Anything else is recorded
    as the source wrote it -- the same rule imports follow, because the
    name is what the file said even when its home is unknown. A base that
    is not a plain name (a subscripted generic, a call) is left out rather
    than guessed at.
    """
    written = _written(node)
    if written is None:
        return None
    if written in named:
        return named[written]
    stem, _, last = written.rpartition(".")
    if written in imported:
        module, inside = imported[written]
        return _joined(module, inside or written)
    if stem and stem in imported:
        module, _inside = imported[stem]
        return _joined(module, last)
    return written


def _written(node: ast.AST) -> Optional[str]:
    """A base as the source spelled it, or None when it is not a name."""
    if isinstance(node, ast.Name):
        return node.id
    if not isinstance(node, ast.Attribute):
        return None
    parts: list[str] = []
    here: ast.AST = node
    while isinstance(here, ast.Attribute):
        parts.append(here.attr)
        here = here.value
    if not isinstance(here, ast.Name):
        return None
    parts.append(here.id)
    return ".".join(reversed(parts))


def _joined(module: str, name: str) -> str:
    """A module and a name in it. A module resolved to a file is joined
    with a colon, the way every declaration here is named; a module that is
    only a dotted name keeps the dotted form."""
    if module.endswith((".py", ".pyi")):
        return f"{module}:{name}"
    return f"{module}.{name}" if module != name else module


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

    held: dict[str, str] = {}
    for item in declarations(content, language="braces"):
        owner = path if "." not in item.name else f"{path}:{item.name.rsplit('.', 1)[0]}"
        # A Rust `impl Shelf` block is a good place to cut a chunk and not
        # a definition of Shelf: the type is defined by its struct, enum or
        # trait, here or in another file. Emitting a define for the impl
        # too made one type look like two. `declarations()` is right to
        # report it -- chunking wants the boundary -- so the distinction
        # belongs here, where the claim is made.
        if path.endswith(".rs") and _at_impl(lines, item.first_line):
            held.setdefault(item.name.rsplit(".", 1)[-1], f"{path}:{item.name}")
            continue
        say(owner, DEFINES, f"{path}:{item.name}", item.first_line)
        held[item.name.rsplit(".", 1)[-1]] = f"{path}:{item.name}"

    # What a class is built on, read from the header line. These languages
    # write it where it can be read; what a name in the body refers to is
    # not written down, and is still not guessed at.
    code = masked(content, prose=False).split("\n")
    if path.endswith(".go"):
        # Go's types, which the brace declaration pattern cannot see.
        for number, line in enumerate(code, start=1):
            named = _GO_TYPE.match(line)
            if named and named.group(1) not in held:
                whole = f"{path}:{named.group(1)}"
                held[named.group(1)] = whole
                say(path, DEFINES, whole, number)
    if path.endswith(".rs"):
        for number, line in enumerate(code, start=1):
            impl = _RUST_IMPL.match(line)
            if not impl:
                continue
            first, second = impl.group(1), impl.group(2)
            # `impl Trait for Type` says the type implements the trait.
            # `impl Type` is an inherent block on a type already defined,
            # so it is neither an edge nor a second definition.
            if second is None:
                continue
            subject = held.get(second, f"{path}:{second}")
            say(subject, MIXES_IN, held.get(first, first), number)
    for number, line in enumerate(code, start=1):
        declares = _DECLARES.search(line)
        if not declares:
            continue
        child = declares.group(1)
        subject = held.get(child, f"{path}:{child}")
        # `None` means the source did not say which relation it is, which
        # only the colon form leaves open.
        clauses: list[tuple[str | None, str]] = [
            (_CLAUSE_MEANS[clause.group(1).lower()], clause.group(2))
            for clause in _BUILT_ON.finditer(line, declares.end())]
        colon = _AFTER_COLON.match(line, declares.end())
        if colon:
            # After a colon the line does not say which is which, except in
            # one real way: Kotlin constructs its superclass and does not
            # construct an interface, so `Base(), Store` distinguishes
            # them. C++ has no interfaces, so a colon base there extends.
            clauses.append((None, colon.group(1)))
        for means, clause in clauses:
            for base in clause.split(","):
                # Kotlin and Scala write `: Base(), Store` and
                # `extends Base(3)`. Keeping the call would make `Base()`
                # and `Base` two entities, and a graph with both cannot
                # answer a question about either.
                constructed = "(" in base
                written = base.strip().split("<")[0].split("(")[0].strip().rstrip("{").strip()
                # A base list may lead with specifiers that name no type.
                words = [word for word in written.split() if word.lower() not in _SPECIFIERS]
                written = words[-1] if words else ""
                if not written or not (written[0].isalpha() or written[0] == "_"):
                    continue
                # A colon clause says less than a keyword does, and how
                # much less depends on the language.
                #
                # Kotlin constructs its superclass and does not construct
                # an interface, so `Base(), Store` really does distinguish
                # them. C++ has no interfaces at all, so `: public Base` is
                # inheritance and reading it as a mixin would be a new
                # error. Everywhere else a colon list is genuinely
                # ambiguous, and inheritance is what it most often is --
                # recorded with that limitation stated rather than guessed
                # at silently.
                if means is not None:
                    relation = means
                elif path.endswith((".kt", ".kts")):
                    relation = INHERITS if constructed else MIXES_IN
                else:
                    relation = INHERITS
                say(subject, relation, held.get(written, written), number)

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
    _meaning(content, path, "braces", say)
    return tuple(found)


def _named(module: str, path: str, resolve: Optional["Resolve"]) -> Optional[str]:
    """An import as it can be named. One written relative to this file is
    left to whoever knows the tree, because this file cannot say what it
    points at; anything else is a package, and names itself."""
    if not module.startswith("."):
        return module
    return resolve(path, 1, module) if resolve is not None else None


def _at_impl(lines: list[str], line: int) -> bool:
    """Whether this declaration is a Rust ``impl`` block rather than a type."""
    if not 1 <= line <= len(lines):
        return False
    return _RUST_IMPL.match(lines[line - 1]) is not None
