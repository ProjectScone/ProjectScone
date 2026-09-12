"""Calls and declarations read from a syntax tree, for the brace family.

The line reader in ``code_graph.py`` cannot bind a call and does not try:
binding one needs **lexical scope**, and a reader that sees masked lines
has none. Six shapes of false edge over three rounds of review said so,
and the brace call graph built on names was withdrawn.

This is the same capability with a parser under it. It claims only what a
syntax tree can settle, which is less than it sounds. Measured over 2,627
call sites in 60 files of this project's web application:

- **44% go through a receiver** (``x.y()``) and need the *type* of the
  thing on the left, which no grammar supplies. Left alone.
- **28% name something from outside the file** -- globals, JSX, values
  from a default import. Left alone.
- **15% name an import** and **5% name a declaration of the file**. These
  are what it records.
- **5% name a local or a parameter**, and are correctly *not* edges.
  That row is exactly what the line reader got wrong.

So a parser buys a fifth of the call sites. The honest claim is "sound
where it speaks", not "a call graph for TypeScript", and the module is
written to make the silence deliberate rather than accidental.

The parser is an optional extra (``scone-memory[code-graph]``). Without
it ``available()`` is False, nothing here runs, and the graph is exactly
what it was.
"""

from __future__ import annotations

from typing import Optional

from .code_graph import CALLS, DEFINES, CodeClaim, _named

try:  # pragma: no cover - exercised by whether the extra is installed
    from tree_sitter import Language, Node, Parser
    import tree_sitter_typescript as _typescript

    _TSX = Language(_typescript.language_tsx())
    _TS = Language(_typescript.language_typescript())
except Exception:  # pragma: no cover - the extra is simply absent
    _TSX = _TS = None  # type: ignore[assignment]

#: Extensions this reader knows a grammar for. A file outside it keeps the
#: line reader's answer rather than getting a worse one from the wrong
#: grammar.
GRAMMARS = {".ts": "ts", ".mts": "ts", ".cts": "ts", ".tsx": "tsx",
            ".js": "tsx", ".jsx": "tsx", ".mjs": "tsx", ".cjs": "tsx"}
#: Nodes that bind their own name in the scope around them.
_BINDS = frozenset({"function_declaration", "generator_function_declaration",
                    "class_declaration", "abstract_class_declaration", "method_definition"})
#: Scopes a `var` hoists to. Everything else is a block, which `let` and
#: `const` respect and `var` does not -- the distinction is the language's
#: and getting it wrong bound a caller to a global it never called.
_FUNCTIONS = frozenset({"function_declaration", "generator_function_declaration",
                        "method_definition", "arrow_function", "function_expression",
                        "program"})
#: Scopes that hold `let`, `const`, a parameter or a catch binding.
_BLOCKS = frozenset({"statement_block", "for_statement", "for_in_statement",
                     "catch_clause", "class_body"})
_OPENS = _FUNCTIONS | _BLOCKS
#: A declaration whose value is a function: `const keep = (p) => p`. Most
#: modern TypeScript writes functions this way and the line reader reports
#: none of them -- 80 in the 60 files first measured.
_FUNCTION_VALUES = frozenset({"arrow_function", "function_expression"})
#: Every shape a binding may take on the left of a parameter or a
#: declarator. Plain identifiers were all this knew, and destructuring, a
#: rest element and a default are ordinary in every file.
_PATTERNS = frozenset({"object_pattern", "array_pattern", "rest_pattern",
                       "assignment_pattern", "object_assignment_pattern", "pair_pattern"})


def available() -> bool:
    """Whether the code-graph extra is installed and its grammar loaded."""
    return _TSX is not None


def _text(node: "Node", source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _pattern(node: "Node", source: bytes) -> list[str]:
    """Every name a binding pattern binds.

    `{leaf}`, `[leaf]`, `...leaf`, `leaf = fallback` and the nesting of
    those are all ordinary, and a reader that only knew a bare identifier
    left each of them unbound -- so the caller stayed bound to whatever
    the name meant outside, which it does not mean inside.
    """
    if node.type == "identifier":
        return [_text(node, source)]
    if node.type in ("shorthand_property_identifier_pattern",
                     "shorthand_property_identifier"):
        return [_text(node, source)]
    if node.type not in _PATTERNS:
        return []
    found: list[str] = []
    for child in node.children:
        # A pair binds its **value**: `{written: local}` binds `local`.
        if node.type == "pair_pattern":
            value = node.child_by_field_name("value")
            return _pattern(value, source) if value is not None else []
        if node.type == "assignment_pattern" and child == node.child_by_field_name("right"):
            continue  # the default is an expression, not a binding
        found.extend(_pattern(child, source))
    return found


def _declares(node: "Node", source: bytes) -> list[str]:
    """The names ``node`` binds in the scope around it."""
    if node.type in _BINDS:
        name = node.child_by_field_name("name")
        return [_text(name, source)] if name is not None else []
    if node.type == "variable_declarator":
        name = node.child_by_field_name("name")
        return _pattern(name, source) if name is not None else []
    if node.type in ("required_parameter", "optional_parameter"):
        pattern = node.child_by_field_name("pattern")
        return _pattern(pattern, source) if pattern is not None else []
    if node.type in _PATTERNS or node.type == "identifier":
        return _pattern(node, source)
    return []


def _line(node: "Node", source: bytes) -> int:
    return source[:node.start_byte].count(b"\n") + 1


def _imports(tree: "Node", source: bytes, path: str) -> dict[str, str]:
    """What each imported name is, as ``local name -> file:name``.

    Only a named import is followed. A default or namespace binding names
    a value whose members are reached through it, and a call through one
    of those is a call through a receiver.
    """
    found: dict[str, str] = {}
    for node in tree.children:
        if node.type != "import_statement":
            continue
        source_node = node.child_by_field_name("source")
        if source_node is None:
            continue
        written = _text(source_node, source).strip("\"'")
        # Relative, and nothing else. `motion/react-mini` contains a
        # slash and is not a path, so testing for one admitted every
        # scoped package subpath as a file of this graph.
        if not written.startswith("."):
            continue
        where = _named(written, path, None)
        if where is None:
            continue
        for clause in (one for one in node.children if one.type == "import_clause"):
            for named in (one for one in clause.children if one.type == "named_imports"):
                for item in (one for one in named.children if one.type == "import_specifier"):
                    original = item.child_by_field_name("name")
                    alias = item.child_by_field_name("alias") or original
                    if original is not None and alias is not None:
                        found[_text(alias, source)] = f"{where}:{_text(original, source)}"
    return found


def _through(node: "Node"):
    """``node``'s children, looking through an `export_statement`.

    `export function leaf() {}` wraps the declaration, so a scan of a
    scope's direct children walks straight past it -- and a call to a
    function declared later bound when the callee was local and vanished
    when it was exported. The same code, read two ways, by a keyword.
    """
    for child in node.children:
        # Recursive, because the wrappers nest: an `export_statement`
        # holds a `lexical_declaration` which holds the declarators.
        # Expanding one layer found `export function` and still missed
        # `export const keep = () => …`, which is most of them.
        if child.type in ("export_statement", "lexical_declaration", "variable_declaration"):
            yield from _through(child)
        else:
            yield child


def _declared(node: "Node") -> Optional[str]:
    """The qualified-name suffix ``node`` declares, if it declares one."""
    if node.type in _BINDS:
        return node.type
    if node.type == "variable_declarator":
        value = node.child_by_field_name("value")
        return node.type if value is not None and value.type in _FUNCTION_VALUES else None
    return None


def _bindings(scope: "Node", source: bytes, function: bool):
    """Every name bound directly in ``scope``, with the node that binds it.

    ``var`` is collected only by a function scope, and is looked for
    through every block inside it, because that is where JavaScript puts
    it. ``let``, ``const``, parameters and catch bindings belong to the
    block that wrote them.
    """
    for child in _through(scope):
        for name in _declares(child, source):
            yield name, child
    for child in scope.children:
        if child.type == "formal_parameters":
            for one in child.children:
                for name in _declares(one, source):
                    yield name, one
    # `p => p()` has no formal_parameters: the parameter is a field.
    parameter = scope.child_by_field_name("parameter")
    if parameter is not None:
        for name in _pattern(parameter, source):
            yield name, parameter
    if function:
        for name, node in _hoisted(scope, source):
            yield name, node


def _hoisted(node: "Node", source: bytes):
    """`var` declared anywhere inside, but not past another function."""
    for child in node.children:
        if child.type in _FUNCTIONS and child is not node:
            continue
        if child.type == "variable_declaration":  # `var`, never let/const
            for one in child.children:
                for name in _declares(one, source):
                    yield name, one
        if child.type == "export_statement":
            for one in child.children:
                if one.type == "variable_declaration":
                    for two in one.children:
                        for name in _declares(two, source):
                            yield name, two
        yield from _hoisted(child, source)


def _walk(node: "Node", source: bytes, path: str, stack: list[dict[str, Optional[str]]],
          imports: dict[str, str], holder: Optional[str], inside: Optional[str],
          seen: set[str], say) -> None:
    opened = node.type in _OPENS
    if opened:
        scope: dict[str, Optional[str]] = {}
        for name, binder in _bindings(node, source, node.type in _FUNCTIONS):
            # A declaration is a thing the graph can name; any other
            # binding is a value, and a call to it is nobody's edge.
            whole = None
            if _declared(binder) is not None:
                whole = f"{path}:{inside}.{name}" if inside else f"{path}:{name}"
                # Keyed on the claim, not on the node that made it: a
                # `var` is bound in its block and hoisted to its
                # function, so the same declaration is reached twice and
                # said twice -- 120 duplicate claims over 85 files.
                if whole not in seen:
                    seen.add(whole)
                    say(f"{path}:{inside}" if inside else path, DEFINES, whole,
                        _line(binder, source))
            scope.setdefault(name, whole)
        stack.append(scope)

    if node.type == "call_expression" and holder is not None and not node.has_error:
        callee = node.child_by_field_name("function")
        if callee is not None and callee.type == "identifier":
            name = _text(callee, source)
            target: Optional[str] = None
            for depth, one in enumerate(reversed(stack)):
                if name in one:
                    # Only the outermost scope holds this file's own
                    # declarations; anything nearer is a value.
                    target = one[name] if depth == len(stack) - 1 else None
                    break
            else:
                target = imports.get(name)
            if target is not None and target != holder:
                say(holder, CALLS, target, _line(node, source))

    for child in node.children:
        within, mine = inside, holder
        if _declared(child) is not None:
            name = next(iter(_declares(child, source)), None)
            if name is not None:
                mine = f"{path}:{inside}.{name}" if inside else f"{path}:{name}"
                if child.type in ("class_declaration", "abstract_class_declaration"):
                    within = f"{inside}.{name}" if inside else name
        _walk(child, source, path, stack, imports, mine, within, seen, say)
    if opened:
        stack.pop()


def syntax_claims(content: str, path: str) -> tuple[CodeClaim, ...]:
    """Declarations and calls this file's syntax settles, or nothing.

    Returns an empty tuple when the extra is absent or the file's
    extension has no grammar here, so a caller can always ask.
    """
    suffix = path[path.rfind("."):].lower() if "." in path.rsplit("/", 1)[-1] else ""
    if not available() or suffix not in GRAMMARS:
        return ()
    source = content.encode()
    lines = content.split("\n")
    found: list[CodeClaim] = []

    def say(subject: str, predicate: str, obj: str, number: int) -> None:
        if not 1 <= number <= len(lines):
            return
        quote = lines[number - 1].strip()
        begins = len("\n".join(lines[:number - 1]).encode()) + (1 if number > 1 else 0)
        found.append(CodeClaim(subject, predicate, obj, quote, number, begins,
                               begins + len(lines[number - 1].encode())))

    tree = Parser(_TSX if GRAMMARS[suffix] == "tsx" else _TS).parse(source)
    _walk(tree.root_node, source, path, [], _imports(tree.root_node, source, path),
          None, None, set(), say)
    return tuple(found)
