"""Declarations and calls read from a syntax tree, for the brace languages a grammar pack knows.

``code_syntax.py`` binds a TypeScript call to the one declaration or
import it can mean, with a parser under it, and says nothing about the
rest. This does the same for the other brace languages -- Go, Rust,
Java, C#, Swift, C, C++, Scala, PHP -- through the grammars of the
``code-languages`` extra, one table per language saying which nodes
declare a name, which open a scope, which bind a parameter or a local,
and what a call looks like. The rule is the TypeScript one: a call to a
bare name is an edge when the name is declared by this file at its top
or in the type that holds the caller, and nothing nearer binds it; a
call through a receiver (``x.y()``, ``T::f()``) needs a type nobody
here has, and a name a parameter or a local shadows is a value, not a
declaration. Where the grammar speaks it decides, and only about what
it speaks about: the line reader keeps imports, inheritance and the
rationale it reads out of comments.

Without the extra ``available()`` is False, nothing here runs, and the
graph is exactly what the line reader made of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

from .code_graph import CALLS, DEFINES, CodeClaim

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tree_sitter import Node

#: Lines a file may run to before the grammar is not asked.
MAX_LINES = 200_000
#: Depth a tree is walked to; a file nested deeper is left to the line reader.
MAX_DEPTH = 400
class _Cut(Exception):
    """The walk reached the depth bound: the grammar says nothing of this file."""


#: Leaves that carry a name as written.
_LEAVES = frozenset({"identifier", "type_identifier", "field_identifier", "name", "simple_identifier",
                     "namespace_identifier", "property_identifier"})


@dataclass(frozen=True)
class Grammar:
    """One language's shape, as its grammar names things."""

    pack: str
    #: Node types that declare a name, with where the name is: a field
    #: (``name``), ``declarator`` for the C family's nested declarators
    #: (``method`` when only a function's declarator counts),
    #: ``receiver`` for a Go method (named by its receiver's type), ``holder``
    #: for a Rust ``impl`` (its methods belong to the type; it declares
    #: nothing), or the kind itself for what has no name of its own
    #: (Swift's ``init``, ``deinit``, ``subscript``).
    declares: Mapping[str, str]
    #: Scopes by node type: ``type`` (a body whose declarations a call may
    #: name, qualified by the holder), ``function`` (its parameters bind),
    #: ``block``.
    scopes: Mapping[str, str]
    #: Nodes binding a parameter, with where the name is.
    parameters: Mapping[str, str]
    #: Nodes binding a local, with where the name is.
    locals: Mapping[str, str]
    #: The call node, the field its callee sits in (``""`` for Java's
    #: ``method_invocation``, whose bare call has no ``object``), and the
    #: leaves a bare callee may be.
    call: str
    callee: str
    bare: frozenset[str] = frozenset({"identifier"})
    #: Parents whose ``type`` body holds members a bare call cannot name
    #: from inside: a Rust ``impl`` or ``trait`` (a sibling method is
    #: ``Self::f`` or ``self.f``), a PHP class (``$this->f``, ``self::f``).
    #: Empty means every type body's members are callable bare, as in
    #: Java, C#, Swift, Scala and C++; ``*`` means all of them.
    members_of: frozenset[str] = frozenset()
    #: Whether a method defined outside its type by a qualified name
    #: (C++'s ``void K::t() { s(); }``) sees its type's other members
    #: bare, as the language says it does; a Go method or a Rust impl
    #: never does.
    siblings_bare: bool = True


def _c_family(pack: str) -> Grammar:
    """C and C++ share a shape: declarators nest, bodies are compound
    statements, and a class body is a field declaration list."""
    return Grammar(
        pack=pack,
        declares={"function_definition": "declarator", "field_declaration": "method", "struct_specifier": "name",
                  "class_specifier": "name", "union_specifier": "name", "enum_specifier": "name",
                  "type_definition": "declarator"},
        scopes={"field_declaration_list": "type", "declaration_list": "type", "function_definition": "function",
                "lambda_expression": "function", "compound_statement": "block"},
        parameters={"parameter_declaration": "declarator", "optional_parameter_declaration": "declarator"},
        locals={"declaration": "declarator"},
        call="call_expression", callee="function")


GRAMMARS: dict[str, Grammar] = {
    "go": Grammar(
        pack="go",
        declares={"function_declaration": "name", "method_declaration": "receiver", "type_spec": "name",
                  "type_alias": "name", "method_elem": "name"},
        scopes={"function_declaration": "function", "method_declaration": "function", "func_literal": "function",
                "interface_type": "type", "block": "block"},
        parameters={"parameter_declaration": "name", "variadic_parameter_declaration": "name"},
        locals={"short_var_declaration": "left", "var_spec": "name", "const_spec": "name", "range_clause": "left"},
        call="call_expression", callee="function", siblings_bare=False),
    "rust": Grammar(
        pack="rust",
        declares={"function_item": "name", "function_signature_item": "name", "struct_item": "name",
                  "enum_item": "name", "union_item": "name", "trait_item": "name", "mod_item": "name",
                  "type_item": "name", "const_item": "name", "static_item": "name", "impl_item": "holder"},
        scopes={"declaration_list": "type", "function_item": "function", "closure_expression": "function",
                "block": "block"},
        parameters={"parameter": "pattern", "closure_parameters": "pattern"},
        locals={"let_declaration": "pattern"},
        call="call_expression", callee="function", members_of=frozenset({"impl_item", "trait_item"}),
        siblings_bare=False),
    "java": Grammar(
        pack="java",
        declares={"class_declaration": "name", "interface_declaration": "name", "enum_declaration": "name",
                  "record_declaration": "name", "annotation_type_declaration": "name",
                  "method_declaration": "name", "constructor_declaration": "name"},
        scopes={"class_body": "type", "interface_body": "type", "enum_body": "type",
                "method_declaration": "function", "constructor_declaration": "function",
                "lambda_expression": "function", "block": "block"},
        parameters={"formal_parameter": "name", "spread_parameter": "declarator", "inferred_parameters": "pattern"},
        locals={"variable_declarator": "name", "catch_formal_parameter": "name", "enhanced_for_statement": "name"},
        call="method_invocation", callee=""),
    "c_sharp": Grammar(
        pack="c_sharp",
        declares={"class_declaration": "name", "struct_declaration": "name", "interface_declaration": "name",
                  "enum_declaration": "name", "record_declaration": "name",
                  "method_declaration": "name", "constructor_declaration": "name",
                  "local_function_statement": "name"},
        scopes={"declaration_list": "type", "method_declaration": "function", "constructor_declaration": "function",
                "local_function_statement": "function", "lambda_expression": "function",
                "anonymous_method_expression": "function", "block": "block"},
        parameters={"parameter": "name", "implicit_parameter": "self"},
        locals={"variable_declarator": "name", "catch_declaration": "name", "foreach_statement": "left"},
        call="invocation_expression", callee="function"),
    "swift": Grammar(
        pack="swift",
        declares={"function_declaration": "name", "protocol_function_declaration": "name",
                  "class_declaration": "name", "protocol_declaration": "name", "typealias_declaration": "name",
                  "init_declaration": "init", "deinit_declaration": "deinit", "subscript_declaration": "subscript"},
        scopes={"class_body": "type", "protocol_body": "type", "enum_class_body": "type",
                "function_declaration": "function", "init_declaration": "function", "deinit_declaration": "function",
                "subscript_declaration": "function", "lambda_literal": "function", "function_body": "block",
                "statements": "block"},
        parameters={"parameter": "name", "lambda_parameter": "name"},
        locals={"property_declaration": "pattern"},
        call="call_expression", callee="", bare=frozenset({"simple_identifier"})),
    "c": _c_family("c"),
    "cpp": _c_family("cpp"),
    "scala": Grammar(
        pack="scala",
        declares={"function_definition": "name", "class_definition": "name", "object_definition": "name",
                  "trait_definition": "name", "enum_definition": "name"},
        # A class's own parameters are in scope through its body: the
        # class opens a scope that holds them, and its body another.
        scopes={"template_body": "type", "class_definition": "type", "trait_definition": "type",
                "enum_definition": "type", "function_definition": "function", "lambda_expression": "function",
                "block": "block"},
        parameters={"parameter": "name", "class_parameter": "name"},
        locals={"val_definition": "pattern", "var_definition": "pattern"},
        call="call_expression", callee="function"),
    "php": Grammar(
        pack="php",
        declares={"function_definition": "name", "method_declaration": "name", "class_declaration": "name",
                  "interface_declaration": "name", "trait_declaration": "name", "enum_declaration": "name"},
        scopes={"declaration_list": "type", "function_definition": "function", "method_declaration": "function",
                "anonymous_function": "function", "arrow_function": "function", "compound_statement": "block"},
        parameters={}, locals={},
        call="function_call_expression", callee="function", bare=frozenset({"name"}), members_of=frozenset({"*"}),
        siblings_bare=False),
}
#: Which grammar reads a suffix.
SUFFIXES: dict[str, str] = {
    ".go": "go", ".rs": "rust", ".java": "java", ".cs": "c_sharp", ".swift": "swift", ".scala": "scala",
    ".php": "php", ".c": "c", ".h": "cpp", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
}


@lru_cache(maxsize=1)
def available() -> bool:
    """Whether the ``code-languages`` extra is installed."""
    try:
        import tree_sitter_language_pack  # noqa: F401
    except Exception:  # pragma: no cover - the extra is simply absent
        return False
    return True


def _text(node: "Node", source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _line(node: "Node") -> int:
    return node.start_point[0] + 1


def _leaf(node: Optional["Node"], source: bytes) -> Optional[str]:
    """The name a node carries: a leaf's text, a qualified name's parts
    joined with dots (`K::m` is `K.m`), a pointer's or generic's base."""
    if node is None:
        return None
    if node.type in _LEAVES or node.type in ("destructor_name", "operator_name"):
        # `~K` and `operator==` are names as written.
        return _text(node, source)
    if node.type == "template_function":
        return _leaf(node.child_by_field_name("name"), source)
    if node.type in ("qualified_identifier", "scoped_identifier", "scoped_type_identifier"):
        scope = _leaf(node.child_by_field_name("scope") or node.child_by_field_name("path"), source)
        name = _leaf(node.child_by_field_name("name"), source)
        return f"{scope}.{name}" if scope and name else name
    if node.type in ("pointer_type", "reference_type", "generic_type", "user_type", "parenthesized_declarator",
                     "reference_declarator", "pointer_declarator", "attributed_declarator"):
        for child in node.children:
            found = _leaf(child, source)
            if found:
                return found
    return None


def _declared(node: "Node", how: str, source: bytes) -> Optional[str]:
    """The name ``node`` declares, found the way ``how`` says."""
    if how == "name":
        return _leaf(node.child_by_field_name("name"), source)
    if how in ("declarator", "method"):
        # A method declared in a C++ class body is a field declaration
        # whose declarator is a function's; a plain field is not a declaration.
        inner: Optional["Node"] = node
        callable_ = False
        while inner is not None and inner.type not in _LEAVES:
            if inner.type in ("qualified_identifier", "scoped_identifier", "destructor_name", "operator_name",
                              "template_function"):
                break
            callable_ = callable_ or inner.type == "function_declarator"
            inner = inner.child_by_field_name("declarator") or next(
                (child for child in inner.children if child.type.endswith("_declarator")), None)
        return _leaf(inner, source) if how == "declarator" or callable_ else None
    if how == "receiver":
        name = _leaf(node.child_by_field_name("name"), source)
        receiver = node.child_by_field_name("receiver")
        owner = None
        if receiver is not None:
            for child in receiver.children:
                if child.type == "parameter_declaration":
                    owner = _leaf(child.child_by_field_name("type"), source)
        return f"{owner}.{name}" if owner and name else name
    if how == "holder":
        return _leaf(node.child_by_field_name("type"), source)
    if how in ("init", "deinit", "subscript"):
        return how
    return None


def _bound(node: "Node", how: str, source: bytes) -> list[str]:
    """The names a parameter or local binds: one from a field, or every
    identifier in a pattern (`let (a, b) = t`, `for x, y := range`)."""
    if how == "self":
        return [_text(node, source)]
    if how == "name":
        # `var a, b int` and `func f(a, b int)` tag every name with the field.
        return [text for child in node.children_by_field_name("name") if (text := _leaf(child, source))]
    if how == "declarator":
        # `int a, b;` and `void (*a)(), (*b)();` tag every declarator.
        declared: list[str] = []
        for child in node.children_by_field_name("declarator") or [node]:
            found = _declared(child, "declarator", source) if child is not node else _declared(node, how, source)
            if found:
                declared.append(found)
        return declared
    where = node.child_by_field_name(how) if how != "pattern" else (node.child_by_field_name("pattern") or node)
    if where is None:
        return []
    names: list[str] = []

    def leaves(inner: "Node") -> None:
        if inner.type in ("identifier", "simple_identifier", "bound_identifier"):
            names.append(_text(inner, source))
        elif inner.type in ("call_expression", "lambda_literal", "closure_expression"):
            return
        else:
            for child in inner.children:
                leaves(child)

    leaves(where)
    return names


def _within(node: "Node", grammar: Grammar, depth: int = 0):
    """Every node of ``node``'s subtree that belongs to ``node``'s scope,
    to the depth bound: an expression nested past it cuts the file."""
    if depth > MAX_DEPTH:
        raise _Cut()
    for child in node.children:
        yield child
        if child.type not in grammar.scopes:
            yield from _within(child, grammar, depth + 1)


def _qualified(holder: Optional[str], path: str, name: str) -> str:
    return f"{holder}.{name}" if holder else f"{path}:{name}"


def _callee(node: "Node", grammar: Grammar) -> Optional["Node"]:
    """The bare name a call names, or None for a call through a receiver."""
    if grammar.callee:
        callee = node.child_by_field_name(grammar.callee)
    elif grammar.pack == "java":
        callee = None if node.child_by_field_name("object") is not None else node.child_by_field_name("name")
    else:
        callee = next((child for child in node.children if child.is_named), None)
    return callee if callee is not None and callee.type in grammar.bare else None


@dataclass
class _Types:
    """Each type's members by bare name, as its body declared them and as
    definitions outside it (`void K::s()`) added them, so a method defined
    outside its type still sees its siblings."""
    members: dict[str, dict[str, str]] = field(default_factory=dict)

    def note(self, name: str, whole: str) -> None:
        if "." in name:
            owner, member = name.rsplit(".", 1)
            self.members.setdefault(owner.rsplit(".", 1)[-1], {}).setdefault(member, whole)

    def of(self, holder: Optional[str]) -> dict[str, str]:
        """The members of the type a qualified holder belongs to."""
        if holder is None or "." not in holder.split(":", 1)[-1]:
            return {}
        return self.members.get(holder.split(":", 1)[-1].rsplit(".", 1)[0].rsplit(".", 1)[-1], {})


def _walk(node: "Node", grammar: Grammar, source: bytes, path: str, stack: list[tuple[str, dict[str, Optional[str]]]],
          holder: Optional[str], seen: set[str], say: Callable[[str, str, str, int], None], depth: int,
          types: _Types) -> None:
    if depth > MAX_DEPTH:
        raise _Cut()
    kind = grammar.scopes.get(node.type)
    if node.parent is None:
        kind = "module"
    elif kind == "type" and ("*" in grammar.members_of or node.parent.type in grammar.members_of):
        kind = "members"
    if kind is not None:
        scope: dict[str, Optional[str]] = {}
        for child in _within(node, grammar):
            how = grammar.declares.get(child.type)
            if how is not None and how != "holder":
                name = _declared(child, how, source)
                if name:
                    whole = _qualified(holder, path, name)
                    if whole not in seen:
                        seen.add(whole)
                        say(holder or path, DEFINES, whole, _line(child))
                    # A declaration a call may name by its bare name: at the
                    # file's top, or in the type that holds the caller. Nearer,
                    # it is a value. A name qualified as written (a Go method,
                    # a C++ `K::m` defined outside its class) is not bare here,
                    # and a member of a body that is members only is named
                    # through its type, never bare.
                    if "." in name and grammar.siblings_bare:
                        types.note(name, whole)
                    if "." in name or kind == "members":
                        continue
                    scope.setdefault(name, whole if kind in ("module", "type") else None)
                    if kind == "type" and holder is not None and grammar.siblings_bare:
                        types.note(f"{holder.split(':', 1)[-1]}.{name}", whole)
            for table in (grammar.parameters, grammar.locals):
                bound = table.get(child.type)
                if bound is not None:
                    for name in _bound(child, bound, source):
                        scope.setdefault(name, None)
        stack.append((kind, scope))

    if node.type == grammar.call and holder is not None and not node.has_error:
        callee = _callee(node, grammar)
        if callee is not None:
            name = _text(callee, source)
            target: Optional[str] = None
            for kind_seen, one in reversed(stack):
                # A method defined outside its type by a qualified name sees
                # the type's other members before the file's top.
                if kind_seen == "module" and name in types.of(holder):
                    target = types.of(holder)[name]
                    break
                if name in one:
                    target = one[name]
                    break
            if target is not None and target != holder:
                say(holder, CALLS, target, _line(node))

    for child in node.children:
        mine = holder
        how = grammar.declares.get(child.type)
        if how is not None and how != "holder":
            # A declaration whose name the grammar does not give holds
            # nothing: a call inside it is nobody's, not the enclosing one's.
            name = _declared(child, how, source)
            mine = _qualified(holder, path, name) if name else None
        elif how == "holder":
            name = _declared(child, how, source)
            mine = _qualified(holder, path, name) if name else None
        _walk(child, grammar, source, path, stack, mine, seen, say, depth + 1, types)
    if kind is not None:
        stack.pop()


def grammar_claims(content: str, path: str) -> tuple[CodeClaim, ...]:
    """Declarations and calls this file's syntax settles, or nothing: an
    empty tuple when the extra is absent, the suffix has no grammar here,
    or the file is longer than the grammar is asked to read."""
    suffix = path[path.rfind("."):].lower() if "." in path.rsplit("/", 1)[-1] else ""
    if suffix not in SUFFIXES or not available():
        return ()
    lines = content.split("\n")
    if len(lines) > MAX_LINES:
        return ()
    from tree_sitter_language_pack import get_parser

    grammar = GRAMMARS[SUFFIXES[suffix]]
    try:
        tree = get_parser(grammar.pack).parse(content.encode())  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - a grammar the pack lacks
        return ()
    source = content.encode()
    found: list[CodeClaim] = []

    def say(subject: str, predicate: str, obj: str, number: int) -> None:
        if not 1 <= number <= len(lines):
            return
        quote = lines[number - 1].strip()
        begins = len("\n".join(lines[:number - 1]).encode()) + (1 if number > 1 else 0)
        found.append(CodeClaim(subject, predicate, obj, quote, number, begins, begins + len(lines[number - 1].encode())))

    try:
        _walk(tree.root_node, grammar, source, path, [], None, set(), say, 0, _Types())
    except (_Cut, RecursionError):
        # Past the depth bound the grammar has read only part of the file;
        # saying nothing leaves the line reader's whole answer standing.
        return ()
    return tuple(found)
