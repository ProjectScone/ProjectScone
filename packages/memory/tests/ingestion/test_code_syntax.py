"""Calls and declarations read from a real syntax tree.

The line reader could not bind a call, because binding one needs lexical
scope and a line reader has none -- six shapes of false edge over three
rounds of review said so, and the brace call graph was withdrawn. This is
the same capability done with a parser, behind an optional extra, and it
claims only what a syntax tree can settle.

Measured over 60 files of this project's web application, 2,627 call
sites: 44% go through a receiver and need **types**, which no grammar
supplies; 28% name something from outside the file. What is left, and
what this reads, is 15% naming an import and 5% naming a declaration of
the file -- plus the 5% that name a local or a parameter, which are
correctly **not** edges and are exactly what the line reader got wrong.
"""

from __future__ import annotations

import pytest

from scone_memory.ingestion import code_syntax

pytestmark = pytest.mark.skipif(not code_syntax.available(),
                                reason="the code-graph extra is not installed")


def claims(source: str, path: str = "web/app.ts"):
    return {(claim.subject, claim.predicate, claim.object)
            for claim in code_syntax.syntax_claims(source, path)}


def test_a_call_names_a_declaration_of_this_file():
    source = ("function helper(p) { return p; }\n"
              "export function drive(p) {\n"
              "  return helper(p);\n"
              "}\n")
    assert ("web/app.ts:drive", "calls", "web/app.ts:helper") in claims(source)


def test_a_parameter_shadows_the_function_it_is_named_after():
    """The case the line reader could never get right: `run(leaf)` calling
    `leaf(1)` calls its argument, and a scope stack knows that."""
    source = ("export function leaf(p) { return p; }\n"
              "export function run(leaf) {\n"
              "  return leaf(1);\n"
              "}\n")
    assert not [one for one in claims(source) if one[1] == "calls"]


def test_a_local_shadows_it_too():
    source = ("export function leaf(p) { return p; }\n"
              "export function run() {\n"
              "  const leaf = () => 2;\n"
              "  return leaf();\n"
              "}\n")
    assert not [one for one in claims(source) if one[1] == "calls"]


def test_a_method_is_not_reached_by_a_bare_name():
    """A method needs a receiver whose type nobody wrote down."""
    source = ("export class A {\n"
              "  save() { return 1; }\n"
              "  run() { return save(); }\n"
              "}\n")
    assert not [one for one in claims(source) if one[1] == "calls"]


def test_a_declaration_header_is_never_a_call_to_itself():
    source = ("export function save(p) { return p; }\n"
              "export class A {\n"
              "  save() {\n"
              "    return 1;\n"
              "  }\n"
              "}\n")
    assert not [one for one in claims(source) if one[1] == "calls"]


def test_an_arrow_const_is_a_declaration():
    """`export const keep = (p) => p` is how most modern TypeScript writes
    a function, and the line reader reports none of them -- 80 in those 60
    files, a silent undercount of `defines` whatever else is true."""
    source = "export const keep = (p: string) => p;\n"
    assert ("web/app.ts", "defines", "web/app.ts:keep") in claims(source)


def test_a_call_to_an_imported_name_reaches_the_file_it_came_from():
    source = ("import {keep, put as hold} from './store';\n"
              "export function drive(p) {\n"
              "  return keep(p) + hold(p);\n"
              "}\n")
    found = claims(source, "web/page.ts")
    assert ("web/page.ts:drive", "calls", "web/store.ts:keep") in found, found
    assert ("web/page.ts:drive", "calls", "web/store.ts:put") in found, found


def test_a_call_through_a_receiver_is_left_alone():
    source = ("import {store} from './store';\n"
              "export function drive(p) {\n"
              "  return store.keep(p);\n"
              "}\n")
    assert not [one for one in claims(source, "web/page.ts") if one[1] == "calls"]


def test_a_name_from_outside_the_file_is_left_alone():
    source = ("export function drive(p) {\n"
              "  return setTimeout(p, 1);\n"
              "}\n")
    assert not [one for one in claims(source) if one[1] == "calls"]


def test_a_package_subpath_is_not_a_file_of_this_graph():
    """`motion/react-mini` contains a slash and is not a path. Testing for
    a slash admitted every scoped package subpath as a file, so a call to
    `useAnimate` was recorded as reaching a file nothing could hold.

    Only an import written **relative** names a file here; that is what
    relative means and it is the check that cannot be fooled by a package
    whose name has a slash in it.
    """
    source = ("import {useAnimate} from 'motion/react-mini';\n"
              "import {keep} from './store';\n"
              "export function drive(p) {\n"
              "  return useAnimate(p) + keep(p);\n"
              "}\n")
    found = claims(source, "web/page.ts")
    assert ("web/page.ts:drive", "calls", "web/store.ts:keep") in found, found
    assert not any("useAnimate" in one[2] for one in found if one[1] == "calls"), found


def test_the_two_readers_never_disagree_about_who_owns_a_declaration():
    """Both readers run for a brace file and their claims are merged, so
    a declaration they both see must be attributed to the same owner --
    otherwise one file's `Shelf.keep` becomes two things and the graph
    holds a disagreement rather than a fact.

    Checked across this project's whole web application as well: 199
    declarations seen by both, and **no** case where they named different
    owners for one of them. The parser sees more; it never sees different.
    """
    from scone_memory.ingestion.code_graph import _brace_claims

    source = ("import {helper} from './helper';\n"
              "export function drive(p) {\n"
              "  return helper(p);\n"
              "}\n"
              "export class Shelf {\n"
              "  keep(p) {\n"
              "    return p;\n"
              "  }\n"
              "}\n"
              "export const hold = (p) => p;\n")
    line = {(c.subject, c.object) for c in _brace_claims(source, "web/app.ts", None)
            if c.predicate == "defines"}
    tree = {(c.subject, c.object) for c in code_syntax.syntax_claims(source, "web/app.ts")
            if c.predicate == "defines"}
    assert line and tree, (line, tree)
    assert line & tree, "this fixture needs a declaration both readers see"
    owners = {obj: subject for subject, obj in line}
    for subject, obj in tree:
        if obj in owners:
            assert owners[obj] == subject, (obj, owners[obj], subject)
    # And the parser sees the arrow const the line reader cannot.
    assert ("web/app.ts", "web/app.ts:hold") in tree - line, tree - line


@pytest.mark.parametrize("source, why", [
    ("export function leaf() { return 1; }\n"
     "export function caller({leaf}: {leaf: () => number}) { return leaf(); }\n",
     "a destructured object parameter"),
    ("export function leaf() { return 1; }\n"
     "export function caller([leaf]: Array<() => number>) { return leaf(); }\n",
     "a destructured array parameter"),
    ("export function leaf() { return 1; }\n"
     "export const caller = leaf => leaf();\n",
     "an arrow parameter written without brackets"),
    ("export function leaf() { return 1; }\n"
     "export function caller() { try { throw null; } catch (leaf) { return leaf(); } }\n",
     "a catch parameter"),
    ("export function leaf() { return 1; }\n"
     "export function caller() { if (true) { var leaf = () => 2; } return leaf(); }\n",
     "a var declared in a nested block, which JavaScript hoists to the function"),
    ("export function leaf(p) { return p; }\n"
     "export function caller() { const {leaf} = deps; return leaf(); }\n",
     "a destructured local"),
    ("export function leaf(p) { return p; }\n"
     "export function caller(...leaf) { return leaf(); }\n",
     "a rest parameter"),
])
def test_every_way_a_name_can_be_bound_shadows_the_declaration_beside_it(source, why):
    """Every one of these was a false edge.

    A scope stack is only as good as its idea of what binds a name, and
    mine knew about plain parameters and `const`. Destructuring, a
    bracketless arrow parameter, `catch`, rest, and `var`'s hoisting to
    the **function** rather than the block are all ordinary JavaScript
    and all of them left the caller bound to a global it never called.
    Found by Codex's reviewer against a tree with no parse error at all,
    which is the point: a syntax tree does not make a reader right, it
    only gives it the chance to be.
    """
    assert not [one for one in claims(source) if one[1] == "calls"], why


def test_a_hoisted_call_is_found_whether_or_not_the_declaration_is_exported():
    """`export function` wraps the declaration in an `export_statement`,
    and a scan of a scope's direct children walks straight past it. So a
    call to a function declared later in the file bound when the callee
    was local and vanished when it was exported -- the same code, read
    two ways, by a rule about a keyword."""
    plain = claims("function caller() { return leaf(); }\n"
                   "function leaf() { return 1; }\n")
    exported = claims("export function caller() { return leaf(); }\n"
                      "export function leaf() { return 1; }\n")
    assert ("web/app.ts:caller", "calls", "web/app.ts:leaf") in plain, plain
    assert ("web/app.ts:caller", "calls", "web/app.ts:leaf") in exported, exported


def test_a_call_this_file_could_not_parse_is_not_a_call():
    """`return leaf( ;` is not a call, it is a file someone is still
    typing. tree-sitter recovers and offers a `call_expression` anyway,
    and a reader that takes it records an edge from something nobody
    wrote."""
    assert not [one for one in claims("function leaf() { return 1; }\n"
                                      "function caller() { return leaf( ; }\n")
                if one[1] == "calls"]


def test_a_declaration_is_claimed_once():
    """`export const caller = leaf => leaf();` was reported twice --
    once from the scope's own scan and once from the pass over its
    children. Two identical facts are not worse than one, but they say
    the reader does not know what it has seen."""
    found = [one for one in claims("export const caller = (p) => p;\n") if one[1] == "defines"]
    assert len(found) == len(set(found)) == 1, found


@pytest.mark.parametrize("source, why", [
    ("export function leaf() { return 1; }\n"
     "export function caller(k: number) { switch (k) { case 1: { const leaf = () => 2;"
     " return leaf(); } default: return 0; } }\n",
     "a const declared inside a switch case"),
    ("export function leaf() { return 1; }\n"
     "export function caller() { const inner = function* (leaf) { return leaf(); };"
     " return inner(() => 2).next().value; }\n",
     "a parameter of a nested generator expression"),
])
def test_an_unmodelled_scope_does_not_fall_through_to_a_global(source, why):
    """The architectural fault under all of these, in Codex's reviewer's
    words: *an unmodelled scope must not silently fall through to a
    global binding*.

    The scope stack was a whitelist of node types I had thought of, so
    anything absent from it -- a switch body, a generator expression --
    bound nothing and the name resolved to the file's own declaration. It
    failed **open**, which for a graph means inventing an edge.

    Every node opens a scope now and binds whatever is declared directly
    in it, so a construct nobody modelled still holds its own names. A
    whitelist can only be as complete as its author; this cannot be
    incomplete in that direction.
    """
    assert not [one for one in claims(source) if one[1] == "calls"], why


def test_a_nested_function_is_not_the_same_thing_as_a_top_level_one():
    """`function worker` inside `caller` was given the same name as an
    unrelated exported `worker` beside it, so the blast radius of a
    global named one when only the other called it. A declaration is
    qualified by what holds it, which is what `path:Class.method` already
    did for classes and nothing did for functions."""
    source = ("export function leaf() { return 1; }\n"
              "export function caller() { function worker() { return leaf(); } return worker(); }\n"
              "export function worker() { return 0; }\n")
    found = claims(source)
    calls = {(one[0], one[2]) for one in found if one[1] == "calls"}
    assert ("web/app.ts:caller.worker", "web/app.ts:leaf") in calls, calls
    assert ("web/app.ts:worker", "web/app.ts:leaf") not in calls, calls
    # And the nested one is declared as belonging to its holder.
    assert ("web/app.ts:caller", "defines", "web/app.ts:caller.worker") in found, found


def test_a_parameter_default_is_evaluated_before_the_body_binds_anything():
    """The one finding in this round that was an **omission**, not a
    false edge, and the only one Codex's reviewer proved by counting
    calls under Node rather than by reading.

    `function caller(x = leaf()) { var leaf = … }` really does call the
    outer `leaf`: a parameter default is evaluated in the parameter
    environment, which cannot see the body's `var`. Hoisting that `var`
    to the *function* rather than to its *body* hid a real call.

    The distinction costs nothing anywhere else, because a body block
    contains every nested block, so hoisting to it reaches the same
    names for every call written inside the body.
    """
    source = ("export function leaf() { return 1; }\n"
              "export function caller(x = leaf()) {\n"
              "  var leaf = () => 2;\n"
              "  return x + leaf();\n"
              "}\n")
    calls = {(one[0], one[2]) for one in claims(source) if one[1] == "calls"}
    assert ("web/app.ts:caller", "web/app.ts:leaf") in calls, calls
    # And exactly once: the body's own `leaf()` is the local var.
    assert len([one for one in claims(source) if one[1] == "calls"]) == 1, claims(source)


def test_one_switch_is_one_scope_however_many_cases_it_has():
    """A `switch` body is a single block in JavaScript: every case shares
    it, and a `const` in one case is in scope for the next.

    Opening a scope for every node fixed a whitelist that failed open and
    replaced it with something that fails **wrong** -- it invented a
    boundary at each case, so case two stopped seeing case one's `leaf`
    and resolved to the global instead. A model that invents a scope is
    as unsound as one that misses a scope.
    """
    # No braces on the case: with them it is a block of its own and the
    # `const` really is scoped to it. Without them the switch body is the
    # block, which is the case that was wrong.
    source = ("export function leaf() { return 1; }\n"
              "export function caller(k: number) {\n"
              "  switch (k) {\n"
              "    case 1:\n"
              "      const leaf = () => 2;\n"
              "      return leaf();\n"
              "    case 2:\n"
              "      return leaf();\n"
              "  }\n"
              "}\n")
    assert not [one for one in claims(source) if one[1] == "calls"], claims(source)


def test_a_generator_expression_keeps_its_own_var():
    """`function*` written as an expression is a function, and a `var`
    inside it belongs to it. Its node type is `generator_function` and my
    list of what counts as a function did not have it, so the `var`
    hoisted out into the enclosing function and suppressed that
    function's genuine call to the global."""
    source = ("export function leaf() { return 1; }\n"
              "export function caller() {\n"
              "  const inner = function* () { var leaf = () => 2; return leaf(); };\n"
              "  return leaf() + inner().next().value;\n"
              "}\n")
    calls = {(one[0], one[2]) for one in claims(source) if one[1] == "calls"}
    assert ("web/app.ts:caller", "web/app.ts:leaf") in calls, calls


@pytest.mark.parametrize("source, name", [
    ("export interface Shelf { keep(p: string): string }\n", "Shelf"),
    ("export type Feature = 'a' | 'b';\n", "Feature"),
    ("export enum Mode { One, Two }\n", "Mode"),
    ("interface Local { x: number }\n", "Local"),
])
def test_a_typescript_type_declaration_is_a_declaration(source, name):
    """Measured against tree-sitter's own view of 200 files of this
    project's web application: the reader emitted no interfaces, type
    aliases or enums at all, and those are 62 of the 378 declarations the
    grammar reports.

    They are the vocabulary a TypeScript codebase is built from -- a
    `type` alias is what a function signature refers to -- so a graph
    that cannot name one cannot answer what depends on it.
    """
    assert ("web/app.ts", "defines", f"web/app.ts:{name}") in claims(source), claims(source)


def test_the_grammar_overrules_the_line_reader_about_declarations():
    """Both readers run and their claims are merged, so the line reader's
    guesses survived alongside the grammar's answers.

    `useEffect(() => {…})` is a call. The line reader reports it as a
    declaration because it matches the shape of one, and over 200 files
    of this project's web application that pattern is most of why its
    precision is 81% against the grammar's 100%. Measured on the same
    files: recall 71% against 100%.

    So where the parser is installed it decides what a **declaration**
    is, and the line reader keeps the claims the parser does not make --
    imports, inheritance, and the rationale notes it reads from comments.
    """
    from scone_memory.ingestion.code_graph import code_claims

    source = ("import {useEffect} from 'react';\n"
              "export function App() {\n"
              "  useEffect(() => {\n"
              "    return undefined;\n"
              "  }, []);\n"
              "  return null;\n"
              "}\n")
    found = code_claims(source, "web/App.tsx", language="braces")
    defines = {claim.object for claim in found if claim.predicate == "defines"}
    assert "web/App.tsx:App" in defines, defines
    assert not any("useEffect" in one for one in defines), defines
    # The line reader's other work is untouched.
    assert any(claim.predicate == "imports" for claim in found), found


def test_a_named_import_of_a_published_package_binds_across_repositories():
    from scone_memory.ingestion.code_graph import code_claims
    from scone_memory.ingestion.code_resolution import file_resolver
    from scone_memory.ingestion.code_syntax import available

    if not available():
        pytest.skip("the code-graph extra is not installed")
    source = 'import { tidy } from "@acme/ui";\nimport { other } from "somewhere-else";\nexport function run() { tidy(); other(); }\n'
    walked = file_resolver(["app/src/run.ts"], {"@acme/ui": "ui"}, known=["ui/src/index.ts"])
    found = {(c.predicate, c.object) for c in code_claims(source, "app/src/run.ts", language="braces", resolve=walked)}
    assert ("imports", "ui/src/index.ts") in found and ("calls", "ui/src/index.ts:tidy") in found
    assert not any(obj.endswith(":other") for predicate, obj in found if predicate == "calls"), \
        "a package nobody here publishes binds nothing"
