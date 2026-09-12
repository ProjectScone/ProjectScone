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
