"""Declarations in languages the line readers do not know, read from the syntax tree."""

from __future__ import annotations

import pytest

from scone_memory.ingestion import code_tree
from scone_memory.ingestion.code import code_language, declaration_at, declarations

pytestmark = pytest.mark.skipif(not code_tree.available(), reason="the tree-sitter grammar pack is not installed")

RUBY = "class Cart\n  def total(items)\n    items.sum\n  end\nend\n\nmodule Shop\n  def self.open; end\nend\n"
LUA = "local function helper()\nend\nfunction Cart.total(items)\n  return 1\nend\n"
BASH = "helper() {\n  echo hi\n}\nfunction other { :; }\n"
PERL = "package Shop::Cart;\nsub total {\n  return 1;\n}\nsub helper { 2 }\n"
FISH = "function total\n  echo 1\nend\nfunction helper\n  echo 2\nend\n"


@pytest.mark.parametrize("source, content, names", [
    ("cart.rb", RUBY, ["Cart", "Cart.total", "Shop", "Shop.open"]),
    ("cart.lua", LUA, ["helper", "Cart.total"]),
    ("run.sh", BASH, ["helper", "other"]),
    ("Cart.pm", PERL, ["Shop::Cart", "total", "helper"]),  # a package statement is a sibling of its subs, not their holder
    ("total.fish", FISH, ["total", "helper"]),
])
def test_a_definition_is_named_by_everything_that_holds_it(source, content, names):
    language = code_language(source)
    assert language is not None and language.startswith("tree:")
    found = declarations(content, language=language)
    assert [d.name for d in found] == names
    for d in found:
        assert content.encode()[d.start:d.end].decode().strip(), "a declaration's span holds its text"
        assert 1 <= d.first_line <= d.last_line


def test_kinds_come_from_the_grammar_and_lines_from_the_tree():
    found = declarations(RUBY, language=code_language("cart.rb"))
    assert [(d.kind, d.first_line, d.last_line) for d in found] == [("class", 1, 5), ("method", 2, 4), ("module", 7, 9), ("singleton_method", 8, 8)]


def test_a_chunk_inside_a_definition_carries_its_name():
    content = PERL
    start = content.index("return 1")
    named = declaration_at(content, start, start + 8, language=code_language("Cart.pm"))
    assert named is not None and named.name == "total" and named.kind == "subroutine_declaration_statement"


def test_the_brace_family_keeps_its_reader_and_unknown_languages_stay_prose():
    for source in ("Main.kt", "cart.php", "Cart.swift", "Cart.scala"):
        assert code_language(source) == "braces", source
    assert code_language("notes.ex") is None and code_language("Makefile") is None


def test_a_file_too_long_or_a_missing_grammar_reads_as_nothing(monkeypatch):
    monkeypatch.setattr(code_tree, "MAX_LINES", 2)
    # Fresh content, so the reader's cache of the last few files cannot answer for the bound.
    assert declarations(RUBY + "# a line past the bound\n", language="tree:ruby") == ()
    monkeypatch.setattr(code_tree, "MAX_LINES", 200_000)
    assert code_tree.tree_declarations("x = 1\n", "no-such-grammar") == ()
