"""Ruby, Lua, shell, Perl and fish files say what they define and load, read from their syntax trees."""
import pytest

from scone_memory.ingestion import code_tree, code_tree_graph
from scone_memory.ingestion.code import code_language
from scone_memory.ingestion.code_graph import code_claims

pytestmark = pytest.mark.skipif(not code_tree.available(), reason="the grammar pack is not installed")


def said(claims, predicate=None):
    return [(c.subject, c.predicate, c.object) for c in claims if predicate is None or c.predicate == predicate]


def test_ruby_defines_by_holder_loads_by_literal_and_keeps_its_notes():
    source = ("require 'json'\nrequire_relative \"lib/util\"\nload \"tasks.rb\"\nrequire name_from_variable\n"
              "# WHY: carts outlive sessions\nmodule Shop\n  class Cart\n    # TODO: expire stale carts, see ADR-12\n    def add(item); end\n  end\nend\n"
              "def top_level; end\n")
    claims = code_claims(source, "app/cart.rb", language=code_language("app/cart.rb"))
    assert said(claims, "defines") == [("app/cart.rb", "defines", "app/cart.rb:Shop"), ("app/cart.rb:Shop", "defines", "app/cart.rb:Shop.Cart"),
                                       ("app/cart.rb:Shop.Cart", "defines", "app/cart.rb:Shop.Cart.add"), ("app/cart.rb", "defines", "app/cart.rb:top_level")]
    assert said(claims, "imports") == [("app/cart.rb", "imports", "json"), ("app/cart.rb", "imports", "lib/util"), ("app/cart.rb", "imports", "tasks.rb")], \
        "a require of a variable is not a claim: the tree cannot say what it names"
    assert said(claims, "notes") == [("app/cart.rb", "notes", "carts outlive sessions")]
    assert said(claims, "flags") == [("app/cart.rb:Shop.Cart", "flags", "expire stale carts, see ADR-12")], \
        "a comment sits in the innermost declaration whose lines hold it: above the method, that is the class, as the Python reader has it"
    assert said(claims, "cites") == [("app/cart.rb:Shop.Cart", "cites", "ADR-12")]
    add = next(c for c in claims if c.object == "app/cart.rb:Shop.Cart.add" and c.predicate == "defines")
    assert add.quote == "def add(item); end" and add.first_line == 9 and source.encode()[add.start:add.end].decode().strip() == add.quote


def test_lua_shell_perl_and_fish_each_say_what_they_define_and_load():
    lua = code_claims('local json = require("json")\nlocal M = {}\n-- NOTE: kept pure\nfunction M.add(a, b) return a + b end\nlocal function helper() end\nreturn M\n',
                      "lib/m.lua", language=code_language("lib/m.lua"))
    assert ("lib/m.lua", "imports", "json") in said(lua) and ("lib/m.lua", "defines", "lib/m.lua:M.add") in said(lua) and ("lib/m.lua", "defines", "lib/m.lua:helper") in said(lua)
    assert said(lua, "notes") == [("lib/m.lua", "notes", "kept pure")], "a `--` comment is read like a `#` one"
    bash = code_claims("#!/bin/bash\nsource ./lib.sh\n. ./other.sh\nsource \"$HOME/x.sh\"\nfunction greet() { echo hi; }\nbuild() { make; }\n",
                       "bin/run.sh", language=code_language("bin/run.sh"))
    assert said(bash, "imports") == [("bin/run.sh", "imports", "./lib.sh"), ("bin/run.sh", "imports", "./other.sh")], "a sourced variable is not a claim"
    assert said(bash, "defines") == [("bin/run.sh", "defines", "bin/run.sh:greet"), ("bin/run.sh", "defines", "bin/run.sh:build")]
    perl = code_claims("package Shop::Cart;\nuse strict;\nuse warnings;\nuse JSON::PP;\nrequire Exporter;\nsub add { return 1; }\n1;\n",
                       "lib/Cart.pm", language=code_language("lib/Cart.pm"))
    assert said(perl, "imports") == [("lib/Cart.pm", "imports", "JSON::PP"), ("lib/Cart.pm", "imports", "Exporter")], "pragmas are not modules"
    assert ("lib/Cart.pm", "defines", "lib/Cart.pm:Shop::Cart") in said(perl, "defines")
    assert ("lib/Cart.pm", "defines", "lib/Cart.pm:add") in said(perl, "defines")
    fish = code_claims("source ~/.config/fish/lib.fish\nfunction greet\n  echo hi\nend\n", "conf.fish", language=code_language("conf.fish"))
    assert said(fish, "imports") == [("conf.fish", "imports", "~/.config/fish/lib.fish")] and said(fish, "defines") == [("conf.fish", "defines", "conf.fish:greet")]


def test_spans_are_bytes_and_a_block_comment_cites_from_its_own_line():
    lua = "-- \u00e9\u00e9 accents before\nlocal json = require(\"json\")\n--[[ setup\n  see ADR-12\n]]\nfunction run() end\n"
    claims = code_tree_graph.tree_claims(lua, "lib/x.lua", grammar="lua")
    loaded = next(c for c in claims if c.predicate == "imports")
    assert lua.encode()[loaded.start:loaded.end].decode() == 'local json = require("json")', \
        "the span is in bytes of the source, as every reader records it; a character offset lands short after an accent"
    cited = next(c for c in claims if c.predicate == "cites")
    assert (cited.object, cited.first_line, cited.quote) == ("ADR-12", 4, "see ADR-12"), \
        "a citation inside a block comment is quoted from the line that names it, not the comment's first line"


def test_a_word_that_begins_with_a_tag_is_prose_and_a_flag_needs_no_colon():
    ruby = "# Notes live beside the code\n# TODO expire carts\n# Whyever: not a note\ndef a; end\n"
    claims = code_tree_graph.tree_claims(ruby, "x.rb", grammar="ruby")
    assert [(c.predicate, c.object) for c in claims if c.predicate in {"notes", "flags"}] == [("flags", "expire carts")]


def test_the_line_bound_bites_before_the_tree_is_read(monkeypatch):
    source = "require 'json'\ndef a; end\n" * 4
    assert any(c.predicate == "imports" for c in code_tree_graph.tree_claims(source, "x.rb", grammar="ruby"))
    monkeypatch.setattr(code_tree_graph, "MAX_LINES", 5)
    assert code_tree_graph.tree_claims(source, "x.rb", grammar="ruby") == ()


def test_without_the_grammar_pack_nothing_is_claimed_and_a_calls_edge_is_never_made(monkeypatch):
    source = "require 'json'\ndef a; b; end\ndef b; end\n"
    claims = code_claims(source, "x.rb", language="tree:ruby")
    assert said(claims, "imports") and not [c for c in claims if c.predicate == "calls"], \
        "a call needs a receiver's type or a cross-file name; the tree has neither"
    monkeypatch.setattr(code_tree, "available", lambda: False)
    monkeypatch.setattr(code_tree_graph, "available", lambda: False)
    assert code_claims(source, "x.rb", language="tree:ruby") == ()
