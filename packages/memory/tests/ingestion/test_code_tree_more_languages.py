"""Elixir, Protobuf and Solidity say what they define and load.

Each language's rule was read from its own grammar's tree on real code,
never by analogy with another's. That is not pedantry: the generic rule
that reads Ruby finds a Julia module and none of its functions, and on R
it produces a declaration named ``function`` -- the keyword, because that
is what sits in R's ``name`` field. Those two are left out for that
reason, and this file holds the check that says so.
"""
import pytest

from scone_memory.ingestion import code_tree
from scone_memory.ingestion.code import code_language
from scone_memory.ingestion.code_graph import code_claims

pytestmark = pytest.mark.skipif(not code_tree.available(), reason="the grammar pack is not installed")


def said(claims, predicate=None):
    return [(c.subject, c.predicate, c.object) for c in claims if predicate is None or c.predicate == predicate]


ELIXIR = '''defmodule Billing.Invoice do
  @moduledoc "Invoices and what they come to."
  alias Billing.Rate
  import Enum, only: [sum: 1]
  require Logger
  use GenServer

  # WHY: a late invoice is still an invoice
  def total(items) do
    sum(items)
  end

  defp rate do
    Rate.current()
  end

  defmacro assert_positive(value) do
    quote do: unquote(value) > 0
  end
end

defmodule Billing.Empty do
end
'''


def test_elixir_says_its_modules_functions_and_what_it_pulls_in():
    claims = code_claims(ELIXIR, "lib/invoice.ex", language=code_language("lib/invoice.ex"))
    assert said(claims, "defines") == [
        ("lib/invoice.ex", "defines", "lib/invoice.ex:Billing.Invoice"),
        ("lib/invoice.ex:Billing.Invoice", "defines", "lib/invoice.ex:Billing.Invoice.total"),
        ("lib/invoice.ex:Billing.Invoice", "defines", "lib/invoice.ex:Billing.Invoice.rate"),
        ("lib/invoice.ex:Billing.Invoice", "defines", "lib/invoice.ex:Billing.Invoice.assert_positive"),
        ("lib/invoice.ex", "defines", "lib/invoice.ex:Billing.Empty"),
    ], "a private function and a macro are declarations; a module holds what it defines"
    assert said(claims, "imports") == [
        ("lib/invoice.ex", "imports", "Billing.Rate"),
        ("lib/invoice.ex", "imports", "Enum"),
        ("lib/invoice.ex", "imports", "Logger"),
        ("lib/invoice.ex", "imports", "GenServer"),
    ], "alias, import, require and use each name a module this file depends on"
    assert said(claims, "notes") == [("lib/invoice.ex:Billing.Invoice", "notes", "a late invoice is still an invoice")]
    total = next(c for c in claims if c.object.endswith(".total") and c.predicate == "defines")
    assert total.quote.startswith("def total(items) do") and total.first_line == 9
    assert ELIXIR.encode()[total.start:total.end].decode().strip() == total.quote, "the span is the bytes of the quote"


def test_an_elixir_call_that_is_not_a_definition_claims_nothing():
    """`def` is a call like any other in this grammar, so the rule has to
    read the word; a call to something else must not become a
    declaration, and neither must a dynamic alias."""
    claims = code_claims('defmodule M do\n  IO.puts("hello")\n  alias unquote(mod)\n  def run, do: :ok\nend\n',
                         "lib/m.ex", language=code_language("lib/m.ex"))
    assert said(claims, "defines") == [("lib/m.ex", "defines", "lib/m.ex:M"),
                                       ("lib/m.ex:M", "defines", "lib/m.ex:M.run")]
    assert said(claims, "imports") == [], "an alias of something the tree cannot read is not a claim"


def test_a_name_elixir_builds_at_compile_time_is_no_name_this_reader_can_read():
    """Metaprogramming names a module or a function from a variable. The
    tree holds `unquote(name)`, which is the call that will make the
    name, not the name; claiming it would put `unquote` in the graph."""
    claims = code_claims('defmodule unquote(mod) do\n  def unquote(fun)(arg), do: arg\n'
                         '  def unquote(name), do: :ok\n  def plain, do: :ok\nend\n',
                         "lib/meta.ex", language=code_language("lib/meta.ex"))
    assert said(claims, "defines") == [("lib/meta.ex", "defines", "lib/meta.ex:plain")], \
        "the one function spelt out is claimed, and the two built at compile time are not"


PROTO = '''syntax = "proto3";
package billing.v1;

import "google/protobuf/timestamp.proto";
import public "common/money.proto";

// WHY: ids are strings so a migration can change their shape
message Invoice {
  string id = 1;
  Money total = 2;

  message Line {
    string sku = 1;
  }
}

enum Kind {
  KIND_UNSPECIFIED = 0;
}

service Billing {
  rpc Total(Invoice) returns (Money);
}
'''


def test_protobuf_says_its_messages_enums_services_and_imports():
    claims = code_claims(PROTO, "proto/billing.proto", language=code_language("proto/billing.proto"))
    assert said(claims, "defines") == [
        ("proto/billing.proto", "defines", "proto/billing.proto:Invoice"),
        ("proto/billing.proto:Invoice", "defines", "proto/billing.proto:Invoice.Line"),
        ("proto/billing.proto", "defines", "proto/billing.proto:Kind"),
        ("proto/billing.proto", "defines", "proto/billing.proto:Billing"),
        ("proto/billing.proto:Billing", "defines", "proto/billing.proto:Billing.Total"),
    ], "a nested message is held by the message around it, and an rpc by its service"
    assert said(claims, "imports") == [
        ("proto/billing.proto", "imports", "google/protobuf/timestamp.proto"),
        ("proto/billing.proto", "imports", "common/money.proto"),
    ], "a public import is still an import"
    assert said(claims, "notes") == [("proto/billing.proto", "notes",
                                      "ids are strings so a migration can change their shape")]


SOLIDITY = '''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./Owned.sol";
import {Money} from "./Money.sol";

interface IBilling {
    function total(uint256 amount) external view returns (uint256);
}

contract Billing is Owned {
    uint256 public rate;

    // TODO: fees are not rounded yet
    function total(uint256 amount) public view returns (uint256) {
        return amount + rate;
    }
}

library Fees {
    function apply(uint256 amount) internal pure returns (uint256) {
        return amount;
    }
}
'''


def test_solidity_says_its_contracts_functions_and_imports():
    claims = code_claims(SOLIDITY, "contracts/Billing.sol", language=code_language("contracts/Billing.sol"))
    defines = said(claims, "defines")
    for expected in ("contracts/Billing.sol:IBilling", "contracts/Billing.sol:Billing",
                     "contracts/Billing.sol:Billing.total", "contracts/Billing.sol:Fees",
                     "contracts/Billing.sol:Fees.apply"):
        assert any(object_ == expected for _, _, object_ in defines), expected
    assert ("contracts/Billing.sol:Billing", "defines", "contracts/Billing.sol:Billing.rate") in defines, \
        "a public state variable is part of the contract's surface"
    assert said(claims, "imports") == [
        ("contracts/Billing.sol", "imports", "./Owned.sol"),
        ("contracts/Billing.sol", "imports", "./Money.sol"),
    ]
    assert said(claims, "flags") == [("contracts/Billing.sol:Billing", "flags", "fees are not rounded yet")]


@pytest.mark.parametrize("path, source", [
    ("Invoice.groovy", "class Invoice { int total(List items) { 0 } }\n"),
    ("build.tcl", "proc total {items} { return 0 }\n"),
], ids=["groovy", "tcl"])
def test_a_language_whose_tree_this_reader_cannot_name_stays_prose(path, source):
    """A rule of their own has not been written and checked against these
    trees. A wrong name in the graph is worse than no name, so they say
    nothing: this is what stops a later `just add the suffix`. R, Julia,
    Haskell, Clojure and OCaml were each here until their own tree was
    read. The Groovy grammar in this pack parses a class as a bare
    `command`, which names nothing at all, and Tcl has no rule here
    either. (Zig, by contrast, is not on this list: the brace family's
    header rule already reads it.)"""
    assert code_language(path) is None, f"{path} is not claimed to be read"
    assert code_claims(source, path, language=None) == ()


# -- Terraform, Erlang and PowerShell ----------------------------------------------


TERRAFORM = '''terraform {
  required_version = ">= 1.5"
}

# WHY: logs outlive the cluster that wrote them
variable "region" {
  default = "us-east-1"
}

resource "aws_s3_bucket" "logs" {
  bucket = "scone-logs"

  lifecycle {
    prevent_destroy = true
  }
}

module "network" {
  source  = "./modules/network"
  version = "1.2.0"
}

module "regional" {
  source = "./modules/${var.env}/network"
}

output "bucket_name" {
  value = aws_s3_bucket.logs.bucket
}
'''


def test_terraform_says_its_blocks_by_type_and_labels_and_the_modules_it_pulls_in():
    claims = code_claims(TERRAFORM, "infra/main.tf", language=code_language("infra/main.tf"))
    assert said(claims, "defines") == [
        ("infra/main.tf", "defines", "infra/main.tf:region"),
        ("infra/main.tf", "defines", "infra/main.tf:aws_s3_bucket.logs"),
        ("infra/main.tf", "defines", "infra/main.tf:network"),
        ("infra/main.tf", "defines", "infra/main.tf:regional"),
        ("infra/main.tf", "defines", "infra/main.tf:bucket_name"),
    ], "a block is named by its labels; `terraform` and `lifecycle` label nothing and declare nothing"
    assert said(claims, "imports") == [("infra/main.tf", "imports", "./modules/network")], \
        "a module's source is what this root depends on; a version is not a module, and a source built " \
        "from a variable is not a path the tree can name"
    assert said(claims, "notes") == [("infra/main.tf", "notes", "logs outlive the cluster that wrote them")]
    bucket = next(c for c in claims if c.object.endswith("aws_s3_bucket.logs"))
    assert bucket.kind == "resource" if hasattr(bucket, "kind") else True


ERLANG = '''-module(billing).
-include("records.hrl").
-include_lib("kernel/include/file.hrl").
-import(lists, [sum/1]).
-export([total/1]).

%% WHY: an invoice is summed once
total(Items) ->
    sum(Items).

rate() -> 0.2.
'''


def test_erlang_says_its_module_functions_includes_and_imports():
    claims = code_claims(ERLANG, "src/billing.erl", language=code_language("src/billing.erl"))
    assert said(claims, "defines") == [
        ("src/billing.erl", "defines", "src/billing.erl:billing"),
        ("src/billing.erl", "defines", "src/billing.erl:total"),
        ("src/billing.erl", "defines", "src/billing.erl:rate"),
    ], "the module attribute declares the module, and each function clause group its function"
    assert said(claims, "imports") == [
        ("src/billing.erl", "imports", "records.hrl"),
        ("src/billing.erl", "imports", "kernel/include/file.hrl"),
        ("src/billing.erl", "imports", "lists"),
    ], "an include is a file and an import attribute is a module; an export is neither"
    assert said(claims, "notes") == [("src/billing.erl", "notes", "an invoice is summed once")]


POWERSHELL = '''Import-Module Billing
Import-Module -Name Reporting
. ./lib/helpers.ps1
. "$PSScriptRoot/dynamic.ps1"

# TODO: paging is not handled
function Get-Total {
    param($Items)
    $Items | Measure-Object -Sum
}

function Set-Rate { param($Rate) }
'''


def test_powershell_says_its_functions_the_modules_it_imports_and_what_it_dot_sources():
    claims = code_claims(POWERSHELL, "tools/billing.ps1", language=code_language("tools/billing.ps1"))
    assert said(claims, "defines") == [
        ("tools/billing.ps1", "defines", "tools/billing.ps1:Get-Total"),
        ("tools/billing.ps1", "defines", "tools/billing.ps1:Set-Rate"),
    ]
    assert said(claims, "imports") == [
        ("tools/billing.ps1", "imports", "Billing"),
        ("tools/billing.ps1", "imports", "./lib/helpers.ps1"),
    ], "a dot-source with a variable in the path is not a literal the tree can name, and -Name is a switch, not a module"
    assert said(claims, "flags") == [("tools/billing.ps1", "flags", "paging is not handled")]


# -- Julia and R, the two the generic rule misreads ---------------------------------


JULIA = '''module Billing

using Statistics
import Dates: now
include("rates.jl")

# WHY: an invoice is summed once
function total(items)
    sum(items)
end

struct Invoice
    id::Int
end

rate(x) = x * 0.2

const CEILING = 10_000

function apply!(invoice)
    invoice.id = 7
end

end
'''


def test_julia_says_its_module_functions_structs_and_what_it_pulls_in():
    claims = code_claims(JULIA, "src/Billing.jl", language=code_language("src/Billing.jl"))
    assert said(claims, "defines") == [
        ("src/Billing.jl", "defines", "src/Billing.jl:Billing"),
        ("src/Billing.jl:Billing", "defines", "src/Billing.jl:Billing.total"),
        ("src/Billing.jl:Billing", "defines", "src/Billing.jl:Billing.Invoice"),
        ("src/Billing.jl:Billing", "defines", "src/Billing.jl:Billing.rate"),
        ("src/Billing.jl:Billing", "defines", "src/Billing.jl:Billing.apply!"),
    ], ("a function's name is in its signature, a struct's in its type head, and `rate(x) = ...` is a function "
        "too -- while `const CEILING = 10_000` assigns a value and `invoice.id = 7` assigns a field, and "
        "neither declares anything")
    assert said(claims, "imports") == [
        ("src/Billing.jl", "imports", "Statistics"),
        ("src/Billing.jl", "imports", "Dates"),
        ("src/Billing.jl", "imports", "rates.jl"),
    ], "using and import name modules, include names a file"
    assert said(claims, "notes") == [("src/Billing.jl:Billing", "notes", "an invoice is summed once")]


R = '''library(stats)
require(utils)
source("helpers.R")
source(file.path(root, "dynamic.R"))

# TODO: weights are not applied
total <- function(items) {
  sum(items)
}

rate = function() 0.2

threshold <- 10
'''


def test_r_names_a_function_after_what_it_is_assigned_to_never_after_the_keyword():
    """R's own `function_definition` node carries the keyword `function`
    in its name field, so the rule that reads Ruby would declare every
    function in the file as `function`. The name is the thing it is
    assigned to."""
    claims = code_claims(R, "R/total.R", language=code_language("R/total.R"))
    assert said(claims, "defines") == [
        ("R/total.R", "defines", "R/total.R:total"),
        ("R/total.R", "defines", "R/total.R:rate"),
    ], "both assignment arrows define; a value that is not a function does not"
    assert not any("function" == object_.rsplit(":", 1)[-1] for _, _, object_ in said(claims, "defines")), \
        "the keyword is never the name"
    assert said(claims, "imports") == [
        ("R/total.R", "imports", "stats"),
        ("R/total.R", "imports", "utils"),
        ("R/total.R", "imports", "helpers.R"),
    ], "library and require name packages, source names a file, and a path built by a call is not a name"
    assert said(claims, "flags") == [("R/total.R", "flags", "weights are not applied")]


# -- one declaration per name ------------------------------------------------------


def test_a_function_written_in_several_clauses_is_declared_once():
    """Elixir writes a function as one clause per shape, and Julia as one
    method per type. Every clause is a definition node, so a reader that
    took each would put the same name in the graph three times and give
    `graph affected` three identical edges to walk."""
    elixir = code_claims('defmodule M do\n  def total(a), do: a\n  def total(a, b), do: a + b\n'
                         '  def total(a, b, c), do: 0\n  def other, do: 1\nend\n',
                         "lib/m.ex", language=code_language("lib/m.ex"))
    assert said(elixir, "defines") == [
        ("lib/m.ex", "defines", "lib/m.ex:M"),
        ("lib/m.ex:M", "defines", "lib/m.ex:M.total"),
        ("lib/m.ex:M", "defines", "lib/m.ex:M.other"),
    ]
    first = next(c for c in elixir if c.object.endswith(".total"))
    assert first.quote == "def total(a), do: a", "the first clause is where the name is declared"

    julia = code_claims('total(x::Int) = x\ntotal(x::Float64) = x\n', "a.jl", language=code_language("a.jl"))
    assert said(julia, "defines") == [("a.jl", "defines", "a.jl:total")]


def test_two_holders_may_each_declare_the_same_name():
    """One name per holder, not one name per file: two modules with a
    `total` of their own are two declarations."""
    claims = code_claims('defmodule A do\n  def total, do: 1\nend\ndefmodule B do\n  def total, do: 2\nend\n',
                         "lib/two.ex", language=code_language("lib/two.ex"))
    assert said(claims, "defines") == [
        ("lib/two.ex", "defines", "lib/two.ex:A"),
        ("lib/two.ex:A", "defines", "lib/two.ex:A.total"),
        ("lib/two.ex", "defines", "lib/two.ex:B"),
        ("lib/two.ex:B", "defines", "lib/two.ex:B.total"),
    ]


# -- Haskell and Clojure -----------------------------------------------------------


HASKELL = '''module Billing.Invoice (total) where

import Data.List (sort)
import qualified Data.Map as M

-- WHY: totals are integers so rounding is the caller's problem
data Invoice = Invoice Int

newtype Rate = Rate Double

type Items = [Int]

total :: Items -> Int
total xs = sum xs
total [] = 0
'''


def test_haskell_says_its_module_types_and_functions_and_what_it_imports():
    claims = code_claims(HASKELL, "src/Invoice.hs", language=code_language("src/Invoice.hs"))
    assert said(claims, "defines") == [
        ("src/Invoice.hs", "defines", "src/Invoice.hs:Billing.Invoice"),
        ("src/Invoice.hs", "defines", "src/Invoice.hs:Invoice"),
        ("src/Invoice.hs", "defines", "src/Invoice.hs:Rate"),
        ("src/Invoice.hs", "defines", "src/Invoice.hs:Items"),
        ("src/Invoice.hs", "defines", "src/Invoice.hs:total"),
    ], "a data type, a newtype and a synonym each declare; a signature is not a second declaration of its function"
    assert said(claims, "imports") == [
        ("src/Invoice.hs", "imports", "Data.List"),
        ("src/Invoice.hs", "imports", "Data.Map"),
    ], "a qualified import names the module, not the alias it is given"
    assert said(claims, "notes") == [("src/Invoice.hs", "notes",
                                      "totals are integers so rounding is the caller's problem")]


CLOJURE = '''(ns billing.core
  (:require [clojure.string :as s]
            [clojure.set :refer [union]])
  (:import java.util.Date))

;; WHY: a reduce reads better than a loop here
(defn total [items]
  (reduce + items))

(defn- rate [] 0.2)

(def ceiling 10000)

(defmacro with-rate [& body] `(do ~@body))
'''


def test_clojure_says_its_namespace_definitions_and_requires():
    claims = code_claims(CLOJURE, "src/core.clj", language=code_language("src/core.clj"))
    assert said(claims, "defines") == [
        ("src/core.clj", "defines", "src/core.clj:billing.core"),
        ("src/core.clj", "defines", "src/core.clj:total"),
        ("src/core.clj", "defines", "src/core.clj:rate"),
        ("src/core.clj", "defines", "src/core.clj:ceiling"),
        ("src/core.clj", "defines", "src/core.clj:with-rate"),
    ], "defn, defn-, def and defmacro all declare, and the namespace declares itself"
    assert said(claims, "imports") == [
        ("src/core.clj", "imports", "clojure.string"),
        ("src/core.clj", "imports", "clojure.set"),
        ("src/core.clj", "imports", "java.util.Date"),
    ], "each required library and each imported class, by name"
    assert said(claims, "notes") == [("src/core.clj", "notes", "a reduce reads better than a loop here")]


def test_a_clojure_call_that_is_not_a_definition_declares_nothing():
    claims = code_claims('(println "hello")\n(let [x 1] x)\n(defn f [] 1)\n', "a.clj",
                         language=code_language("a.clj"))
    assert said(claims, "defines") == [("a.clj", "defines", "a.clj:f")]


# -- OCaml --------------------------------------------------------------------------


OCAML = '''open Printf
open Core.List

(* WHY: totals are integers so rounding is the caller's problem *)
module Billing = struct
  let total items = List.fold_left (+) 0 items
  let rate = 0.2

  type invoice = { id : int }

  type 'a box = { value : 'a }

  exception Missing of string
end

module type Reporter = sig
  val report : string -> unit
end

let apply x = x + 1
'''


def test_ocaml_says_its_modules_values_types_and_what_it_opens():
    claims = code_claims(OCAML, "lib/billing.ml", language=code_language("lib/billing.ml"))
    assert said(claims, "defines") == [
        ("lib/billing.ml", "defines", "lib/billing.ml:Billing"),
        ("lib/billing.ml:Billing", "defines", "lib/billing.ml:Billing.total"),
        ("lib/billing.ml:Billing", "defines", "lib/billing.ml:Billing.rate"),
        ("lib/billing.ml:Billing", "defines", "lib/billing.ml:Billing.invoice"),
        ("lib/billing.ml:Billing", "defines", "lib/billing.ml:Billing.box"),
        ("lib/billing.ml:Billing", "defines", "lib/billing.ml:Billing.Missing"),
        ("lib/billing.ml", "defines", "lib/billing.ml:Reporter"),
        ("lib/billing.ml:Reporter", "defines", "lib/billing.ml:Reporter.report"),
        ("lib/billing.ml", "defines", "lib/billing.ml:apply"),
    ], ("a module holds what its structure defines -- values, types and exceptions alike -- and a module "
        "type holds what its signature promises; a parametrised type is named `box`, never `'a`")
    assert said(claims, "imports") == [
        ("lib/billing.ml", "imports", "Printf"),
        ("lib/billing.ml", "imports", "Core.List"),
    ], "an open names a module by its whole path"
    assert said(claims, "notes") == [("lib/billing.ml", "notes",
                                      "totals are integers so rounding is the caller's problem")]


def test_an_ocaml_interface_file_says_what_it_promises():
    claims = code_claims('val total : int list -> int\ntype invoice\n', "lib/billing.mli",
                         language=code_language("lib/billing.mli"))
    assert said(claims, "defines") == [
        ("lib/billing.mli", "defines", "lib/billing.mli:total"),
        ("lib/billing.mli", "defines", "lib/billing.mli:invoice"),
    ], "an .mli declares the surface, which is what another file can reach"
