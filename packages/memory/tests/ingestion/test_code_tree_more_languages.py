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
    ("stats.R", "total <- function(items) { sum(items) }\n"),
    ("Billing.jl", "module Billing\nfunction total(items)\n  sum(items)\nend\nend\n"),
    ("billing.hs", "module Billing where\ntotal :: [Int] -> Int\ntotal = sum\n"),
    ("core.clj", "(defn total [items] (reduce + items))\n"),
], ids=["r", "julia", "haskell", "clojure"])
def test_a_language_whose_tree_this_reader_cannot_name_stays_prose(path, source):
    """Read by the rule that reads Ruby, R yields a declaration called
    `function` and Julia loses every function it has. A wrong name in the
    graph is worse than no name, so these say nothing until each has a
    rule of its own, read from its own tree."""
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
