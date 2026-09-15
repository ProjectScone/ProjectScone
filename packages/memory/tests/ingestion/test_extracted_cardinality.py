"""Claims the framework reads out of files hold side by side.

A file defines many things and imports many modules; a project depends
on many packages. Read into a ledger whose predicates hold one value at
a time unless configured otherwise, each such claim retired the one
before it: a module with three imports held one, with the other two
closed as "superseded". The graph built on that kept the last line of
every kind and called the rest history.

These predicates are many-valued by their nature, not by configuration,
so the cardinality is declared where the predicates are, and no setting
makes them one-valued.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.extracted import MANY_VALUED
from scone_memory.ingestion import code_graph, manifests, mcp_config
from scone_memory.ingestion.code_graph import record_claims

pytestmark = pytest.mark.asyncio

MODULE = "import os\nimport re\nimport json\n\ndef a():\n    return 1\n\ndef b():\n    return a()\n"


async def engine(**options):
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                              code_graph=True, **options).open()


async def test_a_file_that_imports_three_modules_holds_all_three():
    memory = await engine()
    try:
        await memory.remember("s", MODULE, kind="file", source="pkg/mod.py")
        facts = await memory.facts("s")
        assert sorted(f.object for f in facts if f.predicate == "imports") == ["json", "os", "re"]
        assert sorted(f.object for f in facts if f.predicate == "defines") == ["pkg/mod.py:a", "pkg/mod.py:b"]
        assert await memory.facts("s", status="closed") == [], "nothing was superseded; nothing changed"
    finally:
        await memory.close()


async def test_a_manifest_that_needs_four_packages_holds_all_four():
    memory = await engine()
    try:
        content = '[project]\nname = "x"\ndependencies = ["a", "b", "c", "d"]\n'
        added = await memory.remember("s", content, kind="file", source="pyproject.toml")
        assert await record_claims(memory, "s", episode_id=added.episode_id, content=content,
                                   path="pyproject.toml", when=memory.clock()) == 5
        held = await memory.facts("s")
        assert sorted(f.object for f in held if f.predicate == manifests.DEPENDS_ON) == ["a", "b", "c", "d"]
    finally:
        await memory.close()


async def test_configuration_cannot_make_an_extracted_predicate_one_valued():
    """`many_valued` adds predicates; it does not take these away."""
    memory = await engine(many_valued=["knows"])
    try:
        assert {"imports", "knows"} <= memory.many_valued
        await memory.remember("s", MODULE, kind="file", source="pkg/mod.py")
        assert len([f for f in await memory.facts("s") if f.predicate == "imports"]) == 3
    finally:
        await memory.close()


async def test_every_predicate_the_framework_extracts_is_declared_many_valued():
    """The declaration lives apart from the readers; this keeps them from
    drifting when a reader gains a predicate."""
    from_code = {getattr(code_graph, name) for name in ("DEFINES", "IMPORTS", "IMPORTS_WHEN_CALLED", "IMPORTS_FOR_TYPES",
                                                          "CALLS", "INHERITS", "MIXES_IN", "USES_TYPE", "NOTES", "FLAGS",
                                                          "CITES", "REFERENCES")}
    from_manifests = {manifests.DEPENDS_ON, manifests.DEVELOPS_WITH}
    from_mcp_configs = {mcp_config.RUNS_WITH, mcp_config.REQUIRES_ENV, mcp_config.CONNECTS_TO}
    assert from_code | from_manifests | from_mcp_configs == set(MANY_VALUED)


async def test_a_stated_predicate_still_holds_one_value_at_a_time():
    """The legacy rule for everything a person states is untouched."""
    memory = await engine()
    try:
        await memory.assert_fact("s", "mark", "lives_in", "Austin", valid_from="2022-01-01")
        await memory.assert_fact("s", "mark", "lives_in", "Lisbon", valid_from="2024-01-01")
        assert [f.object for f in await memory.facts("s")] == ["Lisbon"]
    finally:
        await memory.close()
