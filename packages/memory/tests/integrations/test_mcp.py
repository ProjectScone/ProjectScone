"""The MCP server: six tools with the names, arguments and bounds of
crates/scone/src/mcp.rs, and three graph tools beside them, results as
plain text, refusals as ``is_error`` results rather than exceptions."""

from __future__ import annotations

from ..paths import PACKAGE_ROOT

import sys
from pathlib import Path

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.runtime.mcp import create_server
from scone_memory.testing import Clock

PACKAGE_DIR = PACKAGE_ROOT

#: Tool name -> the argument names mcp.rs declares (plus the metadata and
#: where scopes this server adds). The order is not part of the contract.
RUST_ARGUMENTS = {
    "memory_store": {"content", "space", "tags", "metadata"},
    "memory_recall": {"query", "space", "limit", "include_profile", "tags", "as_of", "where", "kind", "source_prefix", "since", "until"},
    "memory_facts_about": {"entity", "space"},
    "memory_pending": {"limit", "space"},
    "memory_store_facts": {"episode_id", "facts", "space"},
    "memory_forget": {"fact_id", "reason", "space"},
}
#: The entity graph, read only: a packet around names or a question, one
#: entity with its relations both ways, and the paths between two.
GRAPH_ARGUMENTS = {
    "memory_graph_context": {"names", "question", "space", "max_bytes", "similar", "min_similarity"},
    "memory_entity": {"name", "space"},
    "memory_connections": {"source", "target", "max_hops", "space"},
    "memory_graph_schema": {"limit", "max_bytes", "space"},
    "memory_graph_match": {"where", "returns", "limit", "status", "as_of", "together", "max_bytes", "space"},
    "memory_graph_overview": {"question", "limit", "facts", "max_bytes", "space"},
}
TOOL_ARGUMENTS = {**RUST_ARGUMENTS, **GRAPH_ARGUMENTS}


@pytest.fixture
async def server():
    engine = await MemoryEngine(
        InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=200, clock=Clock()
    ).open()
    return create_server(engine, "default")


async def call(server, name: str, **arguments) -> tuple[bool, str]:
    result = await server.call_tool(name, arguments)
    return result.is_error, "\n".join(block.text for block in result.content)


async def store_and_distill(server, content: str, subject: str, predicate: str, object: str) -> int:
    """Store one episode and submit one fact for it; return the fact id."""
    error, text = await call(server, "memory_store", content=content)
    assert not error, text
    episode_id = int(text.split("episode ")[1].split(" ")[0])
    error, text = await call(
        server,
        "memory_store_facts",
        episode_id=episode_id,
        facts=[{"subject": subject, "predicate": predicate, "object": object}],
    )
    assert not error, text
    error, text = await call(server, "memory_facts_about", entity=subject)
    assert not error, text
    return int(text.split("fact [")[1].split("]")[0])


# -- the contract with mcp.rs -------------------------------------------------


async def test_exposes_the_six_rust_tools_and_the_graph_tools_with_their_argument_names(server):
    tools = {t.name: t for t in await server.list_tools()}
    assert set(tools) == set(TOOL_ARGUMENTS)
    for name, arguments in TOOL_ARGUMENTS.items():
        assert set(tools[name].input_schema["properties"]) == arguments, name
        assert tools[name].description, name


async def test_store_then_recall_returns_the_text_with_provenance(server):
    error, text = await call(server, "memory_store", content="Moved to Lisbon in March", tags=["life"])
    assert not error
    assert text.startswith("stored episode 1 (1 chunks)")

    error, text = await call(server, "memory_recall", query="where do I live", include_profile=False)
    assert not error
    assert text == "memory [1.00 | 2025-01-01 | episode 1] Moved to Lisbon in March"


async def test_recall_prepends_the_profile_by_default(server):
    await call(server, "memory_store", content="Moved to Lisbon in March")
    error, text = await call(server, "memory_recall", query="where do I live")
    assert not error
    assert text.startswith("## Recent activity\n- Moved to Lisbon in March\nmemory [")


async def test_duplicate_content_is_recognised_not_re_stored(server):
    await call(server, "memory_store", content="Moved to Lisbon in March")
    error, text = await call(server, "memory_store", content="Moved to Lisbon in March")
    assert not error
    assert text == "already stored as episode 1 (deduplicated)"


async def test_recall_with_nothing_stored_says_so(server):
    error, text = await call(server, "memory_recall", query="anything at all")
    assert not error
    assert text == "no matching memory"


# -- bounds, as tool errors ---------------------------------------------------


async def test_more_than_ten_tags_is_a_tool_error(server):
    error, text = await call(server, "memory_store", content="tagged", tags=[f"t{i}" for i in range(11)])
    assert error
    assert text == "at most 10 tags per store"


async def test_query_over_1000_chars_is_a_tool_error(server):
    error, text = await call(server, "memory_recall", query="x" * 1001)
    assert error
    assert text == "query must be 1..=1000 chars, got 1001"


async def test_more_than_fifty_facts_is_a_tool_error(server):
    await call(server, "memory_store", content="a long day")
    facts = [{"subject": "mark", "predicate": f"p{i}", "object": "x"} for i in range(51)]
    error, text = await call(server, "memory_store_facts", episode_id=1, facts=facts)
    assert error
    assert text == "at most 50 facts per submission"


async def test_empty_content_and_long_reason_are_tool_errors(server):
    error, text = await call(server, "memory_store", content="")
    assert error
    assert text == "content must be 1..=100000 bytes, got 0"
    error, text = await call(server, "memory_forget", fact_id=1, reason="r" * 501)
    assert error
    assert text == "reason must be 1..=500 chars, got 501"


async def test_engine_refusal_is_a_tool_error_not_an_exception(server):
    error, text = await call(server, "memory_store", content="x", space="Bad Space")
    assert error
    assert "space name" in text
    error, text = await call(server, "memory_store", content="x", metadata={"Bad Key": "v"})
    assert error
    assert "metadata key" in text
    error, text = await call(server, "memory_forget", fact_id=99, reason="never existed")
    assert error
    assert text == "fact 99 not found in 'default'"


# -- facts --------------------------------------------------------------------


async def test_store_facts_then_facts_about_returns_the_fact(server):
    await call(server, "memory_store", content="Mark moved to Lisbon")
    error, text = await call(
        server,
        "memory_store_facts",
        episode_id=1,
        facts=[{"subject": "Mark", "predicate": "lives_in", "object": "Lisbon"}],
    )
    assert not error
    assert text == "episode 1 distilled: 1 fact added, 0 closed, 0 deduplicated"

    error, text = await call(server, "memory_facts_about", entity="mark")
    assert not error
    assert text == "fact [1] mark lives_in Lisbon (conf 0.80, since 2025-01-01T00:00:00.000Z)"


async def test_store_facts_reports_closure_and_deduplication(server):
    await call(server, "memory_store", content="Mark moved to Lisbon")
    await call(server, "memory_store", content="Mark moved to Porto")
    await call(server, "memory_store_facts", episode_id=1, facts=[{"subject": "mark", "predicate": "lives_in", "object": "Lisbon"}])
    error, text = await call(
        server,
        "memory_store_facts",
        episode_id=2,
        facts=[
            {"subject": "mark", "predicate": "lives_in", "object": "Lisbon"},
            {"subject": "mark", "predicate": "lives_in", "object": "Porto", "valid_from": "2025-02-01"},
        ],
    )
    assert not error
    assert text == "episode 2 distilled: 1 fact added, 1 closed, 1 deduplicated"


async def test_recall_lists_facts_before_memory(server):
    await call(server, "memory_store", content="Mark moved to Lisbon")
    await call(server, "memory_store_facts", episode_id=1, facts=[{"subject": "mark", "predicate": "lives_in", "object": "Lisbon"}])
    error, text = await call(server, "memory_recall", query="where does mark live", include_profile=False)
    assert not error
    assert text.splitlines() == [
        "fact [1] mark lives_in Lisbon (conf 0.80, active)",
        "memory [1.00 | 2025-01-01 | episode 1] Mark moved to Lisbon",
    ]


async def test_forget_closes_the_fact(server):
    fact_id = await store_and_distill(server, "Mark moved to Lisbon", "mark", "lives_in", "Lisbon")
    error, text = await call(server, "memory_forget", fact_id=fact_id, reason="moved away")
    assert not error
    assert text == f"closed fact {fact_id}: moved away"

    error, text = await call(server, "memory_facts_about", entity="mark")
    assert not error
    assert text == "no facts about mark"


async def test_recall_as_of_evaluates_fact_validity_at_that_instant(server):
    await call(server, "memory_store", content="Mark moved to Lisbon")
    await call(
        server,
        "memory_store_facts",
        episode_id=1,
        facts=[{"subject": "mark", "predicate": "lives_in", "object": "Lisbon", "valid_from": "2024-03-02"}],
    )
    before = await call(server, "memory_recall", query="where does mark live", as_of="2023-06-01", include_profile=False)
    after = await call(server, "memory_recall", query="where does mark live", as_of="2024-06-01", include_profile=False)
    assert "fact [1]" not in before[1]
    assert "fact [1] mark lives_in Lisbon" in after[1]


# -- scopes -------------------------------------------------------------------


async def test_where_filter_scopes_recall_to_matching_metadata(server):
    await call(server, "memory_store", content="Alice drinks green tea", metadata={"user_id": "alice"})
    await call(server, "memory_store", content="Bob drinks green tea", metadata={"user_id": "bob"})

    error, text = await call(server, "memory_recall", query="green tea", where={"user_id": "alice"}, include_profile=False)
    assert not error
    assert "Alice drinks green tea" in text
    assert "Bob" not in text


async def test_space_argument_overrides_the_server_default(server):
    await call(server, "memory_store", content="only in the other space", space="other")
    default = await call(server, "memory_recall", query="other space", include_profile=False)
    other = await call(server, "memory_recall", query="other space", space="other", include_profile=False)
    assert default[1] == "no matching memory"
    assert "only in the other space" in other[1]


# -- pending ------------------------------------------------------------------


async def test_pending_lists_episodes_no_fact_cites(server):
    await call(server, "memory_store", content="Mark moved to Lisbon")
    await call(server, "memory_store", content="Mark took up rowing")
    await call(server, "memory_store_facts", episode_id=1, facts=[{"subject": "mark", "predicate": "lives_in", "object": "Lisbon"}])

    error, text = await call(server, "memory_pending")
    assert not error
    assert text == (
        "Episodes awaiting fact extraction (submit via memory_store_facts):\n"
        "--- episode 2 (2025-01-01T00:00:00.000Z)\nMark took up rowing\n"
    )


async def test_pending_is_empty_once_every_episode_is_cited(server):
    await store_and_distill(server, "Mark moved to Lisbon", "mark", "lives_in", "Lisbon")
    error, text = await call(server, "memory_pending")
    assert not error
    assert text == "nothing pending: memory is fully distilled"


async def test_pending_honours_the_limit(server):
    for i in range(4):
        await call(server, "memory_store", content=f"note number {i}")
    error, text = await call(server, "memory_pending", limit=2)
    assert not error
    assert text.count("--- episode") == 2


# -- over stdio ---------------------------------------------------------------


async def test_stdio_server_answers_a_real_client(tmp_path):
    client_module = pytest.importorskip("mcp.client", reason="mcp client API unavailable")
    stdio = pytest.importorskip("mcp.client.stdio", reason="mcp stdio client unavailable")
    params = stdio.StdioServerParameters(
        command=sys.executable,
        args=["-m", "scone_memory.runtime.mcp", "--space", "smoke"],
        env={"SCONE_SQLITE_PATH": str(tmp_path / "memory.db"),
             "PYTHONPATH": str(PACKAGE_DIR / "src")},
        cwd=str(PACKAGE_DIR),
    )
    async with client_module.Client(params) as client:
        listed = await client.list_tools()
        assert {t.name for t in listed.tools} == set(TOOL_ARGUMENTS)

        stored = await client.call_tool("memory_store", {"content": "Moved to Lisbon in March"})
        assert not stored.is_error
        assert stored.content[0].text.startswith("stored episode 1 ")

        recalled = await client.call_tool("memory_recall", {"query": "where do I live", "include_profile": False})
        assert not recalled.is_error
        assert "episode 1] Moved to Lisbon in March" in recalled.content[0].text
    assert (tmp_path / "memory.db").exists()


async def test_recall_tells_the_agent_when_the_evidence_is_weak():
    """Experiment 9 reaches the agent through the tool text: with a floor,
    a weak recall carries a plain warning; a strong one and a server with
    no floor carry nothing extra (the Rust server has no floor)."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock(), similarity_floor=0.5).open()
    server = create_server(engine, "default")
    error, text = await call(server, "memory_store", content="the deploy runbook lives in the ops wiki")
    assert not error
    error, weak = await call(server, "memory_recall", query="zebra quartz umbrella", include_profile=False)
    assert not error and weak.splitlines()[-1].startswith("low confidence: best match similarity 0.") and "say you do not know" in weak
    error, strong = await call(server, "memory_recall", query="the deploy runbook lives in the ops wiki", include_profile=False)
    assert not error and "low confidence" not in strong
    error, empty = await call(server, "memory_recall", query="anything", space="empty", include_profile=False)
    assert not error and "low confidence: nothing was found" in empty

    plain = create_server(await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open(), "default")
    await call(plain, "memory_store", content="the deploy runbook lives in the ops wiki")
    error, text = await call(plain, "memory_recall", query="zebra quartz umbrella", include_profile=False)
    assert not error and "low confidence" not in text, "no floor, no verdict, no line"


async def test_recall_narrows_by_kind_source_prefix_and_dates():
    """The same four narrowing arguments as the Rust server's memory_recall.
    memory_store cannot set kind, source or date, so the episodes are
    seeded through the engine."""
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.runtime.mcp import create_server

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    ids = []
    for kind, text, source, day in [
        ("note", "deploy runbook: rotate the staging keys first", None, "2024-01-10"),
        ("file", "deploy runbook: rotate the staging keys, then restart", "/ops/runbooks/deploy.md", "2024-02-10"),
        ("conversation", "user: where is the deploy runbook for staging keys?", "session-42", "2024-04-10"),
    ]:
        ids.append((await engine.remember("default", text, kind=kind, source=source, created_at=day)).episode_id)
    narrow = create_server(engine, "default")

    async def recalled(**extra):
        error, text = await call(narrow, "memory_recall", query="deploy runbook staging keys", include_profile=False, **extra)
        assert not error, text
        return sorted({int(line.split("episode ")[1].split("]")[0]) for line in text.splitlines() if line.startswith("memory [")})

    assert await recalled() == ids
    assert await recalled(kind="file") == [ids[1]]
    assert await recalled(source_prefix="session-") == [ids[2]]
    assert await recalled(until="2024-01-31") == [ids[0]]
    assert await recalled(since="2024-03-01") == [ids[2]]


async def test_a_propose_gate_parks_low_confidence_agent_facts_for_review():
    """The Rust server can be started with --propose-below so an agent's
    less certain claims wait for a person instead of entering the ledger.
    The Python server runs the live console and could not do that at all,
    which made the safer configuration unavailable exactly where the
    ledger is real."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock()).open()
    gated = create_server(engine, "default", propose_below=0.7)
    await call(gated, "memory_store", content="Ana moved to Lisbon in March and joined Farfetch.")

    error, text = await call(gated, "memory_store_facts", episode_id=1, facts=[
        {"subject": "Ana", "predicate": "lives_in", "object": "Lisbon", "confidence": 0.9},
        {"subject": "Ana", "predicate": "works_at", "object": "Farfetch", "confidence": 0.4},
    ])

    assert not error
    assert [f.object for f in await engine.facts("default")] == ["Lisbon"]
    waiting = await engine.facts("default", status="proposed")
    assert [(f.object, f.confidence) for f in waiting] == [("Farfetch", 0.4)]
    assert "1 proposed" in text


async def test_without_a_gate_every_submitted_fact_still_enters_the_ledger():
    """The default is unchanged, and stays the same as the Rust server's:
    no gate means no parking. Changing that is a policy decision, not
    something this surface should decide on its own."""
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                clock=Clock()).open()
    plain = create_server(engine, "default")
    await call(plain, "memory_store", content="Ana moved to Lisbon in March.")
    error, _ = await call(plain, "memory_store_facts", episode_id=1, facts=[
        {"subject": "Ana", "predicate": "lives_in", "object": "Lisbon", "confidence": 0.1},
    ])

    assert not error
    assert [f.object for f in await engine.facts("default")] == ["Lisbon"]
    assert await engine.facts("default", status="proposed") == []


async def test_the_gate_can_be_turned_on_where_an_operator_can_reach_it(monkeypatch):
    """A gate only the library can set is a gate no deployment has. The
    server reads it from the environment, the same way every other store
    and model setting arrives."""
    from scone_memory.runtime import mcp
    from scone_memory.runtime.cli import settings_for_cli

    seen: dict = {}

    def watching(engine, space, propose_below=None):
        seen.update(space=space, propose_below=propose_below)
        raise RuntimeError("stop before stdio")

    monkeypatch.setattr(mcp, "create_server", watching)
    # Both stores in memory on purpose: the CLI defaults vectors to sqlite,
    # and a test must not open the machine's real ~/.scone-memory store.
    settings = settings_for_cli({"SCONE_DOCUMENTS": "memory", "SCONE_VECTORS": "memory",
                                 "SCONE_EMBEDDER": "hash", "SCONE_MCP_PROPOSE_BELOW": "0.7"})

    with pytest.raises(RuntimeError):
        await mcp.serve(settings, "alpha")

    assert seen == {"space": "alpha", "propose_below": 0.7}


async def test_no_gate_in_the_environment_leaves_the_server_ungated(monkeypatch):
    from scone_memory.runtime import mcp
    from scone_memory.runtime.cli import settings_for_cli

    seen: dict = {}

    def watching(engine, space, propose_below=None):
        seen.update(propose_below=propose_below)
        raise RuntimeError("stop before stdio")

    monkeypatch.setattr(mcp, "create_server", watching)
    with pytest.raises(RuntimeError):
        await mcp.serve(settings_for_cli({"SCONE_DOCUMENTS": "memory", "SCONE_VECTORS": "memory",
                                          "SCONE_EMBEDDER": "hash"}), "alpha")

    assert seen == {"propose_below": None}


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_proposal_threshold_raises_domain_error(threshold):
    from scone_memory.core.errors import InvalidInput

    engine = MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())
    with pytest.raises(InvalidInput, match="confidence"):
        create_server(engine, propose_below=threshold)



async def test_the_graph_tools_read_an_entity_both_ways_and_the_paths_between_two(server):
    await store_and_distill(server, "Alice Chen joined Acme Robotics.", "alice chen", "works_at", "Acme Robotics")
    await store_and_distill(server, "Acme Robotics is based in Lisbon.", "acme robotics", "based_in", "Lisbon")
    result = await server.call_tool("memory_entity", {"name": "acme robotics"})
    error, entity = result.is_error, "\n".join(block.text for block in result.content)
    assert not error and entity.splitlines()[1].startswith("coverage: ")
    assert any(line.startswith("hop 1: ") and "works_at Acme Robotics" in line for line in entity.splitlines())
    error, paths = await call(server, "memory_connections", source="alice chen", target="lisbon")
    assert not error and any(line.startswith("path: ") and "-based_in->" in line for line in paths.splitlines())
    error, packet = await call(server, "memory_graph_context", question="what is in lisbon?")
    assert not error and any(line.startswith("entity: Lisbon") for line in packet.splitlines())
    error, text = await call(server, "memory_graph_context")
    assert error and "names or a question" in text


async def test_the_graph_schema_tool_lists_kinds_and_predicate_shapes(server):
    await store_and_distill(server, "Alice Chen joined Acme Robotics.", "alice chen", "works_at", "Acme Robotics")
    error, text = await call(server, "memory_graph_schema")
    lines = text.splitlines()
    assert not error and lines[0].startswith("schema: space default, current facts as of ")
    assert lines[1] == "coverage: complete" and "kind: person, 1 entity, inferred" in lines
    assert "predicate: works_at, 1 fact: (person) -> (organisation) x1" in lines
    error, text = await call(server, "memory_graph_schema", limit=0)
    assert error and "limit" in text
    error, text = await call(server, "memory_graph_schema", max_bytes=100)
    assert error and "max_bytes" in text


async def test_the_match_tool_answers_a_structured_question_with_cited_rows(server):
    await store_and_distill(server, "Alice Chen joined Acme Robotics.", "alice chen", "works_at", "Acme Robotics")
    await store_and_distill(server, "Acme Robotics is based in Lisbon.", "acme robotics", "based_in", "Lisbon")
    where = [{"subject": "?who", "predicate": "works_at", "object": "?org"},
             {"subject": "?org", "predicate": "based_in", "object": "Lisbon"}]
    error, text = await call(server, "memory_graph_match", where=where, returns=["?who"])
    assert not error and text.splitlines()[0].startswith("match: space default, current facts as of ")
    assert any(line.startswith("row: ?who = alice chen (person) ent:") for line in text.splitlines())
    error, text = await call(server, "memory_graph_match", where=where * 4)
    assert error and "between 1 and 6 patterns" in text
    error, text = await call(server, "memory_graph_match", where=where, status="history", as_of="soon")
    assert error and "RFC 3339" in text


async def test_the_overview_tool_digests_the_communities_a_question_concerns(server):
    await store_and_distill(server, "Alice Chen joined Acme Robotics.", "alice chen", "works_at", "Acme Robotics")
    error, text = await call(server, "memory_graph_overview", question="who works where?")
    assert not error and text.splitlines()[0].startswith("overview: space default, current facts as of ")
    assert any(line.startswith("community: ") and 'matched "works"' in line for line in text.splitlines())
    error, text = await call(server, "memory_graph_overview", limit=0)
    assert error and "limit" in text


async def test_graph_tools_take_a_question_as_long_as_the_other_surfaces_do(server):
    error, text = await call(server, "memory_graph_context", question="who? " * 300)
    assert not error, text
    error, text = await call(server, "memory_graph_context", question="q" * 2001)
    assert error and "2000" in text


@pytest.mark.parametrize("tool, arguments", [
    ("memory_graph_schema", {"limit": True}), ("memory_graph_schema", {"limit": "5"}),
    ("memory_graph_schema", {"max_bytes": "2048"}), ("memory_connections", {"source": "a", "target": "b", "max_hops": True}),
    ("memory_graph_context", {"names": ["a"], "max_bytes": True}),
    ("memory_graph_match", {"where": [{"subject": "?a", "predicate": "knows", "object": "?b"}], "limit": True}),
])
async def test_graph_tools_take_whole_numbers_only_as_the_toolbox_and_http_do(server, tool, arguments):
    """Refused either way the SDK refuses: an error result, or the argument
    error it raises in process and turns into one over the protocol."""
    try:
        result = await server.call_tool(tool, arguments)
    except Exception as refused:
        assert "valid integer" in str(refused)
    else:
        assert result.is_error


async def test_the_graph_report_and_schema_are_resources_a_client_can_attach(server):
    """Read-only context a client can attach without calling a tool: the
    server space's report and schema, and the same for any space by name."""
    await store_and_distill(server, "Alice Chen joined Acme Robotics.", "alice chen", "works_at", "Acme Robotics")
    uris = {str(resource.uri) for resource in await server.list_resources()}
    templates = {template.uri_template for template in await server.list_resource_templates()}
    assert {"scone://graph/report", "scone://graph/schema"} <= uris
    assert {"scone://{space}/graph/report", "scone://{space}/graph/schema"} <= templates
    report = list(await server.read_resource("scone://graph/report"))[0]
    assert report.mime_type == "text/markdown" and str(report.content).startswith("# Knowledge report")
    schema = list(await server.read_resource("scone://graph/schema"))[0]
    assert str(schema.content).startswith("schema: space default,") and "works_at" in str(schema.content)
    other = list(await server.read_resource("scone://elsewhere/graph/schema"))[0]
    assert str(other.content).startswith("schema: space elsewhere,") and "totals: 0 entities" in str(other.content)


async def test_a_resource_for_a_space_that_cannot_exist_is_refused(server):
    with pytest.raises(Exception, match="space"):
        list(await server.read_resource("scone://BAD%20SPACE/graph/report"))


async def test_graph_context_can_seed_by_resemblance(server):
    await store_and_distill(server, "Alice Chen joined Acme Robotics.", "alice chen", "works_at", "Acme Robotics")
    error, text = await call(server, "memory_graph_context", question="which robotics firm?", similar=True)
    assert not error and " similar " in text
    error, text = await call(server, "memory_graph_context", question="which robotics firm?")
    assert not error and " similar " not in text
