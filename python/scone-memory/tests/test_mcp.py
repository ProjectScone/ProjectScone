"""The MCP server: six tools with the names, arguments and bounds of
crates/scone/src/mcp.rs, results as plain text, refusals as ``is_error``
results rather than exceptions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.mcp import create_server
from scone_memory.testing import Clock

PACKAGE_DIR = Path(__file__).resolve().parent.parent

#: Tool name -> the argument names mcp.rs declares (plus the metadata and
#: where scopes this server adds). The order is not part of the contract.
RUST_ARGUMENTS = {
    "memory_store": {"content", "space", "tags", "metadata"},
    "memory_recall": {"query", "space", "limit", "include_profile", "tags", "as_of", "where"},
    "memory_facts_about": {"entity", "space"},
    "memory_pending": {"limit", "space"},
    "memory_store_facts": {"episode_id", "facts", "space"},
    "memory_forget": {"fact_id", "reason", "space"},
}


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


async def test_exposes_the_six_rust_tools_with_their_argument_names(server):
    tools = {t.name: t for t in await server.list_tools()}
    assert set(tools) == set(RUST_ARGUMENTS)
    for name, arguments in RUST_ARGUMENTS.items():
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
        args=["-m", "scone_memory.mcp", "--space", "smoke"],
        env={"SCONE_SQLITE_PATH": str(tmp_path / "memory.db")},
        cwd=str(PACKAGE_DIR),
    )
    async with client_module.Client(params) as client:
        listed = await client.list_tools()
        assert {t.name for t in listed.tools} == set(RUST_ARGUMENTS)

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
