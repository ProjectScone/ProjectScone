"""Shared MCP contracts, result formatting and deny-by-default role checks."""
import functools
import inspect
from typing import Annotated, Awaitable, Callable, Optional, Sequence


from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, Field

from ...core.bearer_keys import KeyHolder
from ...memory.engine import MemoryEngine, Profile, check_space, normalise_term
from ...core.errors import SconeError
from ...core.models import Added, Episode, Fact, RecallResult
from ...entities.mentions import is_mention

MAX_CONTENT = 100_000
MAX_QUERY = 1_000
MAX_ENTITY = 200
MAX_REASON = 500
MAX_LIMIT = 50
MAX_TAGS = 10
MAX_FACTS = 50
MAX_PENDING = 20
#: Rust's SubmittedFact defaults here: an agent's extraction is a
#: proposal, not an observation.
DEFAULT_FACT_CONFIDENCE = 0.8
#: Pending episodes are shown truncated, as the Rust server does.
PENDING_CONTENT_CHARS = 4_000

INSTRUCTIONS = (
    "Persistent memory for this agent. Call memory_recall at task start; "
    "memory_store for durable observations; memory_facts_about before acting "
    "on an entity; memory_forget when the user retracts something. "
    "Periodically (session start or idle), call memory_pending and distill "
    "the returned episodes into subject/predicate/object facts with your own "
    "reasoning, submitting via memory_store_facts; you are the extraction "
    "model and no API key is needed. To see how things connect, call "
    "memory_entity for one entity (its relations both ways), "
    "memory_connections for the paths between two, or memory_graph_context "
    "with names or a question; every line cites its facts. memory_graph_schema "
    "says what kinds of entity and which predicates the graph holds, and "
    "memory_graph_match answers a structured question given as triple patterns "
    "joined by ?variables. For a question about the whole graph, "
    "memory_graph_overview digests each community with cited facts, and "
    "memory_graph_changes says what changed since a moment."
)


class TriplePattern(BaseModel):
    """One pattern of a structured question; a term starting with ? is a variable."""

    model_config = {"extra": "forbid"}

    subject: str
    predicate: str
    object: str


class SubmittedFact(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: Annotated[Optional[float], Field(description="0..=1; defaults to 0.8")] = None
    valid_from: Annotated[
        Optional[str],
        Field(description="RFC 3339 instant the fact holds from; defaults to the episode's date"),
    ] = None


def tool_error(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)


def ok_text(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)])


ToolFn = Callable[..., Awaitable[CallToolResult]]

#: Tools that only read. Everything else is taken to write, so a tool
#: added without a thought about roles is refused to a read-only key
#: rather than quietly allowed; ``test_every_tool_is_classified`` fails
#: until a new name is put in one set or the other.
READING_TOOLS = frozenset({
    "memory_recall", "memory_facts_about", "memory_pending",
    "memory_graph_context", "memory_entity", "memory_connections", "memory_graph_schema",
    "memory_graph_match", "memory_graph_overview", "memory_graph_changes", "memory_entity_duplicates",
    "memory_temporal_answer", "memory_graph_cycles", "memory_graph_stats", "memory_graph_hubs",
    "memory_graph_health", "memory_graph_affected", "memory_activity",
})

#: Tools that change what the space holds.
WRITING_TOOLS = frozenset({"memory_store", "memory_store_facts", "memory_forget", "memory_agent_connect", "memory_record_interaction"})


def refuses_to_write(fn: ToolFn, role: str) -> ToolFn:
    """A key whose role does not write is told so by the tool itself,
    in the words the REST API uses for the same refusal. The tool stays
    in the catalogue: a model that can read the reason corrects itself,
    where a tool that had silently vanished would be guessed at."""

    @functools.wraps(fn)
    async def refused(**arguments: object) -> CallToolResult:
        return tool_error(f"key role {role} cannot write")

    return refused



def refusals_as_tool_errors(fn: ToolFn) -> ToolFn:
    """An engine refusal (bad space name, unparsable date, unknown id)
    becomes an ``is_error`` result, as ``tool_error`` does in mcp.rs.
    ``functools.wraps`` keeps the signature the SDK reads for the schema."""

    @functools.wraps(fn)
    async def guarded(**arguments) -> CallToolResult:
        try:
            return await fn(**arguments)
        except SconeError as e:
            return tool_error(str(e))

    return guarded


def tool(server: MCPServer, name: str, holder: Optional[KeyHolder] = None) -> Callable[[ToolFn], ToolFn]:
    """Register ``fn`` under its mcp.rs name, its docstring (dedented) as
    the description, with engine refusals turned into tool errors.

    ``holder`` is the key this server was built for, when it was built
    for one: a role that does not write reaches no writing tool."""

    def register(fn: ToolFn) -> ToolFn:
        guarded = refusals_as_tool_errors(fn)
        if holder is not None and not holder.may_write and name not in READING_TOOLS:
            guarded = refuses_to_write(guarded, holder.role)
        server.add_tool(guarded, name=name, description=inspect.cleandoc(fn.__doc__ or ""))
        return fn

    return register


# -- result formatting --------------------------------------------------------


def day(timestamp: str) -> str:
    return timestamp[:10]


def describe_added(added: Added) -> str:
    if added.deduplicated:
        return f"already stored as episode {added.episode_id} (deduplicated)"
    return (
        f"stored episode {added.episode_id} ({added.chunks} chunks). "
        "episodic memory stored; distill facts via memory_pending and memory_store_facts"
    )


def profile_lines(profile: Profile) -> list[str]:
    lines: list[str] = []
    if profile.static_facts:
        lines.append("## Profile")
        lines.extend(f"- {f.subject} {f.predicate} {f.object} (conf {f.confidence:.2f})" for f in profile.static_facts)
    if profile.dynamic:
        lines.append("## Recent activity")
        lines.extend(f"- {d.replace(chr(10), ' ')}" for d in profile.dynamic)
    return lines


def recall_lines(result: RecallResult) -> list[str]:
    lines = [
        f"fact [{f.fact_id}] {f.subject} {f.predicate} {f.object} (conf {f.confidence:.2f}, {f.status})"
        for f in result.facts
    ]
    lines.extend(
        f"memory [{item.score:.2f} | {day(item.created_at)} | episode {item.episode_id}] {item.text.strip()}"
        for item in result.items
    )
    lines.extend(f"degraded: {d}" for d in result.degraded)
    if result.low_confidence:
        # The reader is told, in the pack itself, that the evidence is weak
        # (experiment 9): the whole point of the gate is that an agent can
        # decline to answer from it.
        best = "nothing was found" if result.top_similarity is None else f"best match similarity {result.top_similarity:.2f}"
        lines.append(f"low confidence: {best}, below this memory's floor; treat the memories above as weak evidence or say you do not know")
    return lines


def fact_about_line(f: Fact) -> str:
    return f"fact [{f.fact_id}] {f.subject} {f.predicate} {f.object} (conf {f.confidence:.2f}, since {f.valid_from})"


def pending_text(episodes: Sequence[Episode]) -> str:
    if not episodes:
        return "nothing pending: memory is fully distilled"
    out = "Episodes awaiting fact extraction (submit via memory_store_facts):\n"
    for e in episodes:
        out += f"--- episode {e.episode_id} ({e.created_at})\n{e.content[:PENDING_CONTENT_CHARS]}\n"
    return out


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


# -- engine helpers the Rust core has and this engine lacks -------------------


async def facts_about(engine: MemoryEngine, space: str, entity: str) -> list[Fact]:
    """Active facts whose subject is ``entity`` (normalised the way the
    engine normalises subjects), most confident first."""
    check_space(space)
    wanted = normalise_term(entity, "entity")
    found = [f for f in await engine.facts(space) if f.subject == wanted]
    return sorted(found, key=lambda f: (-f.confidence, f.fact_id))


async def pending_episodes(engine: MemoryEngine, space: str, limit: int) -> list[Episode]:
    """The most recent ``limit`` episodes no fact cites as its source.

    An approximation of the Rust distill queue: this engine keeps no
    queue, so "distilled" means "some fact carries this episode's id in
    ``source_episode_id``". An episode that honestly yields no facts
    therefore stays listed. At most ``limit`` of the newest episodes can
    be hidden by distilled ones, so fetching that many more is enough.
    """
    check_space(space)
    facts = await engine.documents.list_facts(space, include_closed=True)
    distilled = {f.source_episode_id for f in facts if f.source_episode_id is not None and not is_mention(f)}
    recent = await engine.documents.recent_episodes(space, limit + len(distilled))
    return [e for e in recent if e.episode_id not in distilled][:limit]


def clamp_confidence(value: Optional[float]) -> float:
    return min(1.0, max(0.0, DEFAULT_FACT_CONFIDENCE if value is None else value))


async def fact_ids_by_status(engine: MemoryEngine, space: str) -> dict[int, str]:
    """Every fact this space holds and what it is. Proposals are included
    on purpose: a parked fact is not in the ledger, but it did land, and
    counting it as deduplicated would tell an agent its claim was already
    known when nobody has looked at it yet."""
    found: dict[int, str] = {f.fact_id: f.status for f in await engine.facts(space, include_closed=True)}
    found.update({f.fact_id: f.status for f in await engine.facts(space, status="proposed")})
    return found


# -- the server ---------------------------------------------------------------
