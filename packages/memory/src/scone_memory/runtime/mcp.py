"""MCP server: the engine as six tools for any MCP agent.

Mirrors ``crates/scone/src/mcp.rs``: the same tool names, the same
argument names, the same input bounds, and the same plain-text result
shape, so an agent configured against the Rust server can point at this
one unchanged. Invalid input comes back as a tool result with
``is_error`` set, never as an exception, so the model can read the
reason and correct the call.

    python -m scone_memory.runtime.mcp --space default

Stores come from the environment exactly as for the CLI (see
``scone_memory.runtime.config``); with nothing set, memory persists to SQLite at
~/.scone-memory/memory.db.

Built on mcp 2.x, where the class the 1.x SDK called ``FastMCP`` is
``mcp.server.mcpserver.MCPServer``.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import inspect
import os
import sys
from typing import Annotated, Awaitable, Callable, Mapping, Optional, Sequence

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, Field, StrictBool, StrictInt

from .. import __version__
from ..core.errors import InvalidInput
from .cli import settings_for_cli
from .config import Settings, build_engine
from ..memory.engine import MemoryEngine, Profile, check_space, normalise_term
from ..core.errors import SconeError
from ..core.models import Added, Episode, Fact, RecallResult
from ..entities.context import MAX_NAME, MAX_NAMES, MAX_QUESTION, ContextLimits, graph_connections, graph_context
from ..entities.report import render_markdown, report_record
from ..entities.schema import MAX_PREDICATES, schema_record, schema_text
from ..retrieval.temporal import (DEFAULT_LIMIT as TEMPORAL_LIMIT, MAX_BYTES as TEMPORAL_BYTES,
                                  MAX_BYTES_LIMIT as TEMPORAL_BYTES_LIMIT,
                                  MAX_LIMIT as TEMPORAL_MAX_LIMIT, TemporalError, temporal_answer)
from ..entities.health import (DEFAULT_EXAMPLES as HEALTH_EXAMPLES, MAX_BYTES as HEALTH_BYTES,
                               MAX_EXAMPLES as HEALTH_MAX_EXAMPLES, HealthError, graph_health)
from ..entities.duplicates import (DEFAULT_MIN_SCORE, DEFAULT_PAIRS, MAX_BYTES as DUPLICATES_BYTES, MAX_PAIRS,
                                    DuplicatesError, likely_duplicates)
from ..entities.changes import DEFAULT_CHANGES, MAX_BYTES as CHANGES_BYTES, MAX_CHANGES, ChangesError, graph_changes
from ..entities.overview import (DEFAULT_COMMUNITIES, DEFAULT_FACTS_EACH, MAX_BYTES as OVERVIEW_BYTES,
                                  MAX_COMMUNITIES, MAX_FACTS_EACH, OverviewError, graph_overview)
from ..entities.match import DEFAULT_ROWS, MAX_BYTES as MATCH_BYTES, MAX_PATTERNS, MAX_ROWS, MatchQueryError, graph_match
from ..entities.view import StatusMode
from ..core.validation import normalise_time

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


def tool(server: MCPServer, name: str) -> Callable[[ToolFn], ToolFn]:
    """Register ``fn`` under its mcp.rs name, its docstring (dedented) as
    the description, with engine refusals turned into tool errors."""

    def register(fn: ToolFn) -> ToolFn:
        server.add_tool(refusals_as_tool_errors(fn), name=name, description=inspect.cleandoc(fn.__doc__ or ""))
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
    distilled = {f.source_episode_id for f in facts if f.source_episode_id is not None}
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


def create_server(engine: MemoryEngine, space: str = "default",
                  propose_below: Optional[float] = None) -> MCPServer:
    """The six tools of mcp.rs and three read-only graph tools, closing over
    one engine and a default space that any call may override with its
    ``space`` argument.

    ``propose_below`` is the Rust server's ``--propose-below``: a fact an
    agent submits under that confidence is parked as a proposal for a
    person instead of entering the ledger. Unset, as there, means every
    submitted fact is a ledger claim.
    """
    if propose_below is not None and not 0.0 <= propose_below <= 1.0:
        raise InvalidInput(f"propose_below must be a confidence in 0..=1, got {propose_below}")
    server = MCPServer(
        name="scone-memory",
        title="Scone memory engine",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    default_space = space

    @tool(server, "memory_store")
    async def memory_store(
        content: Annotated[str, Field(description="The content to remember (1..=100000 bytes)")],
        space: Annotated[Optional[str], Field(description="Space to store into; defaults to the server's space")] = None,
        tags: Annotated[
            Optional[list[str]], Field(description="Tags for focused retrieval later (each 1..=64 chars, max 10)")
        ] = None,
        metadata: Annotated[
            Optional[dict[str, str]],
            Field(description="Scope keys such as user_id or session_id; memory_recall filters on them with `where`"),
        ] = None,
    ) -> CallToolResult:
        """Save content to persistent memory. Returns the episode id; duplicate
        content is recognized, not re-stored. Facts are not distilled here:
        call memory_pending and submit them via memory_store_facts."""
        size = len(content.encode())
        if not content or size > MAX_CONTENT:
            return tool_error(f"content must be 1..={MAX_CONTENT} bytes, got {size}")
        if len(tags or ()) > MAX_TAGS:
            return tool_error(f"at most {MAX_TAGS} tags per store")
        added = await engine.remember(space or default_space, content, tags=tags or (), metadata=metadata)
        return ok_text(describe_added(added))

    @tool(server, "memory_recall")
    async def memory_recall(
        query: Annotated[str, Field(description="Natural-language query (1..=1000 chars)")],
        space: Annotated[Optional[str], Field(description="Space to search; defaults to the server's space")] = None,
        limit: Annotated[Optional[int], Field(description="Max items (1..=50); defaults to 5")] = None,
        include_profile: Annotated[
            Optional[bool],
            Field(description="Prepend the space's profile (identity facts + recent activity). Defaults to true."),
        ] = None,
        tags: Annotated[
            Optional[list[str]], Field(description="Focus recall to episodes carrying ALL of these tags.")
        ] = None,
        as_of: Annotated[
            Optional[str], Field(description="Evaluate fact validity at this RFC 3339 instant (time travel)")
        ] = None,
        where: Annotated[
            Optional[dict[str, str]],
            Field(description="Only episodes whose metadata carries every one of these key=value pairs"),
        ] = None,
        kind: Annotated[Optional[str], Field(description="Only episodes of this kind (note, file, conversation, ...)")] = None,
        source_prefix: Annotated[
            Optional[str],
            Field(description="Only episodes whose source starts with this text (a path, a session id, a URL origin); literal, not a pattern"),
        ] = None,
        since: Annotated[Optional[str], Field(description="Only episodes that happened at or after this RFC 3339 instant")] = None,
        until: Annotated[Optional[str], Field(description="Only episodes that happened at or before this RFC 3339 instant")] = None,
    ) -> CallToolResult:
        """Recall relevant memory: temporal facts first, then episodic chunks,
        each with provenance. `as_of` answers what was true at a past time."""
        if not query or len(query) > MAX_QUERY:
            return tool_error(f"query must be 1..={MAX_QUERY} chars, got {len(query)}")
        target = space or default_space
        lines: list[str] = []
        if include_profile is None or include_profile:
            lines.extend(profile_lines(await engine.profile(target, 5)))
        # Five, not ten, for the same reason the Rust server gives: a
        # smaller pack read better and cost half the bytes.
        result = await engine.recall(
            target,
            query,
            limit=max(1, min(limit or 5, MAX_LIMIT)),
            as_of=as_of,
            tags=tags or (),
            where=where or {},
            kind=kind,
            source_prefix=source_prefix,
            since=since,
            until=until,
        )
        lines.extend(recall_lines(result))
        return ok_text("\n".join(lines) if lines else "no matching memory")

    @tool(server, "memory_facts_about")
    async def memory_facts_about(
        entity: Annotated[str, Field(description="Entity to look up (person, project, tool ...)")],
        space: Annotated[Optional[str], Field(description="Space to search; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """List what is currently known about one entity (active facts only)."""
        if not entity or len(entity) > MAX_ENTITY:
            return tool_error(f"entity must be 1..={MAX_ENTITY} chars, got {len(entity)}")
        found = await facts_about(engine, space or default_space, entity)
        if not found:
            return ok_text(f"no facts about {entity}")
        return ok_text("\n".join(fact_about_line(f) for f in found))

    @tool(server, "memory_graph_context")
    async def memory_graph_context(
        names: Annotated[
            Optional[list[str]], Field(description="Entities to centre on, by name or id (at most 24)")
        ] = None,
        question: Annotated[
            Optional[str], Field(description=f"A question; the entities its words name become the centre (1..={MAX_QUESTION} chars)")
        ] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
        max_bytes: Annotated[
            Optional[StrictInt], Field(description="Byte budget for the packet (512..=64000); defaults to 8000")
        ] = None,
        similar: Annotated[
            Optional[StrictBool],
            Field(description="Also centre on up to three entities the question resembles, each marked with its score"),
        ] = None,
        min_similarity: Annotated[
            Optional[float], Field(description="Keep out resembling entities below this cosine similarity (-1..=1)")
        ] = None,
    ) -> CallToolResult:
        """What the entity graph records around some names or the entities a
        question names: one line per item, coverage first, then the entities,
        paths between them, relations by hop and values. Each line cites its
        facts, re-read now; quotes appear only when they still verify."""
        if not names and not question:
            return tool_error("give names or a question")
        if len(names or ()) > MAX_NAMES or any(not 1 <= len(name) <= MAX_NAME for name in names or ()):
            return tool_error(f"names: at most {MAX_NAMES}, each 1..={MAX_NAME} chars")
        if question is not None and not 1 <= len(question) <= MAX_QUESTION:
            return tool_error(f"question must be 1..={MAX_QUESTION} chars")
        budget = max_bytes if max_bytes is not None else 8_000
        if not 512 <= budget <= 64_000:
            return tool_error("max_bytes must be 512..=64000")
        if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
            return tool_error("min_similarity must be -1..=1")
        packet = await graph_context(engine, space or default_space, names=names or (), question=question,
                                     limits=ContextLimits(max_bytes=budget), similar=bool(similar),
                                     min_similarity=min_similarity)
        return ok_text(packet.text)

    @tool(server, "memory_entity")
    async def memory_entity(
        name: Annotated[str, Field(description="The entity, by name or id")],
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """One entity: its relations in both directions and its values, each
        citing facts re-read now. An ambiguous name lists its candidates."""
        if not name or len(name) > MAX_ENTITY:
            return tool_error(f"name must be 1..={MAX_ENTITY} chars, got {len(name)}")
        packet = await graph_context(engine, space or default_space, names=[name], limits=ContextLimits(max_hops=1))
        return ok_text(packet.text)

    @tool(server, "memory_connections")
    async def memory_connections(
        source: Annotated[str, Field(description="One entity, by name or id")],
        target: Annotated[str, Field(description="The other entity, by name or id")],
        max_hops: Annotated[Optional[StrictInt], Field(description="Longest path to look for (1..=4); defaults to 3")] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """How two entities connect: the shortest paths between them, each hop
        with its direction and the facts behind it, re-read now."""
        if not source or not target or max(len(source), len(target)) > MAX_ENTITY:
            return tool_error(f"source and target must be 1..={MAX_ENTITY} chars")
        hops = max_hops if max_hops is not None else 3
        if not 1 <= hops <= 4:
            return tool_error("max_hops must be 1..=4")
        found = await graph_connections(engine, space or default_space, source, target, max_hops=hops)
        return ok_text(found.text)

    @tool(server, "memory_graph_schema")
    async def memory_graph_schema(
        limit: Annotated[
            Optional[StrictInt], Field(description=f"Predicates to list, most used first (1..={MAX_PREDICATES}); defaults to 200")
        ] = None,
        max_bytes: Annotated[
            Optional[StrictInt], Field(description="Byte budget for the listed predicates (1024..=64000); defaults to 8000")
        ] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """What the entity graph is made of: the kinds its entities have, the
        predicates its facts use and which kinds each joins. Read it first to
        know what the graph could be asked."""
        listed = limit if limit is not None else 200
        budget = max_bytes if max_bytes is not None else 8_000
        if not 1 <= listed <= MAX_PREDICATES:
            return tool_error(f"limit must be 1..={MAX_PREDICATES}")
        if not 1_024 <= budget <= 64_000:
            return tool_error("max_bytes must be 1024..=64000")
        return ok_text(schema_text(await schema_record(engine, space or default_space, limit=listed,
                                                       max_bytes=budget)))

    @tool(server, "memory_graph_match")
    async def memory_graph_match(
        where: Annotated[
            list[TriplePattern],
            Field(description=(f"1..={MAX_PATTERNS} triple patterns joined by shared variables; a term starting "
                               "with ? is a variable, as in {subject: '?who', predicate: 'works_at', object: '?org'}"
                               " and {subject: '?org', predicate: 'based_in', object: 'Lisbon'}")),
        ],
        returns: Annotated[
            Optional[list[str]], Field(description="The variables to answer with; defaults to every variable")
        ] = None,
        limit: Annotated[
            Optional[StrictInt], Field(description=f"Rows to answer (1..={MAX_ROWS}); defaults to {DEFAULT_ROWS}")
        ] = None,
        status: Annotated[
            Optional[StatusMode], Field(description="current (default), history, proposed or all")
        ] = None,
        as_of: Annotated[Optional[str], Field(description="RFC 3339 instant to ask at; defaults to now")] = None,
        together: Annotated[
            Optional[StrictBool],
            Field(description="Join only facts that held at one moment (default true); false joins across time"),
        ] = None,
        max_bytes: Annotated[
            Optional[StrictInt], Field(description="Byte budget for the answer (512..=64000); defaults to 8000")
        ] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """A structured question over the entity graph: every way the patterns
        hold together, one row per answer, each citing its facts re-read now.
        Constants name entities exactly (suggestions come back for near
        misses) and in object position match values too. Read
        memory_graph_schema first to know the predicates."""
        try:
            found = await graph_match(
                engine, space or default_space, [pattern.model_dump() for pattern in where], returns=returns,
                limit=limit if limit is not None else DEFAULT_ROWS, status=status or "current",
                as_of=normalise_time(as_of) if as_of is not None else None,
                together=True if together is None else together,
                max_bytes=max_bytes if max_bytes is not None else MATCH_BYTES)
        except MatchQueryError as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_graph_overview")
    async def memory_graph_overview(
        question: Annotated[
            Optional[str], Field(description=f"A question about the whole graph (1..={MAX_QUESTION} chars); "
                                             "the communities it concerns come first")
        ] = None,
        limit: Annotated[
            Optional[StrictInt], Field(description=f"Communities to digest (1..={MAX_COMMUNITIES}); defaults to "
                                                   f"{DEFAULT_COMMUNITIES}")
        ] = None,
        facts: Annotated[
            Optional[StrictInt], Field(description=f"Facts cited for each (0..={MAX_FACTS_EACH}); defaults to "
                                                   f"{DEFAULT_FACTS_EACH}")
        ] = None,
        max_bytes: Annotated[
            Optional[StrictInt], Field(description="Byte budget for the answer (512..=64000); defaults to 8000")
        ] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """The entity graph at a glance, for questions about the whole of it
        ("what are the main groups here?"): each community's size, kinds,
        predicates, central entities and a few facts, cited and re-read
        now. A question puts the communities it concerns first."""
        try:
            found = await graph_overview(
                engine, space or default_space, question=question,
                limit=limit if limit is not None else DEFAULT_COMMUNITIES,
                facts_each=facts if facts is not None else DEFAULT_FACTS_EACH,
                max_bytes=max_bytes if max_bytes is not None else OVERVIEW_BYTES)
        except OverviewError as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_graph_changes")
    async def memory_graph_changes(
        since: Annotated[str, Field(description="RFC 3339 moment to compare from, as in '2025-01-01T00:00:00Z'")],
        until: Annotated[Optional[str], Field(description="RFC 3339 moment to compare to; defaults to now")] = None,
        limit: Annotated[
            Optional[StrictInt], Field(description=f"Changes to list (1..={MAX_CHANGES}); defaults to {DEFAULT_CHANGES}")
        ] = None,
        max_bytes: Annotated[
            Optional[StrictInt], Field(description="Byte budget for the answer (512..=64000); defaults to 8000")
        ] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """What changed in the entity graph between two moments ("what changed
        since we last spoke?"): claims that moved from one object to
        another, relations that began and ended, values that changed and
        entities that came and went, each citing its facts re-read now."""
        try:
            found = await graph_changes(engine, space or default_space, since=since, until=until,
                                        limit=limit if limit is not None else DEFAULT_CHANGES,
                                        max_bytes=max_bytes if max_bytes is not None else CHANGES_BYTES)
        except ChangesError as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_entity_duplicates")
    async def memory_entity_duplicates(
        limit: Annotated[
            Optional[StrictInt], Field(description=f"Pairs to suggest (1..={MAX_PAIRS}); defaults to {DEFAULT_PAIRS}")
        ] = None,
        min_score: Annotated[
            Optional[float], Field(description="Suggest only pairs at least this likely (0..=1); defaults to 0.5")
        ] = None,
        max_bytes: Annotated[
            Optional[StrictInt], Field(description="Byte budget for the answer (512..=64000); defaults to 8000")
        ] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """Pairs of entities in the graph that may be one thing under two names
        ("Dr. Alice Chen" and "alice chen"), each saying why and citing the
        neighbours they share. Suggestions only: nothing is merged."""
        try:
            found = await likely_duplicates(engine, space or default_space,
                                            limit=limit if limit is not None else DEFAULT_PAIRS,
                                            min_score=min_score if min_score is not None else DEFAULT_MIN_SCORE,
                                            max_bytes=max_bytes if max_bytes is not None else DUPLICATES_BYTES)
        except DuplicatesError as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_graph_health")
    async def memory_graph_health(
        limit: Annotated[
            Optional[StrictInt], Field(description=f"Examples per concern (1..={HEALTH_MAX_EXAMPLES}); "
                                                   f"defaults to {HEALTH_EXAMPLES}")
        ] = None,
        max_bytes: Annotated[
            Optional[StrictInt], Field(description="Byte budget for the answer (512..=64000); defaults to 8000")
        ] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """What in the knowledge graph wants attention: claims resting on
        nothing, kinds that disagree or are missing, entities nothing links
        to, predicates used once, and names that may be one thing. Counts
        with examples; it changes nothing."""
        try:
            found = await graph_health(engine, space or default_space,
                                       limit=limit if limit is not None else HEALTH_EXAMPLES,
                                       max_bytes=max_bytes if max_bytes is not None else HEALTH_BYTES)
        except (HealthError, InvalidInput) as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_temporal_answer")
    async def memory_temporal_answer(
        question: Annotated[str, Field(description="A question about dates: how long between two events, how long "
                                                   "ago one was, which came first, what order they were in")],
        now: Annotated[
            Optional[str], Field(description="The moment to answer from (RFC 3339); defaults to now")
        ] = None,
        limit: Annotated[
            Optional[StrictInt], Field(description=f"Passages read for each event (1..={TEMPORAL_MAX_LIMIT}); "
                                                   f"defaults to {TEMPORAL_LIMIT}")
        ] = None,
        max_bytes: Annotated[
            Optional[StrictInt], Field(description=f"Byte budget for the answer (512..={TEMPORAL_BYTES_LIMIT}); "
                                                   f"defaults to {TEMPORAL_BYTES}")
        ] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """Answer a question about dates by computation rather than guesswork:
        each event named is grounded to a passage and the day it records, and
        the arithmetic is shown for checking. It says instead of computing
        when the question is not one it reads, when an event is not in
        memory, or when an event's day is not decided."""
        try:
            answer = await temporal_answer(engine, space or default_space, question, now=now,
                                           limit=limit if limit is not None else TEMPORAL_LIMIT,
                                           max_bytes=max_bytes if max_bytes is not None else TEMPORAL_BYTES)
        except (TemporalError, InvalidInput) as refused:
            return tool_error(str(refused))
        return ok_text(answer.text)

    def readable(space: str) -> str:
        """A space named in a resource address, refused with the reason."""
        try:
            check_space(space)
        except InvalidInput as refused:
            raise ResourceError(f"space: {refused}") from None
        return space

    async def report_markdown(space: str) -> str:
        return render_markdown(await report_record(engine, readable(space)))

    async def schema_lines_of(space: str) -> str:
        return schema_text(await schema_record(engine, readable(space), max_bytes=8_000))

    # The report and schema as resources a client can attach as context:
    # the server's own space at a fixed address, any other by name.
    @server.resource("scone://graph/report", name="graph-report", mime_type="text/markdown",
                     description="The knowledge report of the server's space: communities, central entities, "
                                 "surprising links and questions, each citing its facts.")
    async def own_report() -> str:
        return await report_markdown(default_space)

    @server.resource("scone://graph/schema", name="graph-schema", mime_type="text/plain",
                     description="What the server's space's graph is made of: kinds and predicates.")
    async def own_schema() -> str:
        return await schema_lines_of(default_space)

    server.resource("scone://{space}/graph/report", name="space-graph-report", mime_type="text/markdown",
                    description="The knowledge report of one space.")(report_markdown)
    server.resource("scone://{space}/graph/schema", name="space-graph-schema", mime_type="text/plain",
                    description="What one space's graph is made of.")(schema_lines_of)

    @tool(server, "memory_pending")
    async def memory_pending(
        limit: Annotated[Optional[int], Field(description="Max episodes to return (1..=20); defaults to 5")] = None,
        space: Annotated[Optional[str], Field(description="Space to inspect; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """List episodes awaiting fact extraction. YOU are the extractor:
        read each episode, distill durable subject/predicate/object facts
        with your own reasoning, then submit them via memory_store_facts."""
        episodes = await pending_episodes(engine, space or default_space, max(1, min(limit or 5, MAX_PENDING)))
        return ok_text(pending_text(episodes))

    @tool(server, "memory_store_facts")
    async def memory_store_facts(
        episode_id: Annotated[int, Field(description="Episode id from memory_pending")],
        facts: Annotated[list[SubmittedFact], Field(description="Extracted facts (max 50)")],
        space: Annotated[Optional[str], Field(description="Space of the episode; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """Submit facts you extracted from a pending episode.

        The engine applies contradiction closure and provenance: it
        decides what each fact supersedes and when it stopped holding,
        not you. What it does NOT do here is wait for a person. These
        become active ledger claims immediately unless this server runs
        with a propose gate, which parks them for review instead. If you
        are unsure of a fact, do not submit it: a wrong claim is easier to
        make than to find later.
        """
        if len(facts) > MAX_FACTS:
            return tool_error(f"at most {MAX_FACTS} facts per submission")
        target = space or default_space
        episode = await engine.episode(target, episode_id)
        before = await fact_ids_by_status(engine, target)
        for fact in facts:
            confidence = clamp_confidence(fact.confidence)
            await engine.assert_fact(
                target,
                fact.subject,
                fact.predicate,
                fact.object,
                valid_from=fact.valid_from or episode.created_at,
                confidence=confidence,
                source_episode_id=episode_id,
                origin="extracted",  # the host agent is a model reading an episode
                proposed=propose_below is not None and confidence < propose_below,
            )
        after = await fact_ids_by_status(engine, target)
        fresh = after.keys() - before.keys()
        # A proposal is not an addition to the ledger, and saying so would
        # tell an agent its claim is held when a person has not seen it.
        proposed = sum(1 for fid in fresh if after[fid] == "proposed")
        added = len(fresh) - proposed
        closed = sum(1 for fid, status in before.items() if status == "active" and after.get(fid) == "closed")
        counted = [f"{plural(added, 'fact')} added"]
        if proposed:  # silent when no gate is configured, which is the default
            counted.append(f"{proposed} proposed")
        counted += [f"{closed} closed", f"{len(facts) - len(fresh)} deduplicated"]
        return ok_text(f"episode {episode_id} distilled: " + ", ".join(counted))

    @tool(server, "memory_forget")
    async def memory_forget(
        fact_id: Annotated[int, Field(description="Fact id to close (from memory_recall / memory_facts_about output)")],
        reason: Annotated[str, Field(description="Why this fact should be forgotten (recorded, never deleted)")],
        space: Annotated[Optional[str], Field(description="Space of the fact; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """Forget a fact: closes its validity interval with your reason.
        History is preserved; nothing is deleted."""
        if not reason or len(reason) > MAX_REASON:
            return tool_error(f"reason must be 1..={MAX_REASON} chars, got {len(reason)}")
        await engine.close_fact(space or default_space, fact_id, reason)
        return ok_text(f"closed fact {fact_id}: {reason}")

    return server


# -- entry point --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scone_memory.runtime.mcp",
        description="Serve the memory engine over MCP stdio. Stores come from SCONE_* variables, as for the CLI.",
    )
    parser.add_argument("--space", default="default", help="space used when a call names none (default: default)")
    return parser


async def close_stores(engine: MemoryEngine) -> None:
    """Kept for callers that import it; the engine closes its own stores
    now, all four of them rather than the two this used to reach."""
    await engine.close()


async def serve(settings: Settings, space: str) -> None:
    engine = await build_engine(settings)
    try:
        await create_server(engine, space, settings.mcp_propose_below).run_stdio_async()
    finally:
        await close_stores(engine)


def main(argv: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_for_cli(os.environ if env is None else env)
    try:
        asyncio.run(serve(settings, args.space))
    except SconeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
