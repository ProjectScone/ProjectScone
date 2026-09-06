"""MCP server: the engine as six tools for any MCP agent.

Mirrors ``crates/scone/src/mcp.rs``: the same tool names, the same
argument names, the same input bounds, and the same plain-text result
shape, so an agent configured against the Rust server can point at this
one unchanged. Invalid input comes back as a tool result with
``is_error`` set, never as an exception, so the model can read the
reason and correct the call.

    python -m scone_memory.mcp --space default

Stores come from the environment exactly as for the CLI (see
``scone_memory.config``); with nothing set, memory persists to SQLite at
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
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, Field

from . import __version__
from .cli import settings_for_cli
from .config import Settings, build_engine
from .engine import MemoryEngine, Profile, check_space, normalise_term
from .errors import SconeError
from .models import Added, Episode, Fact, RecallResult

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
    "model and no API key is needed."
)


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
    return {f.fact_id: f.status for f in await engine.facts(space, include_closed=True)}


# -- the server ---------------------------------------------------------------


def create_server(engine: MemoryEngine, space: str = "default") -> MCPServer:
    """The six tools of mcp.rs, closing over one engine and a default
    space that any call may override with its ``space`` argument."""
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
        """Submit facts you extracted from a pending episode. The engine
        applies contradiction closure and provenance; you only propose."""
        if len(facts) > MAX_FACTS:
            return tool_error(f"at most {MAX_FACTS} facts per submission")
        target = space or default_space
        episode = await engine.episode(target, episode_id)
        before = await fact_ids_by_status(engine, target)
        for fact in facts:
            await engine.assert_fact(
                target,
                fact.subject,
                fact.predicate,
                fact.object,
                valid_from=fact.valid_from or episode.created_at,
                confidence=clamp_confidence(fact.confidence),
                source_episode_id=episode_id,
            )
        after = await fact_ids_by_status(engine, target)
        added = len(after.keys() - before.keys())
        closed = sum(1 for fid, status in before.items() if status == "active" and after.get(fid) == "closed")
        return ok_text(
            f"episode {episode_id} distilled: {plural(added, 'fact')} added, "
            f"{closed} closed, {len(facts) - added} deduplicated"
        )

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
        prog="python -m scone_memory.mcp",
        description="Serve the memory engine over MCP stdio. Stores come from SCONE_* variables, as for the CLI.",
    )
    parser.add_argument("--space", default="default", help="space used when a call names none (default: default)")
    return parser


async def close_stores(engine: MemoryEngine) -> None:
    for store in (engine.documents, engine.vectors):
        if hasattr(store, "close"):
            await store.close()


async def serve(settings: Settings, space: str) -> None:
    engine = await build_engine(settings)
    try:
        await create_server(engine, space).run_stdio_async()
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
