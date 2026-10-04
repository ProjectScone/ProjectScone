"""memory tools registration without transport ownership."""
from typing import Annotated, Callable, Optional


from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult
from pydantic import Field

from ...core.bearer_keys import KeyHolder
from ...memory.engine import MemoryEngine

from .registry import (
    SubmittedFact,
    tool_error,
    ok_text,
    tool,
    describe_added,
    profile_lines,
    recall_lines,
    fact_about_line,
    pending_text,
    plural,
    facts_about,
    pending_episodes,
    clamp_confidence,
    fact_ids_by_status,
    MAX_CONTENT,
    MAX_QUERY,
    MAX_ENTITY,
    MAX_REASON,
    MAX_LIMIT,
    MAX_TAGS,
    MAX_FACTS,
    MAX_PENDING,
)

def register_memory_tools(server: MCPServer, engine: MemoryEngine, chosen: Callable[[Optional[str]], str], holder: Optional[KeyHolder] = None, propose_below: Optional[float] = None) -> None:
    @tool(server, "memory_store", holder)
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
        added = await engine.remember(chosen(space), content, tags=tags or (), metadata=metadata)
        return ok_text(describe_added(added))

    @tool(server, "memory_recall", holder)
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
        target = chosen(space)
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

    @tool(server, "memory_facts_about", holder)
    async def memory_facts_about(
        entity: Annotated[str, Field(description="Entity to look up (person, project, tool ...)")],
        space: Annotated[Optional[str], Field(description="Space to search; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """List what is currently known about one entity (active facts only)."""
        if not entity or len(entity) > MAX_ENTITY:
            return tool_error(f"entity must be 1..={MAX_ENTITY} chars, got {len(entity)}")
        found = await facts_about(engine, chosen(space), entity)
        if not found:
            return ok_text(f"no facts about {entity}")
        return ok_text("\n".join(fact_about_line(f) for f in found))

    @tool(server, "memory_pending", holder)
    async def memory_pending(
        limit: Annotated[Optional[int], Field(description="Max episodes to return (1..=20); defaults to 5")] = None,
        space: Annotated[Optional[str], Field(description="Space to inspect; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """List episodes awaiting fact extraction. YOU are the extractor:
        read each episode, distill durable subject/predicate/object facts
        with your own reasoning, then submit them via memory_store_facts."""
        episodes = await pending_episodes(engine, chosen(space), max(1, min(limit or 5, MAX_PENDING)))
        return ok_text(pending_text(episodes))

    @tool(server, "memory_store_facts", holder)
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
        target = chosen(space)
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

    @tool(server, "memory_forget", holder)
    async def memory_forget(
        fact_id: Annotated[int, Field(description="Fact id to close (from memory_recall / memory_facts_about output)")],
        reason: Annotated[str, Field(description="Why this fact should be forgotten (recorded, never deleted)")],
        space: Annotated[Optional[str], Field(description="Space of the fact; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """Forget a fact: closes its validity interval with your reason.
        History is preserved; nothing is deleted."""
        if not reason or len(reason) > MAX_REASON:
            return tool_error(f"reason must be 1..={MAX_REASON} chars, got {len(reason)}")
        await engine.close_fact(chosen(space), fact_id, reason)
        return ok_text(f"closed fact {fact_id}: {reason}")

