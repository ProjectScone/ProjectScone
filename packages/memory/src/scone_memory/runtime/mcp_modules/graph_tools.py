"""graph tools registration without transport ownership."""
from typing import Annotated, Callable, Optional


from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError
from mcp.types import CallToolResult
from pydantic import Field, StrictBool, StrictInt

from ...core.bearer_keys import KeyHolder
from ...core.errors import InvalidInput, NotFound
from ...memory.engine import MemoryEngine, check_space
from ...entities.context import (
    MAX_NAME,
    MAX_NAMES,
    MAX_QUESTION,
    ContextLimits,
    graph_connections,
    graph_context,
)
from ...entities.report import render_markdown, report_record
from ...entities.schema import MAX_PREDICATES, schema_record, schema_text
from ...retrieval.temporal import (
    DEFAULT_LIMIT as TEMPORAL_LIMIT,
    MAX_BYTES as TEMPORAL_BYTES,
    MAX_BYTES_LIMIT as TEMPORAL_BYTES_LIMIT,
    MAX_LIMIT as TEMPORAL_MAX_LIMIT,
    TemporalError,
    temporal_answer,
)
from ...entities.health import (
    DEFAULT_EXAMPLES as HEALTH_EXAMPLES,
    MAX_BYTES as HEALTH_BYTES,
    MAX_EXAMPLES as HEALTH_MAX_EXAMPLES,
    HealthError,
    graph_health,
)
from ...entities.duplicates import (
    DEFAULT_MIN_SCORE,
    DEFAULT_PAIRS,
    MAX_BYTES as DUPLICATES_BYTES,
    MAX_PAIRS,
    DuplicatesError,
    likely_duplicates,
)
from ...entities.changes import (
    DEFAULT_CHANGES,
    MAX_BYTES as CHANGES_BYTES,
    MAX_CHANGES,
    ChangesError,
    graph_changes,
)
from ...entities.affected import (
    MAX_BYTES as AFFECTED_BYTES,
    MAX_HOPS as AFFECTED_HOPS,
    MAX_REACHED,
    affected,
)
from ...entities.overview import (
    DEFAULT_COMMUNITIES,
    DEFAULT_FACTS_EACH,
    MAX_BYTES as OVERVIEW_BYTES,
    MAX_COMMUNITIES,
    MAX_FACTS_EACH,
    OverviewError,
    graph_overview,
)
from ...entities.match import (
    DEFAULT_ROWS,
    MAX_BYTES as MATCH_BYTES,
    MAX_PATTERNS,
    MAX_ROWS,
    MatchQueryError,
    graph_match,
)
from ...entities.view import StatusMode
from ...core.validation import normalise_time

from .registry import TriplePattern, tool_error, ok_text, tool, MAX_ENTITY

def register_graph_tools(server: MCPServer, engine: MemoryEngine, chosen: Callable[[Optional[str]], str], default_space: str, holder: Optional[KeyHolder] = None) -> None:
    @tool(server, "memory_graph_context", holder)
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
        packet = await graph_context(engine, chosen(space), names=names or (), question=question,
                                     limits=ContextLimits(max_bytes=budget), similar=bool(similar),
                                     min_similarity=min_similarity)
        return ok_text(packet.text)

    @tool(server, "memory_entity", holder)
    async def memory_entity(
        name: Annotated[str, Field(description="The entity, by name or id")],
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """One entity: its relations in both directions and its values, each
        citing facts re-read now. An ambiguous name lists its candidates."""
        if not name or len(name) > MAX_ENTITY:
            return tool_error(f"name must be 1..={MAX_ENTITY} chars, got {len(name)}")
        packet = await graph_context(engine, chosen(space), names=[name], limits=ContextLimits(max_hops=1))
        return ok_text(packet.text)

    @tool(server, "memory_connections", holder)
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
        found = await graph_connections(engine, chosen(space), source, target, max_hops=hops)
        return ok_text(found.text)

    @tool(server, "memory_graph_schema", holder)
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
        return ok_text(schema_text(await schema_record(engine, chosen(space), limit=listed,
                                                       max_bytes=budget)))

    @tool(server, "memory_graph_match", holder)
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
                engine, chosen(space), [pattern.model_dump() for pattern in where], returns=returns,
                limit=limit if limit is not None else DEFAULT_ROWS, status=status or "current",
                as_of=normalise_time(as_of) if as_of is not None else None,
                together=True if together is None else together,
                max_bytes=max_bytes if max_bytes is not None else MATCH_BYTES)
        except MatchQueryError as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_graph_overview", holder)
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
                engine, chosen(space), question=question,
                limit=limit if limit is not None else DEFAULT_COMMUNITIES,
                facts_each=facts if facts is not None else DEFAULT_FACTS_EACH,
                max_bytes=max_bytes if max_bytes is not None else OVERVIEW_BYTES)
        except OverviewError as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_graph_changes", holder)
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
            found = await graph_changes(engine, chosen(space), since=since, until=until,
                                        limit=limit if limit is not None else DEFAULT_CHANGES,
                                        max_bytes=max_bytes if max_bytes is not None else CHANGES_BYTES)
        except ChangesError as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_entity_duplicates", holder)
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
            found = await likely_duplicates(engine, chosen(space),
                                            limit=limit if limit is not None else DEFAULT_PAIRS,
                                            min_score=min_score if min_score is not None else DEFAULT_MIN_SCORE,
                                            max_bytes=max_bytes if max_bytes is not None else DUPLICATES_BYTES)
        except DuplicatesError as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_graph_affected", holder)
    async def memory_graph_affected(
        name: Annotated[str, Field(description="The symbol (`pkg/mod.py:Class.method`), module, file or package",
                                   min_length=1, max_length=200)],
        max_hops: Annotated[
            Optional[StrictInt], Field(description=f"Relationship steps to follow (1..={AFFECTED_HOPS}); defaults to 4")
        ] = None,
        limit: Annotated[
            Optional[StrictInt], Field(description=f"Entities to list (1..={MAX_REACHED}); defaults to 200")
        ] = None,
        max_bytes: Annotated[
            Optional[StrictInt], Field(description=f"Byte budget for the answer's record (1024..={AFFECTED_BYTES}); "
                                                   "its framing takes most of a kilobyte; defaults to 16000")
        ] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """What rests on a symbol, module, file or package: everything that
        calls, imports, inherits, mixes in, depends on or develops with it,
        nearest first, through the code graph's recorded relations. Says how
        deep it walked, what it could not list, and when nothing here rests
        on the name; an ambiguous name is refused with its candidates, an
        unknown one plainly."""
        blast = await affected(engine, chosen(space), name,
                               max_hops=max_hops if max_hops is not None else 4,
                               limit=limit if limit is not None else 200,
                               max_bytes=max_bytes if max_bytes is not None else 16_000)
        if blast.status not in ("found", "nothing"):
            return tool_error(blast.why)
        lines = [blast.why]
        lines += [f"{one.depth}: {one.label} ({one.through} {one.depends_on})" for one in blast.reached]
        if blast.by_depth:
            lines.append("by depth: " + ", ".join(f"{depth} hop(s): {count}"
                                                  for depth, count in sorted(blast.by_depth.items())))
        if blast.not_listed:
            lines.append(f"{blast.not_listed} more reached but not listed (limit or byte budget)")
        if blast.stopped_at_depth:
            lines.append(f"stopped at {blast.deepest} hop(s) with more to follow"
                         + ("; raise max_hops to go further" if blast.deepest < AFFECTED_HOPS else ""))
        return ok_text("\n".join(lines))

    @tool(server, "memory_graph_cycles", holder)
    async def memory_graph_cycles(
        limit: Annotated[
            Optional[StrictInt], Field(description="Groups shown of each kind (1..=100); defaults to 20")
        ] = None,
        max_bytes: Annotated[
            Optional[StrictInt], Field(description="Byte budget for the answer (512..=64000); defaults to 8000")
        ] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """Dependency cycles in the code graph: files that cannot load
        without each other, each with one shortest loop and the facts behind
        every hop; and loops held apart only by imports that run when called
        or never. Counts with examples; it changes nothing."""
        from ...entities.cycles import DEFAULT_LIMIT as CYCLES_LIMIT, MAX_BYTES as CYCLES_BYTES, CyclesError, graph_cycles

        try:
            found = await graph_cycles(engine, chosen(space),
                                       limit=limit if limit is not None else CYCLES_LIMIT,
                                       max_bytes=max_bytes if max_bytes is not None else CYCLES_BYTES)
        except (CyclesError, InvalidInput) as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_graph_stats", holder)
    async def memory_graph_stats(
        max_bytes: Annotated[
            Optional[StrictInt], Field(description="Byte budget for the answer (512..=64000); defaults to 8000")
        ] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """The knowledge graph counted: entities, relations, attributes,
        communities and modularity, isolated and external entities, kinds,
        predicates, and the facts by origin, grounding and standing. Counts
        over recorded data; it changes nothing."""
        from ...entities.stats import MAX_BYTES as STATS_BYTES, StatsError, graph_stats

        try:
            found = await graph_stats(engine, chosen(space),
                                      max_bytes=max_bytes if max_bytes is not None else STATS_BYTES)
        except (StatsError, InvalidInput) as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_graph_hubs", holder)
    async def memory_graph_hubs(
        limit: Annotated[
            Optional[StrictInt], Field(description="Hubs shown, most linked first (1..=100); defaults to 10")
        ] = None,
        above: Annotated[
            Optional[float], Field(description="Only the hubs above this degree percentile (50..=100), as the "
                                               "report holds them apart; defaults to none")
        ] = None,
        max_bytes: Annotated[
            Optional[StrictInt], Field(description="Byte budget for the answer (512..=64000); defaults to 8000")
        ] = None,
        space: Annotated[Optional[str], Field(description="Space to read; defaults to the server's space")] = None,
    ) -> CallToolResult:
        """The entities with the most neighbours, the graph's core abstractions
        or its utility hubs, each with its degree, fact weight, PageRank and
        community. It changes nothing."""
        from ...entities.stats import DEFAULT_HUBS, MAX_BYTES as STATS_BYTES, StatsError, graph_hubs

        try:
            found = await graph_hubs(engine, chosen(space), above=above,
                                     limit=limit if limit is not None else DEFAULT_HUBS,
                                     max_bytes=max_bytes if max_bytes is not None else STATS_BYTES)
        except (StatsError, InvalidInput) as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_graph_health", holder)
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
            found = await graph_health(engine, chosen(space),
                                       limit=limit if limit is not None else HEALTH_EXAMPLES,
                                       max_bytes=max_bytes if max_bytes is not None else HEALTH_BYTES)
        except (HealthError, InvalidInput) as refused:
            return tool_error(str(refused))
        return ok_text(found.text)

    @tool(server, "memory_temporal_answer", holder)
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
            answer = await temporal_answer(engine, chosen(space), question, now=now,
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
        try:
            return chosen(space)
        except NotFound as refused:
            # A resource error travels with its reason; anything else the
            # SDK reports as "error creating resource", which tells the
            # caller nothing about why this address was refused.
            raise ResourceError(str(refused)) from None

    async def report_markdown(space: str) -> str:
        return render_markdown(await report_record(engine, readable(space)))

    async def schema_lines_of(space: str) -> str:
        return schema_text(await schema_record(engine, readable(space), max_bytes=8_000))

    async def health_lines_of(space: str) -> str:
        return (await graph_health(engine, readable(space), max_bytes=8_000)).text

    async def stats_lines_of(space: str) -> str:
        from ...entities.stats import graph_stats

        return (await graph_stats(engine, readable(space), max_bytes=8_000)).text

    async def hubs_lines_of(space: str) -> str:
        from ...entities.stats import graph_hubs

        return (await graph_hubs(engine, readable(space), max_bytes=8_000)).text

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

    @server.resource("scone://graph/health", name="graph-health", mime_type="text/plain",
                     description="What in the server's space's graph wants attention: claims resting on nothing, "
                                 "kinds that disagree or are missing, entities nothing links to, predicates used "
                                 "once, and names that may be one thing.")
    async def own_health() -> str:
        return await health_lines_of(default_space)

    server.resource("scone://{space}/graph/health", name="space-graph-health", mime_type="text/plain",
                    description="What in one space's graph wants attention.")(health_lines_of)

    @server.resource("scone://graph/stats", name="graph-stats", mime_type="text/plain",
                     description="The server's space's graph counted: entities, relations, attributes, communities, "
                                 "kinds, predicates, and the facts by origin, grounding and standing.")
    async def own_stats() -> str:
        return await stats_lines_of(default_space)

    @server.resource("scone://graph/hubs", name="graph-hubs", mime_type="text/plain",
                     description="The server's space's most linked entities, with degree, facts, PageRank and "
                                 "community.")
    async def own_hubs() -> str:
        return await hubs_lines_of(default_space)

    server.resource("scone://{space}/graph/stats", name="space-graph-stats", mime_type="text/plain",
                    description="One space's graph counted.")(stats_lines_of)
    server.resource("scone://{space}/graph/hubs", name="space-graph-hubs", mime_type="text/plain",
                    description="One space's most linked entities.")(hubs_lines_of)

    server.resource("scone://{space}/graph/report", name="space-graph-report", mime_type="text/markdown",
                    description="The knowledge report of one space.")(report_markdown)
    server.resource("scone://{space}/graph/schema", name="space-graph-schema", mime_type="text/plain",
                    description="What one space's graph is made of.")(schema_lines_of)
