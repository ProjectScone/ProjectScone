"""One tool contract, rendered for whichever API is calling.

A model that can search, trace relationships, add memory, read a profile
and read the entity graph needs three things described once: what the
tools are called, what arguments they take, and what running them returns. Keeping that in one place is what
makes the OpenAI-shaped and Anthropic-shaped bindings share a contract.
Hosts choose which tools to offer and execute; this does not register tools
with separate HTTP, MCP or framework services automatically.

Two rules the executor keeps. A mistake is a result, never an exception:
a model that called a tool wrongly can read what went wrong and try
again, while a raised error ends its turn. And a tool reaches exactly
one space, the one the box was built for, because a tool call is not a
place to widen a key's reach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ..core.errors import InvalidInput, SconeError
from ..memory.engine import MemoryEngine

#: The most a search may return however the model asks.
MAX_ITEMS = 20
#: The graph tools' bounds, the same as the HTTP routes and the MCP server.
MAX_NAMES = 24
MAX_NAME = 200
MAX_QUESTION = 2000


@dataclass(frozen=True)
class ToolSpec:
    """A tool as a model sees it: a name, a sentence and a schema."""

    name: str
    summary: str
    parameters: dict

    def as_openai(self) -> dict:
        """The wrapper OpenAI-shaped APIs expect."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.summary, "parameters": self.parameters}}

    def as_anthropic(self) -> dict:
        """The wrapper Anthropic-shaped APIs expect. Same arguments."""
        return {"name": self.name, "description": self.summary, "input_schema": self.parameters}


def _schema(properties: dict, required: Sequence[str]) -> dict:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


MEMORY_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search_memory",
        summary=("Search everything remembered in this space and return the passages that match, "
                 "newest evidence included. Use it before answering from guesswork."),
        parameters=_schema({
            "query": {"type": "string", "description": "What to look for, in plain language."},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ITEMS,
                      "description": f"How many passages to return, 1 to {MAX_ITEMS}. Defaults to 5."},
            "tags": {"type": "array", "items": {"type": "string"},
                     "description": "Only passages carrying every one of these tags."},
        }, ["query"]),
    ),
    ToolSpec(
        name="add_memory",
        summary=("Remember something for later. Say it as a plain statement of fact; "
                 "the same thing twice is stored once."),
        parameters=_schema({
            "content": {"type": "string", "description": "What to remember, in full sentences."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags to find it by later."},
            "source": {"type": "string", "description": "Where it came from, if anywhere."},
        }, ["content"]),
    ),
    ToolSpec(
        name="read_profile",
        summary="Who this space is about: the claims that hold now, and what happened recently.",
        parameters=_schema({
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ITEMS,
                      "description": "How many recent items to include. Defaults to 5."},
        }, []),
    ),
    ToolSpec(
        name="trace_memory",
        summary=("Inspect a fact ID returned by search_memory or read_profile. Return its quoted claims, "
                 "stored relationships and ordered evidence paths within this space. Use this to "
                 "investigate connections before drawing conclusions; coverage can be partial."),
        parameters=_schema({
            "seed_fact_id": {"type": "integer", "minimum": 1, "maximum": 2**63 - 1,
                             "description": "The stored fact ID to start from."},
            "max_hops": {"type": "integer", "minimum": 1, "maximum": 6,
                         "description": "Maximum relationship steps, 1 to 6. Defaults to 3."},
            "tags": {"type": "array", "items": {"type": "string"},
                     "description": "Every supporting source must carry all these tags."},
        }, ["seed_fact_id"]),
    ),
    ToolSpec(
        name="graph_context",
        summary=("What the entity graph records around some names, or around the entities a question "
                 "names: one line per item, coverage first, then entities, the paths between them, "
                 "relations by hop and values. Every line cites the facts behind it, re-read now."),
        parameters=_schema({
            "names": {"type": "array", "items": {"type": "string"},
                      "description": f"Entities to centre on, by name or id; at most {MAX_NAMES}."},
            "question": {"type": "string", "description": "A question; the entities it names become the centre."},
            "max_bytes": {"type": "integer", "minimum": 512, "maximum": 64_000,
                          "description": ("Byte budget for the packet text, 512 to 64000. Defaults to 8000. "
                                          "Candidates and ids around it are capped separately.")},
        }, []),
    ),
    ToolSpec(
        name="explain_entity",
        summary=("One entity: its relations in both directions and its values, each citing its facts. "
                 "An ambiguous name lists the candidates to choose from."),
        parameters=_schema({
            "name": {"type": "string", "description": "The entity, by name or id."},
        }, ["name"]),
    ),
    ToolSpec(
        name="connect_entities",
        summary=("How two entities connect: the shortest paths between them, each hop with its "
                 "direction and the facts behind it. Nothing is shown that no fact supports."),
        parameters=_schema({
            "source": {"type": "string", "description": "One entity, by name or id."},
            "target": {"type": "string", "description": "The other entity, by name or id."},
            "max_hops": {"type": "integer", "minimum": 1, "maximum": 4,
                         "description": "Longest path to look for, 1 to 4. Defaults to 3."},
        }, ["source", "target"]),
    ),
    ToolSpec(
        name="graph_schema",
        summary=("What the entity graph is made of: the kinds its entities have, the predicates its "
                 "facts use and which kinds each joins, most used first. Read it to know what to ask."),
        parameters=_schema({
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000,
                      "description": "How many predicates to list, 1 to 1000. Defaults to 200."},
            "max_bytes": {"type": "integer", "minimum": 1024, "maximum": 64_000,
                          "description": "Byte budget for the listed predicates, 1024 to 64000. Defaults to 16000."},
        }, []),
    ),
)

_GRAPH_TOOLS = frozenset({"graph_context", "explain_entity", "connect_entities", "graph_schema"})

BY_NAME = {tool.name: tool for tool in MEMORY_TOOLS}


def _chosen(names: Optional[Sequence[str]]) -> tuple[ToolSpec, ...]:
    if names is None:
        return MEMORY_TOOLS
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        raise ValueError(f"unknown tool(s) {', '.join(unknown)}; there are {', '.join(BY_NAME)}")
    return tuple(BY_NAME[n] for n in names)


def openai_schema(names: Optional[Sequence[str]] = None) -> list[dict]:
    """The tools as an OpenAI-shaped API wants them."""
    return [tool.as_openai() for tool in _chosen(names)]


def anthropic_schema(names: Optional[Sequence[str]] = None) -> list[dict]:
    """The tools as an Anthropic-shaped API wants them."""
    return [tool.as_anthropic() for tool in _chosen(names)]


def check(spec: ToolSpec, arguments: Mapping[str, Any]) -> Optional[str]:
    """What is wrong with these arguments, or None. Enough of JSON Schema
    to catch what a model actually gets wrong, and no dependency."""
    properties = spec.parameters["properties"]
    for name in spec.parameters["required"]:
        if arguments.get(name) in (None, ""):
            return f"{spec.name} needs {name}"
    for name, value in arguments.items():
        rule = properties.get(name)
        if rule is None:
            return f"{spec.name} has no argument {name}; it takes {', '.join(properties)}"
        kind = rule["type"]
        if kind == "string" and not isinstance(value, str):
            return f"{name} must be text"
        if kind == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            return f"{name} must be a whole number"
        if kind == "array" and not (isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)):
            return f"{name} must be a list of text"
        if kind == "integer" and not rule.get("minimum", 0) <= value <= rule.get("maximum", MAX_ITEMS):
            return f"{name} must be from {rule.get('minimum')} to {rule.get('maximum')}"
    return None


class ToolBox:
    """The tools bound to one engine and one space."""

    def __init__(self, engine: MemoryEngine, space: str, *, tools: Optional[Sequence[str]] = None) -> None:
        self.engine = engine
        self.space = space
        self.tools = _chosen(tools)
        self._by_name = {tool.name: tool for tool in self.tools}

    def openai(self) -> list[dict]:
        return [tool.as_openai() for tool in self.tools]

    def anthropic(self) -> list[dict]:
        return [tool.as_anthropic() for tool in self.tools]

    async def run(self, name: str, arguments: Mapping[str, Any]) -> dict:
        """Run one tool call. The answer is always a dict with ``ok``:
        a model reads a refusal and tries again, where an exception would
        end its turn."""
        spec = self._by_name.get(name)
        if spec is None:
            return {"ok": False, "error": f"there is no tool {name}; there is {', '.join(self._by_name)}"}
        wrong = check(spec, arguments)
        if wrong is not None:
            return {"ok": False, "error": wrong}
        try:
            return {"ok": True, "space": self.space, **await self._call(name, arguments)}
        except SconeError as error:
            return {"ok": False, "error": f"{type(error).__name__}: {error}"}

    async def _call(self, name: str, arguments: Mapping[str, Any]) -> dict:
        if name in _GRAPH_TOOLS:
            return await self._graph(name, arguments)
        if name == "search_memory":
            found = await self.engine.recall(
                self.space, arguments["query"],
                limit=min(int(arguments.get("limit") or 5), MAX_ITEMS),
                tags=tuple(arguments.get("tags") or ()),
            )
            return {
                "items": [{"episode_id": i.episode_id, "chunk_id": i.chunk_id, "text": i.text,
                           "score": i.score, "source": i.source, "created_at": i.created_at}
                          for i in found.items],
                "facts": [self._fact(f) for f in found.facts],
            }
        if name == "add_memory":
            added = await self.engine.remember(
                self.space, arguments["content"],
                tags=tuple(arguments.get("tags") or ()), source=arguments.get("source"),
            )
            return {"episode_id": added.episode_id, "outcome": added.outcome}
        if name == "trace_memory":
            from .relations import trace_memory

            return await trace_memory(self.engine, self.space, arguments["seed_fact_id"],
                                      max_hops=arguments.get("max_hops", 3), tags=arguments.get("tags", ()))
        profile = await self.engine.profile(self.space, limit=min(int(arguments.get("limit") or 5), MAX_ITEMS))
        return {
            "facts": [self._fact(f) for f in profile.static_facts],
            "recent": [{"episode_id": r.episode_id, "text": r.excerpt, "created_at": r.created_at}
                       for r in profile.recent],
        }

    async def _graph(self, name: str, arguments: Mapping[str, Any]) -> dict:
        """The graph tools read the current projection at one instant and
        answer with the packet the HTTP route gives."""
        from ..entities.context import ContextLimits, graph_connections, graph_context
        from ..entities.schema import schema_record

        names = list(arguments.get("names") or ())
        for side in ("name", "source", "target"):
            if side in arguments:
                names.append(arguments[side])
        if len(names) > MAX_NAMES or any(not 1 <= len(entry) <= MAX_NAME for entry in names):
            raise InvalidInput(f"names: at most {MAX_NAMES}, each 1 to {MAX_NAME} characters")
        when = self.engine.clock()
        if name == "graph_schema":
            return await schema_record(self.engine, self.space, as_of=when, limit=arguments.get("limit", 200),
                                       max_bytes=arguments.get("max_bytes", 16_000))
        if name == "connect_entities":
            packet = await graph_connections(self.engine, self.space, arguments["source"], arguments["target"],
                                             max_hops=arguments.get("max_hops", 3), as_of=when)
        elif name == "explain_entity":
            packet = await graph_context(self.engine, self.space, names=[arguments["name"]], as_of=when,
                                         limits=ContextLimits(max_hops=1))
        else:
            question = arguments.get("question")
            if not names and not question:
                raise InvalidInput("give names or a question")
            if question is not None and not 1 <= len(question) <= MAX_QUESTION:
                raise InvalidInput(f"question must be 1 to {MAX_QUESTION} characters")
            packet = await graph_context(self.engine, self.space, names=names, question=question, as_of=when,
                                         limits=ContextLimits(max_bytes=arguments.get("max_bytes", 8_000)))
        return packet.record(self.space, "current", when)

    @staticmethod
    def _fact(fact) -> dict:
        return {"fact_id": fact.fact_id, "subject": fact.subject, "predicate": fact.predicate,
                "object": fact.object, "confidence": fact.confidence, "source_episode_id": fact.source_episode_id}
