"""One tool contract, rendered for whichever API is calling.

A model that can search memory, add to it and read a profile needs three
things described once: what the tools are called, what arguments they
take, and what running them returns. Keeping that in one place is what
makes an OpenAI-shaped binding, an Anthropic-shaped one, MCP and an
agent framework offer the same tools rather than three dialects that
drift apart.

Two rules the executor keeps. A mistake is a result, never an exception:
a model that called a tool wrongly can read what went wrong and try
again, while a raised error ends its turn. And a tool reaches exactly
one space, the one the box was built for, because a tool call is not a
place to widen a key's reach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ..core.errors import SconeError
from ..memory.engine import MemoryEngine

#: The most a search may return however the model asks.
MAX_ITEMS = 20


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
)

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
        profile = await self.engine.profile(self.space, limit=min(int(arguments.get("limit") or 5), MAX_ITEMS))
        return {
            "facts": [self._fact(f) for f in profile.static_facts],
            "recent": [{"episode_id": r.episode_id, "text": r.excerpt, "created_at": r.created_at}
                       for r in profile.recent],
        }

    @staticmethod
    def _fact(fact) -> dict:
        return {"fact_id": fact.fact_id, "subject": fact.subject, "predicate": fact.predicate,
                "object": fact.object, "confidence": fact.confidence, "source_episode_id": fact.source_episode_id}
