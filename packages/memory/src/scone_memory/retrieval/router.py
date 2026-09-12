"""One question, and the machinery that suits it.

This framework has grown a computed temporal answer, an entity graph and
two retrieval lanes. That leaves a caller with a problem it did not have
when there was only recall: knowing which to ask. The leading frameworks
answer that with a model writing a plan, which is expensive, and
inscrutable when it is wrong.

Here the rule is written down, in order, and every answer says which way
it went and why:

1. **A question about dates** goes to the one that computes, but only if
   it can actually ground it. "Looks temporal" is not enough: a question
   whose events are not in the ledger is better served by the passages
   than by a refusal, and the answer says that is what happened.
2. **A question about something the graph knows** goes to the graph, so
   that "what is Alice's employer in?" is answered by the claims rather
   than by hoping a passage says it.
3. **Anything else** is an ordinary search.

A caller who disagrees can name the route; a rule that cannot be
overridden is a rule somebody will work around, and then the framework
learns nothing from being wrong. The answer says the route was asked for
rather than chosen.

Nothing here calls a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from ..core.errors import InvalidInput
from ..core.validation import check_space

if TYPE_CHECKING:
    from ..memory.engine import MemoryEngine

#: The ways a question can be answered, in the order they are tried.
ROUTES = ("temporal", "graph", "recall")
Route = Literal["temporal", "graph", "recall"]
#: Passages an ordinary search answers with, by default and at most.
DEFAULT_LIMIT = 5
MAX_LIMIT = 50


@dataclass(frozen=True)
class Answered:
    """What was asked, which way it went, why, and what came back."""

    question: str
    route: str
    why: str
    text: str
    #: The answer from the machinery that served it, as that machinery
    #: gives it, so nothing is lost by going through here.
    detail: dict[str, object] = field(default_factory=dict)

    def record(self, space: str) -> dict[str, object]:
        return {"schema_version": 1, "space": space, "question": self.question, "route": self.route,
                "why": self.why, "text": self.text, "detail": dict(self.detail)}


async def answer_question(engine: "MemoryEngine", space: str, question: str, *,
                          now: Optional[str] = None, limit: int = DEFAULT_LIMIT,
                          route: Optional[str] = None) -> Answered:
    """Answer a question with whichever machinery suits it, and say which."""
    check_space(space)
    if not isinstance(question, str) or not question.strip():
        raise InvalidInput("a question must have something in it")
    if not 1 <= limit <= MAX_LIMIT:
        raise InvalidInput(f"limit must be from 1 to {MAX_LIMIT}")
    if route is not None and route not in ROUTES:
        raise InvalidInput(f"route must be one of {', '.join(ROUTES)}, not {route!r}")
    when = now or engine.clock()
    if route is not None:
        return await _by(engine, space, question, route, when, limit,
                         why=f"the {route} route was asked for")
    computed, unread = await _temporal(engine, space, question, when, limit)
    if computed is not None:
        return computed
    named = await _graph(engine, space, question, when, limit)
    if named is not None:
        return named
    # The reason a search was reached matters: a question the computer
    # could not ground is a different thing from a question it never read
    # as being about dates, and a caller who cannot tell them apart cannot
    # tell a gap in the ledger from a gap in the rule.
    return await _recall(engine, space, question, limit,
                         why=unread or "nothing it could compute and nothing the graph knows by name")


async def _by(engine: "MemoryEngine", space: str, question: str, route: str, when: str,
              limit: int, why: str) -> Answered:
    """One named route, whatever the rule would have chosen."""
    if route == "temporal":
        found, _ = await _temporal(engine, space, question, when, limit, insist=True)
        return found if found is not None else Answered(question, "temporal", why, "", {})
    if route == "graph":
        found = await _graph(engine, space, question, when, limit, insist=True)
        return found if found is not None else Answered(question, "graph", why, "", {})
    return await _recall(engine, space, question, limit, why=why)


async def _temporal(engine: "MemoryEngine", space: str, question: str, when: str, limit: int,
                    insist: bool = False) -> tuple[Optional[Answered], Optional[str]]:
    """The computed answer when there is one, and otherwise why not — which
    a later route repeats, so a caller can tell a gap in the ledger from a
    gap in the rule."""
    from .temporal import temporal_answer

    answer = await temporal_answer(engine, space, question, now=when, limit=limit)
    if answer.status in ("computed", "recalled"):
        return Answered(question, "temporal",
                        f"it asks about dates and the ledger grounds it ({answer.status})",
                        answer.text, answer.record(space)), None
    if insist:
        return Answered(question, "temporal", f"the temporal route was asked for ({answer.status})",
                        answer.text, answer.record(space)), None
    if answer.status == "not_temporal":
        return None, None
    # It reads as a question about dates and the events are not in the
    # ledger. The passages are a better answer than a refusal, and the
    # caller is told which of the two happened.
    return None, (f"it reads as a question about dates, but the ledger could not ground it "
                  f"({answer.status})")


async def _graph(engine: "MemoryEngine", space: str, question: str, when: str, limit: int,
                 insist: bool = False) -> Optional[Answered]:
    """What the graph records around the things the question names."""
    from ..entities.context import ContextLimits, graph_context

    packet = await graph_context(engine, space, question=question, as_of=when,
                                 limits=ContextLimits(max_hops=1))
    if packet.status != "prepared" and not insist:
        return None
    # A reason somebody reads names the thing, not its identifier.
    named = ", ".join(_labels(packet.text)[:3])
    return Answered(question, "graph",
                    (f"the graph knows {named} by name" if named
                     else "the graph route was asked for"),
                    packet.text, packet.record(space, "current", when))


def _labels(text: str) -> list[str]:
    """The entities a packet says it is about, as it labels them."""
    found = []
    for line in text.splitlines():
        if line.startswith("entity: "):
            found.append(line[len("entity: "):].split(" (")[0])
    return found


async def _recall(engine: "MemoryEngine", space: str, question: str, limit: int, why: str) -> Answered:
    """The ordinary search, which is the answer when nothing else is."""
    found = await engine.recall(space, question, limit=limit)
    lines = [f"{item.score:.2f}  {item.text.strip()[:200]}" for item in found.items]
    return Answered(question, "recall", why, "\n".join(lines) or "nothing found",
                    {"items": [item.model_dump() for item in found.items],
                     "facts": [fact.model_dump() for fact in found.facts]})
