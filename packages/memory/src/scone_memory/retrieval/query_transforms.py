"""A question restated by a model as the words a passage would use, and said so.

The lexical lane finds the words a passage has, and a question is not
written in them: "What happened with the bills?" misses a passage about
the billing run, and no embedder is asked to bridge every such gap. A
model can restate the question as the words the passage that answers
it would contain. That is all it does here, and the restatement is
taken only when it can be read, is not empty, is within the query
bound and still shares a word with the question -- a rewrite that
shares nothing is a different question. In every other case the
question itself is searched, and the record says why: ``applied`` is
false and ``reason`` names the cause. A recall made this way carries
the transform beside its results, so a reader can see what was
actually searched.

Nothing here runs by default. A transform is an explicit call, and its
worth is a number on a benchmark before it is anything else.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..core.errors import InvalidInput
from ..core.validation import MAX_QUERY
from ..providers.llm import ChatModel
from .lexical import tokenize

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.models import RecallResult
    from ..memory.engine import MemoryEngine

#: Bytes a question may carry into a prompt; more is refused.
MAX_QUESTION_BYTES = 8_000

_REWRITE_SYSTEM = (
    "You restate a question as the words the passage that answers it would contain, for a keyword search over "
    "someone's own notes and conversations. Reply with JSON only, of the form {\"query\": \"...\"}: a short search "
    "query holding the question's own key terms and the words a matching passage would use; keep names, dates "
    "and numbers exactly as written; no answer, no explanation."
)


@dataclass(frozen=True)
class Transformed:
    """What was asked, what was searched, and whether the model's version was taken."""

    question: str
    query: str
    applied: bool
    #: Why the question itself was searched, when ``applied`` is false.
    reason: Optional[str]
    model_calls: int
    kind: str = "rewrite"

    def record(self) -> dict[str, object]:
        return {"kind": self.kind, "question": self.question, "query": self.query, "applied": self.applied,
                "reason": self.reason, "model_calls": self.model_calls}


def _object(reply: object) -> Optional[dict[str, object]]:
    """The first JSON object in the reply, or None; a model may wrap it in prose, not omit it."""
    if not isinstance(reply, str):
        return None
    start = reply.find("{")
    while start >= 0:
        depth = 0
        for index in range(start, len(reply)):
            if reply[index] == "{":
                depth += 1
            elif reply[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(reply[start:index + 1])
                    except ValueError:
                        break
                    return value if isinstance(value, dict) else None
        start = reply.find("{", start + 1)
    return None


def _kept(question: str, reason: str, calls: int) -> Transformed:
    return Transformed(question, question, False, reason, calls)


async def rewrite(model: ChatModel, question: str, *, timeout_s: float = 20.0) -> Transformed:
    """The question restated for search by ``model``, or the question itself with the reason."""
    if not isinstance(question, str) or not question.strip():
        raise InvalidInput("a question must have something in it")
    if len(question.encode("utf-8")) > MAX_QUESTION_BYTES:
        raise InvalidInput(f"a question is at most {MAX_QUESTION_BYTES} bytes; refused rather than cut")
    try:
        reply = await asyncio.wait_for(model.complete(_REWRITE_SYSTEM, f"Question: {question}"), timeout_s)
    except TimeoutError:
        return _kept(question, f"timeout after {timeout_s}s; the question was searched as asked", 1)
    except Exception as error:  # a model that fails leaves the question as it was
        return _kept(question, f"model failed: {type(error).__name__}; the question was searched as asked", 1)
    body = _object(reply)
    raw = body.get("query") if body is not None else None
    if not isinstance(raw, str) or not raw.strip():
        return _kept(question, "the reply could not be read as a query; the question was searched as asked", 1)
    query = " ".join(raw.split())
    if len(query) > MAX_QUERY:
        return _kept(question, f"the rewrite is over the {MAX_QUERY}-character bound; the question was searched as asked", 1)
    asked = set(tokenize(question))
    if asked and not asked & set(tokenize(query)):
        return _kept(question, "the rewrite shares no word with the question; the question was searched as asked", 1)
    return Transformed(question, query, True, None, 1)


async def rewritten_recall(engine: "MemoryEngine", model: ChatModel, space: str, question: str,
                           **recall_options: object) -> tuple["RecallResult", Transformed]:
    """Recall over the rewritten question, with the transform beside the result."""
    asked = await rewrite(model, question)
    return await engine.recall(space, asked.query, **recall_options), asked  # type: ignore[arg-type]
