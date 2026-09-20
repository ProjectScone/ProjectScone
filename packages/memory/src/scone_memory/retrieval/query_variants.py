"""One question asked several ways, and the rankings fused.

A question is one wording of a need, and the passage that answers it was
written in another. ``query_transforms`` restates a question once; this
asks for several restatements, searches each, and fuses what each found
by reciprocal rank -- the leading framework's query fusion, with two
rules of its own.

**The question the person asked is always searched, and always counts.**
It leads the fusion at full voice while a variant argues at
``VARIANT_VOICE``, so a model that writes nonsense cannot lose the
question: several variants must agree with each other to outrank what
the question itself found.

**Every refusal is on the record.** A variant is taken only when it can
be read, is not empty, is inside the query bound, still shares a word
with the question, and is not a repeat of the question or of another
variant. Anything else is refused with the reason, beside the ones that
were kept -- so a reader can tell "the model wrote nothing usable" from
"the model was never asked".

Nothing here runs by default and nothing here has moved a number yet. A
transform earns a flag in the conversation path after it moves one
there, and not before; until then this is an explicit call.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from ..core.errors import InvalidInput
from ..core.validation import MAX_QUERY
from ..providers.llm import ChatModel
from . import fusion
from .lexical import tokenize
from .query_transforms import MAX_QUESTION_BYTES, _object

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.models import RecallResult
    from ..memory.engine import MemoryEngine

#: Restatements one call may ask for. More than this is a different
#: technique -- a search over paraphrases -- and costs a search each.
MAX_VARIANTS = 5

#: How loudly a variant argues against the question, in rank fusion. A
#: variant is a guess at what was meant; the question is what was asked.
VARIANT_VOICE = 0.5

_SYSTEM = (
    "You restate a question as several different searches over someone's own notes and conversations. Each "
    "restatement uses the words a passage that answers the question would contain, and a different angle from "
    "the others: one may name the thing, another the event around it, another the words a person would write "
    "about it. Keep names, dates and numbers exactly as written. Reply with JSON only, of the form "
    '{"queries": ["...", "..."]}: no answer, no explanation.'
)


@dataclass(frozen=True)
class Variant:
    """One restatement, kept or refused, and why."""

    query: str
    reason: Optional[str] = None

    def record(self) -> dict[str, object]:
        return {"query": self.query, **({"reason": self.reason} if self.reason else {})}


@dataclass(frozen=True)
class Variants:
    """What was asked, what was searched, and what the model's writing cost."""

    question: str
    kept: list[Variant] = field(default_factory=list)
    refused: list[Variant] = field(default_factory=list)
    #: Why no variant was searched at all, when none was.
    reason: str = ""
    #: Variants beyond the count asked for, dropped unread.
    cut: int = 0
    model_calls: int = 0

    @property
    def searched(self) -> list[str]:
        """The question first, then each variant kept."""
        return [self.question, *(variant.query for variant in self.kept)]

    def record(self) -> dict[str, object]:
        return {"kind": "variants", "question": self.question, "searched": self.searched,
                "searches": len(self.searched), "kept": [v.record() for v in self.kept],
                "refused": [v.record() for v in self.refused], "cut": self.cut,
                "model_calls": self.model_calls, **({"reason": self.reason} if self.reason else {})}


def _folded(query: str) -> str:
    return " ".join(query.split()).casefold()


def _checked(question: str, written: object, seen: set[str]) -> Variant:
    """One restatement against every rule, kept or refused with the reason."""
    if not isinstance(written, str) or not written.strip():
        return Variant(written if isinstance(written, str) else "", "empty; nothing to search")
    query = " ".join(written.split())
    if len(query) > MAX_QUERY:
        return Variant(query, f"over the bound of {MAX_QUERY} characters")
    asked = set(tokenize(question))
    if asked and not asked & set(tokenize(query)):
        return Variant(query, "shares no word with the question, so it asks something else")
    if _folded(query) in seen:
        return Variant(query, "repeats the question or another variant")
    return Variant(query)


async def variants(model: ChatModel, question: str, *, count: int = 3,
                   timeout_s: float = 30.0) -> Variants:
    """``count`` restatements of ``question`` written by ``model``.

    The question is never lost: whatever the model does, it is the first
    thing searched, and ``reason`` says what became of the variants.
    """
    if not isinstance(question, str) or not question.strip():
        raise InvalidInput("a question must have something in it")
    if len(question.encode("utf-8")) > MAX_QUESTION_BYTES:
        raise InvalidInput(f"a question is at most {MAX_QUESTION_BYTES} bytes; refused rather than cut")
    if type(count) is not int or not 1 <= count <= MAX_VARIANTS:
        raise InvalidInput(f"count is a whole number of variants in 1..={MAX_VARIANTS}, got {count!r}")
    try:
        reply = await asyncio.wait_for(
            model.complete(_SYSTEM, f"Question: {question}\nWrite {count} searches."), timeout_s)
    except TimeoutError:
        return Variants(question, reason=f"timeout after {timeout_s}s; the question was searched as asked",
                        model_calls=1)
    except Exception as error:
        return Variants(question, reason=f"model failed: {type(error).__name__}; the question was searched as asked",
                        model_calls=1)
    body = _object(reply)
    written = body.get("queries") if body is not None else None
    if not isinstance(written, list) or any(not isinstance(item, str) for item in written):
        return Variants(question, reason="the reply could not be read as a list of queries; the question "
                                         "was searched as asked", model_calls=1)
    if not written:
        return Variants(question, reason="the model wrote no variants; the question was searched as asked",
                        model_calls=1)
    seen = {_folded(question)}
    kept: list[Variant] = []
    refused: list[Variant] = []
    for item in written[:count]:
        checked = _checked(question, item, seen)
        if checked.reason is None:
            seen.add(_folded(checked.query))
            kept.append(checked)
        else:
            refused.append(checked)
    return Variants(question, kept=kept, refused=refused, cut=max(0, len(written) - count), model_calls=1,
                    reason="" if kept else "no variant was usable; the question was searched as asked")


async def variant_recall(engine: "MemoryEngine", model: ChatModel, space: str, question: str, *,
                         count: int = 3, timeout_s: float = 30.0,
                         **recall_options: object) -> tuple["RecallResult", Variants]:
    """Recall over the question and each variant, fused by reciprocal rank.

    The searches run together; the question's ranking leads at full
    voice and each variant's at ``VARIANT_VOICE``. The result is the
    question's own, re-ordered by the fusion, so everything a caller
    reads of it -- its receipt, its lanes, its citations -- is the
    receipt of the search the person asked for.
    """
    written = await variants(model, question, count=count, timeout_s=timeout_s)
    if not written.kept:
        return await engine.recall(space, question, **recall_options), written  # type: ignore[arg-type]
    found = await asyncio.gather(*(engine.recall(space, query, **recall_options)  # type: ignore[arg-type]
                                   for query in written.searched))
    lanes = [[(item.chunk_id, item.score) for item in result.items] for result in found]
    weights = [1.0, *([VARIANT_VOICE] * (len(lanes) - 1))]
    fused = fusion.rrf(lanes, weights=weights)
    items = {item.chunk_id: item for result in found for item in reversed(result.items)}
    ordered = sorted((items[chunk_id] for chunk_id in fused if chunk_id in items),
                     key=lambda item: -fused[item.chunk_id])
    limit = recall_options.get("limit")
    if type(limit) is int and limit > 0:
        ordered = ordered[:limit]
    # The question's own result, carrying the fused order: its receipt,
    # its event id and its facts are the ones the person asked for.
    return found[0].model_copy(update={"items": ordered}), written
