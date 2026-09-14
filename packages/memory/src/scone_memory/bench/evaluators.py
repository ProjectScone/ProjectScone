"""Judged answer quality: what a local model says about an answer, and what it cannot.

Retrieval metrics say whether the right passages came back; nothing in the
bench said anything about the answer written from them. These four judgments
are the ones the reference framework also asks, done our way: a local judge
(any ``ChatModel``) is handed the exact texts and asked for a small JSON
verdict; the verdict is parsed strictly; and what could not be parsed, what
the judge failed to produce, or what exceeded the input bound comes back as
**unverified** -- never a pass, never a fail. A judgment is the judge's
opinion, not proof; the bench reports it beside the metrics that need no
judge, and says which is which.

- ``faithfulness``: every claim in the answer, and whether a context supports it.
- ``answer_relevancy``: whether the answer addresses the question.
- ``context_relevancy``: the share of contexts that bear on the question.
- ``correctness``: the answer against a reference, on a five-point scale.
- ``pairwise``: which of two answers is better, asked in both orders so a
  judge that prefers the first it reads is caught rather than believed.
- ``semantic_similarity``: how close an answer's embedding is to a
  reference's, which needs no judge and says nothing about truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Optional, Sequence

from ..core.ports import Embedder
from ..providers.llm import ChatModel

#: Contexts one judgment may weigh; more is refused, not cut.
MAX_CONTEXTS = 20
#: Bytes any one text may carry into a prompt; more is refused, not cut.
MAX_TEXT_BYTES = 64_000


@dataclass(frozen=True)
class Judgment:
    kind: str
    #: In [0, 1]; None when there was nothing to score or the judge could not be read.
    score: Optional[float]
    #: None when unverified, or when the judgment is neither (an answer with no claims).
    passing: Optional[bool]
    reasons: tuple[str, ...]
    #: False when the judge failed, could not be read, or was never asked because an input was over the bound.
    verified: bool
    judge_calls: int = 0


def _over_bound(*texts: str) -> Optional[str]:
    for text in texts:
        if not isinstance(text, str):
            return "input is not text"
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            return f"input over the {MAX_TEXT_BYTES}-byte bound; refused rather than cut"
    return None


def _unverified(kind: str, reason: str, calls: int = 0) -> Judgment:
    return Judgment(kind, None, None, (reason,), False, calls)


def _verdict(reply: str) -> Optional[dict[str, object]]:
    """The first JSON object in the reply, or None. A judge may wrap its
    verdict in prose; it may not omit it."""
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


async def _ask(judge: ChatModel, kind: str, system: str, user: str) -> tuple[Optional[dict[str, object]], Optional[Judgment]]:
    try:
        reply = await judge.complete(system, user)
    except Exception as error:  # a judge that fails is not a verdict either way
        return None, _unverified(kind, f"judge failed: {type(error).__name__}", 1)
    verdict = _verdict(reply)
    if verdict is None:
        return None, _unverified(kind, "judge reply could not be read as a JSON verdict", 1)
    return verdict, None


def _numbered(contexts: Sequence[str]) -> str:
    return "\n".join(f"[{index}] {context}" for index, context in enumerate(contexts))


async def faithfulness(judge: ChatModel, *, answer: str, contexts: Sequence[str]) -> Judgment:
    """Every claim the answer makes, and whether one of the contexts supports it."""
    kind = "faithfulness"
    if len(contexts) > MAX_CONTEXTS:
        return _unverified(kind, f"{len(contexts)} contexts exceed the bound of {MAX_CONTEXTS}; refused rather than cut")
    over = _over_bound(answer, *contexts)
    if over:
        return _unverified(kind, over)
    system = ("You judge whether an answer is supported by the given contexts. List every factual claim the "
              "answer makes. For each claim say whether some context supports it, and which one. Reply with "
              'exactly one JSON object: {"claims": [{"claim": "...", "supported": true|false, '
              '"context_index": <integer index or null>}]}. Use only the contexts; do not use outside knowledge.')
    user = f"Contexts:\n{_numbered(contexts)}\n\nAnswer:\n{answer}"
    verdict, failure = await _ask(judge, kind, system, user)
    if failure:
        return failure
    assert verdict is not None
    claims = verdict.get("claims")
    if not isinstance(claims, list):
        return _unverified(kind, "judge verdict lacks a claims list", 1)
    supported = 0
    reasons: list[str] = []
    for item in claims:
        if not isinstance(item, dict) or not isinstance(item.get("claim"), str) or not isinstance(item.get("supported"), bool):
            return _unverified(kind, "judge verdict has a claim without a boolean support", 1)
        if item["supported"]:
            supported += 1
        else:
            reasons.append(f"unsupported: {item['claim']}")
    if not claims:
        return Judgment(kind, None, None, ("no claims to judge",), True, 1)
    score = supported / len(claims)
    return Judgment(kind, score, supported == len(claims), tuple(reasons), True, 1)


async def answer_relevancy(judge: ChatModel, *, question: str, answer: str) -> Judgment:
    """Whether the answer addresses the question that was asked."""
    kind = "answer_relevancy"
    over = _over_bound(question, answer)
    if over:
        return _unverified(kind, over)
    system = ("You judge whether an answer addresses the question asked -- not whether it is true. Reply with "
              'exactly one JSON object: {"addresses": true|false, "reason": "..."}.')
    user = f"Question:\n{question}\n\nAnswer:\n{answer}"
    verdict, failure = await _ask(judge, kind, system, user)
    if failure:
        return failure
    assert verdict is not None
    addresses, reason = verdict.get("addresses"), verdict.get("reason")
    if not isinstance(addresses, bool):
        return _unverified(kind, "judge verdict lacks a boolean 'addresses'", 1)
    reasons = (reason,) if isinstance(reason, str) and reason else ()
    return Judgment(kind, 1.0 if addresses else 0.0, addresses, reasons, True, 1)


async def context_relevancy(judge: ChatModel, *, question: str, contexts: Sequence[str]) -> Judgment:
    """The share of contexts that bear on the question; passing when any does."""
    kind = "context_relevancy"
    if len(contexts) > MAX_CONTEXTS:
        return _unverified(kind, f"{len(contexts)} contexts exceed the bound of {MAX_CONTEXTS}; refused rather than cut")
    over = _over_bound(question, *contexts)
    if over:
        return _unverified(kind, over)
    if not contexts:
        return Judgment(kind, 0.0, False, ("no contexts",), True, 0)
    system = ("You judge which of the numbered contexts bear on the question. Give a verdict for every index. "
              'Reply with exactly one JSON object: {"contexts": [{"index": <integer>, "relevant": true|false}]}.')
    user = f"Question:\n{question}\n\nContexts:\n{_numbered(contexts)}"
    verdict, failure = await _ask(judge, kind, system, user)
    if failure:
        return failure
    assert verdict is not None
    rows = verdict.get("contexts")
    if not isinstance(rows, list):
        return _unverified(kind, "judge verdict lacks a contexts list", 1)
    seen: dict[int, bool] = {}
    for item in rows:
        if (not isinstance(item, dict) or type(item.get("index")) is not int or not isinstance(item.get("relevant"), bool)
                or not 0 <= item["index"] < len(contexts)):
            return _unverified(kind, "judge verdict has a context row without an index and a boolean", 1)
        seen[item["index"]] = item["relevant"]
    if len(seen) != len(contexts):
        return _unverified(kind, f"judge verdict covers {len(seen)} of {len(contexts)} contexts", 1)
    relevant = sum(seen.values())
    reasons = tuple(f"context {index} not relevant" for index, flag in sorted(seen.items()) if not flag)
    return Judgment(kind, relevant / len(contexts), relevant > 0, reasons, True, 1)


async def correctness(judge: ChatModel, *, question: str, answer: str, reference: str, passing_score: int = 4) -> Judgment:
    """The answer against a reference answer, scored 1 to 5 by the judge;
    ``score`` is that over 5, and ``passing`` is the caller's threshold."""
    kind = "correctness"
    if type(passing_score) is not int or not 1 <= passing_score <= 5:
        raise ValueError("passing_score must be an integer from 1 to 5")
    over = _over_bound(question, answer, reference)
    if over:
        return _unverified(kind, over)
    system = ("You judge an answer against a reference answer to the same question. Score 1 (wrong or "
              "unrelated) to 5 (equivalent to the reference in every fact that matters). Reply with exactly one "
              'JSON object: {"score": <integer 1-5>, "reason": "..."}.')
    user = f"Question:\n{question}\n\nReference answer:\n{reference}\n\nAnswer to judge:\n{answer}"
    verdict, failure = await _ask(judge, kind, system, user)
    if failure:
        return failure
    assert verdict is not None
    score, reason = verdict.get("score"), verdict.get("reason")
    if type(score) is not int or not 1 <= score <= 5:
        return _unverified(kind, "judge verdict has no integer score from 1 to 5", 1)
    reasons = (reason,) if isinstance(reason, str) and reason else ()
    return Judgment(kind, score / 5, score >= passing_score, reasons, True, 1)


async def pairwise(judge: ChatModel, *, question: str, answer_a: str, answer_b: str,
                   reference: Optional[str] = None) -> Judgment:
    """Which of two answers to the question is better, in the judge's
    opinion, asked twice with the answers in either order. A judge that
    prefers whichever it read first gives two different verdicts, and
    that disagreement is reported as a tie rather than resolved by a coin.
    ``score`` is 1.0 when A wins, 0.0 when B wins, 0.5 for a tie;
    ``passing`` is whether A won. A reference answer, when given, is shown
    to the judge as what a right answer says."""
    kind = "pairwise"
    over = _over_bound(question, answer_a, answer_b, *(() if reference is None else (reference,)))
    if over:
        return _unverified(kind, over)
    system = ("You judge which of two answers to a question is better: more correct, more complete, and "
              "better grounded. Reply with exactly one JSON object: "
              '{"winner": "first" | "second" | "tie", "reason": "..."}.')
    shown = f"\n\nA right answer says:\n{reference}" if reference is not None else ""
    verdicts: list[str] = []
    reasons: list[str] = []
    calls = 0
    for first, second, label in ((answer_a, answer_b, ("first", "second")), (answer_b, answer_a, ("second", "first"))):
        user = f"Question:\n{question}{shown}\n\nFirst answer:\n{first}\n\nSecond answer:\n{second}"
        verdict, failure = await _ask(judge, kind, system, user)
        calls += 1
        if failure:
            return _unverified(kind, failure.reasons[0], calls)
        assert verdict is not None
        winner, reason = verdict.get("winner"), verdict.get("reason")
        if winner not in ("first", "second", "tie"):
            return _unverified(kind, "judge verdict names no winner among first, second and tie", calls)
        # Translate the judge's positional word into A or B for this order.
        verdicts.append("tie" if winner == "tie" else ("a" if winner == label[0] else "b"))
        if isinstance(reason, str) and reason:
            reasons.append(reason)
    if verdicts[0] != verdicts[1]:
        reasons.append(f"the judge preferred {verdicts[0].upper() if verdicts[0] != 'tie' else 'neither'} in one order and "
                       f"{verdicts[1].upper() if verdicts[1] != 'tie' else 'neither'} in the other: a tie, not a coin")
        return Judgment(kind, 0.5, False, tuple(reasons), True, calls)
    score = {"a": 1.0, "b": 0.0, "tie": 0.5}[verdicts[0]]
    return Judgment(kind, score, verdicts[0] == "a", tuple(reasons), True, calls)


async def semantic_similarity(embedder: Embedder, *, answer: str, reference: str,
                              passing_similarity: float = 0.8) -> Judgment:
    """The cosine between the answer's embedding and the reference's, as a
    score in [0, 1] (a negative cosine is 0), passing at the caller's
    threshold. No judge is called and nothing is checked for truth: two
    fluent wrong answers can sit close together, and what this says
    depends entirely on the embedder, whose id is on every reason."""
    kind = "semantic_similarity"
    if not isinstance(passing_similarity, (int, float)) or isinstance(passing_similarity, bool) or not 0 <= passing_similarity <= 1:
        raise ValueError("passing_similarity must be a number from 0 to 1")
    over = _over_bound(answer, reference)
    if over:
        return _unverified(kind, over)
    if not answer.strip() or not reference.strip():
        return _unverified(kind, "an empty answer or reference has no similarity to measure")
    try:
        first, second = await embedder.embed([answer, reference])
    except Exception as error:  # an embedder that fails is not a verdict either way
        return _unverified(kind, f"embedder failed: {type(error).__name__}")
    if len(first) != len(second) or not first:
        return _unverified(kind, "embedder returned vectors of different or zero width")
    dot = sum(a * b for a, b in zip(first, second))
    norms = sum(a * a for a in first) ** 0.5 * sum(b * b for b in second) ** 0.5
    if norms == 0:
        return _unverified(kind, "embedder returned a zero vector")
    cosine = max(0.0, min(1.0, dot / norms))
    return Judgment(kind, cosine, cosine >= passing_similarity,
                    (f"cosine {cosine:.3f} under embedder {embedder.id}; similarity is not correctness",), True, 0)
