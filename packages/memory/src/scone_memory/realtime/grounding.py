"""Native pre-generation policy for private questions and missing evidence."""
from __future__ import annotations

import asyncio
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..memory.engine import MemoryEngine
from ..retrieval.recall_scope import RecallScope
from .context import ContextReceipt
from .review_evidence import prepare_review_evidence

MISSING_EVIDENCE = (
    "I don't have enough source information to answer that reliably. "
    "Share the relevant document or decision, and I can work from it."
)
CHECK_UNAVAILABLE = "I couldn't verify the information for this reply. Please try again."
GROUNDED_INSTRUCTIONS = (
    "For this answer, use the supplied source evidence and explicit user statements. "
    "Earlier assistant answers are not independent evidence. Apply explicit corrections; "
    "do not infer that a later storage timestamp alone makes a conflicting claim true. "
    "State the actual answer, not just a filename containing it. Cite the supporting source "
    "when using retrieved evidence. If a requested detail is absent, say so instead of guessing."
)


class GroundingJudgment(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    private: float = Field(ge=0, le=1, allow_inf_nan=False)
    supported: float = Field(ge=0, le=1, allow_inf_nan=False)

    @property
    def status(self) -> Literal['general', 'supported', 'insufficient']:
        if self.private <= .2:
            return 'general'
        return 'supported' if self.supported >= .8 else 'insufficient'


class AnswerGrounder(Protocol):
    async def assess(self, question: str, history: list[dict[str, str]], evidence: str | None) -> GroundingJudgment: ...


async def check_answer(grounder: AnswerGrounder, memory: MemoryEngine, space: str,
                       scope: RecallScope, session_id: str, messages: list[dict[str, str]],
                       request: list[dict[str, object]], receipt: ContextReceipt, *,
                       admit_turn_ids: frozenset[str] = frozenset()) -> str | None:
    """Return a fixed public reply when generation must not proceed.

    The judgment concerns input support, not generated-answer correctness.
    Sources are revalidated before and after the judgment. No history summary or
    system instruction is silently promoted to evidence. Provider failure is
    distinct from evidence absence and cancellation always propagates.
    """
    try:
        async with asyncio.timeout(4.0):
            prepared = (await prepare_review_evidence(memory, space, scope, session_id, request, receipt,
                            admit_turn_ids=admit_turn_ids)
                        if receipt['status'] == 'prepared' else None)
            history = [{'role': message['role'], 'content': message['content']} for message in messages[:-1]
                       if message['role'] in ('user', 'assistant')][-16:]
            raw = await grounder.assess(messages[-1]['content'], history, prepared.evidence if prepared else None)
            judgment = GroundingJudgment.model_validate(dict(vars(raw)), strict=True)
            if prepared is not None and not await prepared.validate():
                raise ValueError('source changed')
            receipt['answer_grounding'] = {'status': judgment.status, 'private': judgment.private,
                'supported': judgment.supported, 'source_checked': prepared is not None,
                'history_messages': len(history), 'verified_answer': False}
            return MISSING_EVIDENCE if judgment.status == 'insufficient' else None
    except asyncio.CancelledError:
        raise
    except Exception:
        receipt['answer_grounding'] = {'status': 'unavailable', 'verified_answer': False}
        return CHECK_UNAVAILABLE
