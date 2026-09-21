"""Encrypted history of scoped evidence judgments, revalidated on use.

This stores observations about supplied evidence, never authoritative facts.
Fresh retrieval and source validation belong to AdaptiveRetriever on every use.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import math
import re
from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.validation import check_space
from .adaptive import EvidenceCandidate, EvidenceDecision
from .recall_scope import RecallScope

if TYPE_CHECKING:
    from ..agents._encrypted_store import EncryptedRecordStore


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class RecordedJudgment(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    decision: EvidenceDecision
    model: str = Field(min_length=1, max_length=160)
    probabilities: dict[str, float]

    @model_validator(mode='after')
    def valid_probabilities(self) -> RecordedJudgment:
        if len(self.probabilities) > 101 or any(not math.isfinite(p) or not 0 <= p <= 1
                                               for p in self.probabilities.values()):
            raise ValueError('invalid judgment probabilities')
        return self


class DecisionRevision(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    revision: int = Field(ge=1)
    definition: str = Field(pattern=r'^[0-9a-f]{64}$')
    input_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    evidence: dict[str, str]
    evaluated_at: datetime
    reason: Literal['initial', 'evidence_changed', 'definition_changed', 'expired']
    change: Literal['initial', 'unchanged', 'revised']
    judgment: RecordedJudgment

    @model_validator(mode='after')
    def valid_snapshot(self) -> DecisionRevision:
        if (self.evaluated_at.tzinfo is None or len(self.evidence) > 100
                or any(not re.fullmatch(r'(chunk|fact):[1-9][0-9]*', key)
                       or not re.fullmatch(r'[0-9a-f]{64}', value) for key, value in self.evidence.items())):
            raise ValueError('invalid judgment snapshot')
        if not set(self.judgment.decision.selected_ids).issubset(self.evidence):
            raise ValueError('judgment references absent evidence')
        if set(self.judgment.probabilities) != {*self.evidence, 'sufficient'}:
            raise ValueError('judgment probabilities do not match evidence')
        return self


class DecisionHistory(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid')
    revisions: tuple[DecisionRevision, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode='after')
    def ordered(self) -> DecisionHistory:
        numbers = [r.revision for r in self.revisions]
        if any(right != left + 1 for left, right in zip(numbers, numbers[1:])):
            raise ValueError('invalid decision revision sequence')
        return self


class DecisionConflict(RuntimeError):
    pass


class DecisionMemory:
    """Local encrypted SQLite, bounded per question and globally.

    Short operations own and close their connections. No question or source
    text is duplicated; even identifiers, hashes and probabilities are encrypted.
    History records when a judgment was evaluated, not a promise of freshness.
    """

    def __init__(self, path: str | Path, *, key: bytes, max_records: int = 4096,
                 history_limit: int = 8) -> None:
        if type(key) is not bytes or len(key) != 32:
            raise ValueError('decision memory needs a 32-byte encryption key')
        if type(max_records) is not int or not 1 <= max_records <= 100000:
            raise ValueError('invalid decision memory capacity')
        if type(history_limit) is not int or not 1 <= history_limit <= 8:
            raise ValueError('invalid decision history limit')
        self.path, self._key = Path(path), key
        self._maximum, self._history_limit = max_records, history_limit
        self._open().close()

    def _open(self) -> EncryptedRecordStore:
        # Existing native authenticated storage; crypto remains an optional dependency.
        from ..agents._encrypted_store import EncryptedRecordStore
        return EncryptedRecordStore(self.path, key=self._key, table='evidence_decisions',
            metadata='decision_meta', application_id=0x5343444D,
            domain='scone-evidence-decisions-v1', label='decision')

    def _prefix(self, space: str) -> str:
        check_space(space)
        return hmac.new(self._key, b'decision-space:' + space.encode(), hashlib.sha256).hexdigest() + ':'

    def _token(self, space: str, question: str, scope: RecallScope) -> str:
        if not question.strip() or len(question.encode()) > 8000:
            raise ValueError('invalid decision question')
        family = json.dumps([question, scope.as_dict()], sort_keys=True, separators=(',', ':')).encode()
        return self._prefix(space) + hmac.new(self._key, family, hashlib.sha256).hexdigest()

    def history(self, space: str, question: str, scope: RecallScope) -> DecisionHistory | None:
        token, store = self._token(space, question, scope), self._open()
        try:
            with store._access() as db:
                row = db.execute('SELECT payload FROM evidence_decisions WHERE token=?', (token,)).fetchone()
                return None if row is None else DecisionHistory.model_validate_json(store._unseal(token, row[0]))
        finally:
            store.close()

    def append(self, space: str, question: str, scope: RecallScope, observation: DecisionRevision,
               *, expected_revision: int) -> None:
        observation = DecisionRevision.model_validate(observation.model_dump())
        if type(expected_revision) is not int or expected_revision < 0 or observation.revision != expected_revision + 1:
            raise ValueError('invalid decision revision')
        token, store = self._token(space, question, scope), self._open()
        try:
            with store._access(write=True) as db:
                row = db.execute('SELECT payload FROM evidence_decisions WHERE token=?', (token,)).fetchone()
                old = None if row is None else DecisionHistory.model_validate_json(store._unseal(token, row[0]))
                if (old.revisions[-1].revision if old else 0) != expected_revision:
                    raise DecisionConflict('decision changed during assessment')
                if old is None and db.execute('SELECT COUNT(*) FROM evidence_decisions').fetchone()[0] >= self._maximum:
                    raise RuntimeError('decision memory capacity reached')
                revisions = (*old.revisions, observation) if old else (observation,)
                saved = DecisionHistory(revisions=revisions[-self._history_limit:])
                payload = store._seal(token, saved.model_dump_json().encode())
                db.execute('INSERT INTO evidence_decisions VALUES (?, ?) '
                    'ON CONFLICT(token) DO UPDATE SET payload=excluded.payload', (token, payload))
        finally:
            store.close()

    def forget_space(self, space: str) -> int:
        prefix, store = self._prefix(space), self._open()
        try:
            with store._access(write=True) as db:
                return db.execute('DELETE FROM evidence_decisions WHERE token>=? AND token<?',
                                  (prefix, prefix + 'g')).rowcount
        finally:
            store.close()


class RecordingAssessor(Protocol):
    @property
    def definition(self) -> str: ...
    async def assess(self, question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision: ...
    async def assess_detailed(self, question: str, candidates: tuple[EvidenceCandidate, ...]) -> RecordedJudgment: ...


class RememberedEvidenceAssessor:
    """Reuse only exact, recent, scoped judgments after fresh candidate retrieval."""

    def __init__(self, assessor: RecordingAssessor, memory: DecisionMemory, *,
                 max_age_s: float = 3600, clock: Callable[[], datetime] | None = None) -> None:
        if type(max_age_s) not in (float, int) or not math.isfinite(max_age_s) or not 0 < max_age_s <= 86400:
            raise ValueError('decision age must be positive and at most one day')
        self.assessor, self.memory, self.max_age_s = assessor, memory, max_age_s
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def assess(self, question: str, candidates: tuple[EvidenceCandidate, ...]) -> EvidenceDecision:
        # Unscoped SDK calls may infer, but never share saved judgments.
        return await self.assessor.assess(question, candidates)

    async def assess_scoped(self, question: str, candidates: tuple[EvidenceCandidate, ...], *,
                            space: str, scope: RecallScope) -> EvidenceDecision:
        started = time.perf_counter()
        checked = tuple(EvidenceCandidate.model_validate(c.model_dump()) for c in candidates)
        if len(checked) > 100 or len({c.id for c in checked}) != len(checked):
            raise ValueError('invalid evidence candidates')
        if not checked:
            return await self.assessor.assess(question, checked)
        fingerprints = {c.id: digest(c.model_dump(mode='json')) for c in checked}
        input_digest = digest([c.model_dump(mode='json') for c in checked])
        definition, now = self.assessor.definition, self._clock()
        previous: DecisionRevision | None = None
        try:
            async with asyncio.timeout(.1):
                history = await asyncio.to_thread(self.memory.history, space, question, scope)
            previous = history.revisions[-1] if history else None
        except Exception:
            self._log('read_unavailable', started)
        reason: Literal['initial', 'evidence_changed', 'definition_changed', 'expired'] = 'initial'
        if previous is not None:
            age = (now - previous.evaluated_at).total_seconds()
            if previous.definition != definition:
                reason = 'definition_changed'
            elif previous.input_digest != input_digest:
                reason = 'evidence_changed'
            elif not 0 <= age < self.max_age_s:
                reason = 'expired'
            else:
                self._log('reused', started, previous.revision)
                return previous.judgment.decision.model_copy(deep=True)
        judgment = RecordedJudgment.model_validate((await self.assessor.assess_detailed(question, checked)).model_dump())
        # Persist only a validated observation. Provider failures never replace history.
        observation = DecisionRevision(revision=previous.revision + 1 if previous else 1,
            definition=definition, input_digest=input_digest, evidence=fingerprints,
            evaluated_at=now, reason=reason, judgment=judgment,
            change='initial' if previous is None else
                'unchanged' if previous.judgment.decision == judgment.decision else 'revised')
        try:
            async with asyncio.timeout(.1):
                await asyncio.to_thread(self.memory.append, space, question, scope, observation,
                                        expected_revision=previous.revision if previous else 0)
            self._log('recorded', started, observation.revision)
        except Exception:
            self._log('write_unavailable', started)
        return judgment.decision

    @staticmethod
    def _log(outcome: str, started: float, revision: int | None = None) -> None:
        logging.getLogger(__name__).info('decision_memory.finished', extra={
            'event': 'decision_memory.finished', 'outcome': outcome, 'decision_revision': revision,
            'elapsed_ms': round((time.perf_counter() - started) * 1000, 3)})
