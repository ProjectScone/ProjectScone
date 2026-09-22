"""Explicit evidence-assessment providers for served text conversations."""
from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from ..core.errors import InvalidInput
from ..memory.engine import MemoryEngine
from ..providers.self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier
from ..retrieval.adaptive import AdaptiveLimits, AdaptiveRetriever, EvidenceAssessor
from ..retrieval.multihop import MultiHopLimits

if TYPE_CHECKING:
    from .config import Settings


def retrieval_limits(settings: Settings) -> AdaptiveLimits:
    return AdaptiveLimits(max_rounds=settings.adaptive_max_rounds, max_queries=settings.adaptive_max_queries,
        candidate_limit=settings.adaptive_candidate_limit, max_evidence_bytes=settings.adaptive_max_evidence_bytes,
        timeout_s=settings.adaptive_timeout)


def validate_adaptive_settings(settings: Settings) -> None:
    if (type(settings.decision_memory_max_age) not in (int, float)
            or not math.isfinite(settings.decision_memory_max_age)
            or not 0 < settings.decision_memory_max_age <= 86400):
        raise InvalidInput('SCONE_DECISION_MEMORY_MAX_AGE must be positive and at most 86400 seconds')
    if settings.decision_memory:
        if not settings.adaptive_retrieval or settings.adaptive_provider != 'typesafe':
            raise InvalidInput('decision memory requires TypeSafe adaptive retrieval')
        if not re.fullmatch(r'[0-9a-fA-F]{64}', settings.decision_memory_key or ''):
            raise InvalidInput('SCONE_DECISION_MEMORY_KEY must be a 32-byte hexadecimal key')
    elif settings.decision_memory_key is not None or settings.decision_memory_max_age != 3600.0:
        raise InvalidInput('decision memory settings require SCONE_DECISION_MEMORY')
    if settings.adaptive_provider not in ('self-hosted', 'typesafe'):
        raise InvalidInput('SCONE_ADAPTIVE_PROVIDER must be self-hosted or typesafe')
    if type(settings.adaptive_retrieval) is not bool:
        raise InvalidInput("SCONE_ADAPTIVE_RETRIEVAL must be a boolean")
    if type(settings.adaptive_search_history) is not bool:
        raise InvalidInput("SCONE_ADAPTIVE_SEARCH_HISTORY must be a boolean")
    try:
        limits = retrieval_limits(settings)
        MultiHopLimits(max_hops=settings.adaptive_graph_hops)
    except ValueError:
        raise InvalidInput("adaptive retrieval budgets are outside supported limits") from None
    if not settings.adaptive_retrieval:
        if (settings.adaptive_provider != 'self-hosted' or settings.adaptive_url is not None or settings.adaptive_model is not None
                or settings.adaptive_api_key is not None or settings.adaptive_graph_hops != 0
                or settings.adaptive_search_history
                or limits != AdaptiveLimits(timeout_s=15.0)):
            raise InvalidInput("adaptive retrieval settings require SCONE_ADAPTIVE_RETRIEVAL=1")
        return
    if not settings.conversations_journal:
        raise InvalidInput("adaptive retrieval requires SCONE_CONVERSATIONS_JOURNAL")
    if not settings.adaptive_url or not settings.adaptive_model:
        raise InvalidInput("adaptive retrieval requires SCONE_ADAPTIVE_URL and SCONE_ADAPTIVE_MODEL")
    if settings.adaptive_provider == 'typesafe':
        from ..providers.typesafe_evidence import TypeSafeEvidenceAssessor
        if settings.adaptive_graph_hops or settings.adaptive_search_history:
            raise InvalidInput('TypeSafe assessment currently supports single-pass evidence selection only')
        try:
            TypeSafeEvidenceAssessor(settings.adaptive_url, settings.adaptive_model,
                api_key=settings.adaptive_api_key or '', timeout=settings.adaptive_timeout,
                max_evidence_bytes=settings.adaptive_max_evidence_bytes)
        except ValueError:
            raise InvalidInput('TypeSafe assessment requires a valid HTTPS endpoint, model and credential') from None
        return
    try:
        validate_self_hosted_endpoint(settings.adaptive_url)
        validate_self_hosted_identifier(settings.adaptive_model)
    except ValueError:
        raise InvalidInput("adaptive retrieval requires a valid self-hosted endpoint and model identifier") from None
    key = settings.adaptive_api_key
    if key is not None and (type(key) is not str or not key.strip() or any(ord(char) < 32 or ord(char) == 127 for char in key)):
        raise InvalidInput("SCONE_ADAPTIVE_API_KEY must be nonblank text without control characters")


def build_adaptive_retrieval(settings: Settings, engine: MemoryEngine) -> AdaptiveRetriever | None:
    validate_adaptive_settings(settings)
    if not settings.adaptive_retrieval:
        return None
    from ..providers.evidence_assessor import SelfHostedEvidenceAssessor

    assert settings.adaptive_url is not None and settings.adaptive_model is not None
    limits = retrieval_limits(settings)
    if settings.adaptive_provider == 'typesafe':
        from ..providers.typesafe_evidence import TypeSafeEvidenceAssessor
        direct = TypeSafeEvidenceAssessor(settings.adaptive_url, settings.adaptive_model,
            api_key=settings.adaptive_api_key or '', timeout=settings.adaptive_timeout,
            max_evidence_bytes=settings.adaptive_max_evidence_bytes)
        selected: EvidenceAssessor = direct
        if settings.decision_memory:
            from ..retrieval.decision_memory import DecisionMemory, RememberedEvidenceAssessor
            assert settings.decision_memory_key is not None
            try:
                memory = DecisionMemory(settings.decision_memory, key=bytes.fromhex(settings.decision_memory_key))
            except Exception:
                raise InvalidInput('cannot open encrypted decision memory; check path, key and decision-memory extra') from None
            selected = RememberedEvidenceAssessor(direct, memory, max_age_s=settings.decision_memory_max_age)
        return AdaptiveRetriever(engine, selected, limits=limits,
            failure_policy='retain_verified', empty_selection_policy='empty', evidence_policy='model_selected')
    graph = MultiHopLimits(max_hops=settings.adaptive_graph_hops) if settings.adaptive_graph_hops else None
    assessor = SelfHostedEvidenceAssessor(settings.adaptive_url, settings.adaptive_model,
        api_key=settings.adaptive_api_key, timeout=settings.adaptive_timeout,
        group_relations=graph is not None, max_evidence_bytes=settings.adaptive_max_evidence_bytes)
    return AdaptiveRetriever(engine, assessor, limits=limits, graph_limits=graph,
        include_search_history=settings.adaptive_search_history,
        failure_policy="retain_verified", empty_selection_policy="retain_verified", evidence_policy="original_and_selected")
