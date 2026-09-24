"""Request-local operation timings; no prompts, credentials or source content."""
from __future__ import annotations

import math
import re
import time
from contextvars import ContextVar, Token
from types import TracebackType
from urllib.parse import urlsplit

_active: ContextVar[TurnPerformance | None] = ContextVar('scone_turn_performance', default=None)
_STAGES = frozenset({'capture_user', 'capture_assistant', 'recall', 'embedding',
                     'assessment', 'decision_memory', 'generation', 'model_start_timeout'})
_OUTCOMES = frozenset({'completed', 'failed', 'cancelled', 'timeout', 'timed_out',
                       'prepared', 'empty', 'skipped', 'unavailable', 'degraded',
                       'reused', 'recorded', 'write_unavailable', 'read_unavailable'})
_PROVIDERS = frozenset({'openrouter', 'typesafe', 'ollama', 'remote'})


def provider_name(url: str) -> str:
    try:
        hostname = urlsplit(url).hostname
    except ValueError:
        return 'remote'
    if hostname == 'openrouter.ai':
        return 'openrouter'
    if hostname == 'api.typesafe.ai':
        return 'typesafe'
    return 'remote'


def _number(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    number = float(value)  # type: ignore[arg-type]
    return round(number, 3) if math.isfinite(number) and number >= 0 else None


class TurnPerformance:
    """A bounded timeline inherited by this turn's async child tasks only.

    Durations are inclusive: an embedding span may sit inside capture or recall.
    Consumers must never sum all spans to calculate wall time.
    """
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.spans: list[dict[str, object]] = []
        self.first_text_ms: float | None = None
        self.truncated = False
        self._token: Token[TurnPerformance | None] | None = None
        self._closed = False

    def __enter__(self) -> TurnPerformance:
        self._token = _active.set(self)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self._closed = True
        if self._token is not None:
            _active.reset(self._token)

    def first_text(self) -> None:
        if self.first_text_ms is None and not self._closed:
            self.first_text_ms = self.elapsed()

    def elapsed(self) -> float:
        return round(max(0, time.perf_counter() - self.started) * 1000, 3)

    def snapshot(self) -> dict[str, object]:
        return {'schema_version': 1, 'total_ms': self.elapsed(),
                'first_text_ms': self.first_text_ms, 'truncated': self.truncated,
                'spans': [dict(span) for span in self.spans]}


def observe(stage: str, *, elapsed_ms: object, outcome: object,
            model: object = None, provider: object = None, first_text_ms: object = None,
            mode: object = None,
            **_excluded: object) -> None:
    trace = _active.get()
    elapsed = _number(elapsed_ms)
    if trace is None or trace._closed or stage not in _STAGES or elapsed is None:
        return
    if len(trace.spans) >= 64:
        trace.truncated = True
        return
    span: dict[str, object] = {'stage': stage, 'start_ms': round(max(0, trace.elapsed() - elapsed), 3),
                               'duration_ms': elapsed,
                               'outcome': outcome if isinstance(outcome, str) and outcome in _OUTCOMES else 'unknown'}
    if isinstance(model, str) and re.fullmatch(r'[A-Za-z0-9_./:+-]{1,160}', model):
        span['model'] = model
    if isinstance(provider, str) and provider in _PROVIDERS:
        span['provider'] = provider
    if isinstance(mode, str) and mode in {'stream', 'structured', 'chat'}:
        span['mode'] = mode
    first = _number(first_text_ms)
    if first is not None and first <= elapsed:
        span['first_text_ms'] = first
    trace.spans.append(span)
