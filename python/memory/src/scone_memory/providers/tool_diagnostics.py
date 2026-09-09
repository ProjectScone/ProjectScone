"""Content-free accounting for one self-hosted tool-provider request."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4


ProtocolName = Literal['native', 'structured_action', 'structured_answer']
FailureKind = Literal['http_error', 'transport_timeout', 'transport_error',
                      'request_timeout', 'response_bytes', 'invalid_response', 'internal_error']


def _tokens(value: object) -> int | None:
    # Usage is optional, untrusted provider metadata, never an inferred count.
    return value if type(value) is int and 0 <= value <= 10**9 else None


@dataclass
class ToolCallDiagnostics:
    protocol: ProtocolName
    started: float
    request_bytes: int
    output_token_limit: int
    timeout_s: float
    call_id: str = field(default_factory=lambda: uuid4().hex)
    phase: Literal['transport', 'response', 'parse', 'cleanup', 'completed'] = 'transport'
    failure_kind: FailureKind | None = None
    http_status: int | None = None
    headers_ms: float | None = None
    response_bytes: int = 0
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def observe(self, packet: object) -> None:
        if not isinstance(packet, dict):
            return
        choices = packet.get('choices')
        if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
            reason = choices[0].get('finish_reason')
            allowed = ('stop', 'tool_calls', 'length', 'content_filter', 'function_call')
            self.finish_reason = reason if isinstance(reason, str) and reason in allowed else 'unknown'
        usage = packet.get('usage')
        if isinstance(usage, dict):
            self.prompt_tokens = _tokens(usage.get('prompt_tokens'))
            self.completion_tokens = _tokens(usage.get('completion_tokens'))
            self.total_tokens = _tokens(usage.get('total_tokens'))

    def log_started(self, logger: logging.Logger) -> None:
        logger.info('tool_model.started call_id=%s protocol=%s request_bytes=%s output_token_limit=%s timeout_s=%s',
            self.call_id, self.protocol, self.request_bytes, self.output_token_limit, self.timeout_s,
            extra={'event': 'tool_model.started', 'call_id': self.call_id, 'protocol': self.protocol,
                   'request_bytes': self.request_bytes, 'output_token_limit': self.output_token_limit,
                   'timeout_s': self.timeout_s})

    def log_finished(self, logger: logging.Logger, outcome: str) -> None:
        elapsed_ms = (time.monotonic() - self.started) * 1000
        logger.info(
            'tool_model.finished call_id=%s protocol=%s outcome=%s phase=%s failure_kind=%s '
            'elapsed_ms=%.3f headers_ms=%s http_status=%s response_bytes=%s finish_reason=%s '
            'prompt_tokens=%s completion_tokens=%s total_tokens=%s',
            self.call_id, self.protocol, outcome, self.phase, self.failure_kind, elapsed_ms, self.headers_ms,
            self.http_status, self.response_bytes, self.finish_reason,
            self.prompt_tokens, self.completion_tokens, self.total_tokens,
            extra={'event': 'tool_model.finished', 'call_id': self.call_id, 'protocol': self.protocol,
                   'outcome': outcome, 'phase': self.phase, 'failure_kind': self.failure_kind,
                   'elapsed_ms': elapsed_ms, 'headers_ms': self.headers_ms, 'http_status': self.http_status,
                   'request_bytes': self.request_bytes, 'response_bytes': self.response_bytes,
                   'output_token_limit': self.output_token_limit, 'timeout_s': self.timeout_s,
                   'finish_reason': self.finish_reason, 'prompt_tokens': self.prompt_tokens,
                   'completion_tokens': self.completion_tokens, 'total_tokens': self.total_tokens})
