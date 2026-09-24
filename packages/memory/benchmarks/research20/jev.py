"""Bounded direct Jev requests for synthetic research data, with audit records."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import math
import time

import httpx

from scone_memory.providers.hosted_tool_chat import validate_hosted_endpoint
from scone_memory.providers.typesafe_evidence import _Response


@dataclass(frozen=True)
class Probe:
    key: str
    instruction: str
    sample: dict[str, object]
    yes: str = 'The condition is established by the provided evidence.'
    no: str = 'The condition is not established by the provided evidence.'

    def question(self) -> dict[str, object]:
        return {'type': 'noul', 'instructions': {'question': self.instruction, 'sample': self.sample},
                'criteria': {'true': self.yes, 'false': self.no}}


@dataclass(frozen=True)
class Batch:
    answers: dict[str, float]
    model: str
    elapsed_ms: float
    input_tokens: int
    output_tokens: int


@dataclass
class JevResearchClient:
    endpoint: str
    model: str
    api_key: str = field(repr=False)
    transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False)
    audits: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.endpoint = validate_hosted_endpoint(self.endpoint) + 'v1/systemone'
        if not self.model.strip() or len(self.model) > 160:
            raise ValueError('invalid model')
        if not self.api_key or len(self.api_key) > 4096 or any(not 33 <= ord(c) <= 126 for c in self.api_key):
            raise ValueError('invalid credential')

    async def evaluate(self, probes: tuple[Probe, ...]) -> Batch:
        if not 1 <= len(probes) <= 128 or len({p.key for p in probes}) != len(probes):
            raise ValueError('expected 1..128 distinct probes')
        payload = {'model': self.model, 'state': {'scope': 'synthetic memory research'},
                   'questions': {p.key: p.question() for p in probes}}
        if len(json.dumps(payload, allow_nan=False).encode()) > 512000:
            raise ValueError('research request too large')
        started = time.perf_counter()
        try:
            async with asyncio.timeout(30):
                async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False,
                                            transport=self.transport) as client:
                    async with client.stream('POST', self.endpoint,
                            headers={'Authorization': 'Bearer ' + self.api_key}, json=payload) as response:
                        response.raise_for_status()
                        raw = bytearray()
                        async for chunk in response.aiter_bytes():
                            raw.extend(chunk)
                            if len(raw) > 128000:
                                raise ValueError('research response too large')
                parsed = _Response.model_validate_json(bytes(raw))
                if set(parsed.answers) != {p.key for p in probes}:
                    raise ValueError('missing or extra provider answers')
            answers = {key: answer.noul for key, answer in parsed.answers.items()}
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in answers.values()):
                raise ValueError('invalid probability')
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.audits.append({'request': payload, 'response': parsed.model_dump(), 'elapsed_ms': elapsed_ms})
            return Batch(answers, parsed.model, elapsed_ms, parsed.usage.input_tokens, parsed.usage.output_tokens)
        except Exception as error:
            self.audits.append({'request': payload, 'error_type': type(error).__name__,
                               'elapsed_ms': (time.perf_counter() - started) * 1000})
            raise ValueError('direct Jev research request unavailable') from None
