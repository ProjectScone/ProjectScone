"""Explicit inference selection; storage and self-hosted boundaries stay local."""
import re
from typing import Literal

from .self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier

InferenceProvider = Literal['self_hosted', 'openrouter']
OPENROUTER_BASE = 'https://openrouter.ai/api/v1/'


def inference_endpoint(endpoint: str, model: str, provider: InferenceProvider) -> str:
    validate_self_hosted_identifier(model)
    if provider == 'self_hosted':
        return validate_self_hosted_endpoint(endpoint)
    if provider != 'openrouter' or endpoint not in (OPENROUTER_BASE, OPENROUTER_BASE[:-1]):
        raise ValueError('OpenRouter inference requires https://openrouter.ai/api/v1')
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*', model) is None:
        raise ValueError('OpenRouter inference requires an explicit provider/model identifier')
    return OPENROUTER_BASE
