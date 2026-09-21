"""Explicit HTTPS inference endpoints using Scone's native tool protocol."""
from urllib.parse import unquote, urlsplit

from .inference_endpoint import InferenceProvider, inference_endpoint
from .structured_tool_chat import SelfHostedStructuredToolChat
from .tool_chat import SelfHostedToolChat


def validate_hosted_endpoint(value: str) -> str:
    invalid = 'hosted inference requires an HTTPS URL without credentials, query or fragment'
    if not value or len(value) > 2048 or any(ord(c) <= 32 or ord(c) == 127 for c in value):
        raise ValueError(invalid)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError(invalid) from None
    if (parsed.scheme != 'https' or not parsed.hostname or port == 0
            or any(c in value for c in ('\\', '?', '#', '@')) or '%' in parsed.netloc):
        raise ValueError(invalid)
    path = unquote(parsed.path)
    if (any(part in ('.', '..') for part in path.split('/'))
            or any(ord(c) <= 32 or ord(c) == 127 or c == '\\' for c in path)):
        raise ValueError(invalid)
    return value.rstrip('/') + '/'


def _hosted_inference_endpoint(endpoint: str, model: str, provider: InferenceProvider) -> str:
    if provider != 'self_hosted':
        return inference_endpoint(endpoint, model, provider)
    return validate_hosted_endpoint(endpoint)


class HostedToolChat(SelfHostedToolChat):
    """Native tool calls to an operator-selected HTTPS inference service."""

    _validate_endpoint = staticmethod(_hosted_inference_endpoint)


class HostedStructuredToolChat(SelfHostedStructuredToolChat):
    """Structured actions to an operator-selected HTTPS inference service."""

    _validate_endpoint = staticmethod(_hosted_inference_endpoint)
