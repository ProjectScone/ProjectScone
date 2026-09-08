"""Compatibility names for the self-hosted endpoint boundary.

Legacy validator errors retain their original wording for callers that depend
on them; new integrations should import providers.self_hosted.
"""
from .self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier


def validate_local_endpoint(value: str) -> str:
    try:
        return validate_self_hosted_endpoint(value)
    except ValueError as error:
        raise ValueError(str(error).replace('self-hosted service', 'local service')) from None


def validate_local_identifier(value: str, *, max_length: int = 160) -> str:
    try:
        return validate_self_hosted_identifier(value, max_length=max_length)
    except ValueError as error:
        raise ValueError(str(error).replace('self-hosted model', 'local model')) from None


__all__ = ['validate_local_endpoint', 'validate_local_identifier']
