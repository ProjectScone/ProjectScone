"""Separate destination authority for an operation crossing space boundaries."""
from collections.abc import Mapping

from .app import Forbidden, Unauthorized


def authorize_destination(header: str, into: str, keys: Mapping[str, str], roles: Mapping[str, str]) -> None:
    scheme, _, token = header.partition(' ')
    token = token.strip()
    if scheme.lower() != 'bearer' or not token:
        raise Unauthorized('missing destination bearer key')
    scope = keys.get(token)
    if scope is None:
        raise Unauthorized('unknown destination key')
    if scope != into:
        raise Forbidden('destination key does not authorize the requested space')
    if roles.get(token, 'full') != 'full':
        raise Forbidden('destination key requires the full role for a space merge')
