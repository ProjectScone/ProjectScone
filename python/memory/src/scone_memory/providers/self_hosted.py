"""Endpoint boundary for operator-managed self-hosted services on a private network.

Names in local DNS namespaces are resolved by the host's resolver. Operators
remain responsible for their DNS and for keeping inference within their self-hosted deployment.
"""

from ipaddress import ip_address, ip_network
import re
from urllib.parse import unquote, urlsplit, urlunsplit


_NETWORKS = tuple(ip_network(value) for value in (
    '127.0.0.0/8', '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', '::1/128', 'fc00::/7',
))


def validate_self_hosted_identifier(value: str, *, max_length: int = 160) -> str:
    """A service's JSON model/voice identifier, not a persona alias or URL path."""
    if (not isinstance(value, str) or not value.strip() or len(value) > max_length
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError('self-hosted model and voice identifiers must be bounded nonblank text')
    return value


def validate_self_hosted_endpoint(value: str) -> str:
    """Return an HTTP(S) self-hosted service base URL with one trailing slash.

    Admit loopback, RFC1918 and IPv6 ULA literals, plus localhost and names
    under .localhost, .local and .home.arpa. No public hosts, userinfo, query,
    fragment, control characters or traversal segments are admitted. This
    is a configuration boundary, not a sandbox for the service being called.
    """
    invalid = 'endpoint must be an HTTP(S) self-hosted service URL without credentials, query or fragment'
    if not isinstance(value, str) or not value or len(value) > 2048:
        raise ValueError(invalid)
    if any(ord(char) <= 32 or ord(char) == 127 for char in value) or '\\' in value:
        raise ValueError(invalid)
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ValueError(invalid) from None
    if parsed.scheme not in ('http', 'https') or not host or '@' in parsed.netloc or '?' in value or '#' in value:
        raise ValueError(invalid)
    if port == 0 or '%' in host:
        raise ValueError(invalid)
    try:
        address = ip_address(host)
    except ValueError:
        local_name = host == 'localhost' or host.endswith(('.localhost', '.local', '.home.arpa'))
        if not local_name or not all(re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', label)
                                     for label in host.split('.')):
            raise ValueError(invalid) from None
    else:
        if not any(address in network for network in _NETWORKS):
            raise ValueError(invalid)
    decoded_path = unquote(parsed.path)
    if any(part in ('.', '..') for part in decoded_path.split('/')) or any(
            ord(char) <= 32 or ord(char) == 127 or char == '\\' for char in decoded_path):
        raise ValueError(invalid)
    authority = f'[{host}]' if ':' in host else host
    if port is not None:
        authority += f':{port}'
    return urlunsplit((parsed.scheme, authority, parsed.path.rstrip('/') + '/', '', ''))
