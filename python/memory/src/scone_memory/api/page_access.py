"""Narrow request gate for the explicitly enabled local page bootstrap."""

import re

from starlette.requests import Request


_LOOPBACK_AUTHORITY = re.compile(r"(?:localhost|127\.0\.0\.1|\[::1\])(?::([0-9]{1,5}))?", re.IGNORECASE | re.ASCII)


def permits_local_bootstrap(request: Request) -> bool:
    """Require a loopback peer and one unambiguous loopback Host authority.

    URL parsers may replace an invalid Host with the listening address, so
    only the raw header can authorize bootstrap. Forwarded hosts do not.
    """
    if request.client is None or request.client.host not in {"127.0.0.1", "localhost", "::1"}:
        return False
    hosts = request.headers.getlist("host")
    if len(hosts) != 1:
        return False
    authority = _LOOPBACK_AUTHORITY.fullmatch(hosts[0])
    if authority is None:
        return False
    port = authority.group(1)
    return port is None or 1 <= int(port) <= 65535
