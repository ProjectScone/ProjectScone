"""The bearer key, read once and the same way by every HTTP surface.

A key is not a login: it names a space and carries a role, and that pair
is the whole of what its holder may do. The REST API asks this module
per route, because there the role answer depends on the route. The MCP
transport asks it once per request as ASGI middleware, in front of
everything, because there a refusal has no useful body to write into a
JSON-RPC answer -- a caller with no key never reaches a session at all.

Keeping both on one reading means a key that is refused here is refused
there, in the same words, whichever door it knocks on.
"""

from __future__ import annotations

import json
from typing import Awaitable, Callable, Mapping, NamedTuple, Optional

#: Where an admitted request's holder is left for the app behind the
#: middleware. ASGI reserves ``scope["state"]`` for exactly this.
KEY_STATE = "scone.key_holder"

#: The roles that may write. "review" decides proposals in the REST API
#: and so is not read-only there, but it writes nothing of its own.
WRITING_ROLES = ("write", "full")


class Unauthorized(Exception):
    """No key, or a key this host does not know. Maps to HTTP 401."""


class KeyHolder(NamedTuple):
    """What a key entitles its bearer to: one space, under one role."""

    space: str
    role: str

    @property
    def may_write(self) -> bool:
        return self.role in WRITING_ROLES


def bearer_token(header: str) -> Optional[str]:
    """The token in an Authorization header, or None if there is none to
    read. The scheme is matched without case (RFC 7235 says it is
    case-insensitive, and clients differ on it); any other scheme is not
    a key we can read, so it counts as absent rather than as wrong."""
    scheme, _, rest = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return rest.strip() or None


def key_holder(header: str, keys: Mapping[str, str], roles: Mapping[str, str]) -> KeyHolder:
    """Resolve an Authorization header against the host's key table.

    A key with no recorded role holds the full one: the single-key setup
    (``SCONE_API_KEY``) is one owner on their own memory, and asking them
    to spell that out would be ceremony.
    """
    token = bearer_token(header)
    if token is None:
        raise Unauthorized("missing bearer key")
    space = keys.get(token)
    if space is None:
        # Deliberately not "no such key for space X": an unknown key
        # learns nothing about which spaces exist.
        raise Unauthorized("unknown key")
    return KeyHolder(space, roles.get(token, "full"))


Scope = dict
Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]


class BearerKeys:
    """ASGI middleware: every request carries a key, or it stops here.

    The tables are read on each request rather than copied, so a host
    that adds or withdraws a key while serving is obeyed from the next
    request on. An admitted request reaches the app with its holder in
    ``scope["state"][KEY_STATE]``, which is what decides the space and
    the role behind it.
    """

    def __init__(self, app, keys: Mapping[str, str], roles: Mapping[str, str]) -> None:
        self.app = app
        self.keys = keys
        self.roles = roles

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            # The lifespan is the host starting and stopping its own
            # server; there is no bearer to ask.
            await self.app(scope, receive, send)
            return
        try:
            holder = key_holder(self._authorization(scope), self.keys, self.roles)
        except Unauthorized as refused:
            await self._refuse(scope, send, str(refused))
            return
        scope.setdefault("state", {})[KEY_STATE] = holder
        await self.app(scope, receive, send)

    @staticmethod
    def _authorization(scope: Scope) -> str:
        for name, value in scope.get("headers", ()):
            if name.lower() == b"authorization":
                return value.decode("latin-1")
        return ""

    @staticmethod
    async def _refuse(scope: Scope, send: Send, reason: str) -> None:
        if scope["type"] == "websocket":
            # Closed before the handshake is accepted: 1008 is the
            # policy-violation close, which is what a browser client is
            # shown in place of a status code.
            await send({"type": "websocket.close", "code": 1008, "reason": reason})
            return
        body = json.dumps({"error": reason}).encode()
        await send({"type": "http.response.start", "status": 401, "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"www-authenticate", b"Bearer"),
        ]})
        await send({"type": "http.response.body", "body": body})
