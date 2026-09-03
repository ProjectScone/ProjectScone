"""The single exception the Scone client raises."""

from __future__ import annotations

from typing import Optional

__all__ = ["SconeError"]


class SconeError(Exception):
    """Raised when Scone refuses a request, or the client cannot make one.

    ``status`` is the HTTP status the server answered with, or None when the
    failure happened before a response existed (no API key configured, the
    connection never opened, the request timed out).
    """

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        *,
        body: Optional[str] = None,
    ) -> None:
        super().__init__(message if status is None else f"[{status}] {message}")
        self.message = message
        self.status = status
        self.body = body
