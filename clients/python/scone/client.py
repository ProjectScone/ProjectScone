"""A thin synchronous client for the Scone HTTP API.

One class, one session, one space. The API key a ``Scone`` instance carries
is bound server-side to exactly one space, so there is no space argument
anywhere below: the key decides.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type, Union

import requests

from .errors import SconeError
from .models import Added, Fact, Profile, Recall, Status, Tag

__all__ = ["Scone", "DEFAULT_BASE_URL", "DEFAULT_TIMEOUT"]

DEFAULT_BASE_URL = "http://127.0.0.1:7437"
DEFAULT_TIMEOUT = 30.0

# Server-side bounds from crates/scone/src/serve.rs, mirrored here so a
# doomed request fails locally with a readable message instead of a 422.
MAX_CONTENT_BYTES = 100_000
MAX_QUERY_CHARS = 1_000
MAX_TAGS = 10
MAX_REASON_CHARS = 500

Json = Dict[str, Any]


class Scone:
    """A client for one Scone space, addressed by its Bearer API key."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        timeout: Union[float, Tuple[float, float]] = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Configure a client, reading SCONE_URL and SCONE_API_KEY when unset.

        Raises SconeError when no API key can be found, because every route
        on the server requires one.
        """
        url = base_url or os.environ.get("SCONE_URL") or DEFAULT_BASE_URL
        key = api_key or os.environ.get("SCONE_API_KEY")
        if not key:
            raise SconeError(
                "no API key: pass api_key= or set SCONE_API_KEY "
                "(every Scone route requires a Bearer key)"
            )
        self.base_url = url.rstrip("/")
        self.timeout = timeout
        self._owns_session = session is None
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
                "User-Agent": "scone-client-python/0.2.1",
            }
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "Scone":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Scone(base_url={self.base_url!r})"

    # ------------------------------------------------------------------
    # endpoints
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        tags: Optional[Iterable[str]] = None,
        source: Optional[str] = None,
    ) -> Added:
        """Store a note in the space, optionally tagged, and report the outcome.

        Storing the same text twice deduplicates rather than creating a second
        episode; the returned ``Added.deduplicated`` says which happened.

        ``source`` is accepted only to be refused: POST /v1/episodes has no
        source field, so a server-side source would be silently dropped.
        Passing a non-None value raises rather than lying about it.
        """
        if source is not None:
            raise SconeError(
                "POST /v1/episodes accepts only content and tags; a source "
                "would be dropped on the floor. Ingest with a source via the "
                "CLI (`scone add <file>`) instead."
            )
        if not text:
            raise SconeError("text must not be empty")
        encoded = len(text.encode("utf-8"))
        if encoded > MAX_CONTENT_BYTES:
            raise SconeError(f"text must be 1..={MAX_CONTENT_BYTES} bytes, got {encoded}")
        tag_list = list(tags or [])
        if len(tag_list) > MAX_TAGS:
            raise SconeError(f"at most {MAX_TAGS} tags, got {len(tag_list)}")
        body: Json = {"content": text}
        if tag_list:
            body["tags"] = tag_list
        return Added.from_json(self._request("POST", "/v1/episodes", json=body))

    def recall(
        self,
        query: str,
        limit: Optional[int] = None,
        as_of: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> Recall:
        """Search the space for the text and facts that answer the query.

        ``as_of`` is an ISO-8601 instant that evaluates fact validity at a past
        moment. ``tags`` narrows to episodes carrying all of the given tags.
        """
        # The server only rejects the *empty* string up front; a blank one
        # reaches the engine and comes back as a 500. Refuse it here.
        if not query.strip():
            raise SconeError("query must not be empty or blank")
        if len(query) > MAX_QUERY_CHARS:
            raise SconeError(f"query must be 1..={MAX_QUERY_CHARS} chars, got {len(query)}")
        params: Dict[str, str] = {"q": query}
        if limit is not None:
            params["limit"] = str(limit)
        if as_of is not None:
            params["as_of"] = as_of
        tag_list = list(tags or [])
        if tag_list:
            params["tags"] = ",".join(tag_list)
        return Recall.from_json(self._request("GET", "/v1/recall", params=params))

    def facts(self, include_closed: bool = False) -> List[Fact]:
        """List the space's facts, active ones only unless closed ones are asked for."""
        params = {"all": "true"} if include_closed else None
        payload = self._request("GET", "/v1/facts", params=params)
        return [Fact.from_json(f) for f in payload.get("facts") or []]

    def close_fact(self, fact_id: int, reason: str) -> int:
        """Retire an active fact with a stated reason and return its id.

        A fact that is missing or already closed is not closeable; the server
        reports that as a 500, which surfaces here as a SconeError.
        """
        if not reason:
            raise SconeError("reason must not be empty")
        if len(reason) > MAX_REASON_CHARS:
            raise SconeError(f"reason must be 1..={MAX_REASON_CHARS} chars, got {len(reason)}")
        payload = self._request(
            "POST", f"/v1/facts/{int(fact_id)}/close", json={"reason": reason}
        )
        return int(payload.get("closed", fact_id))

    def profile(self) -> Profile:
        """Fetch the space's durable facts and recent activity summary."""
        return Profile.from_json(self._request("GET", "/v1/profile"))

    def status(self) -> Status:
        """Report the space's size, revision, and whether distillation is running."""
        return Status.from_json(self._request("GET", "/v1/status"))

    def tags(self) -> List[Tag]:
        """List every tag in the space with its episode count, busiest first."""
        payload = self._request("GET", "/v1/tags")
        return [Tag.from_json(t) for t in payload.get("tags") or []]

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, str]] = None,
        json: Optional[Json] = None,
    ) -> Json:
        """Send one request and return its decoded JSON body, or raise SconeError."""
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method, url, params=params, json=json, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise SconeError(f"{method} {url} failed: {exc}") from exc

        if not response.ok:
            raise SconeError(_error_message(response), response.status_code, body=response.text)

        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            raise SconeError(
                f"{method} {path} returned a non-JSON body", response.status_code,
                body=response.text,
            ) from exc
        if not isinstance(payload, dict):
            raise SconeError(
                f"{method} {path} returned {type(payload).__name__}, expected an object",
                response.status_code,
                body=response.text,
            )
        return payload


def _error_message(response: requests.Response) -> str:
    """Pull the server's message out of a failure body.

    Handler failures answer ``{"error": "..."}``; axum's own extractor
    rejections (bad JSON, missing ``q``, wrong content type) answer plain
    text, so fall back to the body and then to the status line.
    """
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    text = (response.text or "").strip()
    return text or f"HTTP {response.status_code}"
