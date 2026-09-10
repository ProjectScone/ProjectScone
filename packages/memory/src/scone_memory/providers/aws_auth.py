"""HTTPX AWS SigV4 signing with explicitly supplied credentials only.

No SDK session, environment credentials, credential chain, IMDS lookup,
refresh, or network access is performed. Replace the auth instance when
short-lived credentials rotate. HTTPS is required. The signer supports the
``es`` and ``aoss`` signing scopes; that does not imply an adapter implements
both services' APIs. OpenSearchVectorIndex targets managed domains (``es``).

https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv-create-signed-request.html
https://docs.aws.amazon.com/opensearch-service/latest/developerguide/managedomains-signing-service-requests.html
"""
from __future__ import annotations

from collections.abc import Callable, Generator
from datetime import datetime, timezone
import hashlib
import hmac
import re
from urllib.parse import quote, urlsplit

import httpx


_UNSIGNED_HEADERS = frozenset({
    "authorization", "connection", "expect", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade", "user-agent",
    "x-amzn-trace-id",
})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_path(path: str) -> str:
    segments: list[str] = []
    for segment in path.split("/"):
        if segment == "..":
            if segments:
                segments.pop()
        elif segment and segment != ".":
            segments.append(segment)
    normalized = "/" + "/".join(segments)
    if segments and path.endswith("/"):
        normalized += "/"
    return quote(normalized, safe="/~")


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


class AwsSigV4Auth(httpx.Auth):
    """Sign complete buffered HTTP requests; secrets are never logged or repr'd."""

    requires_request_body = True

    def __init__(self, access_key_id: str, secret_access_key: str, *, region: str,
                 service: str = "es", session_token: str | None = None,
                 clock: Callable[[], datetime] = _utc_now) -> None:
        if not isinstance(access_key_id, str) or not re.fullmatch(r"[A-Za-z0-9]+", access_key_id):
            raise ValueError("AWS access_key_id must be nonblank alphanumeric text")
        for name, value in (("secret_access_key", secret_access_key), ("session_token", session_token)):
            if name == "session_token" and value is None:
                continue
            if not isinstance(value, str) or not value or any(not 33 <= ord(char) <= 126 for char in value):
                raise ValueError(f"AWS {name} must be nonblank ASCII text without whitespace")
        if not isinstance(region, str) or not re.fullmatch(r"[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+", region):
            raise ValueError("AWS region must be a region identifier")
        if service not in ("es", "aoss"):
            raise ValueError("AWS signing service must be es or aoss")
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self.region = region
        self.service = service
        self._clock = clock

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        if request.url.scheme != "https" or request.url.username or request.url.password:
            raise ValueError("AWS signing requires an HTTPS URL without embedded credentials")
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("AWS signing clock must provide a timezone-aware datetime")
        timestamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for header in ("authorization", "date", "x-amz-security-token"):
            request.headers.pop(header, None)
        request.headers["x-amz-date"] = timestamp
        payload_hash = hashlib.sha256(request.content).hexdigest()
        request.headers["x-amz-content-sha256"] = payload_hash
        if self._session_token is not None:
            request.headers["x-amz-security-token"] = self._session_token
        values: dict[str, list[str]] = {}
        for name, value in request.headers.multi_items():
            if name not in _UNSIGNED_HEADERS:
                values.setdefault(name, []).append(" ".join(value.split()))
        signed_headers = ";".join(sorted(values))
        canonical_headers = "".join(f"{name}:{','.join(values[name])}\n" for name in sorted(values))
        url = urlsplit(str(request.url))
        query_pairs = [pair.partition("=")[::2] for pair in url.query.split("&")] if url.query else []
        canonical_query = "&".join(f"{name}={value}" for name, value in sorted(query_pairs))
        canonical = "\n".join((request.method, _canonical_path(url.path), canonical_query,
                                canonical_headers, signed_headers, payload_hash))
        scope = f"{timestamp[:8]}/{self.region}/{self.service}/aws4_request"
        to_sign = "\n".join(("AWS4-HMAC-SHA256", timestamp, scope,
                              hashlib.sha256(canonical.encode("utf-8")).hexdigest()))
        key = _hmac(("AWS4" + self._secret_access_key).encode("utf-8"), timestamp[:8])
        for component in (self.region, self.service, "aws4_request"):
            key = _hmac(key, component)
        signature = _hmac(key, to_sign).hex()
        request.headers["authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={self._access_key_id}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        yield request
