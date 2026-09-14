"""A page fetched by URL and read as the document it is, with the door it came through bounded.

The document lane reads what a caller hands it; nothing fetched a page.
The leading frameworks' web readers fetch a URL and hand the HTML to a
parser, and take the network on trust. This one fetches with three
rules, each with a test:

- **Only a host on the public internet, unless told otherwise.** A
  server that fetches whatever URL it is given will fetch its own
  metadata service, its database, or the neighbour on its subnet; every
  address a hostname resolves to must be global, at every redirect, or
  the fetch is refused. ``allow_private`` is for a lab and says so.
- **Bounded in bytes, time and hops.** A page is read up to ``max_bytes``
  and refused past it rather than cut; a redirect chain longer than
  ``max_redirects`` is refused; the whole fetch has a deadline.
- **Read as what the server says it is.** The media type chooses the
  reader (HTML, plain text, Markdown, JSON, CSV, PDF); a type the
  document lane does not read is refused, not guessed at.

What was fetched is retained as the original, so the text can be
checked against the bytes later, and the page's URL, its final URL after
redirects, the media type and the moment are on the episode.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import ipaddress
import socket
from typing import Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, Field

from ..core.errors import InvalidInput
from .files import DocumentIngested, ingest_document
from .formats.types import DocumentLimits
from .formats.registry import DocumentParser

#: What a media type is read as. The document lane decides by filename,
#: so the type chooses one.
READ_AS = {"text/html": "page.html", "application/xhtml+xml": "page.html", "text/plain": "page.txt",
           "text/markdown": "page.md", "text/x-markdown": "page.md", "application/json": "page.json",
           "text/csv": "page.csv", "application/pdf": "page.pdf"}
USER_AGENT = "scone-memory/url-import"


class WebLimits(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    max_bytes: int = Field(default=10 * 1024 * 1024, ge=1, le=25 * 1024 * 1024)
    timeout_seconds: float = Field(default=15.0, gt=0, le=120, allow_inf_nan=False)
    max_redirects: int = Field(default=5, ge=0, le=10)
    #: Whether a host that resolves to a private, loopback or link-local address may be fetched.
    allow_private: bool = False


@dataclass(frozen=True)
class FetchedPage:
    url: str
    final_url: str
    media_type: str
    status: int
    fetched_at: str
    data: bytes
    #: The URLs followed to reach the final one, in order.
    redirects: tuple[str, ...] = ()

    @property
    def filename(self) -> str:
        return READ_AS[self.media_type]


def _refuse(url: str, why: str) -> InvalidInput:
    return InvalidInput(f"URL import refused for {url!r}: {why}")


def check_url(url: str, *, allow_private: bool = False) -> str:
    """The URL, when it is one this reader will fetch: http or https, a
    hostname with no credentials in it, and every address the hostname
    resolves to on the public internet unless ``allow_private``."""
    if not isinstance(url, str) or len(url) > 4096:
        raise InvalidInput("a URL is a string of at most 4096 characters")
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise _refuse(url, "only http and https are fetched")
    if not parts.hostname:
        raise _refuse(url, "no host")
    if parts.username or parts.password:
        raise _refuse(url, "credentials in a URL are not sent")
    try:
        addresses = {str(info[4][0]) for info in socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80),
                                                              proto=socket.IPPROTO_TCP)}
    except socket.gaierror as error:
        raise _refuse(url, f"host does not resolve ({error})") from error
    if not addresses:
        raise _refuse(url, "host does not resolve")
    for address in addresses:
        parsed = ipaddress.ip_address(address.split("%", 1)[0])
        if not allow_private and not parsed.is_global:
            raise _refuse(url, f"{parts.hostname} resolves to {address}, which is not on the public internet; "
                               f"a private, loopback or link-local address is fetched only with allow_private")
    return parts.geturl()


class _Redirects(HTTPRedirectHandler):
    """Every hop checked as the first URL was, and counted."""

    def __init__(self, limits: WebLimits, followed: list[str]) -> None:
        super().__init__()
        self.limits = limits
        self.followed = followed

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        if len(self.followed) >= self.limits.max_redirects:
            raise _refuse(req.full_url, f"more than {self.limits.max_redirects} redirect(s)")
        checked = check_url(newurl, allow_private=self.limits.allow_private)
        self.followed.append(checked)
        return super().redirect_request(req, fp, code, msg, headers, checked)


def _fetch(url: str, limits: WebLimits) -> FetchedPage:
    checked = check_url(url, allow_private=limits.allow_private)
    followed: list[str] = []
    opener = build_opener(_Redirects(limits, followed))
    request = Request(checked, headers={"User-Agent": USER_AGENT, "Accept": ", ".join(READ_AS)})
    try:
        with opener.open(request, timeout=limits.timeout_seconds) as response:
            media_type = (response.headers.get_content_type() or "").lower()
            if media_type not in READ_AS:
                raise _refuse(url, f"media type {media_type or 'unknown'!r} is not one the document lane reads")
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > limits.max_bytes:
                raise _refuse(url, f"{declared} bytes declared, over the bound of {limits.max_bytes}")
            data = response.read(limits.max_bytes + 1)
            if len(data) > limits.max_bytes:
                raise _refuse(url, f"more than {limits.max_bytes} bytes; refused rather than cut")
            return FetchedPage(checked, response.geturl(), media_type, response.status, datetime.now(timezone.utc).isoformat(),
                               data, tuple(followed))
    except HTTPError as error:
        raise _refuse(url, f"the server answered {error.code}") from error
    except (URLError, TimeoutError, OSError, ValueError) as error:
        if isinstance(error, InvalidInput):
            raise
        raise _refuse(url, f"fetch failed ({error})") from error


async def fetch_page(url: str, limits: Optional[WebLimits] = None) -> FetchedPage:
    """The page at ``url``, fetched under ``limits``; refused rather than cut or guessed."""
    bound = limits or WebLimits()
    try:
        return await asyncio.wait_for(asyncio.to_thread(_fetch, url, bound), bound.timeout_seconds + 5)
    except asyncio.TimeoutError:
        raise _refuse(url, f"no answer within {bound.timeout_seconds}s") from None


@dataclass(frozen=True)
class UrlIngested:
    document: DocumentIngested
    url: str
    final_url: str
    media_type: str
    fetched_at: str
    bytes: int
    redirects: tuple[str, ...]

    def record(self) -> dict[str, object]:
        return {"episode_id": self.document.added.episode_id, "url": self.url, "final_url": self.final_url,
                "media_type": self.media_type, "fetched_at": self.fetched_at, "bytes": self.bytes,
                "redirects": list(self.redirects), "format": self.document.format, "segments": self.document.segments,
                "original": self.document.original.attachment_id, "manifest": self.document.manifest.attachment_id}


async def ingest_url(engine, space: str, url: str, *, limits: Optional[WebLimits] = None,
                     parser: DocumentParser | None = None, document_limits: DocumentLimits = DocumentLimits(),
                     tags: Sequence[str] = ()) -> UrlIngested:
    """Fetch the page and read it as the document its media type says it is."""
    page = await fetch_page(url, limits)
    ingested = await ingest_document(engine, space, page.data, filename=page.filename, parser=parser, limits=document_limits,
                                     metadata={"document_url": page.url[:2000], "document_final_url": page.final_url[:2000],
                                               "document_media_type": page.media_type, "document_fetched_at": page.fetched_at})
    return UrlIngested(ingested, page.url, page.final_url, page.media_type, page.fetched_at, len(page.data), page.redirects)
