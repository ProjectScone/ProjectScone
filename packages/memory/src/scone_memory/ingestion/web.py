"""A page fetched by URL and read as the document it is, with the door it came through bounded.

The document lane reads what a caller hands it; nothing fetched a page.
The leading frameworks' web readers fetch a URL and hand the HTML to a
parser, and take the network on trust. This one fetches with three
rules, each with a test:

- **Only a host on the public internet, unless told otherwise.** A
  server that fetches whatever URL it is given will fetch its own
  metadata service, its database, or the neighbour on its subnet; every
  address a hostname resolves to must be global, at every redirect, or
  the fetch is refused, and the connection is made to the address that
  was checked rather than to the name again, so a name that changes its
  answer between the check and the connection gains nothing.
  ``allow_private`` is for a lab and says so.
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
from http.client import HTTPConnection, HTTPException, HTTPSConnection
import ipaddress
import socket
import ssl
from typing import Optional, Sequence
from urllib.parse import urljoin, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from ..core import forget_after as schedule
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


def check_url(url: str, *, allow_private: bool = False) -> tuple[str, str, int, str]:
    """The URL, when it is one this reader will fetch, with the address it
    will be fetched at: http or https, a hostname with no credentials in
    it, and every address the hostname resolves to on the public internet
    unless ``allow_private``. Returns (url, address, port, hostname); the
    connection is made to that address, not to the name again, so a name
    that answers differently between the check and the connection gains
    nothing by it."""
    if not isinstance(url, str) or len(url) > 4096:
        raise InvalidInput("a URL is a string of at most 4096 characters")
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise _refuse(url, "only http and https are fetched")
    if not parts.hostname:
        raise _refuse(url, "no host")
    if parts.username or parts.password:
        raise _refuse(url, "credentials in a URL are not sent")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        addresses = [str(info[4][0]) for info in socket.getaddrinfo(parts.hostname, port, proto=socket.IPPROTO_TCP)]
    except socket.gaierror as error:
        raise _refuse(url, f"host does not resolve ({error})") from error
    if not addresses:
        raise _refuse(url, "host does not resolve")
    for address in addresses:
        parsed = ipaddress.ip_address(address.split("%", 1)[0])
        if not allow_private and not parsed.is_global:
            raise _refuse(url, f"{parts.hostname} resolves to {address}, which is not on the public internet; "
                               f"a private, loopback or link-local address is fetched only with allow_private")
    return parts.geturl(), addresses[0], port, parts.hostname


def _connect(scheme: str, address: str, port: int, hostname: str, timeout: float) -> HTTPConnection:
    """A connection to the address that was checked. For https the
    certificate is still checked against the hostname, so pinning the
    address costs nothing in what the server must prove."""
    sock = socket.create_connection((address, port), timeout=timeout)
    if scheme == "https":
        sock = ssl.create_default_context().wrap_socket(sock, server_hostname=hostname)
        connection: HTTPConnection = HTTPSConnection(hostname, port, timeout=timeout)
    else:
        connection = HTTPConnection(hostname, port, timeout=timeout)
    connection.sock = sock
    return connection


def _fetch(url: str, limits: WebLimits) -> FetchedPage:
    checked, address, port, hostname = check_url(url, allow_private=limits.allow_private)
    followed: list[str] = []
    current = checked
    while True:
        parts = urlsplit(current)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        try:
            connection = _connect(parts.scheme, address, port, hostname, limits.timeout_seconds)
        except (OSError, ssl.SSLError) as error:
            raise _refuse(url, f"connection failed ({error})") from error
        try:
            connection.request("GET", path, headers={"Host": parts.netloc, "User-Agent": USER_AGENT,
                                                     "Accept": ", ".join(READ_AS), "Connection": "close"})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location:
                    raise _refuse(url, f"the server redirected ({response.status}) without saying where")
                if len(followed) >= limits.max_redirects:
                    raise _refuse(url, f"more than {limits.max_redirects} redirect(s)")
                current, address, port, hostname = check_url(urljoin(current, location), allow_private=limits.allow_private)
                followed.append(current)
                continue
            if response.status != 200:
                raise _refuse(url, f"the server answered {response.status}")
            media_type = (response.headers.get_content_type() or "").lower()
            if media_type not in READ_AS:
                raise _refuse(url, f"media type {media_type or 'unknown'!r} is not one the document lane reads")
            declared = response.getheader("Content-Length")
            if declared and declared.isdigit() and int(declared) > limits.max_bytes:
                raise _refuse(url, f"{declared} bytes declared, over the bound of {limits.max_bytes}")
            data = response.read(limits.max_bytes + 1)
            if len(data) > limits.max_bytes:
                raise _refuse(url, f"more than {limits.max_bytes} bytes; refused rather than cut")
            return FetchedPage(checked, current, media_type, response.status, datetime.now(timezone.utc).isoformat(),
                               data, tuple(followed))
        except (HTTPException, OSError, ssl.SSLError) as error:
            if isinstance(error, InvalidInput):
                raise
            raise _refuse(url, f"fetch failed ({error})") from error
        finally:
            connection.close()


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
                "original": self.document.original.attachment_id, "manifest": self.document.manifest.attachment_id,
                "forget_after": self.document.added.forget_after}


async def ingest_url(engine, space: str, url: str, *, limits: Optional[WebLimits] = None,
                     parser: DocumentParser | None = None, document_limits: DocumentLimits = DocumentLimits(),
                     tags: Sequence[str] = (), forget_after: Optional[str] = None) -> UrlIngested:
    """Fetch the page and read it as the document its media type says it is.
    ``forget_after`` schedules the episode's forgetting as for ``remember``;
    a refused schedule is refused before anything is fetched."""
    when = schedule.asked(forget_after, engine.clock())
    page = await fetch_page(url, limits)
    ingested = await ingest_document(engine, space, page.data, filename=page.filename, parser=parser, limits=document_limits,
                                     metadata={"document_url": page.url[:2000], "document_final_url": page.final_url[:2000],
                                               "document_media_type": page.media_type, "document_fetched_at": page.fetched_at},
                                     forget_after=when)
    return UrlIngested(ingested, page.url, page.final_url, page.media_type, page.fetched_at, len(page.data), page.redirects)
