"""Bound encoded and decoded response bytes before parsing JSON."""

from __future__ import annotations

import sys
import zlib
from typing import Iterator, Optional

import requests
import urllib3
from urllib3.exceptions import HTTPError
from urllib3.response import HTTPResponse

from .errors import SconeError

DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_READ_BYTES = 65536
_MAX_GZIP_MEMBERS = 1024


def _limit(status: int) -> SconeError:
    return SconeError("response byte limit exceeded", status)


def _length(response: requests.Response) -> Optional[int]:
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    value = value.strip()
    if not value or not value.isascii() or not value.isdecimal():
        raise SconeError("invalid response Content-Length", response.status_code)
    # Avoid converting arbitrarily large integers, including on Python 3.9.
    significant = value.lstrip("0") or "0"
    if len(significant) > 20:
        raise _limit(response.status_code)
    return int(significant)


def _decode(encoded: bytes, coding: str, limit: int, status: int) -> bytes:
    if coding in ("", "identity"):
        if len(encoded) > limit:
            raise _limit(status)
        return encoded
    output = bytearray()
    remaining = encoded
    members = 0
    while remaining:
        members += 1
        if members > _MAX_GZIP_MEMBERS:
            raise SconeError("compressed response member limit exceeded", status)
        if coding == "gzip":
            window = zlib.MAX_WBITS + 16
        else:
            # RFC 1950's two-byte header distinguishes wrapped deflate from
            # the raw deflate sent by some HTTP servers.
            wrapped = (
                len(remaining) >= 2
                and remaining[0] & 15 == 8
                and remaining[0] >> 4 <= 7
                and int.from_bytes(remaining[:2], "big") % 31 == 0
            )
            window = zlib.MAX_WBITS if wrapped else -zlib.MAX_WBITS
        decoder = zlib.decompressobj(window)
        budget = min(sys.maxsize, limit - len(output) + 1)
        try:
            decoded = decoder.decompress(remaining, budget)
        except zlib.error:
            if coding != "deflate" or window != zlib.MAX_WBITS:
                raise
            # Raw deflate can coincidentally start with a valid zlib header.
            # Replaying this bounded buffer never replays the HTTP request.
            decoder = zlib.decompressobj(-zlib.MAX_WBITS)
            decoded = decoder.decompress(remaining, budget)
        output.extend(decoded)
        if len(output) > limit:
            raise _limit(status)
        if not decoder.eof:
            raise SconeError("truncated compressed response", status)
        remaining = decoder.unused_data
        if remaining and coding != "gzip":
            raise SconeError("trailing data in compressed response", status)
    if not encoded:
        raise SconeError("truncated compressed response", status)
    return bytes(output)


def read_response(response: requests.Response, limit: int) -> bytes:
    """Read at most a bounded wire body, then decompress with a bounded output.

    Custom session adapters may already have buffered/decoded their response;
    that allocation is outside this reader's control. It is still checked here.
    """
    status = response.status_code
    if response.raw is None or getattr(response, "_content_consumed", False) is True:
        try:
            body = response.content
        except RuntimeError as exc:
            raise SconeError(
                "response body was consumed by a session hook", status
            ) from exc
        if len(body) > limit:
            raise _limit(status)
        return body
    coding = response.headers.get("Content-Encoding", "").strip().lower()
    if coding not in ("", "identity", "gzip", "deflate"):
        raise SconeError(f"unsupported response Content-Encoding: {coding}", status)
    # Compression metadata may exceed a tiny decoded body. Still cap wire
    # bytes, including metadata and empty concatenated gzip members.
    wire_limit = limit if coding in ("", "identity") else 2 * limit + _READ_BYTES
    declared = _length(response)
    if declared is not None and declared > wire_limit:
        raise _limit(status)
    body_buffer = bytearray()
    try:
        while True:
            amount = min(_READ_BYTES, wire_limit - len(body_buffer) + 1)
            if isinstance(response.raw, HTTPResponse):
                part = response.raw.read(amount, decode_content=False)
            else:
                part = response.raw.read(amount)
            if not isinstance(part, bytes):
                raise SconeError("response stream did not return bytes", status)
            if not part:
                break
            body_buffer.extend(part)
            if len(body_buffer) > wire_limit:
                raise _limit(status)
        if declared is not None and len(body_buffer) != declared:
            raise SconeError("truncated response body", status)
        return _decode(bytes(body_buffer), coding, limit, status)
    except (HTTPError, requests.RequestException, OSError, zlib.error) as exc:
        raise SconeError(f"response body failed: {exc}", status) from exc


def read_lines(response: requests.Response, *, line_limit: int, total_limit: int) -> Iterator[bytes]:
    """Yield one line at a time from a streaming response, bounded twice.

    A frame is checked before it is parsed: a single line longer than
    ``line_limit`` is refused rather than buffered, and a stream that has
    delivered more than ``total_limit`` in all is refused rather than kept
    open. The wire must be uncompressed -- an encoded stream cannot be
    bounded per line -- so anything but ``identity`` is refused here.
    """
    status = response.status_code
    coding = response.headers.get("Content-Encoding", "").strip().lower()
    if coding not in ("", "identity"):
        raise SconeError("streamed response must not be content-encoded", status)
    if response.raw is None:
        raise SconeError("streamed response has no readable body", status)
    total = 0
    pending = b""
    try:
        for chunk in response.raw.stream(8192, decode_content=False):
            if not chunk:
                continue
            total += len(chunk)
            if total > total_limit:
                raise _limit(status)
            pending += chunk
            if len(pending) > line_limit and b"\n" not in pending:
                raise SconeError("streamed line exceeds its limit", status)
            while True:
                cut = pending.find(b"\n")
                if cut < 0:
                    break
                line, pending = pending[:cut], pending[cut + 1:]
                if len(line) > line_limit:
                    raise SconeError("streamed line exceeds its limit", status)
                yield line.rstrip(b"\r")
    except (requests.RequestException, urllib3.exceptions.HTTPError, TimeoutError, OSError) as exc:
        # urllib3 raises its own ReadTimeoutError from `raw.stream()`, which
        # is not a requests exception; a stalled server must still surface
        # as a refusal rather than as a raw transport error.
        raise SconeError("streamed read failed: " + str(exc), status) from exc
    if pending:
        if len(pending) > line_limit:
            raise SconeError("streamed line exceeds its limit", status)
        yield pending.rstrip(b"\r")
