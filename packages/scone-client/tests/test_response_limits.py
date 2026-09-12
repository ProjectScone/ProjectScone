"""Real streaming transport: response budgets, compressed bodies and no replay."""

from contextlib import contextmanager
import gzip
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import zlib
from unittest.mock import Mock

import pytest
import requests

from scone import Scone, SconeError


@contextmanager
def endpoint(
    payload, *, coding=None, status=200, declared=True, chunked=False, stall=False
):
    observed = []
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.do_GET()

        def do_GET(self):
            observed.append((self.command, self.headers.get("Accept-Encoding")))
            self.send_response(status)
            if coding:
                self.send_header("Content-Encoding", coding)
            self.send_header("Content-Type", "application/json")
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            elif declared is not False:
                self.send_header(
                    "Content-Length",
                    str(len(payload) if declared is True else declared),
                )
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                if stall:
                    release.wait(2)
                if chunked:
                    for offset in range(0, len(payload), 13):
                        part = payload[offset : offset + 13]
                        self.wfile.write(
                            ("%x\r\n" % len(part)).encode() + part + b"\r\n"
                        )
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port), observed
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, None])
def test_response_budget_is_validated_at_construction(value):
    with pytest.raises(SconeError):
        Scone("http://127.0.0.1:1", "key", max_response_bytes=value)


@pytest.mark.parametrize(
    "declared,chunked", [(True, False), (False, False), (False, True)]
)
def test_large_success_and_error_bodies_are_bounded_without_post_retry(
    declared, chunked
):
    for status in (200, 503):
        with endpoint(
            b'{"message":"' + b"x" * 4096 + b'"}',
            declared=declared,
            chunked=chunked,
            status=status,
        ) as (url, observed):
            with Scone(url, "key", max_response_bytes=128) as client:
                with pytest.raises(SconeError, match="response byte limit") as error:
                    client._request("POST", "/mutation", json={"write": True})
            assert error.value.status == status and error.value.body is None
            assert len(observed) == 1


@pytest.mark.parametrize(
    "coding,encode",
    [
        ("gzip", gzip.compress),
        ("deflate", zlib.compress),
        ("deflate", lambda value: zlib.compress(value)[2:-4]),
    ],
)
def test_compression_is_bounded_by_decoded_bytes(coding, encode):
    with endpoint(encode(b'{"text":"' + b"x" * 1000000 + b'"}'), coding=coding) as (
        url,
        _,
    ):
        with Scone(url, "key", max_response_bytes=128) as client:
            with pytest.raises(SconeError, match="response byte limit"):
                client._request("GET", "/value")


@pytest.mark.parametrize(
    "coding,encode",
    [
        (None, lambda value: value),
        ("gzip", gzip.compress),
        ("deflate", zlib.compress),
        ("deflate", lambda value: zlib.compress(value)[2:-4]),
    ],
)
def test_exact_unicode_budget_and_explicit_compression_negotiation(coding, encode):
    raw = json.dumps({"text": "é" * 100}, ensure_ascii=False).encode()
    with endpoint(encode(raw), coding=coding, chunked=True) as (url, observed):
        with Scone(url, "key", max_response_bytes=len(raw)) as client:
            assert client._request("GET", "/value") == {"text": "é" * 100}
        assert observed[0][1] == "gzip, deflate"


def test_concatenated_gzip_members_remain_supported():
    with endpoint(
        gzip.compress(b'{"ok":') + gzip.compress(b"true}"), coding="gzip"
    ) as (url, _):
        with Scone(url, "key", max_response_bytes=100) as client:
            assert client._request("GET", "/value") == {"ok": True}


@pytest.mark.parametrize(
    "payload,coding,declared",
    [
        (gzip.compress(b"{}")[:-3], "gzip", True),
        (b"{}", "unknown", True),
        (b"{}", None, 1000),
    ],
)
def test_truncated_or_unsupported_response_is_not_a_success(payload, coding, declared):
    with endpoint(payload, coding=coding, declared=declared) as (url, observed):
        with Scone(url, "key", max_response_bytes=2000) as client:
            with pytest.raises(SconeError):
                client._request("POST", "/mutation", json={})
        assert len(observed) == 1


def test_injected_session_remains_open_after_response_limit_error():
    class Session(requests.Session):
        closed = False

        def close(self):
            self.closed = True
            super().close()

    session = Session()
    try:
        with endpoint(b"x" * 1024) as (url, _):
            with Scone(url, "key", session=session, max_response_bytes=128) as client:
                with pytest.raises(SconeError):
                    client._request("GET", "/value")
        assert not session.closed
    finally:
        session.close()


@pytest.mark.parametrize(
    "payload,status,coding",
    [
        (b'{"ok":true}', 200, None),
        (b'{"error":"unavailable"}', 503, None),
        (b"bad json", 200, None),
        (b"{}", 302, None),
        (b"x" * 4096, 200, None),
        (b"{}", 200, "unknown"),
    ],
)
def test_response_is_closed_on_every_exit(payload, status, coding):
    closed = []

    def capture(response, **kwargs):
        response.close = Mock(wraps=response.close)
        closed.append(response.close)

    with requests.Session() as session:
        session.hooks["response"] = [capture]
        with endpoint(payload, status=status, coding=coding) as (url, _):
            with Scone(url, "key", session=session, max_response_bytes=128) as client:
                if payload == b'{"ok":true}':
                    assert client._request("GET", "/value") == {"ok": True}
                else:
                    with pytest.raises(SconeError):
                        client._request("GET", "/value")
    assert len(closed) == 1
    closed[0].assert_called_once()


def test_stalled_body_keeps_status_and_never_replays_post():
    with endpoint(b"{}", stall=True) as (url, observed):
        with Scone(url, "key", timeout=(1, 0.02)) as client:
            with pytest.raises(SconeError) as error:
                client._request("POST", "/mutation", json={})
        assert error.value.status == 200
        assert len(observed) == 1


@pytest.mark.parametrize("declared", ["bad", "-1", "9" * 5000])
def test_invalid_or_huge_content_length_is_refused(declared):
    with endpoint(b"{}", declared=declared) as (url, _):
        with Scone(url, "key") as client:
            with pytest.raises(SconeError):
                client._request("GET", "/value")


def test_long_zero_padded_length_does_not_escape_as_python_integer_error():
    with endpoint(b"{}", declared="0" * 5000 + "2") as (url, _):
        with Scone(url, "key") as client:
            assert client._request("GET", "/value") == {}


def test_compression_metadata_is_bounded_even_when_decoded_body_is_empty():
    with endpoint(gzip.compress(b"") * 4000, coding="gzip", declared=False) as (url, _):
        with Scone(url, "key", max_response_bytes=128) as client:
            with pytest.raises(SconeError, match="response byte limit"):
                client._request("GET", "/value")


def test_gzip_member_count_is_bounded():
    with endpoint(gzip.compress(b"") * 1025, coding="gzip") as (url, _):
        with Scone(url, "key", max_response_bytes=128) as client:
            with pytest.raises(SconeError, match="member limit"):
                client._request("GET", "/value")


@pytest.mark.parametrize(
    "payload,coding",
    [
        (gzip.compress(b"{}") + b"garbage", "gzip"),
        (zlib.compress(b"{}") + b"garbage", "deflate"),
        (b"", "gzip"),
        (gzip.compress(b"{}")[:-8] + b"badcrc!!", "gzip"),
    ],
)
def test_compression_integrity_is_checked(payload, coding):
    with endpoint(payload, coding=coding) as (url, _):
        with Scone(url, "key") as client:
            with pytest.raises(SconeError):
                client._request("GET", "/value")


@pytest.mark.parametrize(
    "coding,encode", [(None, lambda value: value), ("gzip", gzip.compress)]
)
def test_injected_session_hook_can_cache_response_body(coding, encode):
    def cache(response, **kwargs):
        assert response.content

    with requests.Session() as session:
        session.hooks["response"] = [cache]
        for raw in (b'{"ok":true}', b"x" * 1024):
            with endpoint(encode(raw), coding=coding) as (url, _):
                with Scone(
                    url, "key", session=session, max_response_bytes=128
                ) as client:
                    if len(raw) > 128:
                        with pytest.raises(SconeError, match="response byte limit"):
                            client._request("GET", "/value")
                    else:
                        assert client._request("GET", "/value") == {"ok": True}


def test_unknown_charset_does_not_break_valid_json():
    def encoding(response, **kwargs):
        response.encoding = "x-unknown-review-charset"

    with requests.Session() as session:
        session.hooks["response"] = [encoding]
        with endpoint(b'{"ok":true}') as (url, _):
            with Scone(url, "key", session=session) as client:
                assert client._request("GET", "/value") == {"ok": True}


@pytest.mark.parametrize("raw", [b'{"ok":true}', b"x" * 1024])
def test_injected_adapter_can_return_a_file_like_raw_body(raw):
    class Adapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            response = requests.Response()
            response.status_code = 200
            response.raw = io.BytesIO(raw)
            response.headers["Content-Length"] = str(len(raw))
            return response

        def close(self):
            pass

    with requests.Session() as session:
        session.mount("http://", Adapter())
        with Scone(
            "http://127.0.0.1:1", "key", session=session, max_response_bytes=128
        ) as client:
            if len(raw) > 128:
                with pytest.raises(SconeError, match="response byte limit"):
                    client._request("GET", "/value")
            else:
                assert client._request("GET", "/value") == {"ok": True}


def test_raw_deflate_with_a_coincidental_zlib_header():
    # A non-final stored block with 156 bytes, then a final empty block.
    encoded = (
        bytes.fromhex("789c0063ff") + b"{}" + b" " * 154 + bytes.fromhex("010000ffff")
    )
    assert zlib.decompress(encoded, -zlib.MAX_WBITS) == b"{}" + b" " * 154
    with endpoint(encoded, coding="deflate") as (url, _):
        with Scone(url, "key", max_response_bytes=156) as client:
            assert client._request("GET", "/value") == {}


def test_large_configured_byte_budget_does_not_overflow_decoder():
    with endpoint(gzip.compress(b"{}"), coding="gzip") as (url, _):
        with Scone(url, "key", max_response_bytes=2**100) as client:
            assert client._request("GET", "/value") == {}


def test_successful_json_does_not_decode_an_error_body(monkeypatch):
    import codecs
    decoded = []
    def record_codec(name):
        if name == 'scone_error_body_probe':
            decoded.append(name)
            return codecs.lookup('utf-8')
    codecs.register(record_codec)
    response = requests.Response()
    response.status_code = 200
    response.encoding = 'scone-error-body-probe'
    response.raw = io.BytesIO(b'{"ok":true}')
    response.headers['Content-Type'] = 'application/json'
    session = Mock(spec=requests.Session)
    session.headers = {}
    session.request.return_value = response
    try:
        assert Scone('http://127.0.0.1:1', 'key', session=session)._request('GET', '/fixture') == {'ok': True}
        assert decoded == []
    finally:
        unregister = getattr(codecs, 'unregister', None)
        if unregister is not None:
            unregister(record_codec)
