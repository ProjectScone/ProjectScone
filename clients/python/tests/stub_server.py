"""A tiny scriptable HTTP server for exercising the client without Scone.

Tests hand it a list of canned responses; it records every request it
received so assertions can inspect headers, query strings, and bodies.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


class RecordedRequest:
    """One request the stub saw, decoded far enough to assert against."""

    def __init__(
        self, method: str, path: str, query: Dict[str, List[str]],
        headers: Dict[str, str], body: bytes,
    ) -> None:
        self.method = method
        self.path = path
        self.query = query
        self.headers = headers
        self.body = body

    @property
    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8")) if self.body else None

    def __repr__(self) -> str:
        return f"<{self.method} {self.path} query={self.query}>"


class StubScone:
    """A stub Scone server: queue responses, then read back what arrived."""

    def __init__(self) -> None:
        self.requests: List[RecordedRequest] = []
        # Explicit per-(method, path) routes win; otherwise the queue is
        # drained in order, and a `default` covers anything left.
        self.routes: Dict[Tuple[str, str], Tuple[int, Any]] = {}
        self.queue: List[Tuple[int, Any]] = []
        self.default: Tuple[int, Any] = (200, {})
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- scripting -----------------------------------------------------

    def route(self, method: str, path: str, status: int, payload: Any) -> "StubScone":
        """Answer every `method path` with this status and JSON payload."""
        self.routes[(method.upper(), path)] = (status, payload)
        return self

    def enqueue(self, status: int, payload: Any) -> "StubScone":
        """Answer the next unrouted request with this status and payload."""
        self.queue.append((status, payload))
        return self

    def set_default(self, status: int, payload: Any) -> "StubScone":
        self.default = (status, payload)
        return self

    def _respond_with(self, method: str, path: str) -> Tuple[int, Any]:
        keyed = self.routes.get((method, path))
        if keyed is not None:
            return keyed
        if self.queue:
            return self.queue.pop(0)
        return self.default

    # -- lifecycle -----------------------------------------------------

    def start(self) -> str:
        """Start on an ephemeral port and return the base URL."""
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # keep pytest output clean
                pass

            def _handle(self, method: str) -> None:
                parsed = urlparse(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                stub.requests.append(
                    RecordedRequest(
                        method=method,
                        path=parsed.path,
                        query=parse_qs(parsed.query),
                        headers={k.lower(): v for k, v in self.headers.items()},
                        body=body,
                    )
                )
                status, payload = stub._respond_with(method, parsed.path)
                if isinstance(payload, (bytes, str)):
                    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
                    content_type = "text/plain; charset=utf-8"
                else:
                    raw = json.dumps(payload).encode("utf-8")
                    content_type = "application/json"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                self._handle("GET")

            def do_POST(self) -> None:
                self._handle("POST")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # serve_forever polls at 0.5s by default, which would put half a
        # second of teardown on every test that uses the stub.
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> "StubScone":
        self.base_url = self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
