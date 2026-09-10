"""Executed inside the disposable smoke container, with synthetic input only."""

from __future__ import annotations

import json
from importlib import import_module
from importlib.metadata import version
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def fetch(path: str, *, authenticated: bool = False,
          body: dict[str, object] | None = None) -> tuple[int, bytes]:
    headers = {"Authorization": "Bearer " + os.environ["SCONE_API_KEY"]} if authenticated else {}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = Request("http://127.0.0.1:7437" + path, data=data, headers=headers)
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, response.read(4_000_000)
    except HTTPError as error:
        return error.code, error.read(4096)


def main() -> None:
    import_module("boto3")
    import_module("qdrant_client")
    assert sys.version_info[:2] == (3, 14), "container must use Python 3.14"
    assert version("boto3"), "AWS SDK must be installed"
    assert os.getuid() == 10001, "container must run as the unprivileged application user"
    deadline = time.monotonic() + 25
    while True:
        try:
            status, body = fetch("/healthz")
            if status == 200 and json.loads(body) == {"ok": True}:
                break
        except (OSError, URLError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("container did not become healthy")
        time.sleep(0.2)
    status, page = fetch("/memory")
    assert status == 404, "framework must not bundle browser pages"
    assert os.environ["SCONE_API_KEY"].encode() not in page, "missing routes must not disclose keys"
    assert fetch("/v1/capabilities")[0] == 401, "memory API must require authentication"
    assert fetch("/v1/capabilities", authenticated=True)[0] == 200
    content = "Synthetic container smoke: amber otter carries the cobalt lantern."
    status, body = fetch("/v1/episodes", authenticated=True,
                         body={"content": content, "source": "synthetic-container-smoke"})
    assert status == 200, "synthetic episode write failed"
    stored = json.loads(body)
    assert isinstance(stored, dict) and isinstance(stored.get("episode_id"), int)
    status, body = fetch("/v1/recall?q=amber%20otter%20cobalt%20lantern", authenticated=True)
    assert status == 200 and content.encode() in body, "synthetic stored text must be recalled verbatim"


if __name__ == "__main__":
    main()
