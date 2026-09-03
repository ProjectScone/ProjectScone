"""Integration test: the client against a real `scone serve` process.

Builds the release binary, writes a config.toml with one [[server.keys]]
entry into a temp data dir, starts the server on a free port, and drives a
full round trip through it. Skipped when cargo cannot produce the binary,
so a checkout without a Rust toolchain still runs the unit tests.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import pytest

from scone import Scone, SconeError

REPO_ROOT = Path(__file__).resolve().parents[3]
API_KEY = "sk-integration-test"
SPACE = "default"
BUILD_TIMEOUT = 900
STARTUP_TIMEOUT = 60


def _target_dir() -> Path:
    """Where cargo puts its artifacts, honoring CARGO_TARGET_DIR."""
    override = os.environ.get("CARGO_TARGET_DIR")
    return Path(override) if override else REPO_ROOT / "target"


def _free_port() -> int:
    """Ask the OS for an unused loopback port and hand it back."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _accepts_connections(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="module")
def scone_binary() -> Path:
    """Build `scone-cli` in release mode and return the binary's path."""
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not on PATH; cannot build the scone binary")
    build = subprocess.run(
        ["cargo", "build", "--release", "-p", "scone-cli"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
    )
    binary = _target_dir() / "release" / "scone"
    if build.returncode != 0 or not binary.exists():
        pytest.skip(f"cargo build --release -p scone-cli failed:\n{build.stderr[-2000:]}")
    return binary


@pytest.fixture(scope="module")
def live_server(scone_binary: Path, tmp_path_factory) -> str:
    """Run `scone serve` on a free port for the module and return its URL."""
    data_dir = tmp_path_factory.mktemp("scone-data")
    port = _free_port()
    (data_dir / "config.toml").write_text(
        "[server]\n"
        f'listen = "127.0.0.1:{port}"\n'
        "\n"
        "[[server.keys]]\n"
        f'key = "{API_KEY}"\n'
        f'space = "{SPACE}"\n',
        encoding="utf-8",
    )

    log = open(data_dir / "serve.log", "w+", encoding="utf-8")
    process: Optional[subprocess.Popen] = subprocess.Popen(
        [
            str(scone_binary),
            "--data-dir", str(data_dir),
            # The deterministic hash embedder keeps the test off the network:
            # the default local embedder downloads an ONNX model on first use.
            "--embedder", "hash",
            "serve",
            "--listen", f"127.0.0.1:{port}",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            if process.poll() is not None:
                log.seek(0)
                pytest.fail(f"scone serve exited early:\n{log.read()}")
            if _accepts_connections(port):
                break
            time.sleep(0.1)
        else:
            log.seek(0)
            pytest.fail(f"scone serve never accepted connections:\n{log.read()}")
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log.close()


@pytest.fixture
def client(live_server: str):
    with Scone(live_server, API_KEY) as c:
        yield c


@pytest.mark.integration
def test_round_trip_against_a_live_server(client: Scone):
    """add -> recall -> tags -> status -> facts, all against a real engine."""

    # -- add ----------------------------------------------------------
    stored = client.add("the deploy checklist lives in the wiki", tags=["ops"])
    assert stored.episode_id >= 1
    assert stored.deduplicated is False
    assert stored.chunks is not None and stored.chunks >= 1

    other = client.add("the checklist for baking bread")
    assert other.episode_id != stored.episode_id

    # Storing the same text again deduplicates onto the same episode.
    again = client.add("the deploy checklist lives in the wiki")
    assert again.deduplicated is True
    assert again.episode_id == stored.episode_id

    # -- recall -------------------------------------------------------
    pack = client.recall("checklist")
    assert len(pack) >= 2, pack
    assert any("wiki" in memory.text for memory in pack)
    for memory in pack:
        assert memory.episode_id >= 1
        assert memory.created_at, "the server must date every recalled chunk"
        assert memory.day == memory.created_at.split("T")[0]
    assert pack.space_bytes > 0
    assert 0.0 <= pack.context_reduction <= 1.0

    # The tag filter narrows to the tagged episode only.
    tagged = client.recall("checklist", tags=["ops"])
    assert len(tagged) == 1, tagged
    assert "wiki" in tagged.items[0].text

    # limit is honored.
    assert len(client.recall("checklist", limit=1)) == 1

    # as_of is accepted and does not error.
    client.recall("checklist", as_of="2030-01-01T00:00:00Z")

    # -- tags ---------------------------------------------------------
    tags = client.tags()
    assert [t.name for t in tags] == ["ops"], tags
    assert tags[0].count == 1

    # -- status -------------------------------------------------------
    status = client.status()
    assert status.space == SPACE
    assert status.episodes == 2, "two distinct episodes, the third deduplicated"
    assert status.chunks >= 2
    assert status.revision >= 1
    # No LLM is configured, so the semantic lane is paused.
    assert status.semantic_lane == "paused"
    assert status.semantic_lane_active is False

    # -- facts --------------------------------------------------------
    # The lane is paused, so nothing has been distilled: the list is empty
    # but the endpoint must still answer with a well-formed body.
    assert client.facts() == []
    assert client.facts(include_closed=True) == []

    # -- profile ------------------------------------------------------
    profile = client.profile()
    assert any("wiki" in line for line in profile.dynamic), profile.dynamic


@pytest.mark.integration
def test_a_wrong_key_is_refused_by_the_live_server(live_server: str):
    with Scone(live_server, "sk-not-a-real-key") as impostor:
        with pytest.raises(SconeError) as excinfo:
            impostor.status()
    assert excinfo.value.status == 401
    assert excinfo.value.message == "unknown key"


@pytest.mark.integration
def test_the_live_server_enforces_its_bounds(client: Scone):
    # Bypass the client's own guard to prove the server's 422 shape.
    with pytest.raises(SconeError) as excinfo:
        client._request("POST", "/v1/episodes", json={"content": ""})
    assert excinfo.value.status == 422
    assert "content must be" in excinfo.value.message


@pytest.mark.integration
def test_closing_an_absent_fact_reports_500_not_404(client: Scone):
    """Documents real server behavior: NotFound is mapped to 500 by serve.rs."""
    with pytest.raises(SconeError) as excinfo:
        client.close_fact(4242, "no such fact")
    assert excinfo.value.status == 500, "serve.rs maps every engine error to 500"
    assert "4242" in excinfo.value.message


@pytest.mark.integration
def test_a_blank_query_500s_on_the_server_so_the_client_refuses_it_first(client: Scone):
    """serve.rs checks q.is_empty(), not q.trim(); a blank q reaches the engine."""
    with pytest.raises(SconeError) as excinfo:
        client._request("GET", "/v1/recall", params={"q": "   "})
    assert excinfo.value.status == 500
    assert "query is empty" in excinfo.value.message

    # The client stops it locally, before a request exists.
    with pytest.raises(SconeError) as local:
        client.recall("   ")
    assert local.value.status is None


@pytest.mark.integration
def test_axum_rejections_arrive_as_plain_text_and_still_parse(client: Scone):
    with pytest.raises(SconeError) as missing_q:
        client._request("GET", "/v1/recall")
    assert missing_q.value.status == 400
    assert "missing field `q`" in missing_q.value.message

    with pytest.raises(SconeError) as unknown_route:
        client._request("GET", "/v1/nope")
    assert unknown_route.value.status == 404
    assert unknown_route.value.message == "HTTP 404", "404 has an empty body"
