"""The operator command must launch the real service, not a demo runtime."""
import asyncio
import builtins
from contextlib import closing
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from scone_memory.cli import build_parser, main


def env_for(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SCONE_")}
    env.update(SCONE_SQLITE_PATH=str(tmp_path / "memory.db"), SCONE_EMBEDDER="hash",
               SCONE_API_KEYS="launcher-alpha:alpha,launcher-beta:beta", SCONE_HOST="127.0.0.1")
    return env


def test_parser_requires_journal_and_an_explicit_model_or_history_choice():
    parser = build_parser()
    args = parser.parse_args(["serve-conversations", "--journal", "sessions.db", "--history-only", "--console"])
    assert args.history_only is True and args.console is True
    args = parser.parse_args(["serve-conversations", "--journal", "sessions.db", "--model-factory", "local_model:create"])
    assert args.model_factory == "local_model:create" and args.console is False
    for arguments in [["--history-only"], ["--journal", "sessions.db"],
                      ["--journal", "sessions.db", "--history-only", "--model-factory", "local_model:create"]]:
        with pytest.raises(SystemExit) as error:
            parser.parse_args(["serve-conversations", *arguments])
        assert error.value.code == 2


def test_missing_keys_and_alias_paths_fail_before_creating_state(tmp_path, capsys):
    env = env_for(tmp_path)
    del env["SCONE_API_KEYS"]
    assert main(["serve-conversations", "--journal", str(tmp_path / "sessions.db"), "--history-only"], env=env) == 2
    assert not (tmp_path / "memory.db").exists() and not (tmp_path / "sessions.db").exists()
    assert "key" in capsys.readouterr().err.lower()
    assert main(["serve-conversations", "--journal", str(tmp_path / "memory.db"), "--history-only"], env=env_for(tmp_path)) == 2
    assert not (tmp_path / "memory.db").exists()


def test_bad_factory_is_rejected_without_invocation_or_secret_diagnostics(tmp_path, capsys):
    env = env_for(tmp_path)
    for factory in ["not a factory", "missing_module:create", "builtins:42", "builtins:None", "builtins:len"]:
        assert main(["serve-conversations", "--journal", str(tmp_path / "sessions.db"), "--model-factory", factory], env=env) == 2
    assert not (tmp_path / "memory.db").exists()
    assert "launcher-alpha" not in capsys.readouterr().err


def test_factory_loading_does_not_call_provider(monkeypatch):
    from scone_memory.api.conversation_server import load_model_factory
    module = ModuleType("launcher_test_provider")
    def create():
        raise AssertionError("loading must not invoke the model")
    module.create = create
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert load_model_factory("launcher_test_provider:create") is create
    async def asynchronous():
        pass
    module.create = asynchronous
    with pytest.raises(ValueError):
        load_model_factory("launcher_test_provider:create")


def test_journal_aliases_are_rejected_before_open(tmp_path, capsys):
    memory = tmp_path / "memory.db"
    memory.write_bytes(b"untouched")
    for link in ("symlink.db", "hardlink.db"):
        alias = tmp_path / link
        if link == "symlink.db":
            alias.symlink_to(memory)
        else:
            os.link(memory, alias)
        assert main(["serve-conversations", "--journal", str(alias), "--history-only"], env=env_for(tmp_path)) == 2
        assert memory.read_bytes() == b"untouched"


@pytest.mark.parametrize("started", [True, False])
def test_server_builds_and_closes_resources_on_one_loop(tmp_path, monkeypatch, capsys, started):
    import uvicorn
    from scone_memory.api import conversation_server as launcher
    from scone_memory.cli import settings_for_cli
    loops, closed = [], []
    class Backend:
        async def close(self):
            loops.append(asyncio.get_running_loop())
            closed.append(self)
    document, event = Backend(), Backend()
    engine = SimpleNamespace(documents=document, vectors=document, events=event)
    async def build(settings):
        loops.append(asyncio.get_running_loop())
        return engine
    class Server:
        def __init__(self, config):
            self.started = started
        async def serve(self):
            loops.append(asyncio.get_running_loop())
    monkeypatch.setattr(launcher, "build_engine", build)
    monkeypatch.setattr("scone_memory.api.conversations.create_conversation_app", lambda *a, **kw: object())
    monkeypatch.setattr(uvicorn, "Server", Server)
    assert launcher.main(settings_for_cli(env_for(tmp_path)), journal=str(tmp_path / "sessions.db")) == (0 if started else 2)
    assert len(set(loops)) == 1 and closed == [document, event]


def test_cleanup_continues_after_one_backend_fails():
    from scone_memory.api.conversation_server import close_engine
    closed = []
    class Broken:
        async def close(self):
            raise RuntimeError("provider-secret")
    class Healthy:
        async def close(self):
            closed.append(True)
    with pytest.raises(RuntimeError, match="backend cleanup failed"):
        asyncio.run(close_engine(SimpleNamespace(documents=Broken(), vectors=Healthy())))
    assert closed == [True]


def test_optional_api_import_failure_is_actionable(tmp_path, monkeypatch, capsys):
    real_import = builtins.__import__
    def without_api(name, *args, **kwargs):
        if name.endswith("api.conversation_server"):
            raise ImportError("fastapi unavailable")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_api)
    assert main(["serve-conversations", "--journal", str(tmp_path / "sessions.db"), "--history-only"], env=env_for(tmp_path)) == 2
    assert "api" in capsys.readouterr().err
    assert not (tmp_path / "memory.db").exists()


@pytest.mark.parametrize("configured", [{"SCONE_API_KEY": "   "}, {"SCONE_API_KEYS": "valid:"},
                                      {"SCONE_API_KEYS": "valid:bad/space"}])
def test_invalid_key_mapping_is_rejected_before_state(tmp_path, monkeypatch, configured):
    from scone_memory.api import conversation_server as launcher
    env = env_for(tmp_path)
    del env["SCONE_API_KEYS"]
    env.update(configured)
    async def must_not_build(settings):
        pytest.fail("invalid authentication must not open stores")
    monkeypatch.setattr(launcher, "build_engine", must_not_build)
    assert main(["serve-conversations", "--journal", str(tmp_path / "sessions.db"), "--history-only"], env=env) == 2


@pytest.fixture
def launched(tmp_path):
    processes = []
    clients = []
    def start(*options):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = env_for(tmp_path)
        env["SCONE_PORT"] = str(port)
        root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join([root, str(Path(__file__).parent), env.get("PYTHONPATH", "")])
        process = subprocess.Popen([sys.executable, "-m", "scone_memory.cli", "serve-conversations",
                                    "--journal", str(tmp_path / "sessions.db"), *options],
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(process)
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": "Bearer launcher-alpha"}, timeout=1)
        clients.append(client)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError("launcher exited: " + process.communicate()[1])
            try:
                if client.get("/healthz").status_code == 200:
                    return client, process
            except httpx.HTTPError:
                pass
            time.sleep(.025)
        raise AssertionError("launcher did not become ready")
    yield start
    for client in clients:
        client.close()
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            _, stderr = process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill(); process.communicate()
            raise AssertionError("launcher required forced termination")
        assert process.returncode in (0, 130), stderr
        assert "launcher-alpha" not in stderr and "Traceback" not in stderr


def test_history_only_launch_has_real_memory_and_opt_in_pages(launched):
    client, _ = launched("--history-only")
    with closing(client):
        cap = client.get("/v1/conversations/capabilities").json()
        assert cap["text_configured"] is False and cap["voice"] is False
        assert cap["streaming"] is False
        assert client.get("/conversations").status_code == 404
        assert client.post("/v1/conversations", json={"request_id": "no-model", "capture": True}).status_code == 503
        added = client.post("/v1/episodes", json={"content": "Launcher retained source"})
        assert added.status_code == 200
        assert client.get("/v1/episodes/" + str(added.json()["episode_id"]), headers={"Authorization": "Bearer launcher-beta"}).status_code == 404


def test_configured_launcher_runs_native_scone_and_serves_keyless_pages(launched):
    client, _ = launched("--model-factory", "test_text_conversation:ScriptedModel", "--console")
    with closing(client):
        page = client.get("/conversations")
        assert page.status_code == 200 and "launcher-alpha" not in page.text
        assert client.get("/v1/conversations/capabilities").json()["recall_scope"] is True
        assert client.get("/v1/conversations/capabilities").json()["streaming"] is True
        session = client.post("/v1/conversations", json={"request_id": "new", "capture": True, "recall_scope": {"kind": "file"}}).json()
        route = "/v1/conversations/" + session["session_id"]
        assert client.post(route + "/turns", json={"request_id": "turn", "text": "Hello", "expected_revision": session["revision"]}).status_code == 202
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            receipt = client.get(route + "/turns/turn").json()
            if receipt["status"] != "pending":
                break
            time.sleep(.025)
        assert receipt["status"] == "completed", receipt
        assert len(client.get(route + "/transcript").json()["episodes"]) == 2
