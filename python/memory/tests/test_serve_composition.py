"""``serve`` composes the conversation service on the memory origin when the
operator names a journal; without one it is the memory-only app, unchanged."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import __main__ as serve
from scone_memory.runtime.config import Settings

AUTH = {"Authorization": "Bearer solo"}


def engine_for():
    return asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open())


def settings_for(tmp_path, **env):
    return Settings.from_env({"SCONE_API_KEY": "solo", "SCONE_SQLITE_PATH": str(tmp_path / "memory.db"), **env})


def test_serve_without_a_journal_is_the_memory_only_app(tmp_path):
    with TestClient(serve.build_app(settings_for(tmp_path), engine_for())) as c:
        miss = c.get("/v1/conversations/capabilities", headers=AUTH)
        assert miss.status_code == 404 and miss.headers["content-type"].startswith("application/json")
        assert "conversations" not in c.get("/v1/capabilities", headers=AUTH).json()["features"]
        assert "solo" in c.get("/memory").text, "a single key is still baked into the memory-only console"
    assert not (tmp_path / "sessions.db").exists()


def test_a_journal_composes_the_conversation_service_on_the_memory_origin(tmp_path):
    settings = settings_for(tmp_path, SCONE_CONVERSATIONS_JOURNAL=str(tmp_path / "sessions.db"))
    with TestClient(serve.build_app(settings, engine_for())) as c:
        ready = c.get("/v1/conversations/capabilities", headers=AUTH)
        assert ready.status_code == 200 and ready.json()["text_configured"] is False
        assert c.get("/v1/capabilities", headers=AUTH).json()["features"]["conversations"] is True
        assert c.get("/v1/status", headers=AUTH).status_code == 200
        assert c.get("/healthz").status_code == 200
        for path in ("/", "/memory", "/playground", "/conversations", "/conversations/session-one"):
            page = c.get(path)
            assert page.status_code == 200 and page.headers["content-type"].startswith("text/html"), path
            assert "solo" not in page.text, "a composed host never bakes a key into its shell"
        assert c.get("/v1/conversations", headers=AUTH).status_code == 200
    assert (tmp_path / "sessions.db").exists(), "the service owned its journal for the app's life"


@pytest.mark.parametrize("composed", [False, True])
def test_the_consolidation_worker_runs_for_the_life_of_either_app(tmp_path, monkeypatch, composed):
    class Worker:
        started = stopped = False

        def start(self):
            self.started = True

        async def stop(self):
            self.stopped = True

    worker = Worker()
    monkeypatch.setattr(serve, "build_worker", lambda engine, settings, spaces: worker)
    env = {"SCONE_CONVERSATIONS_JOURNAL": str(tmp_path / "sessions.db")} if composed else {}
    with TestClient(serve.build_app(settings_for(tmp_path, **env), engine_for())) as c:
        assert c.get("/healthz").status_code == 200
        assert worker.started and not worker.stopped
    assert worker.stopped


def test_a_trusted_model_factory_marks_text_configured_without_being_called(tmp_path, monkeypatch):
    module = ModuleType("compose_test_provider")

    def create():
        raise AssertionError("building the app must not invoke the model factory")

    module.create = create
    monkeypatch.setitem(sys.modules, module.__name__, module)
    settings = settings_for(tmp_path, SCONE_CONVERSATIONS_JOURNAL=str(tmp_path / "sessions.db"),
                            SCONE_CONVERSATIONS_MODEL_FACTORY="compose_test_provider:create")
    with TestClient(serve.build_app(settings, engine_for())) as c:
        ready = c.get("/v1/conversations/capabilities", headers=AUTH).json()
        assert ready["text_configured"] is True and ready["streaming"] is True


def test_reloading_pages_re_reads_the_composed_shell(tmp_path, monkeypatch):
    shell = tmp_path / "shell.html"
    shell.write_text('<div id="root">first</div>')
    monkeypatch.setattr("scone_memory.api.conversations.PLAYGROUND", shell)
    settings = settings_for(tmp_path, SCONE_CONVERSATIONS_JOURNAL=str(tmp_path / "sessions.db"), SCONE_UI_DEV="1")
    with TestClient(serve.build_app(settings, engine_for())) as c:
        assert "first" in c.get("/conversations").text
        shell.write_text('<div id="root">second</div>')
        assert "second" in c.get("/conversations").text


def test_a_journal_that_is_the_memory_database_is_refused_before_serving(tmp_path, monkeypatch, capsys):
    memory = tmp_path / "memory.db"
    settings = settings_for(tmp_path, SCONE_DOCUMENTS="sqlite", SCONE_CONVERSATIONS_JOURNAL=str(memory))
    with pytest.raises(ValueError):
        serve.build_app(settings, engine_for())

    async def fake_build(settings):
        return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

    monkeypatch.setattr(serve, "build_engine", fake_build)
    with pytest.raises(SystemExit) as stop:
        serve.main(settings)
    assert stop.value.code == 2
    assert "journal" in capsys.readouterr().err.lower()
    assert not memory.exists()
