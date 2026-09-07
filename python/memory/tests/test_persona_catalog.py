"""A saved persona catalog: parsed strictly, bound once at startup, and never
leaking instructions or provider configuration past the operator."""

from __future__ import annotations

import json
import sqlite3

import pytest

from scone_memory.core.errors import Conflict
from scone_memory.realtime.catalog import bind_catalog, load_personas
from scone_memory.realtime.persona import Persona
from scone_memory.realtime.providers import ProviderRegistry
from scone_memory.realtime.session_journal import SessionJournal


def persona(id="helper", **over):
    document = {"schema_version": 1, "id": id, "name": id.title(), "instructions": "Answer briefly.",
                "reply": {"provider": "stub", "model": "echo"},
                "transcription": {"provider": "stub", "model": "ears"},
                "speech": {"provider": "stub", "model": "mouth", "voice": "alto"}}
    document.update(over)
    return document


def registry():
    return ProviderRegistry(reply={("stub", "echo"): lambda: None},
                            transcription={("stub", "ears"): lambda: None},
                            speech={("stub", "mouth", "alto"): lambda: None})


def test_the_catalog_file_is_a_strict_list_read_in_order(tmp_path):
    path = tmp_path / "personas.json"
    path.write_text(json.dumps([persona("second"), persona("first")]))
    assert [p.id for p in load_personas(path)] == ["second", "first"]

    path.write_text(json.dumps([persona("twin"), persona("twin")]))
    with pytest.raises(ValueError, match="twin"):
        load_personas(path)

    path.write_text(json.dumps([persona("leaky", instructions="TOP-SECRET-PROMPT", endpoint="https://x")]))
    with pytest.raises(ValueError) as error:
        load_personas(path)
    assert "leaky" in str(error.value) and "TOP-SECRET" not in str(error.value)

    for broken in ('{"not": "a list"}', "[1]", "not json"):
        path.write_text(broken)
        with pytest.raises(ValueError):
            load_personas(path)


def test_binding_refuses_an_unregistered_choice_naming_persona_and_stage():
    helper = Persona.model_validate(persona("helper"))
    singer = Persona.model_validate(persona("singer", speech={"provider": "stub", "model": "mouth", "voice": "tenor"}))
    with pytest.raises(ValueError) as error:
        bind_catalog([helper, singer], registry())
    assert "singer" in str(error.value) and "speech" in str(error.value)

    catalog = bind_catalog([helper], registry())
    assert catalog.get("helper").persona == helper and catalog.get("nobody") is None
    assert catalog.name("helper") == "Helper" and catalog.name("nobody") is None
    assert catalog.public() == [{
        "id": "helper", "name": "Helper",
        "reply": {"provider": "stub", "model": "echo"},
        "transcription": {"provider": "stub", "model": "ears"},
        "speech": {"provider": "stub", "model": "mouth", "voice": "alto"},
        "activity": None, "text_ready": True, "voice_ready": False,
    }]
    assert "instructions" not in json.dumps(catalog.public())


def test_the_journal_records_a_session_persona_and_migrates_older_files(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionJournal(path) as journal:
        plain = journal.create("alpha", "one")["session_id"]
        bound = journal.create("alpha", "two", persona="helper")["session_id"]
        assert journal.get("alpha", plain)["persona"] is None
        assert journal.get("alpha", bound)["persona"] == "helper"
        listed = {s["session_id"]: s["persona"] for s in journal.sessions("alpha")["items"]}
        assert listed == {plain: None, bound: "helper"}
        # A retry naming another persona is a different command, not a replay.
        with pytest.raises(Conflict):
            journal.create("alpha", "two", persona="other")
        assert journal.create("alpha", "two", persona="helper")["session_id"] == bound

    older = tmp_path / "older.db"
    with SessionJournal(older) as journal:
        sid = journal.create("alpha", "one")["session_id"]
    db = sqlite3.connect(older)
    db.execute("ALTER TABLE sessions DROP COLUMN persona")
    db.execute("PRAGMA user_version=3")
    db.commit()
    db.close()
    with SessionJournal(older) as journal:
        assert journal.get("alpha", sid)["persona"] is None, "rows from before the column read as unbound"
        assert journal.create("alpha", "two", persona="helper")["session_id"]
    assert sqlite3.connect(older).execute("PRAGMA user_version").fetchone()[0] == 4
