"""The Obsidian notes written into a vault a person already keeps: theirs untouched, ours kept current."""
import io
import json
import zipfile

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.core.models import Fact
from scone_memory.entities.export import export_graph, obsidian_files
from scone_memory.entities.project import project_entities
from scone_memory.entities.vault import DEFAULT_FOLDER, MANIFEST, SIGNATURE, signed, write_vault


def fact(number: int, subject: str, predicate: str, object_: str) -> Fact:
    return Fact(fact_id=number, space="alpha", subject=subject.casefold(), predicate=predicate, object=object_,
                valid_from="2025-01-01T00:00:00Z")


def projection_of(*rows):
    return project_entities("alpha", list(rows), revision=1)


ABOUT = {"status": "current", "as_of": "2025-06-01T00:00:00.000Z"}


def test_every_note_carries_the_signature_and_the_zip_still_holds_the_same_files():
    projection = projection_of(fact(1, "alice chen", "works_at", "Acme"), fact(2, "acme", "based_in", "Lisbon"))
    files = obsidian_files(projection, ABOUT)
    assert set(files) >= {"index.md", "graph.canvas"} and all(f"{SIGNATURE}: 1" in text for name, text in files.items() if name.endswith(".md"))
    assert f"scone_projection: {projection.digest}" in files["index.md"], "the digest is the index's alone"
    assert not any("scone_projection" in text for name, text in files.items() if name.startswith("entities/"))
    bundle = zipfile.ZipFile(io.BytesIO(export_graph(projection, "obsidian").body))
    assert set(bundle.namelist()) == set(files)
    cards = [node for node in json.loads(files["graph.canvas"])["nodes"] if "file" in node]
    assert cards and all(str(node["file"]).startswith("entities/") for node in cards)
    placed = obsidian_files(projection, ABOUT, root="scone/")
    assert all(str(node["file"]).startswith("scone/entities/") for node in json.loads(placed["graph.canvas"])["nodes"] if "file" in node), \
        "written into a vault, the canvas names its notes by their path under the vault"


def test_writing_into_a_vault_keeps_their_notes_updates_ours_and_removes_what_we_no_longer_write(tmp_path):
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / ".obsidian" / "app.json").write_text('{"theme": "moonstone"}', encoding="utf-8")
    (vault / "Daily.md").write_text("# my own note\n", encoding="utf-8")
    home = vault / DEFAULT_FOLDER / "entities"
    home.mkdir(parents=True)
    first = projection_of(fact(1, "alice chen", "works_at", "Acme"), fact(2, "acme", "based_in", "Lisbon"),
                          fact(3, "bob", "knows", "Alice Chen"))
    alice = next(name for name in obsidian_files(first, ABOUT) if name.lower() == "entities/alice chen.md")
    (vault / DEFAULT_FOLDER / alice).write_text("# Alice, as I know her\n\nmine\n", encoding="utf-8")
    receipt = write_vault(obsidian_files(first, ABOUT, root="scone/"), vault, projection=first.digest)
    assert receipt.kept_theirs == (alice,), "a note this did not write is never written over"
    assert (vault / DEFAULT_FOLDER / alice).read_text(encoding="utf-8") == "# Alice, as I know her\n\nmine\n"
    assert receipt.written == len(obsidian_files(first, ABOUT)) - 1 and receipt.updated == receipt.unchanged == receipt.removed == 0
    assert signed(home / "Bob.md") and not signed(vault / DEFAULT_FOLDER / alice) and (vault / DEFAULT_FOLDER / MANIFEST).exists()
    assert (vault / ".obsidian" / "app.json").read_text(encoding="utf-8") == '{"theme": "moonstone"}' and (vault / "Daily.md").exists()
    again = write_vault(obsidian_files(first, ABOUT, root="scone/"), vault, projection=first.digest)
    assert (again.written, again.updated, again.removed) == (0, 0, 0) and again.unchanged == receipt.written, "the same notes are left as they are"
    later = projection_of(fact(1, "alice chen", "works_at", "Acme"), fact(2, "acme", "based_in", "Porto"))
    third = write_vault(obsidian_files(later, ABOUT, root="scone/"), vault, projection=later.digest)
    assert not any(name.lower() == "bob.md" for name in (n.name for n in home.iterdir())) and third.removed == 2, \
        "the notes written earlier for Bob and Lisbon, forgotten since, are removed"
    assert third.updated >= 1 and "mine" in (vault / DEFAULT_FOLDER / alice).read_text(encoding="utf-8")
    manifest = json.loads((vault / DEFAULT_FOLDER / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["projection"] == later.digest and alice not in manifest["files"]
    assert not any(name.lower() in ("entities/bob.md", "entities/lisbon.md") for name in manifest["files"])
    assert [name.lower() for name in third.record()["kept_theirs"]] == [alice.lower()], \
        "the later projection spells the name in lower case; on a disk that ignores case it is still their note, still kept"


def test_a_note_unchanged_by_a_change_elsewhere_stays_unchanged_and_a_recased_note_is_one_note(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    first = projection_of(fact(1, "alice chen", "works_at", "Acme"), fact(2, "bob", "knows", "Cho"))
    write_vault(obsidian_files(first, ABOUT, root="scone/"), vault, projection=first.digest)
    later = projection_of(fact(1, "alice chen", "works_at", "Acme"), fact(2, "bob", "knows", "Cho"), fact(3, "cho", "lives_in", "Porto"))
    receipt = write_vault(obsidian_files(later, ABOUT, root="scone/"), vault, projection=later.digest)
    names = {name.lower(): name for name in obsidian_files(later, ABOUT)}
    assert receipt.unchanged >= 3 and receipt.removed == 0, "Alice's note has nothing to do with Porto and is left as it was"
    assert receipt.updated >= 2, "the index, the canvas and Cho's note changed"
    recased = projection_of(fact(1, "alice chen", "works_at", "Acme"), fact(2, "bob", "knows", "Alice Chen"),
                            fact(3, "cho", "lives_in", "Porto"))
    spelled = {name.lower(): name for name in obsidian_files(recased, ABOUT)}
    assert spelled["entities/alice chen.md"] != names["entities/alice chen.md"], "the fixture recases the note's name"
    third = write_vault(obsidian_files(recased, ABOUT, root="scone/"), vault, projection=recased.digest)
    assert third.removed == 0 and third.kept_theirs == (), "a note whose spelling changed case is one note, not a stale one and a new one"
    assert any(name.lower() == "alice chen.md" for name in (n.name for n in (vault / DEFAULT_FOLDER / "entities").iterdir()))


def test_a_note_whose_signature_was_taken_out_is_the_persons_again(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    projection = projection_of(fact(1, "acme", "based_in", "Lisbon"))
    write_vault(obsidian_files(projection, ABOUT, root="scone/"), vault, projection=projection.digest)
    [acme] = [name for name in obsidian_files(projection, ABOUT) if name.startswith("entities/") and "acme" in name.lower()]
    (vault / DEFAULT_FOLDER / acme).write_text("# Acme, rewritten by hand\n", encoding="utf-8")
    again = write_vault(obsidian_files(projection, ABOUT, root="scone/"), vault, projection=projection.digest)
    assert again.kept_theirs == (acme,) and (vault / DEFAULT_FOLDER / acme).read_text(encoding="utf-8") == "# Acme, rewritten by hand\n", \
        "listed in the manifest or not, a note without the signature is the person's"
    manifest = json.loads((vault / DEFAULT_FOLDER / MANIFEST).read_text(encoding="utf-8"))
    assert acme not in manifest["files"]
    gone = write_vault(obsidian_files(projection_of(fact(1, "bob", "knows", "Cho")), ABOUT, root="scone/"), vault, projection="p")
    assert (vault / DEFAULT_FOLDER / acme).exists() and gone.removed == 1, "Lisbon's note goes as stale; the reclaimed note never does"


def test_the_manifest_is_written_before_the_notes_so_a_crash_orphans_nothing(tmp_path, monkeypatch):
    from scone_memory.entities import vault as vault_module

    vault = tmp_path / "vault"
    vault.mkdir()
    projection = projection_of(fact(1, "acme", "based_in", "Lisbon"))
    files = obsidian_files(projection, ABOUT, root="scone/")
    real = vault_module._atomic_write
    calls = []

    def crashing(target, content):
        calls.append(target.name)
        if target.name == "graph.canvas" and len(calls) > 1:
            real(target, content)
            raise OSError("disk full")
        real(target, content)

    monkeypatch.setattr(vault_module, "_atomic_write", crashing)
    with pytest.raises(OSError):
        write_vault(files, vault, projection=projection.digest)
    assert calls[0] == MANIFEST, "the manifest goes first"
    monkeypatch.setattr(vault_module, "_atomic_write", real)
    receipt = write_vault(files, vault, projection=projection.digest)
    assert receipt.kept_theirs == () and receipt.written + receipt.updated + receipt.unchanged == len(files), \
        "after the crash every file, the canvas included, is still this writer's"


def test_a_signed_note_is_adopted_without_a_manifest_and_prose_that_names_the_key_is_not(tmp_path):
    vault = tmp_path / "vault"
    home = vault / DEFAULT_FOLDER / "entities"
    home.mkdir(parents=True)
    projection = projection_of(fact(1, "acme", "based_in", "Lisbon"), fact(2, "bob", "knows", "Acme"))
    names = {name.lower(): name for name in obsidian_files(projection, ABOUT)}
    acme, bob = names["entities/acme.md"], names["entities/bob.md"]
    (vault / DEFAULT_FOLDER / acme).write_text(f"\ufeff---\nid: x\n{SIGNATURE}: 1\n---\n\n# Acme\n", encoding="utf-8")
    (vault / DEFAULT_FOLDER / bob).write_text(f"# Bob\n\nthe key {SIGNATURE}: appears in prose only\n", encoding="utf-8")
    receipt = write_vault(obsidian_files(projection, ABOUT, root="scone/"), vault, projection=projection.digest)
    assert receipt.kept_theirs == (bob,) and receipt.updated == 1, "a signed note is ours to update; prose is not a signature"
    assert "the key" in (vault / DEFAULT_FOLDER / bob).read_text(encoding="utf-8")


def test_a_target_that_is_not_a_directory_or_a_folder_that_escapes_is_refused(tmp_path):
    (tmp_path / "file").write_text("x", encoding="utf-8")
    projection = projection_of(fact(1, "acme", "based_in", "Lisbon"))
    with pytest.raises(InvalidInput, match="not a directory"):
        write_vault(obsidian_files(projection, ABOUT), tmp_path / "file", projection=projection.digest)
    with pytest.raises(InvalidInput, match="one plain name"):
        write_vault(obsidian_files(projection, ABOUT), tmp_path, projection=projection.digest, folder="../out")
    with pytest.raises(InvalidInput, match="inside the folder"):
        write_vault({"../escape.md": "x"}, tmp_path, projection=projection.digest)
    with pytest.raises(InvalidInput, match="inside the folder"):
        write_vault({"/tmp/escape.md": "x"}, tmp_path, projection=projection.digest)
