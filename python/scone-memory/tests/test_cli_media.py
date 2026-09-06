"""Explicit CLI originals must reach persisted, scoped episode attachments."""

import asyncio
import base64
import hashlib
import io
import json

import pytest

from scone_memory.cli import main, settings_for_cli
from scone_memory.config import build_engine
from scone_memory.errors import NotFound

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=")


def environment(tmp_path):
    return {"SCONE_SQLITE_PATH": str(tmp_path / "memory.db"), "SCONE_EMBEDDER": "hash"}


def invoke(env, arguments, note="Juniper calibration screenshot"):
    output = io.StringIO()
    status = main(["--space", "photos", "--json", "remember", *arguments], env=env, stdin=io.StringIO(note), out=output)
    return status, output.getvalue()


def test_cli_original_survives_reopen_and_is_not_shared_across_spaces(tmp_path):
    original = tmp_path / "source.bin"  # type comes from bytes, not the suffix
    original.write_bytes(PNG)
    env = environment(tmp_path)
    code, raw = invoke(env, ["--image", str(original), "--meta", "session_id=explicit-session"])
    assert code == 0
    receipt = json.loads(raw)
    assert receipt["attachments"][0]["attachment_id"] == hashlib.sha256(PNG).hexdigest()
    assert receipt["attachments"][0]["media_type"] == "image/png"
    assert receipt["attachments"][0]["filename"] == "source.bin"

    async def inspect():
        engine = await build_engine(settings_for_cli(env))
        try:
            source = await engine.episode("photos", receipt["episode_id"])
            assert source.metadata["session_id"] == "explicit-session"
            assert source.content == "Juniper calibration screenshot"
            assert len(source.attachments) == 1
            _, stored = await engine.attachment("photos", source.attachments[0].attachment_id)
            assert stored == PNG
            with pytest.raises(NotFound):
                await engine.attachment("other", source.attachments[0].attachment_id)
            result = await engine.recall("photos", "Juniper")
            assert result.items[0].episode_id == receipt["episode_id"]
        finally:
            await engine.documents.close()
            await engine.vectors.close()

    asyncio.run(inspect())
    code, raw = invoke(env, ["--image", str(original)])
    assert code == 0
    assert json.loads(raw)["deduplicated"] is True
    assert len(json.loads(raw)["attachments"]) == 1


@pytest.mark.parametrize("case", ["empty", "html", "missing", "directory", "oversized", "jsonl"])
def test_invalid_originals_fail_without_storing_an_episode(tmp_path, case, capsys):
    image = tmp_path / "original.png"
    if case == "directory":
        image.mkdir()
    elif case == "oversized":
        with image.open("wb") as file:
            file.truncate(25 * 1024 * 1024 + 1)
    elif case != "missing":
        image.write_bytes(b"" if case == "empty" else b"<html>not a raster</html>" if case == "html" else PNG)
    env = environment(tmp_path)
    code, output = invoke(env, ["--image", str(image), *(["--jsonl"] if case == "jsonl" else [])])
    assert code == 2
    assert output == ""
    assert "error:" in capsys.readouterr().err
    result = io.StringIO()
    assert main(["--space", "photos", "--json", "status"], env=env, out=result) == 0
    assert json.loads(result.getvalue())["episodes"] == 0


@pytest.mark.parametrize("data,media_type", [
    (b"\xff\xd8\xff\xe0" + b"fixture", "image/jpeg"),
    (b"GIF89a" + b"fixture", "image/gif"),
    (b"RIFF\x04\x00\x00\x00WEBP" + b"fixture", "image/webp"),
])
def test_cli_detects_supported_signatures_without_claiming_decode(tmp_path, data, media_type):
    image = tmp_path / "selected.data"
    image.write_bytes(data)
    code, raw = invoke(environment(tmp_path), ["--image", str(image)])
    assert code == 0
    assert json.loads(raw)["attachments"][0]["media_type"] == media_type


def test_cli_rejects_empty_note_before_storing_image(tmp_path, capsys):
    original = tmp_path / "source.png"
    original.write_bytes(PNG)
    code, raw = invoke(environment(tmp_path), ["--image", str(original)], note=" \n")
    assert code == 2 and raw == ""
    assert "nonempty source note" in capsys.readouterr().err
    assert not list(tmp_path.rglob(hashlib.sha256(PNG).hexdigest()))


async def test_missing_attachment_link_is_unconfirmed_not_success(tmp_path):
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.blobs import InMemoryBlobStore
    from scone_memory.cli import build_parser, run
    from scone_memory.errors import InvalidInput

    class DroppedLink(InMemoryBlobStore):
        async def link(self, space, attachment_id, episode_id):
            pass  # emulate a blob backend losing an acknowledged link

    original = tmp_path / "source.png"
    original.write_bytes(PNG)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), blobs=DroppedLink()).open()
    args = build_parser().parse_args(["remember", "--image", str(original), "--json"])
    out = io.StringIO()
    with pytest.raises(InvalidInput, match="source save unconfirmed"):
        await run(args, engine, io.StringIO("A selected original"), out)
    assert out.getvalue() == ""
    assert (await engine.status("default")).episodes == 1  # partial side effect acknowledged


def test_plain_text_remember_receipt_is_unchanged(tmp_path):
    code, raw = invoke(environment(tmp_path), [])
    assert code == 0
    assert "attachments" not in json.loads(raw)


@pytest.mark.parametrize("extra", [[], ["--jsonl"]])
def test_explicit_empty_image_argument_never_silently_saves_text(tmp_path, extra, capsys):
    env = environment(tmp_path)
    code, raw = invoke(env, ["--image", "", *extra], note='{"content":"test source"}')
    assert code == 2
    assert raw == ""
    assert "error:" in capsys.readouterr().err
    output = io.StringIO()
    assert main(["--space", "photos", "--json", "status"], env=env, out=output) == 0
    assert json.loads(output.getvalue())["episodes"] == 0
