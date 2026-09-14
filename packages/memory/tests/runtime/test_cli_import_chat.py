"""`import-chat` turns an export on disk into conversation memories and says what it counted instead."""
import io
import json

from scone_memory.runtime.cli import main

WHATSAPP = ("12/03/2024, 14:05 - Messages and calls are end-to-end encrypted.\n"
            "12/03/2024, 14:05 - Alice: The harbour closes in November\n"
            "12/03/2024, 14:06 - Bob: <Media omitted>\n"
            "13/03/2024, 09:03 - Bob: Noted, moving the boat\n")


def environment(tmp_path):
    return {"SCONE_SQLITE_PATH": str(tmp_path / "memory.db"), "SCONE_EMBEDDER": "hash"}


def test_import_chat_stores_messages_and_reports_the_rest(tmp_path):
    export = tmp_path / "Harbour crew.txt"
    export.write_text(WHATSAPP, encoding="utf-8")
    env = environment(tmp_path)
    output = io.StringIO()
    assert main(["--space", "chats", "--json", "import-chat", str(export), "--time-zone", "Europe/London", "--meta", "user_id=mark"],
                env=env, out=output) == 0
    receipt = json.loads(output.getvalue())
    assert receipt["platform"] == "whatsapp" and receipt["chat"] == "Harbour crew"
    assert (receipt["stored"], receipt["messages"], receipt["sessions"], receipt["episodes"]) == (2, 2, 2, 2)
    assert receipt["system_messages"] == 1 and receipt["media_only_messages"] == 1 and receipt["time_zone"] == "Europe/London"
    assert receipt["first_at"] == "2024-03-12T14:05:00.000Z" and "episode_ids" not in receipt
    output = io.StringIO()
    assert main(["--space", "chats", "recall", "harbour closing", "--json"], env=env, out=output) == 0
    assert "harbour closes in november" in output.getvalue().lower(), output.getvalue()

    output = io.StringIO()
    assert main(["--space", "chats", "import-chat", str(export), "--chat", "crew"], env=env, out=output) == 0
    said = output.getvalue()
    assert "imported 2 of 2 messages from the whatsapp chat 'crew'" in said and "as 2 session(s), 2 speaker(s)" in said
    assert "counted, not stored: 1 system line(s), 1 media-only message(s), 1 attachment(s)" in said
    assert "clock times read in UTC (assumed)" in said, "no zone was given this time, and the receipt says what was assumed"

    output = io.StringIO()
    assert main(["--space", "chats", "import-chat", str(export), "--chat", "crew"], env=env, out=output) == 0
    assert "imported 0 of 2 messages" in output.getvalue() and "2 already known" in output.getvalue()


def test_import_chat_exits_nonzero_when_nothing_was_stored_and_names_the_reason(tmp_path):
    export = tmp_path / "long.txt"
    export.write_text("13/03/2024, 14:05 - Alice: " + "x" * 2_000_100 + "\n", encoding="utf-8")
    env = environment(tmp_path)
    output = io.StringIO()
    assert main(["--space", "chats", "import-chat", str(export)], env=env, out=output) == 1
    assert "imported 0 of 1 messages" in output.getvalue() and "1 refused (content exceeds" in output.getvalue()
    quiet = tmp_path / "quiet.txt"
    quiet.write_text("01/02/2024, 10:00 - Alice: hi\n", encoding="utf-8")
    output = io.StringIO()
    assert main(["--space", "chats", "import-chat", str(quiet), "--date-order", "month-first"], env=env, out=output) == 0
    assert "dates read month-first as told; the file itself did not decide" in output.getvalue()


def test_import_chat_refuses_what_it_cannot_read(tmp_path, capsys):
    env = environment(tmp_path)
    table = tmp_path / "table.csv"
    table.write_text("a,b\n1,2\n")
    assert main(["--space", "chats", "import-chat", str(table)], env=env, out=io.StringIO()) == 2
    assert "must be a WhatsApp .txt" in capsys.readouterr().err
    assert main(["--space", "chats", "import-chat", str(tmp_path / "missing.txt")], env=env, out=io.StringIO()) == 2
    assert "cannot read" in capsys.readouterr().err
    quiet = tmp_path / "quiet.txt"
    quiet.write_text("01/02/2024, 10:00 - Alice: hi\n", encoding="utf-8")
    assert main(["--space", "chats", "import-chat", str(quiet)], env=env, out=io.StringIO()) == 2
    assert "say which with date_order" in capsys.readouterr().err
    assert main(["--space", "chats", "import-chat", str(quiet), "--date-order", "day-first", "--gap-hours", "nan"], env=env, out=io.StringIO()) == 2
    assert "--gap-hours must be zero or more" in capsys.readouterr().err
