"""`scone-memory sync-directory --forget-after`: the run's instant on every
source it writes, named in the JSON result and in the text summary."""
import io
import json

from scone_memory.runtime.cli import main

from .test_cli_directory_sync import setup


def test_the_command_schedules_what_it_writes_and_says_so(tmp_path, capsys):
    root, _, arguments, env = setup(tmp_path)
    (root / "report.txt").write_text("Telescope schedule")
    text = [argument for argument in arguments if argument != "--json"]
    out = io.StringIO()
    assert main(text + ["--forget-after", "2099-01-01"], env=env, out=out) == 0
    assert "every source written is to be forgotten after 2099-01-01T00:00:00.000Z" in out.getvalue()
    assert "unchanged source(s) keep" not in out.getvalue()
    out = io.StringIO()
    assert main(arguments + ["--forget-after", "2099-06-01"], env=env, out=out) == 0
    result = json.loads(out.getvalue())
    assert result["forget_after"] == "2099-06-01T00:00:00.000Z" and result["schedule_kept"] == 1
    assert result["receipts"][0]["forget_after"] == "2099-01-01T00:00:00.000Z"
    out = io.StringIO()
    assert main(text, env=env, out=out) == 0
    assert "forgotten after" not in out.getvalue() and "1 unchanged source(s) keep" in out.getvalue()
    assert main(arguments + ["--forget-after", "2020-01-01"], env=env, out=io.StringIO()) == 2
    assert "forget_after" in capsys.readouterr().err
