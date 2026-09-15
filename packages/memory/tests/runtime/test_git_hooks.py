"""Git's own hooks keep a repository's graph current: installed beside a person's hooks, never over them."""
import json
import os
import shutil
import stat
import subprocess
import sys
import time

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.runtime import githooks
from scone_memory.runtime.cli import main

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed here")


def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "t@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "T"], check=True)
    (root / "app.py").write_text("def go():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "app.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "first"], check=True)
    return root


def test_install_writes_the_runner_and_the_hooks_and_carries_only_store_settings(tmp_path):
    root = repo(tmp_path)
    env = {"SCONE_DOCUMENTS": "sqlite", "SCONE_SQLITE_PATH": str(tmp_path / "m.db"), "SCONE_POSTGRES_URL": "postgres://u:secret@h/db",
           "SCONE_API_KEY": "sk-not-a-real-key", "PYTHONPATH": "/src:/tests"}
    receipt = githooks.install(root / "sub" if (root / "sub").mkdir() is None else root, space="alpha", env=env,
                               interpreter="/opt/py/bin/python")
    hooks = root / ".git" / "hooks"
    assert receipt.hooks_written == ("post-commit", "post-checkout", "post-merge") and receipt.repository == str(root.resolve())
    runner = (hooks / "scone-map").read_text(encoding="utf-8")
    assert "secret" not in runner and "sk-not" not in runner and "SCONE_POSTGRES_URL" not in runner and "SCONE_API_KEY" not in runner, \
        "a URL or a key is never copied into a file under .git"
    assert "export SCONE_DOCUMENTS=sqlite" in runner and "SCONE_SQLITE_PATH" in runner and "export PYTHONPATH=/src:/tests" in runner
    assert receipt.carried == ("SCONE_DOCUMENTS", "SCONE_SQLITE_PATH", "PYTHONPATH")
    assert "PY=/opt/py/bin/python" in runner and "--space alpha map" in runner and "--graph" in runner
    for hook in githooks.HOOKS:
        text = (hooks / hook).read_text(encoding="utf-8")
        assert text.startswith("#!/bin/sh\n") and githooks.BLOCK_START in text and f"scone-map {hook}" in text
        assert (hooks / hook).stat().st_mode & stat.S_IXUSR
    seen = githooks.status(root)
    assert seen.installed and seen.hooks == {h: "installed" for h in githooks.HOOKS} and seen.interpreter == "/opt/py/bin/python"
    assert seen.interpreter_exists is False and seen.carried == ("SCONE_DOCUMENTS", "SCONE_SQLITE_PATH", "PYTHONPATH") and seen.log is None


def test_a_persons_hook_is_appended_to_never_replaced_and_uninstall_gives_it_back(tmp_path):
    root = repo(tmp_path)
    hooks = root / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    own = "#!/bin/sh\necho theirs >> /dev/null\n"
    (hooks / "post-commit").write_text(own, encoding="utf-8")
    first = githooks.install(root, env={})
    assert first.hooks_appended == ("post-commit",) and first.hooks_written == ("post-checkout", "post-merge")
    text = (hooks / "post-commit").read_text(encoding="utf-8")
    assert text.startswith(own) and text.count(githooks.BLOCK_START) == 1
    again = githooks.install(root, env={}, space="beta")
    assert again.hooks_replaced == ("post-commit", "post-checkout", "post-merge") and not again.hooks_written
    assert (hooks / "post-commit").read_text(encoding="utf-8").count(githooks.BLOCK_START) == 1, "a second install replaces its own block only"
    assert "--space beta" in (hooks / "scone-map").read_text(encoding="utf-8")
    assert githooks.status(root).hooks["post-commit"] == "installed"
    gone = githooks.uninstall(root)
    assert gone["hooks_cleaned"] == ["post-commit"] and sorted(gone["hooks_removed"]) == ["post-checkout", "post-merge"] and gone["runner_removed"]
    assert (hooks / "post-commit").read_text(encoding="utf-8") == own and not (hooks / "post-merge").exists() and not (hooks / "scone-map").exists()
    assert githooks.status(root).installed is False and githooks.status(root).hooks["post-commit"] == "other"


def test_a_block_is_replaced_where_it_stands_a_broken_one_is_refused_and_a_mention_is_no_block(tmp_path):
    root = repo(tmp_path)
    hooks = root / ".git" / "hooks"
    githooks.install(root, env={})
    after = (hooks / "post-commit").read_text(encoding="utf-8") + "echo after-ours\n"
    (hooks / "post-commit").write_text(after, encoding="utf-8")
    githooks.install(root, env={}, space="beta")
    text = (hooks / "post-commit").read_text(encoding="utf-8")
    assert text.index(githooks.BLOCK_END) < text.index("echo after-ours"), "what the person put after the block stays after it"
    assert text.count(githooks.BLOCK_START) == 1
    (hooks / "post-merge").write_text(f"#!/bin/sh\n{githooks.BLOCK_START}\necho mine\n", encoding="utf-8")
    with pytest.raises(InvalidInput, match="not whole"):
        githooks.install(root, env={})
    with pytest.raises(InvalidInput, match="not whole"):
        githooks.uninstall(root)
    assert githooks.status(root).hooks["post-merge"] == "installed", "a broken block still counts as ours to report"
    assert "echo mine" in (hooks / "post-merge").read_text(encoding="utf-8"), "nothing of the person's is touched"
    (hooks / "post-merge").write_text(f"#!/bin/sh\n# {githooks.BLOCK_START} is not a block when it is a comment\necho mine\n", encoding="utf-8")
    assert githooks.status(root).hooks["post-merge"] == "other"
    receipt = githooks.install(root, env={})
    assert "post-merge" in receipt.hooks_appended
    (hooks / "post-checkout").write_text("#!/bin/sh\necho no newline at the end", encoding="utf-8")
    githooks.install(root, env={})
    assert "echo no newline at the end\n" + githooks.BLOCK_START in (hooks / "post-checkout").read_text(encoding="utf-8")


def test_status_shows_how_the_log_ends():
    receipt = '{\n "read": 3,\n "claims": 40\n}\n'
    assert json.loads(githooks._last_receipt(receipt)) == {"read": 3, "claims": 40}
    failed = receipt + "Traceback (most recent call last):\n  boom\nModuleNotFoundError: No module named scone_memory\n"
    assert githooks._last_receipt(failed) == "ModuleNotFoundError: No module named scone_memory", "a failed run is not hidden behind the success before it"
    long = json.dumps({"read": 3, "unbound_calls": ["x" * 3_000]}) + "\n"
    shown = json.loads(githooks._last_receipt(long))
    assert shown["read"] == 3 and "unbound_calls" not in shown and "counts only" in shown["shown"]
    assert githooks._last_receipt("") is None and githooks._last_receipt("not json\n") == "not json"


def test_not_a_repository_is_refused_and_the_cli_prints_receipts(tmp_path, capsys):
    with pytest.raises(InvalidInput, match="not a git repository"):
        githooks.install(tmp_path, env={})
    root = repo(tmp_path)
    code = main(["--json", "hooks", "install", "--root", str(root), "--no-graph"], env={"SCONE_DOCUMENTS": "sqlite"})
    printed = json.loads(capsys.readouterr().out)
    assert code == 0 and printed["graph"] is False and printed["carried_settings"] == ["SCONE_DOCUMENTS"]
    assert "--graph" not in (root / ".git" / "hooks" / "scone-map").read_text(encoding="utf-8")
    code = main(["--json", "hooks", "status", "--root", str(root)], env={})
    assert code == 0 and json.loads(capsys.readouterr().out)["installed"] is True
    code = main(["hooks", "uninstall", "--root", str(root)], env={})
    assert code == 0 and json.loads(capsys.readouterr().out)["runner_removed"] is True
    assert main(["hooks", "status", "--root", str(tmp_path / "nowhere")], env={}) == 2
    with pytest.raises(SystemExit):
        main(["hooks", "status", "--root", str(root), "--spce", "x"], env={})


def test_a_commit_maps_the_repository_when_the_hook_is_told_to_wait(tmp_path):
    root = repo(tmp_path)
    env = {"SCONE_DOCUMENTS": "sqlite", "SCONE_VECTORS": "sqlite", "SCONE_SQLITE_PATH": str(tmp_path / "hook.db")}
    githooks.install(root, env=env, interpreter=sys.executable)
    (root / "more.py").write_text("import app\n\ndef more():\n    return app.go()\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "more.py"], check=True)
    # git runs a hook from the repository's top, so the source tree the
    # tests import from must be named absolutely for the hook's interpreter.
    run_env = {**os.environ, "SCONE_HOOK_WAIT": "1", "SCONE_EMBEDDER": os.environ.get("SCONE_EMBEDDER", "hash"),
               "PYTHONPATH": os.pathsep.join(os.path.abspath(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)}
    done = subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "second"], capture_output=True, text=True,
                          env=run_env, timeout=180)
    assert done.returncode == 0, done.stderr
    log = (root / ".git" / "scone-map.log").read_text(encoding="utf-8")
    seen = githooks.status(root)
    assert seen.log and seen.last_receipt and seen.last_receipt.startswith("{"), log
    receipt = json.loads(seen.last_receipt)
    assert receipt.get("read", 0) >= 2, log
    # A checkout of files (branch flag 0) maps nothing: the tree did not move.
    before = len(log.splitlines())
    subprocess.run(["git", "-C", str(root), "checkout", "-q", "--", "app.py"], check=True, env=run_env)
    assert len((root / ".git" / "scone-map.log").read_text(encoding="utf-8").splitlines()) == before
    # Not told to wait, the commit returns at once and the map lands in the log soon after.
    (root / "third.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "third.py"], check=True)
    quick = {key: value for key, value in run_env.items() if key != "SCONE_HOOK_WAIT"}
    started = time.monotonic()
    done = subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "third"], capture_output=True, text=True, env=quick, timeout=180)
    assert done.returncode == 0, done.stderr
    assert time.monotonic() - started < 60, "the commit did not wait for the map"
    for _ in range(600):
        if len((root / ".git" / "scone-map.log").read_text(encoding="utf-8").splitlines()) > before:
            break
        time.sleep(0.2)
    else:
        pytest.fail("the background map wrote nothing within two minutes")
    assert json.loads(githooks.status(root).last_receipt).get("found", 0) >= 3, "the map after the third commit saw all three files"
