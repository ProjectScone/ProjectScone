"""Interpreter selection is data, and launching tests never read private config."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).parents[3]


def project(tmp_path: Path) -> Path:
    root = tmp_path / "project with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/serve-self-hosted.sh", scripts / "serve-self-hosted.sh")
    return root


def interpreter(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$SCONE_TEST_CAPTURE"\n')
    path.chmod(0o755)


@pytest.mark.parametrize("arguments", [["--check"], ["--host", "127.0.0.1", "--port", "9999"]])
def test_explicit_python_override_is_one_quoted_executable(tmp_path: Path, arguments: list[str]) -> None:
    root = project(tmp_path)
    selected = root / "runtime with spaces/bin/python"
    interpreter(selected)
    capture = root / "arguments.txt"
    env = {**os.environ, "SCONE_PYTHON": str(selected), "SCONE_TEST_CAPTURE": str(capture)}
    result = subprocess.run([str(root / "scripts/serve-self-hosted.sh"), *arguments], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = capture.read_text().splitlines()
    assert actual[:3] == [str(root / "scripts/local_env.py"), "--env-file", str(root / ".env.local")]
    if arguments == ["--check"]:
        assert actual[3:] == ["--check"]
    else:
        assert actual[3:] == ["--", str(selected), "-m", "scone_memory.runtime.cli", "serve", *arguments]


@pytest.mark.parametrize("override", [None, ""])
def test_default_python_path_is_preserved(tmp_path: Path, override: str | None) -> None:
    root = project(tmp_path)
    selected = root / "python/memory/.venv/bin/python"
    interpreter(selected)
    capture = root / "arguments.txt"
    env = {key: value for key, value in os.environ.items() if key != "SCONE_PYTHON"}
    env["SCONE_TEST_CAPTURE"] = str(capture)
    if override is not None:
        env["SCONE_PYTHON"] = override
    subprocess.run([str(root / "scripts/serve-self-hosted.sh")], env=env, check=True, capture_output=True)
    assert capture.read_text().splitlines()[4] == str(selected)


def test_interpreter_override_is_not_shell_code(tmp_path: Path) -> None:
    root = project(tmp_path)
    marker = root / "must-not-exist"
    capture = root / "arguments.txt"
    env = {**os.environ, "SCONE_PYTHON": f"$(touch '{marker}')", "SCONE_TEST_CAPTURE": str(capture)}
    result = subprocess.run([str(root / "scripts/serve-self-hosted.sh"), "--check"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert not marker.exists()
    assert not capture.exists()
