"""Canonical self-hosted examples retain old entrypoints without starting services."""
from __future__ import annotations

from ..paths import PACKAGE_ROOT, REPO_ROOT

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scone_memory.providers.self_hosted_reranker import SelfHostedLLMReranker
from scone_memory.testing.self_hosted_reranking import CASES, parse_options

EXAMPLES = PACKAGE_ROOT / "examples"
ROOT = REPO_ROOT


def test_reranker_implementation_is_packaged():
    assert SelfHostedLLMReranker.__module__ == "scone_memory.providers.self_hosted_reranker"
    with pytest.raises(ValueError):
        SelfHostedLLMReranker("https://api.example.com/v1", "model")


def test_packaged_diagnostic_preserves_options(tmp_path):
    options = parse_options([
        "--model", "served-model", "--embedding-cache", str(tmp_path),
        "--output", str(tmp_path / "fresh.json"),
    ])
    assert options.endpoint == "http://127.0.0.1:11434/v1/"
    assert options.model == "served-model"
    assert len(CASES) == 3


@pytest.mark.parametrize("entrypoint", [
    ["-m", "scone_memory.testing.self_hosted_reranking"],
    [str(EXAMPLES / "evaluate_self_hosted_reranking.py")],
    [str(EXAMPLES / "evaluate_local_reranking.py")],
])
def test_packaged_and_compatibility_cli_help_need_no_model(entrypoint):
    if entrypoint[0] != "-m" and not Path(entrypoint[0]).is_file():
        pytest.skip("Private example wrappers are not included in the source checkout")
    result = subprocess.run([sys.executable, *entrypoint, "--help"], capture_output=True, text=True, check=True)
    assert "--embedding-cache" in result.stdout and "--model" in result.stdout


@pytest.mark.parametrize("arguments", [["--check"], ["--host", "127.0.0.1", "--port", "12345"]])
def test_legacy_launcher_forwards_exactly_to_canonical_without_launching_service(tmp_path, arguments):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    interpreter = tmp_path / "packages/memory/.venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$SCONE_TEST_CAPTURE"\n')
    interpreter.chmod(0o755)
    for name in ("serve-self-hosted.sh", "serve-local.sh"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    # The launchers prefer SCONE_PYTHON over the checkout's interpreter, and
    # the gate exports it, so an inherited value would bypass the stub.
    environment = {key: value for key, value in os.environ.items() if key != "SCONE_PYTHON"}
    outputs = []
    for name in ("serve-self-hosted.sh", "serve-local.sh"):
        capture = tmp_path / f"{name}.txt"
        subprocess.run([str(scripts / name), *arguments], check=True,
                       env={**environment, "SCONE_TEST_CAPTURE": str(capture)}, capture_output=True)
        outputs.append(capture.read_text().splitlines())
    assert outputs[0] == outputs[1]
    assert outputs[0][:3] == [str(scripts / "local_env.py"), "--env-file", str(tmp_path / ".env.local")]
    if arguments == ["--check"]:
        assert outputs[0][3:] == ["--check"]
    else:
        assert outputs[0][3:7] == ["--", str(interpreter), "-m", "scone_memory.runtime.cli"]
        assert outputs[0][7:] == ["serve", *arguments]
