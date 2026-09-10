"""Private environment loading is explicit, data-only, and secret-safe."""

from ..paths import REPO_ROOT

from pathlib import Path
import ast
import re
import runpy
import subprocess
import sys

import pytest

from scone_memory.runtime.config import Settings

ROOT = REPO_ROOT
LOADER = ROOT / "scripts" / "local_env.py"


def test_template_parses_as_actual_local_settings():
    parse = runpy.run_path(str(LOADER))["parse_environment"]
    values = parse((ROOT / ".env.example").read_text())
    settings = Settings.from_env(values)
    assert settings.documents == settings.vectors == "sqlite"
    assert settings.embedder == "hash" and settings.host == "127.0.0.1"
    assert settings.keys == {} and not settings.chat_url and not settings.derive
    assert settings.distill_accept_at is None and settings.retention == {}
    assert values["LANGSMITH_TRACING"] == "false" and values["OPENAI_AGENTS_DISABLE_TRACING"] == "true"


def test_template_covers_every_named_python_setting():
    import scone_memory.runtime.config as config

    tree = ast.parse(Path(config.__file__).read_text())
    names = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
             and isinstance(node.value, str) and re.fullmatch(r"SCONE_[A-Z_]+", node.value)}
    template = (ROOT / ".env.example").read_text()
    assert names and all(name in template for name in names)


def test_loader_treats_shell_substitution_as_literal_data(tmp_path):
    parse = runpy.run_path(str(LOADER))["parse_environment"]
    marker = tmp_path / "never-created"
    text = f"VALUE='$(touch {marker})'\nEMPTY=\nQUOTED='literal # value'\n"
    result = parse(text)
    assert result == {"VALUE": f"$(touch {marker})", "EMPTY": "", "QUOTED": "literal # value"}
    assert not marker.exists()


@pytest.mark.parametrize("text", ["not an assignment", "A=one two", "A='unclosed", "A=1\nA=2"])
def test_loader_rejects_invalid_files_without_echoing_values(text):
    parse = runpy.run_path(str(LOADER))["parse_environment"]
    with pytest.raises(ValueError) as error:
        parse(text + " private-secret-marker")
    assert "private-secret-marker" not in str(error.value)


def test_private_file_overrides_inherited_settings_without_mutating_them(tmp_path):
    load = runpy.run_path(str(LOADER))["private_environment"]
    path = tmp_path / ".env.local"
    path.write_text("SCONE_API_KEY='local-private-value'\nLANGSMITH_TRACING=false\n")
    path.chmod(0o600)
    inherited = {"LANGSMITH_TRACING": "true", "PATH": "/usr/bin"}
    loaded = load(path, inherited)
    assert loaded["SCONE_API_KEY"] == "local-private-value"
    assert loaded["LANGSMITH_TRACING"] == "false" and inherited["LANGSMITH_TRACING"] == "true"
    assert loaded["PATH"] == "/usr/bin"
    path.chmod(0o644)
    with pytest.raises(ValueError):
        load(path, {})
    path.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        load(link, {})


def test_explicit_loader_executes_target_without_leaking_environment(tmp_path):
    path = tmp_path / ".env.local"
    path.write_text("SCONE_API_KEY='local-private-value'\nLANGSMITH_TRACING=false\n")
    path.chmod(0o600)
    command = [sys.executable, str(LOADER), "--env-file", str(path)]
    checked = subprocess.run([*command, "--check"], text=True, capture_output=True, check=True)
    assert "no process started" in checked.stdout and "local-private-value" not in checked.stdout + checked.stderr
    child = subprocess.run([*command, "--", sys.executable, "-c",
        "import os; assert os.environ['SCONE_API_KEY']=='local-private-value'; "
        "assert os.environ['LANGSMITH_TRACING']=='false'; print('configured')"],
        text=True, capture_output=True, check=True)
    assert child.stdout == "configured\n" and child.stderr == ""
