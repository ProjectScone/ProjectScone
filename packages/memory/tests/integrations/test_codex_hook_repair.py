"""Codex-cache repairs preserve security decisions without host environment flags."""

from ..paths import REPO_ROOT
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = REPO_ROOT/'scripts/repair-codex-hooks.py'
spec = importlib.util.spec_from_file_location('hook_repair',SCRIPT)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)

SOURCE = '''import json, os
def emit_metrics(payload):
    print(json.dumps(payload), flush=True)
'''


@pytest.mark.parametrize('host',['CODEX_THREAD_ID','CODEX_SESSION_ID','no_codex_environment'])
def test_metadata_conversion_preserves_findings_and_blocking(monkeypatch,host):
    monkeypatch.delenv('CODEX_THREAD_ID',raising=False)
    monkeypatch.delenv('CODEX_SESSION_ID',raising=False)
    if host!='no_codex_environment':monkeypatch.setenv(host,'synthetic')
    payload={'metrics':{'skipped':False},'rewakeSummary':'summary','decision':'block','reason':'review this',
             'hookSpecificOutput':{'hookEventName':'PostToolUse','additionalContext':'security finding'}}
    namespace={}
    exec(repair.compatible_security_source(SOURCE),namespace)
    output=io.StringIO()
    with contextlib.redirect_stdout(output):namespace['emit_metrics'](payload)
    expected={k:v for k,v in payload.items() if k not in ('metrics','rewakeSummary')}
    assert json.loads(output.getvalue())==expected


def test_metrics_only_result_becomes_valid_empty_object(monkeypatch):
    monkeypatch.setenv('CODEX_THREAD_ID','synthetic')
    namespace={}
    exec(repair.compatible_security_source(SOURCE),namespace)
    output=io.StringIO()
    with contextlib.redirect_stdout(output):namespace['emit_metrics']({'metrics':{'skip':True}})
    assert output.getvalue()=='{}\n'


def test_repair_is_idempotent_and_rejects_unrecognized_output_layout():
    fixed=repair.compatible_security_source(SOURCE)
    assert repair.compatible_security_source(fixed)==fixed
    with pytest.raises(ValueError):repair.compatible_security_source('print("unexpected")')


def test_bootstrap_emits_one_response_and_preserves_startup_notice():
    source='''import json
def main():
    print(json.dumps({"async": True, "asyncTimeout": 180000}), flush=True)
    print(json.dumps({"metrics": {"sdk_bootstrap": 1}, "systemMessage": "reviewer notice"}), flush=True)
'''
    namespace={}
    exec(repair.compatible_security_source(source),namespace)
    output=io.StringIO()
    with contextlib.redirect_stdout(output):namespace['main']()
    assert json.loads(output.getvalue())=={'systemMessage':'reviewer notice'}


def test_upgrade_v1_removes_environment_dependency():
    source='''import json, os
# Scone Codex hook-output compatibility v1
def _scone_print_hook_json(serialized, *, flush=False):
    print(serialized, flush=flush)
def emit_metrics(payload):
    _scone_print_hook_json(json.dumps(payload), flush=True)
'''
    updated=repair.compatible_security_source(source)
    assert 'compatibility v1' not in updated
    assert repair.MARKER in updated


def test_cli_backs_up_hook_and_restores_missing_version_without_changing_trust(tmp_path):
    cache=tmp_path/'plugins/cache/claude-plugins-official'
    hook=cache/'security-guidance/2.0.7/hooks/security_reminder_hook.py'
    hook.parent.mkdir(parents=True)
    hook.write_text(SOURCE)
    hook.chmod(0o755)
    bootstrap=hook.with_name('ensure_agent_sdk.py')
    bootstrap.write_text(SOURCE.replace('emit_metrics', 'main'))
    claude_hook=tmp_path/'claude/plugins/security_reminder_hook.py'
    claude_hook.parent.mkdir(parents=True)
    claude_hook.write_text(SOURCE)
    current=cache/'remember/0.29.1'
    current.mkdir(parents=True)
    config=tmp_path/'config.toml'
    config.write_text('# Existing trust configuration\n')
    command=[sys.executable,str(SCRIPT),'--codex-dir',str(tmp_path),
             '--remember-legacy-version','0.25.0']
    subprocess.run(command,check=True,capture_output=True)
    assert hook.read_text()==SOURCE
    assert not (current.parent/'0.25.0').exists()
    for _ in range(2):
        subprocess.run([*command,'--apply'],check=True,capture_output=True)
    assert (current.parent/'0.25.0').resolve()==current
    assert repair.MARKER in hook.read_text()
    assert hook.stat().st_mode & 0o777==0o755
    backups=list((tmp_path/'backups').glob('*/security-guidance/*/hooks/*.py'))
    assert len(backups)==2
    assert next(p for p in backups if p.name==hook.name).read_text()==SOURCE
    assert repair.MARKER in bootstrap.read_text()
    assert claude_hook.read_text()==SOURCE
    assert config.read_text()=='# Existing trust configuration\n'
