"""Installed-hook repair preserves security decisions and Claude output."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[3]/'scripts/repair-codex-hooks.py'
spec = importlib.util.spec_from_file_location('hook_repair',SCRIPT)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)

SOURCE = '''import json, os
def emit_metrics(payload):
    print(json.dumps(payload), flush=True)
'''


@pytest.mark.parametrize('host',['CODEX_THREAD_ID','CODEX_SESSION_ID','claude'])
def test_metadata_conversion_preserves_findings_and_blocking(monkeypatch,host):
    monkeypatch.delenv('CODEX_THREAD_ID',raising=False)
    monkeypatch.delenv('CODEX_SESSION_ID',raising=False)
    if host!='claude':monkeypatch.setenv(host,'synthetic')
    payload={'metrics':{'skipped':False},'rewakeSummary':'summary','decision':'block','reason':'review this',
             'hookSpecificOutput':{'hookEventName':'PostToolUse','additionalContext':'security finding'}}
    namespace={}
    exec(repair.compatible_security_source(SOURCE),namespace)
    output=io.StringIO()
    with contextlib.redirect_stdout(output):namespace['emit_metrics'](payload)
    expected=payload if host=='claude' else {k:v for k,v in payload.items() if k not in ('metrics','rewakeSummary')}
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


def test_cli_backs_up_hook_and_restores_missing_version_without_changing_trust(tmp_path):
    cache=tmp_path/'plugins/cache/claude-plugins-official'
    hook=cache/'security-guidance/2.0.7/hooks/security_reminder_hook.py'
    hook.parent.mkdir(parents=True)
    hook.write_text(SOURCE)
    hook.chmod(0o755)
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
    assert len(backups)==1
    assert backups[0].read_text()==SOURCE
    assert config.read_text()=='# Existing trust configuration\n'
