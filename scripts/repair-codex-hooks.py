#!/usr/bin/env python3
"""Repair installed hook compatibility without disabling hooks or trusting new commands."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil

MARKER = '# Scone Codex hook-output compatibility v1'
COMPAT = '''
# Scone Codex hook-output compatibility v1
def _scone_print_hook_json(serialized, *, flush=False):
    # Preserve decisions, reasons and findings. Only Claude-specific telemetry
    # fields are omitted from Codex's response; Claude keeps its original output.
    if os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_SESSION_ID"):
        output = json.loads(serialized)
        if isinstance(output, dict):
            output.pop("metrics", None)
            output.pop("rewakeSummary", None)
            serialized = json.dumps(output)
    print(serialized, flush=flush)

'''


def compatible_security_source(source: str) -> str:
    if MARKER in source:
        return source
    tree = ast.parse(source)
    replacements = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print'
                and node.args and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Attribute)
                and isinstance(node.args[0].func.value, ast.Name)
                and node.args[0].func.value.id == 'json' and node.args[0].func.attr == 'dumps'):
            replacements += 1
    if not replacements or source.count('print(json.dumps(') != replacements:
        raise ValueError('unrecognized security hook output layout; no files changed')
    anchor = 'def emit_metrics('
    if anchor not in source:
        raise ValueError('security hook metrics function not found; no files changed')
    updated = source.replace('print(json.dumps(', '_scone_print_hook_json(json.dumps(')
    updated = updated.replace(anchor, COMPAT + anchor, 1)
    ast.parse(updated)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex-dir', type=Path, default=Path.home()/'.codex')
    parser.add_argument('--remember-legacy-version', action='append', default=[])
    parser.add_argument('--apply', action='store_true', help='Apply repairs; default is read-only inspection.')
    args = parser.parse_args()
    cache = args.codex_dir/'plugins/cache/claude-plugins-official'
    changes = []
    for hook in (cache/'security-guidance').glob('*/hooks/security_reminder_hook.py'):
        source = hook.read_text()
        updated = compatible_security_source(source)
        if updated != source:
            changes.append((hook, updated))
    remember = cache/'remember'
    versions = [p for p in remember.iterdir() if p.is_dir() and not p.is_symlink()
                and re.fullmatch(r'\d+\.\d+\.\d+', p.name)] if remember.exists() else []
    aliases = []
    for legacy in args.remember_legacy_version:
        if not re.fullmatch(r'\d+\.\d+\.\d+', legacy) or not versions:
            raise ValueError('valid legacy version and installed Remember release required')
        old = remember/legacy
        if old.exists() or old.is_symlink():
            continue
        current = max(versions, key=lambda p: tuple(map(int, p.name.split('.'))))
        aliases.append((old, current))
    for hook, _ in changes:
        print(f'Normalize Codex JSON metadata: {hook}')
    for old, current in aliases:
        print(f'Restore session hook path: {old} -> {current.name}')
    if not args.apply:
        return
    backup = args.codex_dir/'backups'/('hook-compat-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if changes:
        backup.mkdir(parents=True, mode=0o700)
    for hook, updated in changes:
        saved = backup/hook.relative_to(cache)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hook, saved)
        temporary = hook.with_suffix('.py.scone-tmp')
        temporary.write_text(updated)
        temporary.chmod(hook.stat().st_mode & 0o777)
        temporary.replace(hook)
    for old, current in aliases:
        old.symlink_to(current.name, target_is_directory=True)
    print('Repairs applied; hook definitions and trust configuration unchanged.')


if __name__ == '__main__':
    main()
