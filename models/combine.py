#!/usr/bin/env python3
"""Rejoin each model's weight file from the chunks committed in models/<name>/parts/, checking every byte.

    python3 models/combine.py            # every model
    python3 models/combine.py nvidia     # only models whose folder name contains "nvidia"
    python3 models/combine.py --check    # verify joined files only, change nothing

No network and no packages beyond the standard library. GitHub refuses files over 100 MB, so every weight file was
cut into chunks of 90 MiB. MANIFEST.json lists each chunk's sha256 in order and the sha256 of the whole file, which is
Hugging Face's own LFS hash of the original. The joined file is written next to the parts as models/<name>/<file>,
where transformers and sentence-transformers expect it; .gitignore keeps it out of git.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parent
ROOT = MODELS.parent  # chunk paths in MANIFEST.json are relative to the repository root
BLOCK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()


def combine(folder: str, weights: dict) -> bool:
    target = MODELS / folder / weights['file']
    if target.exists() and target.stat().st_size == weights['bytes'] and sha256_of(target) == weights['sha256']:
        print(f'{folder}: {weights["file"]} already joined and verified')
        return True
    partial = target.with_name(target.name + '.partial')
    whole = hashlib.sha256()
    with partial.open('wb') as out:
        for chunk in weights['chunks']:
            path = ROOT / chunk['path']
            if not path.exists():
                print(f'{folder}: MISSING {chunk["path"]} (did the clone or pull finish?)', file=sys.stderr)
                partial.unlink()
                return False
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != chunk['sha256']:
                print(f'{folder}: CORRUPT {chunk["path"]}', file=sys.stderr)
                partial.unlink()
                return False
            whole.update(data)
            out.write(data)
    if whole.hexdigest() != weights['sha256']:
        print(f'{folder}: joined file does not match {weights["sha256"]}', file=sys.stderr)
        partial.unlink()
        return False
    partial.replace(target)
    print(f'{folder}: joined {len(weights["chunks"])} chunks into {weights["file"]} ({weights["bytes"]:,} bytes, verified)')
    return True


def check(folder: str, weights: dict) -> bool:
    target = MODELS / folder / weights['file']
    ok = target.exists() and sha256_of(target) == weights['sha256']
    print(f'{folder}: {"ok" if ok else "NOT JOINED OR WRONG"}')
    return ok


def main(argv: list[str]) -> int:
    only_check = '--check' in argv
    wanted = next((a for a in argv if not a.startswith('--')), '')
    models = json.loads((MODELS / 'MANIFEST.json').read_text())['models']
    chosen = {f: m for f, m in models.items() if wanted in f}
    if not chosen:
        print(f'no model folder matches {wanted!r}; folders: {", ".join(models)}', file=sys.stderr)
        return 2
    results = [(check if only_check else combine)(folder, model['weights']) for folder, model in chosen.items()]
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
