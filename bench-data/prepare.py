#!/usr/bin/env python3
"""Make the benchmark datasets ready to use. Standard library only.

    python3 bench-data/prepare.py              # unpack LongMemEval-S (no network)
    python3 bench-data/prepare.py --m          # also download LongMemEval-M (2.7 GB) and convert it to JSONL
    python3 bench-data/prepare.py --ontonotes  # also download the OntoNotes 5 test split
    python3 bench-data/prepare.py --check      # verify every committed file against MANIFEST.json

Most datasets are committed here as they are. Three are not:

- ``longmemeval_s.json`` (278 MB) is over GitHub's 100 MB file limit, so it is committed gzipped and unpacked here.
- ``longmemeval_m_cleaned.jsonl`` (2.4 GB) is too large to commit. It is downloaded from Hugging Face at a pinned
  sha256 and rewritten as one question per line, which is what the harness reads without loading it all at once.
- ``ner/ontonotes5-test.parquet`` is under a licence that does not allow redistribution, so it is downloaded.

Every download and every unpacked file is checked against a sha256 recorded below or in MANIFEST.json.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent
BLOCK = 1 << 20

M_URL = ('https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/'
         '98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_m_cleaned.json')
M_SHA256 = '9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f'
M_QUESTIONS = 500
ONTONOTES_URL = 'https://huggingface.co/datasets/tner/ontonotes5/resolve/refs%2Fconvert%2Fparquet/ontonotes5/test/0000.parquet'
ONTONOTES_SHA256 = '1fa4f7070147c2ccd84ece6e8642486d86e56ee2d69153edc5b271306bd5b6dc'


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, expected: str) -> None:
    """Fetch ``url`` to ``target``, refusing a file whose sha256 is not ``expected``."""
    if target.exists() and sha256_of(target) == expected:
        print(f'{target.name}: already present and verified')
        return
    partial = target.with_name(target.name + '.partial')
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, partial.open('wb') as out:
        for block in iter(lambda: response.read(BLOCK), b''):
            digest.update(block)
            out.write(block)
    if digest.hexdigest() != expected:
        partial.unlink()
        raise SystemExit(f'{target.name}: download does not match sha256 {expected}')
    partial.replace(target)
    print(f'{target.name}: downloaded and verified')


def array_items(path: Path) -> Iterator[object]:
    """The items of a top-level JSON array, one at a time, without holding the file in memory."""
    decoder = json.JSONDecoder()
    with path.open('r', encoding='utf-8') as handle:
        buffer = handle.read(BLOCK).lstrip()
        if not buffer.startswith('['):
            raise ValueError(f'{path.name} is not a JSON array')
        buffer = buffer[1:]
        while True:
            buffer = buffer.lstrip().removeprefix(',').lstrip()
            if buffer.startswith(']'):
                return
            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                more = handle.read(BLOCK)
                if not more:
                    raise ValueError(f'{path.name} ends inside an item') from None
                buffer += more
                continue
            if end == len(buffer):  # a number or literal may continue in the next block; read on before trusting it
                more = handle.read(BLOCK)
                if more:
                    buffer += more
                    continue
            yield item
            buffer = buffer[end:]


def to_jsonl(source: Path, target: Path) -> int:
    count = 0
    partial = target.with_name(target.name + '.partial')
    with partial.open('w', encoding='utf-8') as out:
        for item in array_items(source):
            out.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    partial.replace(target)
    return count


def unpack_s(manifest: dict) -> None:
    target = HERE / 'longmemeval_s.json'
    expected = manifest['generated']['longmemeval_s.json']['sha256']
    if target.exists() and sha256_of(target) == expected:
        print('longmemeval_s.json: already unpacked and verified')
        return
    partial = target.with_name(target.name + '.partial')
    with gzip.open(HERE / 'longmemeval_s.json.gz', 'rb') as packed, partial.open('wb') as out:
        shutil.copyfileobj(packed, out, BLOCK)
    if sha256_of(partial) != expected:
        partial.unlink()
        raise SystemExit('longmemeval_s.json: unpacked file does not match its sha256')
    partial.replace(target)
    print('longmemeval_s.json: unpacked and verified')


def fetch_m() -> None:
    target = HERE / 'longmemeval_m_cleaned.jsonl'
    if target.exists():
        with target.open('rb') as handle:
            lines = sum(1 for _ in handle)
        if lines == M_QUESTIONS:
            print(f'longmemeval_m_cleaned.jsonl: already present with {lines} questions')
            return
    source = HERE / 'longmemeval_m_cleaned.json'
    download(M_URL, source, M_SHA256)
    count = to_jsonl(source, target)
    if count != M_QUESTIONS:
        raise SystemExit(f'longmemeval_m_cleaned.jsonl: {count} questions, expected {M_QUESTIONS}')
    source.unlink()
    print(f'longmemeval_m_cleaned.jsonl: {count} questions written')


def check(manifest: dict) -> bool:
    ok = True
    for name, entry in manifest['files'].items():
        path = HERE / name
        good = path.exists() and path.stat().st_size == entry['bytes'] and sha256_of(path) == entry['sha256']
        print(f'{"ok " if good else "BAD"} {name}')
        ok = ok and good
    return ok


def main(argv: list[str]) -> int:
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())
    if '--check' in argv:
        return 0 if check(manifest) else 1
    unpack_s(manifest)
    if '--m' in argv:
        fetch_m()
    if '--ontonotes' in argv:
        download(ONTONOTES_URL, HERE / 'ner' / 'ontonotes5-test.parquet', ONTONOTES_SHA256)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
