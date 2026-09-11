"""Local directory command; encryption keys are read from private files only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
from typing import TextIO

from pydantic import ValidationError

from ..core.errors import InvalidInput
from ..memory.engine import MemoryEngine


def read_journal_key(path: str, root: str) -> bytes:
    target = Path(path).absolute()
    try:
        if target.resolve(strict=True).is_relative_to(Path(root).resolve(strict=True)):
            raise InvalidInput('directory journal key must be outside the source root')
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or info.st_nlink != 1 or info.st_size != 32):
                raise InvalidInput('directory key requires a private, single-link regular file containing exactly 32 bytes')
            key = stream.read(33)
            if len(key) != 32:
                raise InvalidInput('directory key must contain exactly 32 bytes')
            return key
    except (OSError, ValueError):
        raise InvalidInput('directory journal key file cannot be read safely') from None


async def run_directory_sync(args: argparse.Namespace, engine: MemoryEngine, out: TextIO) -> int:
    try:
        from ..ingestion.directory_sync import DirectorySync
        from ..ingestion.source_scan import ScanLimits
    except ImportError:
        raise InvalidInput("directory synchronization requires scone-memory[document-workflows]") from None
    key = read_journal_key(args.key_file, args.root)
    try:
        limits = ScanLimits(max_files=args.max_files, max_total_bytes=args.max_total_bytes)
    except ValidationError:
        raise InvalidInput('directory scan limits are invalid') from None
    sync = DirectorySync(engine, args.root, space=args.space, journal=args.journal,
                         key=key, store_id=args.store_id, parser_revision=args.parser_revision,
                         scan_limits=limits,
                         extensions=frozenset(args.extension) if args.extension else None)
    result = await sync.synchronize(delete_missing=args.delete_missing)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=True), file=out)
    else:
        print(f"directory sync {'complete' if result.complete else 'partial'}; collection {result.collection_id}", file=out)
        for receipt in result.receipts:
            identity = f' episode {receipt.episode_id}' if receipt.episode_id is not None else ''
            path = json.dumps(receipt.path, ensure_ascii=True)
            print(f'{receipt.status}: {path}{identity}' + (f' ({receipt.code})' if receipt.code else ''), file=out)
        for issue in result.issues:
            print(f'{issue.code}: {json.dumps(issue.path or ".", ensure_ascii=True)}', file=out)
        if result.skipped:
            print(f'{result.skipped} unsupported file(s) skipped', file=out)
    return 0 if result.complete else 1
