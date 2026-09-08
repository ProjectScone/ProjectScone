"""Launch a command with a private environment file, without evaluating shell code."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import re
import shlex
import stat
import sys

_ASSIGNMENT = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)")
_MAX_BYTES = 128 * 1024


def parse_environment(text: str) -> dict[str, str]:
    """Single-line KEY=value data; quotes supported, expansion and execution absent."""
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _ASSIGNMENT.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid environment assignment on line {number}")
        name, raw_value = match.groups()
        if name in values:
            raise ValueError(f"duplicate environment assignment on line {number}")
        try:
            tokens = shlex.split(raw_value, comments=True, posix=True)
        except ValueError:
            raise ValueError(f"invalid environment quoting on line {number}") from None
        if len(tokens) > 1 or any("\0" in token for token in tokens):
            raise ValueError(f"invalid environment value on line {number}")
        values[name] = tokens[0] if tokens else ""
    return values


def private_environment(path: Path, inherited: Mapping[str, str]) -> dict[str, str]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid():
            raise ValueError("environment file must be an owned regular file with mode 0600")
        raw = stream.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise ValueError("environment file exceeds 128 KiB")
    return {**inherited, **parse_environment(raw.decode("utf-8"))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="validate file without starting any command")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        environment = private_environment(args.env_file, os.environ)
    except (OSError, ValueError) as error:
        print(f"cannot load private environment ({type(error).__name__}); check file format and 0600 permissions", file=sys.stderr)
        return 2
    if args.check:
        print("Private environment file validated; no process started.")
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --, or use --check")
    try:
        os.execvpe(command[0], command, environment)
    except OSError as error:
        print(f"cannot start configured command ({type(error).__name__})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
