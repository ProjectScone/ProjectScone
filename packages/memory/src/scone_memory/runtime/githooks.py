"""``scone-memory hooks``: keep a repository's graph current from git's own hooks.

``map --watch`` follows a tree by polling it. A repository has a better
signal than the clock: git says when a commit lands, a branch is checked
out or a merge completes, and those are the moments the graph goes stale.
``hooks install`` writes one runner script into the repository's hooks
directory (wherever ``core.hooksPath`` puts it) and a guarded block into
``post-commit``, ``post-checkout`` and ``post-merge`` that calls it; each
call maps the tree with ``--graph`` in the background, so a commit is not
made to wait, and appends the receipt to ``scone-map.log`` in the
repository's git directory (a linked worktree's own, so two worktrees do
not share a log). ``hooks status`` says what is installed, with which
interpreter and which settings, and how the log ends: the last receipt
when the last run finished, or the last line when it did not; ``hooks
uninstall`` takes the blocks out and leaves whatever else the hook files
held.

Rules that keep this safe to run twice and safe beside a person's own
hooks: a hook file that exists is appended to, never replaced, and a
second install replaces its own block where it stands, so what the
person put after it stays after it; a file with a block start and no
end is refused rather than guessed at; the interpreter path is written
in full, as the leading tool's hooks do, so a commit from an editor with
no shell environment still finds it, and ``PYTHONPATH`` goes with it
when the install ran with one, for a source tree that is not installed;
and only the settings that name a store kind or a local path
(``SCONE_DOCUMENTS``, ``SCONE_VECTORS``, ``SCONE_EVENTS``,
``SCONE_SQLITE_PATH``) are written into the runner -- a connection URL
or a key is never copied into a file under ``.git``; the runner reads
those from the environment it runs in, or from a file the person names
with ``--env-file``. A checkout of files (``post-checkout`` with its
branch flag 0) maps nothing: the tree did not move. ``SCONE_HOOK_WAIT=1``
makes the runner wait for the map, for a test or a script that wants
the receipt before going on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
from typing import Mapping, Optional

from ..core.errors import InvalidInput

HOOKS = ("post-commit", "post-checkout", "post-merge")
RUNNER = "scone-map"
LOG = "scone-map.log"
BLOCK_START = "# >>> scone-memory hooks >>>"
BLOCK_END = "# <<< scone-memory hooks <<<"
#: The settings copied into the runner: a store's kind and a local path,
#: never a URL or a key; and the module path, for a source tree.
CARRIED_SETTINGS = ("SCONE_DOCUMENTS", "SCONE_VECTORS", "SCONE_EVENTS", "SCONE_SQLITE_PATH", "PYTHONPATH")
#: Characters of a receipt `status` shows before it shows a summary instead.
MAX_RECEIPT = 2_000


@dataclass(frozen=True)
class HooksReceipt:
    repository: str
    hooks_dir: str
    runner: str
    log: str
    interpreter: str
    space: str
    graph: bool
    carried: tuple[str, ...]
    env_file: Optional[str]
    hooks_written: tuple[str, ...] = ()
    hooks_appended: tuple[str, ...] = ()
    hooks_replaced: tuple[str, ...] = ()

    def record(self) -> dict[str, object]:
        return {"repository": self.repository, "hooks_dir": self.hooks_dir, "runner": self.runner, "log": self.log,
                "interpreter": self.interpreter, "space": self.space, "graph": self.graph,
                "carried_settings": list(self.carried), "env_file": self.env_file,
                "hooks_written": list(self.hooks_written), "hooks_appended": list(self.hooks_appended),
                "hooks_replaced": list(self.hooks_replaced)}


@dataclass(frozen=True)
class HooksStatus:
    repository: str
    hooks_dir: str
    installed: bool
    hooks: dict[str, str] = field(default_factory=dict)
    interpreter: Optional[str] = None
    interpreter_exists: Optional[bool] = None
    carried: tuple[str, ...] = ()
    log: Optional[str] = None
    last_receipt: Optional[str] = None

    def record(self) -> dict[str, object]:
        return {"repository": self.repository, "hooks_dir": self.hooks_dir, "installed": self.installed,
                "hooks": dict(self.hooks), "interpreter": self.interpreter, "interpreter_exists": self.interpreter_exists,
                "carried_settings": list(self.carried), "log": self.log, "last_receipt": self.last_receipt}


def _git(root: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as failed:
        raise InvalidInput(f"git could not be run: {failed}") from None
    if done.returncode != 0:
        raise InvalidInput(f"{root} is not a git repository, or git refused: {done.stderr.strip()[:200]}")
    return done.stdout.strip()


def _places(root: str | Path) -> tuple[Path, Path, Path]:
    """The repository's top, its hooks directory (honouring core.hooksPath)
    and its git directory, resolved through git itself."""
    top = Path(_git(Path(root), "rev-parse", "--show-toplevel"))
    hooks = Path(_git(top, "rev-parse", "--git-path", "hooks"))
    git_dir = Path(_git(top, "rev-parse", "--git-dir"))
    hooks = hooks if hooks.is_absolute() else top / hooks
    git_dir = git_dir if git_dir.is_absolute() else top / git_dir
    return top, hooks, git_dir


def _runner_text(top: Path, interpreter: str, space: str, graph: bool, carried: Mapping[str, str],
                 log: Path, env_file: Optional[str]) -> str:
    settings = "".join(f"export {name}={shlex.quote(value)}\n" for name, value in carried.items())
    source = f'[ -f {shlex.quote(env_file)} ] && . {shlex.quote(env_file)}\n' if env_file else ""
    flag = " --graph" if graph else ""
    return (f"#!/bin/sh\n{BLOCK_START}\n"
            f"# Written by `scone-memory hooks install`; `scone-memory hooks uninstall` removes it.\n"
            f"# Maps the repository after a commit, a branch checkout or a merge; the receipt goes to the log.\n"
            f"HOOK=\"$1\"\n"
            f"if [ \"$HOOK\" = post-checkout ] && [ \"$4\" != 1 ]; then exit 0; fi\n"
            f"ROOT={shlex.quote(str(top))}\nPY={shlex.quote(interpreter)}\nLOG={shlex.quote(str(log))}\n"
            f"{source}{settings}"
            f"run() {{ \"$PY\" -m scone_memory.runtime.cli --json --space {shlex.quote(space)} map \"$ROOT\"{flag} >> \"$LOG\" 2>&1; }}\n"
            f"if [ -n \"$SCONE_HOOK_WAIT\" ]; then run; else ( run & ); fi\n"
            f"exit 0\n{BLOCK_END}\n")


def _block(runner: Path, hook: str) -> str:
    return f"{BLOCK_START}\n{shlex.quote(str(runner))} {hook} \"$@\"\n{BLOCK_END}\n"


def _span(text: str, path: Path) -> Optional[tuple[int, int]]:
    """Where this writer's block sits in a hook file: the start of its
    first marker line to the end of its end marker line (with the line
    break), or None when there is no block. A marker is a whole line, so
    a comment that mentions one is not a block; a start without an end,
    or a second start, is refused rather than guessed at."""
    starts, ends = [], []
    offset = 0
    for line in text.splitlines(keepends=True):
        bare = line.strip()
        if bare == BLOCK_START:
            starts.append(offset)
        elif bare == BLOCK_END:
            ends.append(offset + len(line))
        offset += len(line)
    if not starts and not ends:
        return None
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise InvalidInput(f"{path} holds a scone-memory hook block that is not whole (a start without its end, or two); "
                           "put it right by hand before installing or uninstalling")
    return starts[0], ends[0]


def _has_block(text: str, path: Path) -> bool:
    try:
        return _span(text, path) is not None
    except InvalidInput:
        return True


def _executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install(root: str | Path, *, space: str = "default", graph: bool = True, env: Optional[Mapping[str, str]] = None,
            interpreter: Optional[str] = None, env_file: Optional[str] = None) -> HooksReceipt:
    """Write the runner and the three hook blocks, and say what was done."""
    top, hooks, git_dir = _places(root)
    environment = os.environ if env is None else env
    carried = {name: environment[name] for name in CARRIED_SETTINGS if environment.get(name)}
    python = interpreter or sys.executable
    if env_file is not None:
        env_file = str(Path(env_file).expanduser().resolve())
    hooks.mkdir(parents=True, exist_ok=True)
    runner, log = hooks / RUNNER, git_dir / LOG
    runner.write_text(_runner_text(top, python, space, graph, carried, log, env_file), encoding="utf-8")
    _executable(runner)
    written, appended, replaced = [], [], []
    for hook in HOOKS:
        path = hooks / hook
        block = _block(runner, hook)
        if not path.exists():
            path.write_text("#!/bin/sh\n" + block, encoding="utf-8")
            written.append(hook)
        else:
            current = path.read_text(encoding="utf-8", errors="replace")
            span = _span(current, path)
            if span is not None:
                # Replaced where it stands: what the person put after it stays after it.
                path.write_text(current[:span[0]] + block + current[span[1]:], encoding="utf-8")
                replaced.append(hook)
            else:
                joint = "" if current.endswith("\n") or not current else "\n"
                path.write_text(current + joint + block, encoding="utf-8")
                appended.append(hook)
        _executable(path)
    return HooksReceipt(str(top), str(hooks), str(runner), str(log), python, space, graph, tuple(carried),
                        env_file, tuple(written), tuple(appended), tuple(replaced))


def status(root: str | Path) -> HooksStatus:
    """What is installed: each hook's state (`installed`, `absent`, or
    `other`, a hook of the person's own without the block), the runner's
    interpreter and whether it still exists, the settings carried, and
    the log's last line."""
    top, hooks, git_dir = _places(root)
    runner, log = hooks / RUNNER, git_dir / LOG
    states: dict[str, str] = {}
    for hook in HOOKS:
        path = hooks / hook
        if not path.exists():
            states[hook] = "absent"
        else:
            states[hook] = "installed" if _has_block(path.read_text(encoding="utf-8", errors="replace"), path) else "other"
    interpreter: Optional[str] = None
    carried: list[str] = []
    if runner.exists():
        for line in runner.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("PY="):
                interpreter = "".join(shlex.split(line[3:]))
            elif line.startswith("export ") and "=" in line:
                carried.append(line[7:].split("=", 1)[0])
    last = _last_receipt(log.read_text(encoding="utf-8", errors="replace")) if log.exists() else None
    return HooksStatus(str(top), str(hooks), runner.exists() and any(s == "installed" for s in states.values()),
                       states, interpreter, None if interpreter is None else Path(interpreter).exists(),
                       tuple(carried), str(log) if log.exists() else None, last)


def _last_receipt(text: str) -> Optional[str]:
    """How the log ends: the last receipt, compact, when the log ends with
    one; the log's last line when it ends with something else (a
    traceback from a run that failed), so a failed run is never hidden
    behind the success before it. A receipt too long to show whole is
    shown by its counts."""
    decoder = json.JSONDecoder()
    at = text.rfind("\n{") + 1 if text.rfind("\n{") >= 0 else (0 if text.startswith("{") else -1)
    if at >= 0:
        try:
            value, end = decoder.raw_decode(text, at)
        except ValueError:
            value, end = None, at
        if value is not None and not text[end:].strip():
            shown = json.dumps(value, ensure_ascii=False)
            if len(shown) > MAX_RECEIPT and isinstance(value, dict):
                shown = json.dumps({key: item for key, item in value.items() if isinstance(item, (int, float, str, bool))}
                                   | {"shown": "counts only; the whole receipt is in the log"}, ensure_ascii=False)
            return shown[:MAX_RECEIPT]
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1][:MAX_RECEIPT] if lines else None


def uninstall(root: str | Path) -> dict[str, object]:
    """Take the blocks out of the hook files (a file left with only its
    shebang is removed), remove the runner, and say what was done. The
    log is left, being a record."""
    top, hooks, _ = _places(root)
    removed, cleaned = [], []
    for hook in HOOKS:
        path = hooks / hook
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8", errors="replace")
        span = _span(current, path)
        if span is None:
            continue
        rest = current[:span[0]] + current[span[1]:]
        if rest.strip() in ("", "#!/bin/sh"):
            path.unlink()
            removed.append(hook)
        else:
            path.write_text(rest, encoding="utf-8")
            cleaned.append(hook)
    runner = hooks / RUNNER
    runner_removed = runner.exists()
    if runner_removed:
        runner.unlink()
    return {"repository": str(top), "hooks_dir": str(hooks), "hooks_removed": removed, "hooks_cleaned": cleaned,
            "runner_removed": runner_removed}
