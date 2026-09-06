"""``scone-memory agent-hook``: turn an agent's hook payload into an agent
event, and optionally capture the text as memory.

Observe only. The hook never changes permissions, never blocks the
agent (it exits 0 whatever happens, with details on stderr only when
SCONE_HOOK_DEBUG=1), never prints the key, and sends nothing for a
project that is not on its allowlist. The default feed is metadata
(who, when, which tool, how long); text travels only with
``--feed full``, redacted here and again by the server. Hidden
reasoning is not in any hook payload we read, so it cannot be sent.

Configuration (flags win over environment):
  --agent claude-code|codex        SCONE_HOOK_AGENT
  --server http://127.0.0.1:7437   SCONE_HOOK_SERVER
  --key-env SCONE_API_KEY          SCONE_HOOK_KEY_ENV   (name of the env var holding the key)
  --space default                  (informational; the key decides the space)
  --feed metadata|full             SCONE_HOOK_FEED      (default metadata)
  --projects name=/abs/path,...    SCONE_HOOK_PROJECTS  (required; cwd must sit under one)
  --capture                        SCONE_HOOK_CAPTURE=1 (store prompt/response text as episodes; full feed only)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from .engine import redact_secrets
from .timeutil import now_rfc3339

Transport = Callable[[str, dict, str], dict]

CLAUDE_EVENTS = {
    "SessionStart": "session_start",
    "UserPromptSubmit": "prompt",
    "Stop": "response",
    "PreToolUse": "tool_use",
    "PostToolUse": "tool_result",
    "PostToolUseFailure": "tool_result",
    "SessionEnd": "session_end",
}
MAX_TEXT_BYTES = 60_000


@dataclass(frozen=True)
class HookConfig:
    agent: str
    server: str
    key: str
    feed: str
    projects: Mapping[str, str]  # canonical name -> absolute path
    capture: bool


def parse_projects(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, path = entry.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"projects entries are name=/abs/path, got {entry!r}")
        out[name.strip()] = str(Path(path.strip()).expanduser().resolve())
    return out


def project_for(cwd: Optional[str], projects: Mapping[str, str]) -> Optional[str]:
    """The allowlisted project whose path contains cwd, else None."""
    if not cwd:
        return None
    here = Path(cwd).resolve()
    for name, root in projects.items():
        root_path = Path(root)
        if here == root_path or root_path in here.parents:
            return name
    return None


def clip(text: object, limit: int = MAX_TEXT_BYTES) -> tuple[str, bool]:
    """Cut on UTF-8 bytes, never inside a character; says whether it cut."""
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
    raw = s.encode()
    if len(raw) <= limit:
        return s, False
    return raw[:limit].decode(errors="ignore"), True


def normalise_claude(payload: Mapping) -> Optional[dict]:
    event = CLAUDE_EVENTS.get(str(payload.get("hook_event_name")))
    if event is None:
        return None
    out: dict = {
        "agent": "claude-code",
        "session_id": str(payload.get("session_id") or ""),
        "event": event,
        "source_event_id": None,
        "text": None,
    }
    if event == "prompt":
        out["text"] = payload.get("user_input", payload.get("prompt"))  # newer docs, older versions
        out["source_event_id"] = f"prompt:{payload.get('prompt_id') or ''}" if payload.get("prompt_id") else None
    elif event == "response":
        out["text"] = payload.get("last_assistant_message")
        out["source_event_id"] = f"stop:{payload.get('prompt_id') or ''}" if payload.get("prompt_id") else None
    elif event == "tool_use":
        out["tool_name"] = payload.get("tool_name")
        out["tool_use_id"] = payload.get("tool_use_id")
        out["text"] = payload.get("tool_input")
        out["source_event_id"] = f"pre:{payload.get('tool_use_id')}" if payload.get("tool_use_id") else None
    elif event == "tool_result":
        out["tool_name"] = payload.get("tool_name")
        out["tool_use_id"] = payload.get("tool_use_id")
        out["text"] = payload.get("tool_output", payload.get("tool_response"))
        out["ok"] = payload.get("hook_event_name") != "PostToolUseFailure"
        out["source_event_id"] = f"post:{payload.get('tool_use_id')}" if payload.get("tool_use_id") else None
    if payload.get("model"):
        out["model"] = str(payload["model"])
    return out


def normalise_codex(payload: Mapping) -> Optional[dict]:
    """Best effort until Codex's exact hook fields are confirmed: keyed on
    which fields are present. Anything unrecognised is ignored."""
    session = str(payload.get("session_id") or payload.get("thread_id") or "")
    out: dict = {"agent": "codex", "session_id": session, "source_event_id": None, "text": None}
    turn = payload.get("turn_id")
    if "prompt" in payload:
        out.update(event="prompt", text=payload.get("prompt"), source_event_id=f"prompt:{turn}" if turn else None)
    elif "tool_response" in payload or "tool_output" in payload:
        out.update(event="tool_result", tool_name=payload.get("tool_name"), tool_use_id=payload.get("call_id") or payload.get("tool_use_id"),
                   text=payload.get("tool_response", payload.get("tool_output")), ok=payload.get("ok", True),
                   source_event_id=f"post:{payload.get('call_id') or payload.get('tool_use_id') or turn}" if (payload.get("call_id") or payload.get("tool_use_id") or turn) else None)
    elif "last_assistant_message" in payload:
        out.update(event="response", text=payload.get("last_assistant_message"), source_event_id=f"stop:{turn}" if turn else None)
    elif str(payload.get("hook_event_name", "")).lower() in ("sessionstart", "session_start"):
        out.update(event="session_start")
    elif str(payload.get("hook_event_name", "")).lower() in ("sessionend", "session_end"):
        out.update(event="session_end")
    else:
        return None
    if payload.get("model"):
        out["model"] = str(payload["model"])
    return out


def build_event(normalised: dict, project: str, feed: str) -> dict:
    """The agent event to post. Metadata feed drops every text field."""
    event = {k: v for k, v in normalised.items() if v is not None and k != "text"}
    event["project"] = project
    if feed == "full" and normalised.get("text") is not None:
        text, truncated = clip(normalised["text"])
        event["text"] = redact_secrets(text)
        if truncated:
            event["text_truncated"] = True
    return event


def http_transport(path: str, body: dict, key: str, server: str = "", timeout: float = 2.0) -> dict:
    req = urllib.request.Request(
        server.rstrip("/") + path, data=json.dumps(body).encode(), method="POST",
        headers={"content-type": "application/json", "authorization": "Bearer " + key},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - server chosen by the operator
        return json.loads(resp.read().decode() or "{}")


def deliver(normalised: dict, config: HookConfig, project: str, transport: Transport, ts: Optional[str] = None) -> dict:
    """Capture first (idempotent by content), then the event that links to it."""
    event = build_event(normalised, project, config.feed)
    if config.capture and config.feed == "full" and normalised.get("event") in ("prompt", "response") and event.get("text"):
        added = transport("/v1/episodes", {
            "content": event["text"], "kind": "conversation", "created_at": ts or now_rfc3339(),
            "source": f"{config.agent}://{normalised['session_id']}",
            "metadata": {"agent": config.agent, "session_id": normalised["session_id"][:256], "project": project},
        }, config.key)
        if isinstance(added, dict) and added.get("episode_id") is not None:
            event["episode_id"] = int(added["episode_id"])
    return transport("/v1/events", {"kind": "agent", "payload": event}, config.key)


def run_hook(
    argv: Sequence[str], stdin_text: str, env: Mapping[str, str], transport: Optional[Transport] = None, stdout=None
) -> int:
    """Everything that can go wrong exits 0: the hook observes, it never
    gets in the agent's way. Stdout is always one JSON object, because the
    host parses it: {} for an observation, or the compiled prompt context
    for UserPromptSubmit (SCONE_HOOK_COMPILE=1). Diagnostics name only
    the exception type, so a malformed payload cannot echo secrets."""
    out = stdout or sys.stdout
    debug = env.get("SCONE_HOOK_DEBUG") == "1"
    emitted = False
    try:
        args = _parser().parse_args(argv)
        payload = json.loads(stdin_text or "{}")
        if env.get("SCONE_HOOK_COMPILE", "1") == "1" and payload.get("hook_event_name") == "UserPromptSubmit":
            request = payload.get("user_input", payload.get("prompt"))
            if isinstance(request, str) and request.strip():
                from .prompting import hook_output

                print(json.dumps(hook_output(request), ensure_ascii=False), file=out)
                emitted = True
        agent = args.agent or env.get("SCONE_HOOK_AGENT") or "claude-code"
        server = args.server or env.get("SCONE_HOOK_SERVER") or "http://127.0.0.1:7437"
        key_env = args.key_env or env.get("SCONE_HOOK_KEY_ENV") or "SCONE_API_KEY"
        key = env.get(key_env, "")
        feed = args.feed or env.get("SCONE_HOOK_FEED") or "metadata"
        projects = parse_projects(args.projects or env.get("SCONE_HOOK_PROJECTS") or "")
        capture = args.capture or env.get("SCONE_HOOK_CAPTURE") == "1"
        if feed not in ("metadata", "full"):
            raise ValueError("feed must be metadata or full")
        if not key:
            raise ValueError(f"no key in ${key_env}")
        if not projects:
            raise ValueError("no projects allowlisted; nothing is sent")
        normalised = normalise_claude(payload) if agent == "claude-code" else normalise_codex(payload)
        if normalised is None or not normalised.get("session_id"):
            return 0
        project = project_for(payload.get("cwd"), projects)
        if project is None:
            return 0
        wanted_sessions = env.get("SCONE_HOOK_SESSIONS", "")
        if wanted_sessions and normalised["session_id"] not in {s.strip() for s in wanted_sessions.split(",") if s.strip()}:
            return 0
        config = HookConfig(agent, server, key, feed, projects, capture)
        send = transport or (lambda path, body, k: http_transport(path, body, k, server))
        deliver(normalised, config, project, send)
    except SystemExit:  # argparse error
        if debug:
            print("agent-hook: bad arguments", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - fail open by design
        if debug:
            print(f"agent-hook: {type(e).__name__}", file=sys.stderr)
    finally:
        if not emitted:
            print("{}", file=out)
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scone-memory agent-hook", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", choices=["claude-code", "codex"])
    p.add_argument("--server")
    p.add_argument("--key-env")
    p.add_argument("--space")
    p.add_argument("--feed", choices=["metadata", "full"])
    p.add_argument("--projects")
    p.add_argument("--capture", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_hook(list(sys.argv[1:] if argv is None else argv), sys.stdin.read(), os.environ)
