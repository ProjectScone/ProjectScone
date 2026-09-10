"""The hook observes and never gets in the way."""

from __future__ import annotations

import json
import subprocess
import sys

from scone_memory.capture.agent_hook import build_event, normalise_claude, normalise_codex, project_for, run_hook

PROJECTS = "scone=/Users/me/ProjectScone,examples=/Users/me/zdeceptron/examples"
ENV = {"SCONE_API_KEY": "k", "SCONE_HOOK_PROJECTS": PROJECTS, "SCONE_HOOK_FEED": "full", "SCONE_HOOK_CAPTURE": "1"}


def test_observer_module_entrypoint_accepts_host_flags_and_returns_valid_hook_json():
    result = subprocess.run([sys.executable, "-m", "scone_memory.capture.agent_hook", "--agent", "codex", "--feed", "metadata"], input="{}", text=True, capture_output=True)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {}


class Recorder:
    def __init__(self, episode_id=41):
        self.calls = []
        self.episode_id = episode_id

    def __call__(self, path, body, key):
        self.calls.append((path, body, key))
        return {"episode_id": self.episode_id} if path == "/v1/episodes" else {"recorded": 7}


def claude(event, **fields):
    return json.dumps({"hook_event_name": event, "session_id": "sess-1", "cwd": "/Users/me/ProjectScone/python", **fields})


def test_claude_events_normalise_to_the_agent_schema():
    prompt = normalise_claude(json.loads(claude("UserPromptSubmit", user_input="hello", prompt_id="p1")))
    assert (prompt["event"], prompt["text"], prompt["source_event_id"]) == ("prompt", "hello", "prompt:p1")
    stop = normalise_claude(json.loads(claude("Stop", last_assistant_message="done", prompt_id="p1")))
    assert (stop["event"], stop["text"], stop["source_event_id"]) == ("response", "done", "stop:p1")
    tool = normalise_claude(json.loads(claude("PostToolUse", tool_name="Bash", tool_use_id="t9", tool_output={"stdout": "x"})))
    assert (tool["event"], tool["tool_name"], tool["ok"], tool["source_event_id"]) == ("tool_result", "Bash", True, "post:t9")
    failed = normalise_claude(json.loads(claude("PostToolUseFailure", tool_name="Bash", tool_use_id="t9")))
    assert failed["ok"] is False
    assert normalise_claude(json.loads(claude("Notification"))) is None


def test_codex_tool_metadata_survives_without_tool_response_text():
    pre = normalise_codex({"hook_event_name": "PreToolUse", "session_id": "c1", "tool_name": "Bash", "tool_use_id": "call-7"})
    post = normalise_codex({"hook_event_name": "PostToolUse", "session_id": "c1", "tool_name": "Bash", "tool_use_id": "call-7"})
    assert pre is not None and post is not None
    assert (pre["event"], pre["source_event_id"], post["event"], post["source_event_id"]) == ("tool_use", "pre:call-7", "tool_result", "post:call-7")
    assert pre["text"] is None and post["text"] is None


def test_codex_events_normalise_best_effort():
    assert normalise_codex({"session_id": "c1", "prompt": "hi", "turn_id": "u1"})["event"] == "prompt"
    assert normalise_codex({"thread_id": "c1", "tool_response": "out", "tool_name": "shell", "call_id": "x"})["event"] == "tool_result"
    assert normalise_codex({"session_id": "c1", "last_assistant_message": "ok", "turn_id": "u1"})["source_event_id"] == "stop:u1"
    assert normalise_codex({"session_id": "c1", "unrelated": 1}) is None


def test_project_allowlist_uses_canonical_names_not_paths():
    projects = {"scone": "/Users/me/ProjectScone", "examples": "/Users/me/zdeceptron/examples"}
    assert project_for("/Users/me/ProjectScone/packages/memory", projects) == "scone"
    assert project_for("/Users/me/zdeceptron/examples", projects) == "examples"
    assert project_for("/Users/me/zdeceptron", projects) is None
    assert project_for("/Users/me/ProjectSconeX", projects) is None
    assert project_for(None, projects) is None


def test_metadata_feed_sends_no_text_and_full_feed_redacts():
    n = normalise_claude(json.loads(claude("UserPromptSubmit", user_input="use token=sk-live-ABCDEFGHIJKLMNOPQRSTUV now", prompt_id="p1")))
    meta = build_event(n, "scone", "metadata")
    assert "text" not in meta and meta["project"] == "scone" and meta["event"] == "prompt"
    full = build_event(n, "scone", "full")
    assert "sk-live-" not in full["text"] and "[redacted]" in full["text"]


def test_capture_posts_the_episode_then_the_linked_event():
    rec = Recorder(episode_id=41)
    code = run_hook([], claude("UserPromptSubmit", user_input="remember the harbour", prompt_id="p1"), ENV, transport=rec)
    assert code == 0
    assert [c[0] for c in rec.calls] == ["/v1/episodes", "/v1/events"]
    episode = rec.calls[0][1]
    assert (episode["kind"], episode["metadata"]["project"], episode["metadata"]["agent"]) == ("conversation", "scone", "claude-code")
    event = rec.calls[1][1]
    assert event["kind"] == "agent" and event["payload"]["episode_id"] == 41 and event["payload"]["source_event_id"] == "prompt:p1"
    assert all(c[2] == "k" for c in rec.calls)


def test_hook_is_silent_outside_the_allowlist_and_fails_open():
    rec = Recorder()
    outside = json.dumps({"hook_event_name": "UserPromptSubmit", "session_id": "s", "cwd": "/tmp/elsewhere", "user_input": "x"})
    assert run_hook([], outside, ENV, transport=rec) == 0 and rec.calls == []
    assert run_hook([], "not json at all", ENV, transport=rec) == 0
    assert run_hook(["--feed", "loud"], claude("Stop"), ENV, transport=rec) == 0
    assert run_hook([], claude("Stop", last_assistant_message="x"), {**ENV, "SCONE_API_KEY": ""}, transport=rec) == 0
    assert run_hook([], claude("Stop", last_assistant_message="x"), {k: v for k, v in ENV.items() if k != "SCONE_HOOK_PROJECTS"}, transport=rec) == 0
    assert rec.calls == []

    def exploding(path, body, key):
        raise ConnectionError("server down")

    assert run_hook([], claude("Stop", last_assistant_message="x"), ENV, transport=exploding) == 0


def test_metadata_default_without_capture_posts_one_event_only():
    rec = Recorder()
    env = {"SCONE_API_KEY": "k", "SCONE_HOOK_PROJECTS": PROJECTS}
    assert run_hook([], claude("PostToolUse", tool_name="Read", tool_use_id="t1", tool_output="secret sk-live-ABCDEFGHIJKLMNOPQRSTUV"), env, transport=rec) == 0
    assert [c[0] for c in rec.calls] == ["/v1/events"]
    payload = rec.calls[0][1]["payload"]
    assert "text" not in payload and payload["tool_name"] == "Read" and payload["event"] == "tool_result"


def test_the_hook_imports_nothing_heavy():
    """The hook runs on every prompt and tool call and the host waits on it
    with a timeout of a few seconds. Under load a pydantic import alone can
    take that long, so the hook's import chain must be standard library
    plus the two small scone modules it uses."""
    import subprocess
    import sys

    probe = (
        "import sys; import scone_memory.capture.agent_hook, scone_memory.capture.prompting; "
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in ('pydantic', 'fastapi', 'httpx', 'fastembed', 'qdrant_client', 'pymongo', 'numpy')); "
        "scone = sorted(m for m in sys.modules if m.startswith('scone_memory')); "
        "print(repr(heavy)); print(repr(scone))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout.splitlines()
    heavy, scone = eval(out[0]), eval(out[1])  # noqa: S307 - our own repr output
    assert heavy == [], f"the hook dragged in {heavy}"
    assert set(scone) <= {"scone_memory", "scone_memory.capture", "scone_memory.capture.agent_hook", "scone_memory.capture.prompting", "scone_memory.capture.redact"}, scone
