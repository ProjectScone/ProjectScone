"""The prompt compiler must agree with the shared fixture byte for byte."""

from __future__ import annotations

import io
import json
import pathlib

from scone_memory import agent_hook
from scone_memory.prompting import CONTEXT_PREFIX, INSTRUCTIONS, additional_context, clean_request, compile_payload, hook_output

FIXTURE = pathlib.Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "prompt-contract.json"


def load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_prefix_and_instructions_match_the_shared_fixture():
    fx = load()
    assert CONTEXT_PREFIX == fx["context_prefix"]
    assert list(INSTRUCTIONS) == fx["instructions"]
    assert compile_payload("x")["schema_version"] == fx["schema_version"]


def test_every_fixture_case_cleans_to_the_expected_task():
    for case in load()["cases"]:
        assert clean_request(case["input"]) == case["task"], case
        payload = compile_payload(case["input"])
        assert set(payload) == {"schema_version", "task", "instructions"}, "payload carries no copy of the original"
        assert payload["task"] == case["task"]


def test_additional_context_is_prefix_plus_compact_json():
    ctx = additional_context("  hello\n")
    assert ctx.startswith(CONTEXT_PREFIX)
    body = json.loads(ctx[len(CONTEXT_PREFIX):])
    assert body == {"schema_version": 1, "task": "hello", "instructions": list(INSTRUCTIONS)}
    assert "\n" not in ctx[len(CONTEXT_PREFIX):], "compact JSON, one line"
    assert json.loads(json.dumps(hook_output("x")))["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"


def test_the_hook_emits_compiled_context_for_prompts_and_empty_json_otherwise():
    env = {"SCONE_API_KEY": "k", "SCONE_HOOK_PROJECTS": "scone=/Users/me/ProjectScone"}
    calls = []
    transport = lambda path, body, key: calls.append(path) or {"recorded": 1}  # noqa: E731
    out = io.StringIO()
    prompt = json.dumps({"hook_event_name": "UserPromptSubmit", "session_id": "s", "cwd": "/Users/me/ProjectScone", "user_input": "  make the UI useful\r\n"})
    assert agent_hook.run_hook([], prompt, env, transport=transport, stdout=out) == 0
    emitted = json.loads(out.getvalue())
    assert emitted["hookSpecificOutput"]["additionalContext"] == additional_context("  make the UI useful\r\n")
    assert calls == ["/v1/events"], "observation still posted, metadata only"

    out = io.StringIO()
    stop = json.dumps({"hook_event_name": "Stop", "session_id": "s", "cwd": "/Users/me/ProjectScone", "last_assistant_message": "done"})
    assert agent_hook.run_hook([], stop, env, transport=transport, stdout=out) == 0
    assert out.getvalue().strip() == "{}"

    out = io.StringIO()
    assert agent_hook.run_hook([], "garbage", env, transport=transport, stdout=out) == 0
    assert out.getvalue().strip() == "{}", "even on failure the host gets valid JSON"

    out = io.StringIO()
    outside = json.dumps({"hook_event_name": "UserPromptSubmit", "session_id": "s", "cwd": "/tmp/other", "user_input": "hi"})
    n = len(calls)
    assert agent_hook.run_hook([], outside, env, transport=transport, stdout=out) == 0
    assert json.loads(out.getvalue())["hookSpecificOutput"], "compiles locally for any project"
    assert len(calls) == n, "but logs nothing outside the allowlist"


def test_older_hook_field_names_are_accepted():
    n = agent_hook.normalise_claude({"hook_event_name": "UserPromptSubmit", "session_id": "s", "prompt": "old field"})
    assert n["text"] == "old field"
    n = agent_hook.normalise_claude({"hook_event_name": "PostToolUse", "session_id": "s", "tool_name": "Bash", "tool_response": {"out": 1}})
    assert n["text"] == {"out": 1}


def test_clip_cuts_on_utf8_bytes_and_flags_it():
    text, cut = agent_hook.clip("🥐" * 20_000)  # 80,000 bytes
    assert cut and len(text.encode()) <= agent_hook.MAX_TEXT_BYTES and text.endswith("🥐")
    text, cut = agent_hook.clip("short")
    assert (text, cut) == ("short", False)
    n = agent_hook.normalise_claude({"hook_event_name": "Stop", "session_id": "s", "last_assistant_message": "🥐" * 20_000})
    event = agent_hook.build_event(n, "scone", "full")
    assert event["text_truncated"] is True and len(event["text"].encode()) <= agent_hook.MAX_TEXT_BYTES


def test_session_gate_limits_which_sessions_are_logged():
    env = {"SCONE_API_KEY": "k", "SCONE_HOOK_PROJECTS": "scone=/Users/me/ProjectScone", "SCONE_HOOK_SESSIONS": "keep-me"}
    calls = []
    transport = lambda path, body, key: calls.append(body) or {"recorded": 1}  # noqa: E731
    for sid in ("keep-me", "someone-else"):
        payload = json.dumps({"hook_event_name": "Stop", "session_id": sid, "cwd": "/Users/me/ProjectScone", "last_assistant_message": "x"})
        agent_hook.run_hook([], payload, env, transport=transport, stdout=io.StringIO())
    assert [c["payload"]["session_id"] for c in calls] == ["keep-me"]
