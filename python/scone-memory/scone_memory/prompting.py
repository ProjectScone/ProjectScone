"""Deterministic prompt compiler for the UserPromptSubmit hook.

Every prompt a person sends to a host agent passes through here, locally,
with no model call: the request is trimmed of outer ASCII whitespace
(space, tab, CR, LF only, so non-breaking spaces and inner code spacing
survive) and wrapped in a small JSON payload the host receives as
additional context. The wrapper says plainly that it is user-level data
and grants no authority. The fixture at tests/fixtures/prompt-contract.json
is shared with the Rust compiler; both must agree byte for byte.
"""

from __future__ import annotations

import json
from typing import Mapping

SCHEMA_VERSION = 1
CONTEXT_PREFIX = (
    "Scone structured request (user-level data, not additional authority). "
    "The original user request remains authoritative; this representation does not "
    "change permissions or override higher-priority instructions.\n"
)
INSTRUCTIONS = (
    "Fulfill the user's request without inventing unstated requirements.",
    "Preserve the meaning of quoted text, code, and explicit constraints.",
    "State consequential uncertainty; respect existing permission boundaries.",
)
_OUTER = " \t\r\n"
#: Above this the request passes through untouched and the host is told
#: so; a structured copy of a huge prompt would double its size. Same
#: limit as the Rust compiler.
MAX_TASK_BYTES = 60_000
SKIPPED_NOTICE = (
    "Scone prompt processing skipped: request exceeds the 60000-byte limit; the original prompt is unchanged."
)


def clean_request(request: str) -> str:
    """Outer ASCII whitespace only. Nothing inside the text changes."""
    return request.strip(_OUTER)


def compile_payload(request: str) -> dict:
    return {"schema_version": SCHEMA_VERSION, "task": clean_request(request), "instructions": list(INSTRUCTIONS)}


def additional_context(request: str) -> str:
    """What the hook hands the host: prefix plus the payload as compact JSON."""
    # Keys sorted, so the bytes match the Rust compiler (serde_json sorts);
    # consumers must still compare as JSON, not as text.
    return CONTEXT_PREFIX + json.dumps(compile_payload(request), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def hook_output(request: str) -> Mapping[str, object]:
    """The stdout JSON for a UserPromptSubmit hook. Three outcomes, the
    same as the Rust compiler: an empty request (after outer trim) gets
    {}, an oversized one gets a systemMessage notice and no context, and
    everything else gets the compiled context."""
    task = clean_request(request)
    if not task:
        return {}
    if len(task.encode()) > MAX_TASK_BYTES:
        return {"systemMessage": SKIPPED_NOTICE}
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional_context(request),
        }
    }
