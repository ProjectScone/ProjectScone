"""Secret scrubbing shared by the hook and the engine.

Standard library only, on purpose: the hook runs on every prompt and
tool call, so its import must cost nothing beyond Python itself.
"""

from __future__ import annotations

import re

#: Defence in depth for the agent feed: the hook redacts before posting,
#: and the engine scrubs again. Patterns cover the common key shapes;
#: this is a net, not a guarantee, and the docs say so.
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(?:sk|rk|pk)[-_](?:live|test|proj|ant|or)?[-_]?[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bnpg_[A-Za-z0-9]{12,}\b"),
    re.compile(r"(?i)\b(bearer|token|api[_-]?key|secret|password)\b(\s*[:=]\s*|\s+)[\"']?([A-Za-z0-9._\-/+]{12,})"),
    re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^@\s]+@"),
)


def redact_secrets(text: str) -> str:
    """Replace key-shaped substrings with [redacted]. A net, not a proof."""
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 3:
            text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]", text)
        else:
            text = pattern.sub("[redacted]", text)
    return text
