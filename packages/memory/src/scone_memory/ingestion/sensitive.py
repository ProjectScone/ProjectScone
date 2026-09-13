"""A source that is a credential is refused on the way in, and says so.

`sync` admits `.json`, `.yaml`, `.toml`, `.ini`, `.txt`, `.log` and `.sh`
among many others, and its walk descends every directory. Nothing on that
path looked at a file's name or its bytes, so `credentials.json`, a
`service-account.json`, a `config.toml` holding a token and a `.log` that
printed one went straight into the ledger and the embedding index.
`retrieval/withhold` exists for the way *out*, and it is opt-in; by then
the bytes are stored, embedded and searchable.

Two stages, cheap one first.

**The name.** Decided before a byte is read: a dedicated credential
store as a parent directory; a key or credential filename; or a keyword
that is load-bearing in a short name. Three things are spared on
purpose, because each is a real file the rule would otherwise eat: a
committed template (`.env.example` is the convention for "safe to
share"), a programming-language source file (`token.py` is a module, not
a store), and a keyword buried in a long slug (`token-economics-of-recall`
is a note *about* tokens).

**The content.** Decides what no name can: a private-key block, or a
recognised secret, in the first `SCAN_BYTES`. It uses the same
`SECRET_PATTERNS` the agent feed already scrubs with, so there is one
list of what a secret looks like and not two that drift.

Two rules about the result. Refusal never rewrites a byte -- invariant I1
says a stored chunk is its file unchanged, so a file is taken whole or
not taken. And the scan is bounded, so every result carries how much was
examined and whether the bound was reached: a secret past `SCAN_BYTES`
is genuinely unseen, and a reader told the scan was cut short can weigh
"nothing found" for what it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from posixpath import basename

from ..capture.redact import SECRET_PATTERNS
from .code import BRACE_SUFFIXES, PYTHON_SUFFIXES

#: Bytes of one file examined for a secret. Beyond this a secret is not
#: seen, and `Screened.bounded` says so.
SCAN_BYTES = 262_144

#: Directories that exist to hold credentials. Anything beneath one is a
#: credential store's contents, whatever it is called.
_STORE_DIRS = frozenset({".ssh", ".aws", ".gnupg", ".kube", ".azure", ".docker", ".gcloud"})

#: Files that are keys by convention of name.
_KEY_NAME = re.compile(r"^id_(rsa|dsa|ecdsa|ed25519)(\.|$)")
_KEY_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".ppk"})

#: Files that are credential stores by convention of name.
_CREDENTIAL_NAMES = frozenset({"credentials", ".netrc", ".npmrc", ".pypirc", ".htpasswd",
                               ".git-credentials"})
_CREDENTIAL_NAME = re.compile(r"^(credentials\.|service[-_]?account.*\.json$|secrets?\.)")

#: A committed template is the convention for "safe to share".
_TEMPLATE_SUFFIXES = (".example", ".sample", ".template", ".dist")

#: Languages whose files are modules. A keyword in one of these names a
#: thing in the program, not a store; the content stage still reads it.
_SOURCE_SUFFIXES = (frozenset(PYTHON_SUFFIXES) | frozenset(BRACE_SUFFIXES)
                    | frozenset({".rb", ".java", ".kt", ".kts", ".scala", ".swift", ".php", ".c",
                                 ".h", ".cc", ".cpp", ".hpp", ".cs", ".ex", ".exs", ".erl", ".hs",
                                 ".lua", ".pl", ".r"}))
_KEYWORD = re.compile(r"^(credential|secret|passwd|password|private_key|token)s?$")

#: A keyword is load-bearing at the end of a short name or as the whole
#: of it: `api_token` and `db-password` are stores; `token.data` is the
#: `token` module's data, and `token-economics-of-recall` is a topic.
_SHORT_NAME = 3

#: Directories a tool writes for itself. Their filenames describe the
#: modules they cached -- `token` and `secrets` are both in the standard
#: library -- so the name stage is skipped beneath one. The bytes are
#: still screened.
_TOOL_CACHES = frozenset({".mypy_cache", "__pycache__", ".pytest_cache", ".ruff_cache",
                          "node_modules", ".tox", ".venv", "venv", ".nox"})

#: The generic `token = value` pattern was written for redaction, where an
#: over-match costs a few characters. Here it costs the whole file, and
#: over this repository it matched 43 lines of code and prose -- `api_key
#: = os.environ.get`, `token = self._runs._token`, `bearer authentication.`
#: -- so it refuses only when the value is *shaped* like a secret.
_GENERIC = 5
_PLACEHOLDER = re.compile(r"[<>{}$]|changeme|example|your[_-]|(.)\1{7,}", re.I)
_SECRET_LENGTH = 16


@dataclass(frozen=True)
class Screened:
    """What was decided, and how much it took to decide it.

    ``reason`` names the rule -- ``name:…`` when the filename decided
    and nothing was read, ``content:…`` when the bytes did. ``scanned``
    is how many bytes were examined and ``bounded`` whether more existed
    than the scan reached, so that a ``None`` reason can never be read
    as a promise about bytes nobody looked at.
    """

    reason: str | None
    scanned: int
    bounded: bool


def screen(path: str, content: bytes) -> Screened:
    """Decide whether ``path`` with ``content`` is a credential."""
    by_name = _by_name(path)
    if by_name is not None:
        return Screened(by_name, 0, False)
    examined = content[:SCAN_BYTES]
    text = examined.decode("utf-8", errors="replace")
    reason: str | None = None
    if SECRET_PATTERNS[0].search(text):
        reason = "content:private_key"
    elif any(_matches(index, pattern, text) for index, pattern in enumerate(SECRET_PATTERNS) if index):
        reason = "content:secret"
    return Screened(reason, len(examined), len(content) > SCAN_BYTES)


def _matches(index: int, pattern: re.Pattern[str], text: str) -> bool:
    if index != _GENERIC:
        return pattern.search(text) is not None
    return any(_secret_shaped(found.group(3)) for found in pattern.finditer(text))


def _secret_shaped(value: str) -> bool:
    """Letters and digits together, long, and not a name or a placeholder.

    `os.environ.get` and `self._runs._token` are identifiers: no digit.
    `authentication.` is a word: no digit. `<your-password>`, `${SECRET}`,
    `changeme` and `xxxxxxxxxxxxxxxx` are placeholders by their own
    convention. What is left has the shape of something generated.
    """
    if len(value) < _SECRET_LENGTH or _PLACEHOLDER.search(value):
        return False
    return any(ch.isdigit() for ch in value) and any(ch.isalpha() for ch in value)


def _by_name(path: str) -> str | None:
    parts = path.replace("\\", "/").split("/")
    if any(part.lower() in _TOOL_CACHES for part in parts[:-1]):
        return None
    if any(part.lower() in _STORE_DIRS for part in parts[:-1]):
        return "name:credential_store"
    name = basename(path)
    lower = name.lower()
    if any(lower.endswith(suffix) for suffix in _TEMPLATE_SUFFIXES):
        return None
    if lower == ".env" or lower.startswith(".env."):
        return "name:environment_file"
    if _KEY_NAME.match(lower) or _suffix(lower) in _KEY_SUFFIXES:
        return "name:private_key"
    if lower in _CREDENTIAL_NAMES or _CREDENTIAL_NAME.match(lower):
        return "name:credential_file"
    if _suffix(lower) in _SOURCE_SUFFIXES:
        return None
    stem = lower[: -len(_suffix(lower))] if _suffix(lower) else lower
    segments = [segment for segment in re.split(r"[-_. ]+", stem) if segment]
    if segments and len(segments) <= _SHORT_NAME and _KEYWORD.match(segments[-1]):
        return "name:credential_keyword"
    return None


def _suffix(lower: str) -> str:
    dot = lower.rfind(".")
    return lower[dot:] if dot > 0 else ""
