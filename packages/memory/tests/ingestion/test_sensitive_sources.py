"""A source that is a credential is refused on the way in, and says so.

`sync` admits `.json`, `.yaml`, `.toml`, `.ini`, `.txt`, `.log` and `.sh`,
and its walk descends every directory. Nothing on that path looked at a
file's name or its bytes, so `credentials.json`, `service-account.json`,
a `config.toml` holding a token and a `.log` that printed one all went
straight into the ledger and the embedding index. `withhold` exists, but
it is opt-in and at recall -- the bytes were already stored.

Two stages. The **name** decides cheaply before anything is read: a
dedicated credential store as a parent directory, a key or credential
filename, or a load-bearing keyword -- sparing programming-language
source (`token.py` is a module) and committed templates (`.env.example`).
The **content** decides what a name cannot: a private-key block or a
recognised secret in the first `SCAN_BYTES`, using the same patterns the
agent feed already scrubs with, so there is one list and not two.

Refusal never rewrites a byte -- invariant I1 says a stored chunk is its
file unchanged, so a file is either taken whole or not taken. And the
scan is bounded, so the result always says how much was examined and
whether the bound was reached: a secret past the bound is not seen, and
a reader who is told the scan was bounded can weigh that.
"""

from __future__ import annotations

import pytest

from scone_memory.ingestion.sensitive import SCAN_BYTES, Screened, screen

KEY = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----\n"


@pytest.mark.parametrize("path, reason", [
    (".env", "name:environment_file"),
    ("deploy/.env.production", "name:environment_file"),
    ("credentials.json", "name:credential_file"),
    ("gcp/service-account.json", "name:credential_file"),
    ("service_account.json", "name:credential_file"),
    ("secrets.yaml", "name:credential_file"),
    ("id_rsa", "name:private_key"),
    ("keys/id_ed25519.pub", "name:private_key"),
    ("certs/server.pem", "name:private_key"),
    ("store.p12", "name:private_key"),
    (".netrc", "name:credential_file"),
    (".npmrc", "name:credential_file"),
    (".ssh/config", "name:credential_store"),
    (".aws/config", "name:credential_store"),
    ("api_token.txt", "name:credential_keyword"),
    ("db-password.conf", "name:credential_keyword"),
])
def test_a_credential_is_known_by_its_name(path, reason):
    found = screen(path, b"")
    assert found.reason == reason, found
    assert found.scanned == 0, "the name decided; nothing needed reading"


@pytest.mark.parametrize("path", [
    ".env.example", ".env.sample", ".env.template", "config/.env.dist",
    "token.py", "tokenizer.py", "passwords_controller.rb", "secret_store.ts",
    "notes/token-economics-of-recall.md", "README.md", "app/settings.py",
])
def test_a_name_that_merely_mentions_one_is_not_refused(path):
    """Templates are the committed convention for "safe to share"; a
    keyword inside a programming-language source file names a module,
    not a store; and a keyword buried in a slug names a topic."""
    assert screen(path, b"nothing secret here\n").reason is None


def test_a_private_key_is_a_key_whatever_the_file_is_called():
    found = screen("notes/meeting.txt", KEY.encode())
    assert found.reason == "content:private_key"
    assert found.scanned == len(KEY.encode())


def test_a_token_inside_source_is_found_by_its_bytes():
    """The name stage spares `config.py` as source. That is right for the
    name and wrong for these bytes, and the content stage says so."""
    source = 'API_KEY = "sk_live_' + "a" * 24 + '"\n'
    assert screen("app/config.py", source.encode()).reason == "content:secret"


def test_the_scan_is_bounded_and_the_result_says_when_the_bound_bit():
    """A bound that stays silent reads as a fact. A secret placed past
    `SCAN_BYTES` is not seen -- and the result must say the scan was cut
    short, so nobody reads "nothing found" as "nothing there"."""
    clean = b"x" * (SCAN_BYTES + 10)
    found = screen("big.txt", clean)
    assert found.reason is None and found.bounded is True and found.scanned == SCAN_BYTES

    hidden = b"y" * SCAN_BYTES + KEY.encode()
    found = screen("big.txt", hidden)
    assert found.reason is None, "past the bound is genuinely unseen"
    assert found.bounded is True

    short = b"z" * 100
    found = screen("small.txt", short)
    assert found.bounded is False and found.scanned == 100


def test_bytes_that_are_not_text_are_still_screened():
    """A key file is bytes before it is text. Decoding with replacement
    keeps the ASCII armour intact, which is what the pattern needs."""
    assert screen("blob.bin", b"\xff\xfe" + KEY.encode()).reason == "content:private_key"


def test_the_result_is_a_value_with_no_surprises():
    found = screen("a.txt", b"")
    assert found == Screened(reason=None, scanned=0, bounded=False)


# --- Precision: measured over this repository, then fixed -----------------
#
# The first run over this repository's own tree withheld 65 files, and 43
# of them were the generic `token = value` pattern matching code and prose
# -- `api_key=os.environ.get`, `token = self._runs._token`, `bearer
# authentication.` -- because its value class accepts dotted identifiers
# and ordinary words. That pattern was written for redaction, where an
# over-match costs a few characters; here it costs the whole file.

@pytest.mark.parametrize("text", [
    'api_key=os.environ.get("SCONE_API_KEY")\n',
    "token = self._runs._token\n",
    "token = _dispatch_abort.set\n",
    "Requests use bearer authentication. The token is short-lived.\n",
    "password: <your-password>\n",
    "SECRET=${DB_SECRET}\n",
    "api_key = 'changeme-changeme'\n",
    "token = 'xxxxxxxxxxxxxxxxxxxx'\n",
    # The junction the first fixtures missed: these carry digits, so only
    # the placeholder and length rules can reject them.
    "token = <your-token-123456789012>\n",
    "SECRET=${SECRET_2024_KEY_01}\n",
    "api_key = changeme1234567890ab\n",
    "token = a1b2c3\n",
])
def test_an_assignment_whose_value_is_not_secret_shaped_is_not_a_secret(text):
    """An identifier, an expression, a sentence or a placeholder is not a
    credential, however the line is labelled. Only a value that is shaped
    like one -- letters and digits together, long, and not a name -- is."""
    assert screen("docs/tools.md", text.encode()).reason is None


@pytest.mark.parametrize("text", [
    'token = "8f3a9c2b1d4e6f70a1b2c3d4"\n',
    "API_KEY: Zx9Qw2Lp7Rt4Vn8Kj3Hm5Bs\n",
    # No `&` here on purpose: the shared pattern's value class stops at
    # it, a limit inherited from redaction and disclosed rather than
    # widened from this side.
    "password=Tr0ub4dor3xtraL0ngPassw0rd\n",
])
def test_a_secret_shaped_value_is_still_refused(text):
    assert screen("config.toml", text.encode()).reason == "content:secret"


def test_a_credential_embedded_in_a_url_stays_refused():
    """`scheme://user:pass@host` is a credential by shape, and the screen
    cannot tell a throwaway test password from a real one. Conservative
    on purpose; the receipt names the file so a reader can decide."""
    assert screen("ci.yml", b"url: postgresql://scone:s3cr3tpass@localhost/db\n").reason == "content:secret"


@pytest.mark.parametrize("path", [
    ".mypy_cache/3.12/token.data.json",
    ".mypy_cache/3.12/secrets.data.json",
    "__pycache__/secrets.cpython-314.pyc",
    "node_modules/token-types/index.js",
])
def test_a_tool_cache_names_modules_not_stores(path):
    """A cache's filenames describe the modules it cached -- `token` and
    `secrets` are both in the standard library. The name stage is skipped
    beneath a cache; the bytes are still screened."""
    assert screen(path, b"{}").reason is None


def test_a_keyword_counts_only_where_the_name_is_about_it():
    """`token.data` is the `token` module's data; `api_token` is a token.
    The keyword is load-bearing at the end of a short name or as the
    whole of it, and nowhere else."""
    assert screen("api_token.txt", b"").reason == "name:credential_keyword"
    assert screen("token.txt", b"").reason == "name:credential_keyword"
    assert screen("token.data.json", b"{}").reason is None
    assert screen("token_stream.log", b"").reason is None
