"""What a project's MCP configuration says its agents run, read as claims.

An MCP configuration is where a project writes down the tool servers it
hands its agents: ``.mcp.json`` (Claude Code), ``claude_desktop_config.json``
(Claude Desktop), ``mcp.json`` (Cursor, Windsurf, VS Code under
``.vscode/``), ``mcp_servers.json``, ``mcp_config.json``,
``cline_mcp_settings.json``, ``.gemini/settings.json`` and Codex's
``.codex/config.toml``. Left unread, the graph knew a project's packages
and imports and nothing about the servers its agents talk to, what those
servers run on, or which variables must be set before they start. Read
here, a configuration makes claims the way a manifest does -- quoted from
the file, cited to it, extracted rather than stated:

- ``defines``: the file defines each server it configures, named by the
  file and the server's key (``.mcp.json:filesystem``), so two files that
  both configure a ``filesystem`` stay two things.
- ``runs_with``: the executable a local server starts with, by its base
  name (``npx``, ``uvx``, ``docker``, ``node``), so "everything that runs
  through docker" is one question.
- ``depends_on``: the package the server runs, when the executable says
  which index it comes from. The first positional argument of ``npx``,
  ``bunx`` or ``pnpx`` is an npm package; of ``uvx`` or ``pipx`` a PyPI
  distribution (``--from`` and ``--with`` name distributions too). It is
  spelled as its index spells it, so a server and a ``package.json`` or
  ``pyproject.toml`` that name one package meet at one entity. A docker
  image is left out: the ``run`` line's flags cannot be told from the
  image without knowing every flag, and a guess would name the wrong
  thing.
- ``requires_env``: the environment variables a server needs, by name and
  never by value: the keys of its ``env`` map; a ``${NAME}``,
  ``${NAME:-default}`` or ``${env:NAME}`` reference in its command,
  arguments, URL, headers or env values; a ``-e NAME`` or ``--env NAME``
  a docker run passes through from the host; and Codex's
  ``bearer_token_env_var`` and ``env_http_headers``. The object is
  ``$NAME``, as a shell writes it. A reference in lower or mixed case
  (``${workspaceFolder}``, ``${input:token}``) is a tool's own variable,
  not the environment's, unless it says ``env:``.
- ``connects_to``: the origin (scheme, host and port) of a remote
  server's URL, when the host is written out rather than referenced. A
  user name or password written into the URL is not part of the origin
  and never reaches a claim.

A value is never quoted. What an ``env`` map holds is what a
configuration keeps secret, an argument or header may carry one too, and
a URL's query can; so every claim quotes the token that grounds it -- the
server's key, the command, the package as written, the variable's name,
the URL's host -- and its byte span covers that token alone, found in
the field it came from: a JSON file is scanned for where each key sits,
so a server named ``env``, two servers on one line, or a value that
happens to contain a command's name cannot move a claim onto the wrong
line.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
import re
import tomllib
from typing import Any, Iterator, Optional
from urllib.parse import urlsplit

from .code import MAX_LINES, _line_starts
from .code_graph import DEFINES, MAX_CLAIMS, CodeClaim
from .manifests import DEPENDS_ON, python_name

#: The executable a local server starts with.
RUNS_WITH = "runs_with"
#: An environment variable a server needs set, by name.
REQUIRES_ENV = "requires_env"
#: The origin a remote server is reached at.
CONNECTS_TO = "connects_to"

_JSON_NAMES = frozenset({".mcp.json", "mcp.json", "mcp_servers.json", "mcp_config.json",
                         "claude_desktop_config.json", "cline_mcp_settings.json"})
#: A generic name that is a configuration only under its tool's directory.
_UNDER = {"settings.json": ".gemini", "config.toml": ".codex"}
#: The dot-named entries a walk over a tree keeps although it passes the
#: rest by: the configuration itself, and the tools' directories that
#: hold one. What else those directories hold is judged as any file is.
WALKED_DOT_NAMES = frozenset({".mcp.json", ".vscode", ".cursor", ".gemini", ".codex"})
#: The keys a file keeps its servers under, by tool.
_MAPS = ("mcpServers", "servers", "mcp_servers")
_NPM = frozenset({"npx", "bunx", "pnpx"})
_PYPI = frozenset({"uvx", "pipx"})
#: A `${NAME}`, `${NAME:-default}` or `${env:NAME}` reference.
_REFERENCE = re.compile(r"\$\{(env:)?([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: Flags whose next argument is a value and not a package. A flag not
#: listed here is read as taking none, so an unknown flag that does take
#: one would name its value as the package; the sets are the tools' own.
_NPM_VALUED = frozenset({"--loglevel", "--registry", "--cache", "--userconfig", "--shell", "-c", "--node-arg", "-n",
                         "--call", "--prefix", "--workspace", "-w", "--script-shell", "--node-options"})
_PYPI_VALUED = frozenset({"--python", "-p", "--index", "--index-url", "--extra-index-url", "--constraint", "-c",
                          "--exclude-newer", "--cache-dir", "--directory", "--project", "--env-file", "--spec",
                          "--with-requirements", "--with-editable", "--refresh-package", "--find-links", "-f",
                          "--index-strategy", "--python-preference", "--resolution", "--link-mode", "--config-file",
                          "--prerelease", "--keyring-provider", "--reinstall-package", "--upgrade-package",
                          "--python-platform", "--no-binary-package", "--only-binary-package", "--color", "--pip-args"})
_TOML_HEADER = re.compile(r"^\s*\[\s*mcp_servers\.(.+?)\s*\]\s*(?:#.*)?$")
_TOML_FIELD = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*=", re.MULTILINE)
#: A key of an inline table or of a sub-table's line: `{ KEY = ` or `, KEY = ` or a line's own `KEY = `.
_TOML_KEY = re.compile(r'(?:^|[{,])\s*("?)([A-Za-z0-9_-]+)\1\s*=', re.MULTILINE)


def is_mcp_config(path: str) -> bool:
    """Known by its name wherever it sits, or by its name under its
    tool's directory (``.gemini/settings.json``, ``.codex/config.toml``)."""
    if not path:
        return False
    parts = path.replace("\\", "/").lower().rsplit("/", 2)
    name = parts[-1]
    if name in _JSON_NAMES:
        return True
    return len(parts) > 1 and _UNDER.get(name) == parts[-2]


Span = tuple[int, int]


class _Claims:
    """Claims about one configuration, each quoting the token it rests on,
    placed by character offsets into the text."""

    def __init__(self, content: str, path: str) -> None:
        self.path = path
        self.content = content
        self.starts = _line_starts(content)
        self.found: list[CodeClaim] = []
        self.made: set[tuple[str, str, str]] = set()

    def at(self, subject: str, predicate: str, obj: str, offset: int, quote: str) -> None:
        """Claim ``subject predicate obj`` quoted from ``quote`` at ``offset``."""
        if len(self.found) >= MAX_CLAIMS or (subject, predicate, obj) in self.made or offset < 0:
            return
        line = bisect_right(self.starts, offset)
        begins = len(self.content[:offset].encode())
        self.found.append(CodeClaim(subject, predicate, obj, quote, line, begins, begins + len(quote.encode())))
        self.made.add((subject, predicate, obj))

    def say(self, subject: str, predicate: str, obj: str, token: str, span: Span, quote: Optional[str] = None) -> None:
        """Claim quoted from the first ``token`` within ``span``, or from
        ``quote`` inside that token when the grounding text is shorter."""
        start, stop = span
        offset = self.content.find(token, start, stop)
        if offset < 0:
            return
        shown = quote if quote is not None else token
        if shown != token:
            offset = self.content.find(shown, offset, offset + len(token))
        self.at(subject, predicate, obj, offset, shown)


def _strings(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _references(text: str) -> Iterator[tuple[str, str]]:
    """Every environment variable a string references, with the token as
    written: capitals name the environment; anything else is a tool's
    own variable unless it says ``env:``."""
    for found in _REFERENCE.finditer(text):
        explicit, name = found.group(1), found.group(2)
        if explicit or name == name.upper():
            yield name, found.group(0)


def _npm_package(spec: str) -> Optional[str]:
    """The npm package a spec names, its version dropped: ``name``,
    ``name@1.2``, ``@scope/name``, ``@scope/name@latest``. None for a
    path, a URL or a reference the index would not know."""
    if not spec or "${" in spec or "://" in spec or spec[0] in "./~":
        return None
    cut = spec.rfind("@")
    name = spec[:cut] if cut > 0 else spec
    scoped = name.startswith("@")
    if not name or (scoped and name.count("/") != 1) or (not scoped and "/" in name):
        return None
    return name.lower()


def _pypi_package(spec: str) -> Optional[str]:
    """The distribution a requirement names, as the index spells it."""
    if "${" in spec or "/" in spec:
        return None
    return python_name(spec)


def _packages(command: str, args: list[str]) -> Iterator[tuple[str, str]]:
    """The packages the arguments name, given what the command fetches
    from, each with the argument that spells it."""
    if command in _NPM:
        # `npx [-y] [--package=X | -p X] <package>[@version] ...`: the
        # package is the first positional, unless a flag names it.
        expecting: Optional[str] = None
        for arg in args:
            if expecting is not None:
                flag, expecting = expecting, None
                if flag in ("--package", "-p"):
                    package = _npm_package(arg)
                    if package:
                        yield package, arg
                    return
                continue
            if arg.startswith("--package=") or arg.startswith("-p="):
                package = _npm_package(arg.split("=", 1)[1])
                if package:
                    yield package, arg
                return
            if arg in ("--package", "-p", *_NPM_VALUED):
                expecting = arg
                continue
            if arg == "--":
                continue
            if arg.startswith("-"):
                continue
            package = _npm_package(arg)
            if package:
                yield package, arg
            return
    elif command in _PYPI:
        # `uvx [--from X] [--with Y] <package> ...`, `pipx run <package>`:
        # `--from` and `--with` name distributions; the first positional
        # is one unless `--from` already said which.
        positional_is_package = True
        awaiting: Optional[str] = None
        for arg in args:
            if awaiting is not None:
                flag, awaiting = awaiting, None
                if flag in ("--from", "--with"):
                    package = _pypi_package(arg)
                    if package:
                        yield package, arg
                    positional_is_package = positional_is_package and flag != "--from"
                continue
            if arg in ("--from", "--with", *_PYPI_VALUED):
                awaiting = arg
                continue
            if arg.startswith("--from=") or arg.startswith("--with="):
                flag, spec = arg.split("=", 1)
                package = _pypi_package(spec)
                if package:
                    yield package, arg
                positional_is_package = positional_is_package and flag != "--from"
                continue
            if arg.startswith("-") or arg == "run":
                continue
            if positional_is_package:
                package = _pypi_package(arg)
                if package:
                    yield package, arg
            return


def _docker_env(args: list[str]) -> Iterator[str]:
    """The variables a ``docker run`` passes through from the host: ``-e
    NAME`` and ``--env NAME`` without a value of their own."""
    expecting = False
    for arg in args:
        if expecting:
            expecting = False
            if _ENV_NAME.match(arg):
                yield arg
            continue
        if arg in ("-e", "--env"):
            expecting = True
        elif arg.startswith("--env=") and _ENV_NAME.match(arg[6:]):
            yield arg[6:]


def _origin(url: str) -> Optional[tuple[str, str]]:
    """The scheme, host and port of a URL and the host as written, or
    None when the host is a reference or the URL does not parse. A user
    name or password in the URL is not part of the origin."""
    try:
        parts = urlsplit(url.strip())
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not host or "${" in parts.netloc:
        return None
    written = parts.netloc.rsplit("@", 1)[-1]
    shown = f"[{host}]" if ":" in host else host
    return f"{parts.scheme}://{shown}" + (f":{port}" if port is not None else ""), written


@dataclass(frozen=True)
class _Key:
    """An object key in a JSON text: its name, nesting depth and place."""

    name: str
    depth: int
    offset: int


def _json_keys(text: str) -> list[_Key]:
    """Every object key in a JSON text that parses, by a scan that follows
    strings and their escapes, so a key is known by where it sits and
    not by the first line that happens to spell it."""
    keys: list[_Key] = []
    depth, index, length = 0, 0, len(text)
    while index < length:
        char = text[index]
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        elif char == '"':
            start = index
            index += 1
            while index < length and text[index] != '"':
                index += 2 if text[index] == "\\" else 1
            raw = text[start + 1:index]
            index += 1
            after = index
            while after < length and text[after] in " \t\r\n":
                after += 1
            if after < length and text[after] == ":":
                try:
                    name = json.loads('"' + raw + '"')
                except ValueError:
                    name = raw
                keys.append(_Key(str(name), depth, start))
            continue
        index += 1
    return keys


def _server(text: _Claims, server: str, spec: dict[str, Any], fields: dict[str, Span], env_keys: list[tuple[str, int]],
            whole: Span, *, quoted: str) -> None:
    """The claims one server's block makes: each grounded in the field it
    came from (``fields`` maps a field's name to its span), an ``env``
    key at the place the scan found it."""
    subject = f"{text.path}:{server}"

    def field(name: str) -> Span:
        return fields.get(name, (whole[1], whole[1]))

    def token(name: str) -> str:
        return quoted.format(name)

    command = spec.get("command")
    args = _strings(spec.get("args"))
    if isinstance(command, str) and command.strip():
        for name, reference in _references(command):
            text.say(subject, REQUIRES_ENV, f"${name}", reference, field("command"))
        if "${" not in command:
            executable = command.strip().replace("\\", "/").rsplit("/", 1)[-1]
            text.say(subject, RUNS_WITH, executable.lower(), token(command), field("command"), quote=executable)
            for package, spelled in _packages(executable.lower(), args):
                text.say(subject, DEPENDS_ON, package, token(spelled), field("args"), quote=spelled)
            if executable.lower() == "docker":
                for name in _docker_env(args):
                    text.say(subject, REQUIRES_ENV, f"${name}", token(name), field("args"))
    for arg in args:
        for name, reference in _references(arg):
            text.say(subject, REQUIRES_ENV, f"${name}", reference, field("args"))
    env = spec.get("env")
    if isinstance(env, dict):
        placed = dict(env_keys)
        for key, value in env.items():
            if isinstance(key, str) and _ENV_NAME.match(key):
                if key in placed:
                    text.at(subject, REQUIRES_ENV, f"${key}", placed[key], token(key))
                else:
                    text.say(subject, REQUIRES_ENV, f"${key}", token(key), field("env"))
            if isinstance(value, str):
                for name, reference in _references(value):
                    text.say(subject, REQUIRES_ENV, f"${name}", reference, field("env"))
    url = spec.get("url")
    if isinstance(url, str):
        for name, reference in _references(url):
            text.say(subject, REQUIRES_ENV, f"${name}", reference, field("url"))
        origin = _origin(url)
        if origin:
            text.say(subject, CONNECTS_TO, origin[0], origin[1], field("url"))
    for header_map in ("headers", "http_headers"):
        headers = spec.get(header_map)
        if isinstance(headers, dict):
            for value in headers.values():
                if isinstance(value, str):
                    for name, reference in _references(value):
                        text.say(subject, REQUIRES_ENV, f"${name}", reference, field(header_map))
    bearer = spec.get("bearer_token_env_var")
    if isinstance(bearer, str) and _ENV_NAME.match(bearer):
        text.say(subject, REQUIRES_ENV, f"${bearer}", token(bearer), field("bearer_token_env_var"))
    env_headers = spec.get("env_http_headers")
    if isinstance(env_headers, dict):
        for value in env_headers.values():
            if isinstance(value, str) and _ENV_NAME.match(value):
                text.say(subject, REQUIRES_ENV, f"${value}", token(value), field("env_http_headers"))


def _servers(data: object) -> Optional[tuple[str, dict[str, Any]]]:
    if not isinstance(data, dict):
        return None
    for key in _MAPS:
        servers = data.get(key)
        if isinstance(servers, dict):
            return key, servers
    return None


def _json_claims(content: str, path: str) -> tuple[CodeClaim, ...]:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, RecursionError):
        return ()
    found = _servers(data)
    if not found:
        return ()
    map_name, servers = found
    keys = _json_keys(content)
    top = [key for key in keys if key.depth == 1]
    map_key = next((key for key in top if key.name == map_name), None)
    if map_key is None:
        return ()
    later = [key.offset for key in top if key.offset > map_key.offset]
    map_end = later[0] if later else len(content)
    server_keys = [key for key in keys if key.depth == 2 and map_key.offset < key.offset < map_end]
    text = _Claims(content, path)
    for index, key in enumerate(server_keys):
        spec = servers.get(key.name)
        if not key.name.strip() or not isinstance(spec, dict):
            continue
        end = server_keys[index + 1].offset if index + 1 < len(server_keys) else map_end
        field_keys = [k for k in keys if k.depth == 3 and key.offset < k.offset < end]
        fields: dict[str, Span] = {}
        for position, field_key in enumerate(field_keys):
            stop = field_keys[position + 1].offset if position + 1 < len(field_keys) else end
            fields.setdefault(field_key.name, (field_key.offset, stop))
        env_span = fields.get("env")
        env_keys = [(k.name, k.offset) for k in keys if env_span and k.depth == 4 and env_span[0] < k.offset < env_span[1]]
        text.at(path, DEFINES, f"{path}:{key.name}", key.offset, content[key.offset:content.find('"', key.offset + 1) + 1])
        _server(text, key.name, spec, fields, env_keys, (key.offset, end), quoted='"{}"')
    return tuple(text.found)


def _toml_claims(content: str, path: str) -> tuple[CodeClaim, ...]:
    try:
        data = tomllib.loads(content)
    except (tomllib.TOMLDecodeError, RecursionError):
        return ()
    servers = data.get("mcp_servers")
    if not isinstance(servers, dict) or not servers:
        return ()
    text = _Claims(content, path)
    starts = text.starts
    headers = [(starts[index], found.group(1).strip().strip("'\""))
               for index, line in enumerate(content.split("\n")) if (found := _TOML_HEADER.match(line))]
    for server, spec in servers.items():
        if not isinstance(server, str) or not isinstance(spec, dict):
            continue
        # A server's text runs from its own header to the first header of
        # another server; its sub-tables (`[mcp_servers.git.env]`) stay inside.
        own = [offset for offset, name in headers if name == server or name.startswith(server + ".")]
        if not own:
            continue
        start = min(own)
        later = [offset for offset, name in headers if offset > start and not (name == server or name.startswith(server + "."))]
        stop = min(later) if later else len(content)
        fields: dict[str, Span] = {}
        placed = [(found.group(1), found.start(1)) for found in _TOML_FIELD.finditer(content, start, stop)]
        for position, (name, offset) in enumerate(placed):
            end = placed[position + 1][1] if position + 1 < len(placed) else stop
            fields.setdefault(name, (offset, end))
        for offset, name in headers:
            if name == server + ".env" and start <= offset < stop:
                # The sub-table form: its lines run to the next header.
                after = [other for other, _ in headers if other > offset]
                fields["env"] = (offset, min(after) if after else stop)
        env_span = fields.get("env")
        env_keys = ([(found.group(2), found.start(2)) for found in _TOML_KEY.finditer(content, env_span[0], env_span[1])
                     if found.group(2) != "env"] if env_span else [])
        text.at(path, DEFINES, f"{path}:{server}", content.find("mcp_servers." + server, start, stop), "mcp_servers." + server)
        _server(text, server, spec, fields, env_keys, (start, stop), quoted="{}")
    return tuple(text.found)


def mcp_config_claims(content: str, path: str) -> tuple[CodeClaim, ...]:
    """What an MCP configuration configures, as claims about it. Empty for
    a file that is not one, does not parse, keeps no servers, or is
    longer than a source file is read."""
    if not content or not is_mcp_config(path) or content.count("\n") > MAX_LINES:
        return ()
    if path.replace("\\", "/").lower().endswith(".toml"):
        return _toml_claims(content, path)
    return _json_claims(content, path)
