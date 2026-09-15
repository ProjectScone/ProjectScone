"""An MCP configuration says which servers a project hands its agents, what they run on, and what they need."""
import io
import json

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion import mcp_config
from scone_memory.ingestion.code_graph import DEFINES
from scone_memory.ingestion.manifests import DEPENDS_ON, is_manifest, manifest_claims
from scone_memory.ingestion.mcp_config import CONNECTS_TO, REQUIRES_ENV, RUNS_WITH, is_mcp_config, mcp_config_claims

CLAUDE = '''{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem@0.6.2", "${HOME}/docs"]
    },
    "git": {
      "command": "uvx",
      "args": ["--from", "mcp_server_git", "mcp-server-git", "--repository", "."],
      "env": {"GIT_TOKEN": "ghp_not_a_real_value", "GIT_AUTHOR": "${USER:-nobody}"}
    },
    "github": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "GITHUB_TOKEN", "-e", "MODE=readonly", "ghcr.io/github/github-mcp-server"]
    },
    "remote": {
      "type": "http",
      "url": "https://MCP.example.com:8443/sse?key=${REMOTE_KEY}",
      "headers": {"Authorization": "Bearer ${REMOTE_KEY}", "X-Team": "literal-team-value"}
    },
    "tool-var": {
      "command": "${TOOL_HOME}/bin/server",
      "args": ["${workspaceFolder}", "${input:token}"]
    },
    "broken": "not a server"
  }
}
'''

VSCODE = '''{
  "inputs": [{"id": "token", "type": "promptString"}],
  "servers": {
    "docs": {"type": "stdio", "command": "/usr/local/bin/node", "args": ["./build/index.js", "${env:DOCS_ROOT}", "${input:token}"]},
    "search": {"type": "sse", "url": "${env:SEARCH_URL}"}
  }
}
'''

CODEX = '''model = "o3"

[mcp_servers.git]
command = "npx"
args = ["-y", "@scope/git-mcp"]

[mcp_servers.git.env]
GIT_TOKEN = "not-a-real-token"

[mcp_servers.remote]
url = "https://mcp.example.com/mcp"
bearer_token_env_var = "REMOTE_TOKEN"
env_http_headers = { "X-Org" = "ORG_ID" }
'''


def said(claims, predicate=None):
    return [(c.subject, c.predicate, c.object) for c in claims if predicate is None or c.predicate == predicate]


def test_a_claude_code_configuration_says_what_each_server_runs_needs_and_reaches():
    claims = mcp_config_claims(CLAUDE, "app/.mcp.json")
    assert claims and manifest_claims(CLAUDE, "app/.mcp.json") == claims, "a configuration is read wherever a manifest is"
    assert said(claims, DEFINES) == [("app/.mcp.json", DEFINES, f"app/.mcp.json:{name}")
                                     for name in ("filesystem", "git", "github", "remote", "tool-var")], \
        "a server is named by its file and key; an entry that is not an object is no server"
    assert said(claims, RUNS_WITH) == [("app/.mcp.json:filesystem", RUNS_WITH, "npx"), ("app/.mcp.json:git", RUNS_WITH, "uvx"),
                                       ("app/.mcp.json:github", RUNS_WITH, "docker")], \
        "a command that is a reference names no executable"
    assert said(claims, DEPENDS_ON) == [("app/.mcp.json:filesystem", DEPENDS_ON, "@modelcontextprotocol/server-filesystem"),
                                        ("app/.mcp.json:git", DEPENDS_ON, "mcp-server-git")], \
        "the version is dropped, a distribution is spelled as its index spells it, a docker image is not a package"
    assert said(claims, REQUIRES_ENV) == [
        ("app/.mcp.json:filesystem", REQUIRES_ENV, "$HOME"),
        ("app/.mcp.json:git", REQUIRES_ENV, "$GIT_TOKEN"), ("app/.mcp.json:git", REQUIRES_ENV, "$GIT_AUTHOR"),
        ("app/.mcp.json:git", REQUIRES_ENV, "$USER"),
        ("app/.mcp.json:github", REQUIRES_ENV, "$GITHUB_TOKEN"),
        ("app/.mcp.json:remote", REQUIRES_ENV, "$REMOTE_KEY"),
        ("app/.mcp.json:tool-var", REQUIRES_ENV, "$TOOL_HOME"),
    ], "-e NAME=value sets a value and needs nothing; ${workspaceFolder} and ${input:token} are the tool's own; one variable twice is one claim"
    assert said(claims, CONNECTS_TO) == [("app/.mcp.json:remote", CONNECTS_TO, "https://mcp.example.com:8443")]


def test_no_claim_quotes_a_value_and_each_quotes_the_token_that_grounds_it():
    claims = mcp_config_claims(CLAUDE, "app/.mcp.json")
    for claim in claims:
        assert CLAUDE.encode()[claim.start:claim.end].decode() == claim.quote, "the span is the quote's bytes"
        for kept_secret in ("ghp_not_a_real_value", "literal-team-value", "key=", "readonly"):
            assert kept_secret not in claim.quote
    by = {(c.predicate, c.object): c for c in claims}
    assert (by[REQUIRES_ENV, "$GIT_TOKEN"].quote, by[REQUIRES_ENV, "$GIT_TOKEN"].first_line) == ('"GIT_TOKEN"', 10)
    assert by[REQUIRES_ENV, "$USER"].quote == "${USER:-nobody}"
    assert by[DEPENDS_ON, "@modelcontextprotocol/server-filesystem"].quote == "@modelcontextprotocol/server-filesystem@0.6.2"
    assert by[DEPENDS_ON, "mcp-server-git"].quote == "mcp_server_git", "quoted as written, named as the index names it"
    assert by[CONNECTS_TO, "https://mcp.example.com:8443"].quote == "MCP.example.com:8443", "the host as written, never the query"
    assert (by[DEFINES, "app/.mcp.json:git"].quote, by[DEFINES, "app/.mcp.json:git"].first_line) == ('"git"', 7)
    assert by[RUNS_WITH, "npx"].quote == "npx" and by[RUNS_WITH, "docker"].first_line == 13


def test_vs_code_gemini_and_codex_spellings_are_read_and_a_generic_name_only_under_its_tool():
    vscode = mcp_config_claims(VSCODE, ".vscode/mcp.json")
    assert said(vscode, DEFINES) == [(".vscode/mcp.json", DEFINES, ".vscode/mcp.json:docs"), (".vscode/mcp.json", DEFINES, ".vscode/mcp.json:search")]
    assert said(vscode, RUNS_WITH) == [(".vscode/mcp.json:docs", RUNS_WITH, "node")], \
        "an executable is known by its base name; a script path is not a package"
    assert said(vscode, REQUIRES_ENV) == [(".vscode/mcp.json:docs", REQUIRES_ENV, "$DOCS_ROOT"), (".vscode/mcp.json:search", REQUIRES_ENV, "$SEARCH_URL")], \
        "`${env:NAME}` names the environment whatever its case; `${input:token}` is a prompt, not a variable"
    assert said(vscode, CONNECTS_TO) == [] and said(vscode, DEPENDS_ON) == [], "a URL that is a reference reaches no origin the file names"
    gemini = '{"theme": "dark", "mcpServers": {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}}\n'
    assert said(mcp_config_claims(gemini, "home/.gemini/settings.json"), DEPENDS_ON) == \
        [("home/.gemini/settings.json:fetch", DEPENDS_ON, "mcp-server-fetch")], "a one-line file keeps every key on its line"
    assert mcp_config_claims(gemini, "home/settings.json") == () and not is_mcp_config("settings.json")
    codex = mcp_config_claims(CODEX, "home/.codex/config.toml")
    assert said(codex, DEFINES) == [("home/.codex/config.toml", DEFINES, "home/.codex/config.toml:git"),
                                    ("home/.codex/config.toml", DEFINES, "home/.codex/config.toml:remote")]
    assert said(codex, DEPENDS_ON) == [("home/.codex/config.toml:git", DEPENDS_ON, "@scope/git-mcp")]
    assert said(codex, REQUIRES_ENV) == [("home/.codex/config.toml:git", REQUIRES_ENV, "$GIT_TOKEN"),
                                         ("home/.codex/config.toml:remote", REQUIRES_ENV, "$REMOTE_TOKEN"),
                                         ("home/.codex/config.toml:remote", REQUIRES_ENV, "$ORG_ID")], \
        "a sub-table stays with its server; Codex names a variable by value, and the value is the name"
    assert said(codex, CONNECTS_TO) == [("home/.codex/config.toml:remote", CONNECTS_TO, "https://mcp.example.com")]
    token = next(c for c in codex if c.object == "$GIT_TOKEN")
    assert (token.quote, token.first_line, CODEX.encode()[token.start:token.end].decode()) == ("GIT_TOKEN", 8, "GIT_TOKEN")


def test_package_spellings_flags_and_paths():
    def packages(command, *args):
        return [package for package, _ in mcp_config._packages(command, list(args))]

    assert packages("npx", "-y", "server-memory@latest") == ["server-memory"]
    assert packages("npx", "--package=@Scope/cli@2", "cli-cmd") == ["@scope/cli"]
    assert packages("npx", "-p", "@scope/cli", "cli-cmd") == ["@scope/cli"]
    assert packages("npx", "-y", "./local/server.js") == [] and packages("npx", "https://x/y.tgz") == []
    assert packages("bunx", "@scope/pkg") == ["@scope/pkg"] and packages("npx", "${PKG}") == []
    assert packages("uvx", "--python", "3.12", "Mcp_Server.Fetch==1.0") == ["mcp-server-fetch"]
    assert packages("uvx", "--from", "git+https://x/y", "tool") == [], "a VCS spec is not a distribution name"
    assert packages("uvx", "--with", "extra-pkg", "main-pkg") == ["extra-pkg", "main-pkg"]
    assert packages("pipx", "run", "mcp-server-time") == ["mcp-server-time"]
    assert packages("node", "some-package") == [] and packages("python", "-m", "server") == []
    assert packages("npx", "--loglevel", "silent", "some-real-pkg") == ["some-real-pkg"], "a flag's value is not a package"
    assert packages("uvx", "--index-strategy", "unsafe-best-match", "mcp-server-fetch") == ["mcp-server-fetch"]
    assert packages("npx", "-y", "--", "@scope/pkg") == ["@scope/pkg"]
    assert list(mcp_config._docker_env(["run", "-e", "A_B", "--env", "C", "--env=D", "-e", "E=1", "-e", "not valid"])) == ["A_B", "C", "D"]


@pytest.mark.parametrize("path,expected", [
    (".mcp.json", True), ("app/.mcp.json", True), (".cursor/mcp.json", True), (".vscode/MCP.json", True),
    ("mcp_servers.json", True), ("windsurf/mcp_config.json", True), ("claude_desktop_config.json", True),
    ("cline_mcp_settings.json", True), (".gemini/settings.json", True), ("home/.codex/config.toml", True),
    ("settings.json", False), ("app/config.toml", False), ("mcp.jsonc", False), ("package.json", False), ("", False),
])
def test_a_configuration_is_known_by_its_name_and_a_generic_one_by_its_directory(path, expected):
    assert is_mcp_config(path) is expected
    assert is_manifest(path) is (expected or path == "package.json")


@pytest.mark.parametrize("path,content", [
    (".mcp.json", '{"mcpServers": {'), (".mcp.json", '["not", "an", "object"]'), (".mcp.json", '{"mcpServers": []}'),
    ("mcp.json", '{"servers": {}}'), (".codex/config.toml", "[mcp_servers\ncommand = 'x'"),
    (".codex/config.toml", "mcp_servers = 3\n"), (".mcp.json", ""), ("mcp_servers.json", "null"),
    ("mcp.json", '{"mcpServers": {"x": 1, "": {"command": "npx"}}}'),
])
def test_a_configuration_that_does_not_parse_or_keeps_no_server_claims_nothing_and_never_raises(path, content):
    assert mcp_config_claims(content, path) == ()


def test_a_url_with_credentials_keeps_them_out_of_the_origin_and_the_quote():
    line = '{"mcpServers": {"r": {"url": "https://User:hunter2@API.example.com:8443/mcp?key=hunter2"}}}\n'
    claims = mcp_config_claims(line, ".mcp.json")
    [origin] = [c for c in claims if c.predicate == CONNECTS_TO]
    assert (origin.object, origin.quote) == ("https://api.example.com:8443", "API.example.com:8443")
    assert not any("hunter2" in c.quote or "hunter2" in c.object for c in claims)
    assert line.encode()[origin.start:origin.end].decode() == "API.example.com:8443"
    v6 = mcp_config_claims('{"mcpServers": {"r": {"url": "http://[::1]:8000/mcp"}}}\n', ".mcp.json")
    assert [c.object for c in v6 if c.predicate == CONNECTS_TO] == ["http://[::1]:8000"]


def test_claims_are_placed_by_where_each_key_sits_not_by_the_first_line_that_spells_it():
    one_line = '{"mcpServers": {"a": {"command": "npx", "args": ["x"]}, "b": {"command": "uvx", "args": ["y"]}}}\n'
    assert said(mcp_config_claims(one_line, ".mcp.json"), RUNS_WITH) == [(".mcp.json:a", RUNS_WITH, "npx"), (".mcp.json:b", RUNS_WITH, "uvx")], \
        "two servers on one line are two servers"
    named_env = ('{"mcpServers": {\n  "first": {\n    "command": "npx",\n    "args": ["-y", "a"],\n    "env": {"K": "v"}\n  },\n'
                 '  "env": {\n    "command": "uvx",\n    "args": ["b"]\n  }\n}}\n')
    claims = mcp_config_claims(named_env, ".mcp.json")
    assert said(claims, DEFINES) == [(".mcp.json", DEFINES, ".mcp.json:first"), (".mcp.json", DEFINES, ".mcp.json:env")]
    env_defined = next(c for c in claims if c.object == ".mcp.json:env")
    assert env_defined.first_line == 7, "a server named like a field is found where it sits, not at the field"
    assert (".mcp.json:first", REQUIRES_ENV, "$K") in said(claims) and (".mcp.json:env", RUNS_WITH, "uvx") in said(claims)
    value_first = ('{"mcpServers": {"docs": {\n  "env": {"AUTH_TOKEN": "nodeSECRETabc123"},\n  "command": "node",\n'
                   '  "args": ["./build/index.js"]\n}}}\n')
    runs = next(c for c in mcp_config_claims(value_first, ".mcp.json") if c.predicate == RUNS_WITH)
    assert (runs.first_line, runs.quote) == (3, "node"), "the command is quoted from the command field, not from a value that contains its name"
    key = next(c for c in mcp_config_claims(value_first, ".mcp.json") if c.predicate == REQUIRES_ENV)
    assert (key.first_line, key.quote) == (2, '"AUTH_TOKEN"') and value_first.encode()[key.start:key.end].decode() == '"AUTH_TOKEN"'


def test_byte_spans_hold_past_text_that_is_not_ascii():
    accented = '{"description": "café — crème brûlée", "mcpServers": {"x": {"command": "npx", "args": ["-y", "@scope/pkg"], "env": {"KEY": "v"}}}}\n'
    claims = mcp_config_claims(accented, ".mcp.json")
    assert len(claims) >= 4
    for claim in claims:
        assert accented.encode()[claim.start:claim.end].decode() == claim.quote, "spans are bytes of the UTF-8 source"
    assert claims[0].start > len('{"description": "café — crème brûlée"'), "the offset counts the accents' extra bytes"


def test_a_server_of_the_wrong_shapes_is_still_defined_and_claims_nothing_else():
    odd = '{"mcpServers": {"x": {"command": 7, "args": "not a list", "env": [], "url": 5, "headers": "no"}}}\n'
    assert said(mcp_config_claims(odd, ".mcp.json")) == [(".mcp.json", DEFINES, ".mcp.json:x")]


def test_the_claim_cap_holds_for_a_configuration_too(monkeypatch):
    assert len(mcp_config_claims(CLAUDE, ".mcp.json")) > 4
    monkeypatch.setattr(mcp_config, "MAX_CLAIMS", 4)
    assert len(mcp_config_claims(CLAUDE, ".mcp.json")) == 4


@pytest.mark.asyncio
async def test_map_reads_a_configuration_and_its_servers_are_no_call_targets(tmp_path):
    from scone_memory.runtime.cli import build_parser, run

    root = tmp_path / "repo"
    (root / ".vscode").mkdir(parents=True)
    (root / ".mcp.json").write_text('{"mcpServers": {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "env": {"FETCH_KEY": "x"}}}}\n',
                                    encoding="utf-8")
    (root / ".vscode" / "mcp.json").write_text('{"servers": {"docs": {"command": "node", "args": ["build/index.js"]}}}\n', encoding="utf-8")
    (root / ".vscode" / "settings.json").write_text('{"editor.tabSize": 2}\n', encoding="utf-8")
    (root / "app.py").write_text("def go(x):\n    return x.fetch()\n", encoding="utf-8")
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), code_graph=True).open()
    try:
        out = io.StringIO()
        code = await run(build_parser().parse_args(["--json", "map", str(root), "--graph"]), engine, io.StringIO(""), out)
        receipt = json.loads(out.getvalue())
        assert code == 0 and receipt["unconfirmed_call_candidates"] == [], "`.mcp.json:fetch` is no `fetch()`"
        facts = await engine.facts("default")
        assert sorted((f.subject, f.predicate, f.object) for f in facts if f.predicate in (RUNS_WITH, REQUIRES_ENV, DEPENDS_ON)) == [
            (".mcp.json:fetch", DEPENDS_ON, "mcp-server-fetch"), (".mcp.json:fetch", REQUIRES_ENV, "$FETCH_KEY"),
            (".mcp.json:fetch", RUNS_WITH, "uvx"), (".vscode/mcp.json:docs", RUNS_WITH, "node"),
        ], "a dot-named configuration is read although dot-named files are otherwise passed over"
        assert all(f.origin == "extracted" for f in facts if f.predicate == REQUIRES_ENV)
    finally:
        await engine.close()
