"""CLI configuration and ownership of standalone engine lifetimes."""
import argparse
import asyncio
import secrets
import os
import sys
from typing import Mapping, Optional, Sequence

from dataclasses import replace
from pathlib import Path


from ..cli import settings_for_cli
from ..config import Settings, build_engine
from ...memory.engine import MemoryEngine
from ...core.errors import SconeError

from .factory import create_server
from .transport import DEFAULT_HTTP_PORT, check_bind, http_app
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scone_memory.runtime.mcp",
        description="Serve the memory engine over MCP, on stdio or over HTTP. Stores come from SCONE_* "
                    "variables, as for the CLI.",
    )
    parser.add_argument("--space", default="default", help="space used when a call names none (default: default)")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio",
                        help="stdio for a client that starts this process (default), http for one that connects")
    parser.add_argument("--host", default=None,
                        help=f"address to listen on with --transport http (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None,
                        help=f"port to listen on with --transport http (default: {DEFAULT_HTTP_PORT})")
    parser.add_argument("--allow-anonymous", action="store_true",
                        help="serve HTTP with no key at all, as stdio does. Loopback only, and no key is issued")
    return parser


#: The key a self-hosted server issues itself, kept beside the memory it
#: opens so that moving the store moves the key with it.
KEY_FILE = "mcp-key"


def key_file(settings: Settings) -> Path:
    return Path(settings.sqlite_path).expanduser().parent / KEY_FILE


def self_hosted_key(settings: Settings, space: str, url: str, out) -> Settings:
    """The key a host serving HTTP with none configured is given.

    Self-hosting should not begin with inventing a secret and an
    environment variable to put it in. The first start issues one, saves
    it where only this user can read it, and prints it with the address
    to paste it into; later starts read it back, so a client configured
    once keeps working. A host that sets SCONE_API_KEYS never reaches
    here, and neither does stdio, where the client is the process that
    started this one and has nothing to prove.
    """
    path = key_file(settings)
    saved = path.read_text().strip() if path.exists() else ""
    if saved:
        print(f"scone: serving with the key saved at {path} (delete it to issue a new one)", file=out)
        if path.stat().st_mode & 0o077:
            print(f"scone: warning: {path} is readable by others; other users of this machine can read this "
                  f"key. chmod 600 it, or delete it for a new one.", file=out)
        key = saved
    else:
        key = "sk-scone-" + secrets.token_urlsafe(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:  # an empty file, or another start won the race
            os.chmod(path, 0o600)
            handle = os.open(path, os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(handle, "w") as writing:
            writing.write(key + "\n")
        print(f"scone: no SCONE_API_KEYS set, so this server issued a key for space {space!r}:\n"
              f"  key:  {key}\n"
              f"  file: {path} (yours alone; delete it to issue a new one)\n"
              f"  url:  {url}\n"
              f'  in an MCP client: "headers": {{"Authorization": "Bearer {key}"}}', file=out)
    return replace(settings, keys={key: space}, roles={key: "full"})


async def close_stores(engine: MemoryEngine) -> None:
    """Kept for callers that import it; the engine closes its own stores
    now, all four of them rather than the two this used to reach."""
    await engine.close()


async def serve(settings: Settings, space: str, *, engine_builder=build_engine, server_factory=create_server) -> None:
    engine = await engine_builder(settings)
    try:
        await server_factory(engine, space, settings.mcp_propose_below).run_stdio_async()
    finally:
        await close_stores(engine)


async def serve_http(settings: Settings, space: str, host: str, port: int, *, engine_builder=build_engine, app_factory=http_app) -> None:
    import uvicorn

    engine = await engine_builder(settings)
    try:
        app = app_factory(engine, space, settings.keys, settings.roles, settings.mcp_propose_below, host)
        await uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info")).serve()
    finally:
        await close_stores(engine)


def main(argv: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None, *, stdio_runner=serve, http_runner=serve_http) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.transport == "stdio" and (args.host is not None or args.port is not None):
        parser.error("--host and --port apply to --transport http")
    settings = settings_for_cli(os.environ if env is None else env)
    try:
        if args.transport == "stdio":
            asyncio.run(stdio_runner(settings, args.space))
        else:
            host, port = args.host or "127.0.0.1", args.port or DEFAULT_HTTP_PORT
            if args.allow_anonymous:
                settings = replace(settings, keys={}, roles={})
            elif not settings.keys:
                settings = self_hosted_key(settings, args.space, f"http://{host}:{port}/mcp", sys.stderr)
            # Before the stores are opened: a server that must not listen
            # should not have touched anything first.
            check_bind(settings.keys, host)
            asyncio.run(http_runner(settings, args.space, host, port))
    except SconeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
