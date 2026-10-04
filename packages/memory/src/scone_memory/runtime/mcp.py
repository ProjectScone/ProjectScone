"""Backward-compatible MCP entrypoint; implementation lives in mcp_modules."""
import sys
from typing import Mapping, Optional, Sequence
from .config import Settings, build_engine
from .mcp_modules.registry import (
    TriplePattern,
    SubmittedFact,
    tool_error,
    ok_text,
    refuses_to_write,
    refusals_as_tool_errors,
    tool,
    day,
    describe_added,
    profile_lines,
    recall_lines,
    fact_about_line,
    pending_text,
    plural,
    facts_about,
    pending_episodes,
    clamp_confidence,
    fact_ids_by_status,
    MAX_CONTENT,
    MAX_QUERY,
    MAX_ENTITY,
    MAX_REASON,
    MAX_LIMIT,
    MAX_TAGS,
    MAX_FACTS,
    MAX_PENDING,
    DEFAULT_FACT_CONFIDENCE,
    PENDING_CONTENT_CHARS,
    INSTRUCTIONS,
    ToolFn,
    READING_TOOLS,
    WRITING_TOOLS,
)
from .mcp_modules.factory import create_server
from .mcp_modules.transport import (
    HttpTransport,
    DEFAULT_HTTP_PORT,
    LOOPBACK_NAMES,
    is_loopback,
    check_bind,
)
from .mcp_modules.cli import build_parser, KEY_FILE, key_file, self_hosted_key, close_stores
from .mcp_modules import cli, transport
from ..memory.engine import MemoryEngine

def http_app(engine: MemoryEngine, space: str = "default", keys: Optional[Mapping[str, str]] = None,
             roles: Optional[Mapping[str, str]] = None, propose_below: Optional[float] = None,
             host: str = "127.0.0.1", *, streamable_http_path: str = "/mcp"):
    return transport.http_app(engine, space, keys, roles, propose_below, host,
                              streamable_http_path=streamable_http_path, server_factory=create_server)

async def serve(settings: Settings, space: str) -> None:
    await cli.serve(settings, space, engine_builder=build_engine, server_factory=create_server)

async def serve_http(settings: Settings, space: str, host: str, port: int) -> None:
    await cli.serve_http(settings, space, host, port, engine_builder=build_engine, app_factory=http_app)

def main(argv: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None) -> int:
    return cli.main(argv, env, stdio_runner=serve, http_runner=serve_http)

if __name__ == "__main__":
    sys.exit(main())
