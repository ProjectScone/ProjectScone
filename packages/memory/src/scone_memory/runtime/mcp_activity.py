"""MCP registration for the transport-independent recorded-activity reader."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult
from pydantic import Field, StrictInt

from ..core.bearer_keys import KeyHolder
from ..core.errors import SconeError
from ..memory.engine import MemoryEngine
from ..observability.activity import activity
from .mcp_modules.registry import ok_text, tool, tool_error


def register_activity_tool(
    server: MCPServer, engine: MemoryEngine, chosen: Callable[[str | None], str], holder: KeyHolder | None = None,
) -> None:
    cursor_key = secrets.token_bytes(32)

    @tool(server, "memory_activity", holder)
    async def memory_activity(
        hours: Annotated[StrictInt, Field(description="Past server-relative hours, 1..720; defaults to 2. Use 48 for two days.")] = 2,
        limit: Annotated[StrictInt, Field(description="Maximum events in this page, 1..50; defaults to 20.")] = 20,
        project: Annotated[str | None, Field(description="Exact recorded project name, 1..120 characters.")] = None,
        session_id: Annotated[str | None, Field(description="Exact recorded session ID, 1..128 characters.")] = None,
        cursor: Annotated[str | None, Field(description="Returned next_cursor; repeat the original hours/project/session filters. Keeps the original window and receipt high-water mark.")] = None,
        space: Annotated[str | None, Field(description="Defaults to this server's space; HTTP keys cannot cross their authorized space.")] = None,
    ) -> CallToolResult:
        """Read recorded agent activity for the past 2 or 48 hours, without a model.

        Returns JSON with receipt timestamps, session/project/tool metadata,
        retained source identities and bounded excerpts, and explicit coverage.
        These are recorded interactions, not proof of completed tasks. Capture
        may be incomplete. Raw event text and unfiltered profiles are never used.
        Follow next_cursor with the same filters to continue the frozen window.
        """
        target = chosen(space)
        try:
            result = await activity(engine.events, engine.documents, target, now=engine.clock(), cursor_key=cursor_key,
                                    hours=hours, limit=limit, project=project, session_id=session_id, cursor=cursor)
        except SconeError:
            raise
        except Exception:
            # Backend exceptions can contain private paths or connection details.
            return tool_error("activity is unavailable because the event or source reader failed")
        return ok_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
