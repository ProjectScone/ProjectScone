"""Compose an MCP server over an already-owned memory engine."""
from typing import Optional


from mcp.server.mcpserver import MCPServer

from ... import __version__
from ...core.bearer_keys import KeyHolder
from ...core.errors import InvalidInput, NotFound
from ...memory.engine import MemoryEngine

from .registry import INSTRUCTIONS
from .memory_tools import register_memory_tools
from .graph_tools import register_graph_tools
from ..mcp_activity import register_activity_tool
from ..mcp_agent import register_agent_tools

def create_server(engine: MemoryEngine, space: str = "default",
                  propose_below: Optional[float] = None,
                  holder: Optional[KeyHolder] = None) -> MCPServer:
    """Compose memory tools, graph resources and activity tools over
    one engine and a default space that any call may override with its
    ``space`` argument.

    ``propose_below`` is the Rust server's ``--propose-below``: a fact an
    agent submits under that confidence is parked as a proposal for a
    person instead of entering the ledger. Unset, as there, means every
    submitted fact is a ledger claim.

    ``holder`` is the bearer key this server answers to, when it answers
    to one: over stdio nobody presents a key and the space argument goes
    anywhere, exactly as before; over HTTP the key names the space and
    the role, and both bound what these tools can reach. A server is
    built per key rather than per request, so the bound is fixed at the
    tool, not re-decided on each call.
    """
    if propose_below is not None and not 0.0 <= propose_below <= 1.0:
        raise InvalidInput(f"propose_below must be a confidence in 0..=1, got {propose_below}")
    server = MCPServer(
        name="scone-memory",
        title="Scone memory engine",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    default_space = space

    def chosen(named: Optional[str]) -> str:
        """The space a call names, or this server's own when it names
        none. A server built for a key reaches that key's space only:
        another name is as unknown as a space that never existed, which
        is what the REST API tells the same key."""
        if named is None or named == default_space:
            return default_space
        if holder is not None:
            raise NotFound(f"space {named!r} is not this key's")
        return named

    register_memory_tools(server, engine, chosen, holder, propose_below)
    register_graph_tools(server, engine, chosen, default_space, holder)
    register_activity_tool(server, engine, chosen, holder)
    register_agent_tools(server, engine, chosen, holder)
    return server
