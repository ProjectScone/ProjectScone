"""Intentional MCP session registration and explicit public interaction writes."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Callable, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult

from ..capture.redact import redact_secrets
from ..core.bearer_keys import KeyHolder
from ..core.errors import Gone, InvalidInput, NotFound, SconeError
from ..core.models import EpisodeKind
from ..core.validation import check_space, normalise_metadata
from ..memory.engine import MemoryEngine
from .mcp_modules.registry import ok_text, tool, tool_error

MAX_PUBLIC_BYTES = 60_000
_IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z')


def _identifier(value: str, field: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise InvalidInput(f'{field} must be 1..128 ASCII letters, digits, dots, underscores, colons or hyphens')
    return value


def _label(value: str, field: str, limit: int = 120) -> str:
    if not value or value != value.strip() or len(value) > limit or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise InvalidInput(f'{field} must be a nonblank single line of at most {limit} characters')
    try:
        if len(value.encode('utf-8')) > 256:
            raise InvalidInput(f'{field} must be at most 256 UTF-8 bytes')
    except UnicodeError as error:
        raise InvalidInput(f'{field} must be valid UTF-8') from error
    return redact_secrets(value)


def _client_metadata(ctx: Context[object, object]) -> dict[str, str]:
    """SDK initialize metadata is optional, bounded, and never authority."""
    try:
        params: object = ctx.session.client_params
    except (AttributeError, ValueError):
        return {}
    info: object = getattr(params, 'client_info', None)
    result: dict[str, str] = {}
    for attribute, field, limit in [('name', 'client_name', 120), ('version', 'client_version', 64)]:
        value: object = getattr(info, attribute, None)
        if isinstance(value, str):
            try:
                result[field] = _label(value, field, limit)
            except InvalidInput:
                continue
    return result


def _metadata(ctx: Context[object, object], session_id: str, project: str | None,
              agent_name: str | None) -> dict[str, str]:
    metadata = {'session_id': _identifier(session_id, 'session_id'), 'identity_basis': 'self_declared',
                **_client_metadata(ctx)}
    if project is not None:
        metadata['project'] = _label(project, 'project')
    if agent_name is not None:
        metadata['agent_name'] = _label(agent_name, 'agent_name')
    elif 'client_name' in metadata:
        metadata['agent_name'] = metadata['client_name']
    name = metadata.get('agent_name', '').casefold()
    metadata['agent'] = 'codex' if name == 'codex' else 'claude-code' if name in {'claude-code', 'claude code'} else 'other'
    return metadata


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


async def _retain_and_record(engine: MemoryEngine, space: str, *, operation: str, identity: list[str],
                             content: str, kind: EpisodeKind, metadata: dict[str, str],
                             event_name: str, redacted: bool) -> CallToolResult:
    check_space(space)
    if engine.events is None:
        raise InvalidInput('no event log is attached; agent registration and interaction receipts are unavailable')
    metadata = {**metadata, 'mcp_operation': operation}
    metadata['request_sha256'] = _digest({'content': content, 'metadata': metadata})
    metadata = normalise_metadata(metadata)
    identity_hash = _digest(identity)
    source_key = f'mcp-agent:{operation}:{identity_hash}'
    source = f'mcp-agent://{identity_hash}'
    try:
        episode = await engine.episode_by_key(space, source_key)
    except Gone:
        raise
    except NotFound:
        added = await engine.remember(space, content, kind=kind, source=source, metadata=metadata, dedup_key=source_key)
        episode = await engine.episode(space, added.episode_id)
    # remember() deduplicates by key, so a competing write must be checked too.
    if (episode.content != content or episode.source != source or episode.kind != kind
            or any(episode.metadata.get(name) != value for name, value in metadata.items())):
        raise InvalidInput('this session/request identity was already retained with different content or metadata')
    payload: dict[str, object] = {'agent': metadata['agent'], 'session_id': metadata['session_id'],
                                  'event': event_name, 'source_event_id': f'mcp:{operation}:{identity_hash}',
                                  'episode_id': episode.episode_id}
    if 'project' in metadata:
        payload['project'] = metadata['project']
    try:
        event = await engine.record(space, 'agent', payload)
    except SconeError:
        raise
    except Exception:
        return tool_error(f'Source retained as episode {episode.episode_id}, but its event receipt is unavailable. '
                          'Retry the same session/request and public text; do not invent a new request ID.')
    return ok_text(json.dumps({'status': 'registered' if event_name == 'session_start' else 'recorded',
                              'space': space, 'session_id': metadata['session_id'], 'agent': metadata['agent'],
                              'identity_basis': 'self_declared', 'episode_id': episode.episode_id,
                              'event_id': event.event_id, 'recorded_at': event.ts,
                              'redaction_applied': redacted, 'host_capture_configured': False,
                              'capture_complete': False}, sort_keys=True))


def register_agent_tools(server: MCPServer, engine: MemoryEngine, chosen: Callable[[str | None], str],
                         holder: KeyHolder | None = None) -> None:
    @tool(server, 'memory_agent_connect', holder)
    async def memory_agent_connect(session_id: str, ctx: Context[object, object], project: str | None = None,
                                   agent_name: str | None = None, space: str | None = None) -> CallToolResult:
        """Register this intentionally selected agent session in memory.

        Any MCP harness may call this. Names and SDK clientInfo are self-declared,
        not authentication. The bearer key fixes the HTTP space and write role.
        This records a session-start receipt; it installs no hooks, scans no
        transcripts, and does not establish automatic or live capture.
        """
        target = chosen(space)
        metadata = _metadata(ctx, session_id, project, agent_name)
        content = json.dumps({'observation': 'Explicit MCP session registration', **metadata}, sort_keys=True)
        return await _retain_and_record(engine, target, operation='connect', identity=[session_id], content=content,
                                        kind='observation', metadata=metadata, event_name='session_start', redacted=False)

    @tool(server, 'memory_record_interaction', holder)
    async def memory_record_interaction(session_id: str, request_id: str, role: Literal['user', 'assistant'],
                                        public_text: str, ctx: Context[object, object], project: str | None = None,
                                        agent_name: str | None = None, space: str | None = None) -> CallToolResult:
        """Retain only the public interaction text explicitly supplied in this call.

        No hidden reasoning, transcript scanning, or agent control. Provide a
        stable request ID; the session/request/role tuple identifies one source.
        Exact retained-payload replays reuse the source and event; changed
        payloads are refused. User and assistant sides may share a request ID.
        Common secret patterns are redacted, which is not a guarantee that the
        supplied text is secret-free. Ordinary memory embedding may run.
        """
        target = chosen(space)
        metadata = _metadata(ctx, session_id, project, agent_name)
        metadata.update(request_id=_identifier(request_id, 'request_id'), role=role)
        try:
            size = len(public_text.encode('utf-8'))
        except UnicodeError as error:
            raise InvalidInput('public_text must be valid UTF-8') from error
        if not public_text.strip() or size > MAX_PUBLIC_BYTES:
            raise InvalidInput(f'public_text must be nonblank and at most {MAX_PUBLIC_BYTES} UTF-8 bytes')
        content = redact_secrets(public_text)
        return await _retain_and_record(engine, target, operation='interaction', identity=[session_id, request_id, role],
                                        content=content, kind='conversation', metadata=metadata,
                                        event_name='prompt' if role == 'user' else 'response', redacted=content != public_text)
