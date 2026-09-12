"""Optional single-process conversation service; native memory routes are mounted.

Linux/macOS local-filesystem journal ownership only. Provider factories and
engines are server configuration. Turn results are cached for this process,
not reconstructed or retried after restart; lifecycle receipts are durable.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hmac
import os
from pathlib import Path
import stat
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket
from starlette.websockets import WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from ..runtime.conversation_review import ConversationReview
from ..retrieval.adaptive import AdaptiveRetriever

from ..memory.engine import check_space, normalise_time
from ..core.errors import Conflict, InvalidInput, NotFound
from ..core.models import RecallItem, RecallResult
from ..core.ports import TextFilter
from ..realtime.session_journal import SessionJournal
from ..retrieval.recall_scope import RecallScope
from ..retrieval.evidence_graph import MAX_CHUNKS, MAX_FACTS, build_query_evidence_graph
from ..retrieval.evidence_records import canonical_evidence, fingerprint, restrict_graph
from .app import create_app, episode_json, permitted
from ._lifecycle import finish_host_cleanup
from ..realtime.catalog import PersonaCatalog
from ..realtime.websocket import WebSocketAudioTransport

# What a host without a catalog reports: the revision of an empty one.
_EMPTY_REVISION = PersonaCatalog((), {}).revision
from .text_stream import MAX_BYTES, MAX_CHUNKS, TextWindow, sse


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


class Create(Command):
    capture: bool
    recall_scope: dict[str, object] = Field(default_factory=dict)
    # A voice session waits for its audio socket instead of taking text turns.
    mode: Literal["text", "voice"] = "text"
    # A catalog id; fixed for the session's life and echoed in its receipts.
    persona: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    # The configuration the client displayed; a mismatch refuses the session.
    persona_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")


class Stop(Command):
    expected_revision: int = Field(ge=1, le=2**63 - 2)


class Turn(Stop):
    text: str = Field(min_length=1, max_length=32000)


@dataclass
class OwnedSession:
    runtime: object
    turns: dict = field(default_factory=dict)
    stops: dict = field(default_factory=dict)
    active: asyncio.Task | None = None
    active_id: str | None = None
    cleanup_task: asyncio.Task | None = None


def _without_cached_graph(result):
    """Receipts retain identifiers and fingerprints, never source text copies."""
    if not isinstance(result, dict) or not isinstance(result.get("memory_context"), dict):
        return result
    context = dict(result["memory_context"])
    if context.pop("evidence_graph", None) is not None or "evidence_graph_status" in context:
        context["evidence_graph_status"] = "unavailable"
    if isinstance(context.get('tool_retrieval'), dict):
        tools = dict(context['tool_retrieval'])
        tools.pop('packets', None)
        tools['packets_status'] = 'unavailable'
        context['tool_retrieval'] = tools
    return {**result, "memory_context": context}


def create_conversation_app(engine, keys, journal_path, runtime_factory, *, scoped_runtime_factory=None,
                            max_sessions=100, max_turns=100, public_text_streaming=False,
                            worker=None, catalog=None, ingest_concurrency=4, roles=None,
                            runtime_available=None, model_connections_available=False,
                            vision_available=None, answer_review=None, adaptive_retriever=None, tool_retrieval=None,
                            agent_catalog=None, agent_plan_store=None, agent_run_service=None, document_ocr=None, document_import_service=None, document_media=None,
                            directory_sync_service=None):
    """The caller owns engine lifecycle; service owns journal and runtime tasks.

    runtime_factory(space, sid) supplies async reply(text) and close(). None
    advertises unavailable text. Do not use non-cooperative/untrusted runtimes:
    cancellation and cleanup use cooperative asyncio, not process termination.
    scoped_runtime_factory(space, sid, scope) explicitly opts into fixed recall
    constraints. It receives an immutable RecallScope; pass scope.kwargs() to
    the native runtime. When configured it takes precedence for every new session.
    public_text_streaming=True requires every runtime to accept an async on_text
    reply keyword delivering public chunks. Defaults off for custom runtimes.
    worker is a ConsolidationWorker started once the journal is owned and
    stopped with the service; a mounted memory app's own lifespan never runs,
    so the service must carry it. The independent Webapp serves browser pages.
    catalog is a bound PersonaCatalog: sessions may name one of its personas at
    creation and then run that persona's native text runtime. A host with a
    catalog and no bare runtime requires the choice; nothing is chosen for it.
    A voice session names a persona and is spoken to over its audio socket.
    answer_review configures native text sessions only. When supplied, custom
    runtime factories must accept answer_reviewer, review_policy and review_limits
    keywords and honor the native TextConversation review contract.
    adaptive_retriever must use this engine. Text runtime factories receive
    adaptive_retriever and recall_timeout keywords when it is configured.
    tool_retrieval describes the explicit ConversationTools binding installed by
    the host on its text factories. It advertises configuration, not model health.
    """
    from ..runtime.conversation_tools import ConversationTools

    keys = dict(keys)
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("configure nonempty bearer keys")
    for space in keys.values():
        check_space(space)
    for limit in (max_sessions, max_turns):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("service admission limits must be integers in 1..100")
    if runtime_factory is not None and not callable(runtime_factory):
        raise ValueError("runtime_factory must be callable or None")
    if scoped_runtime_factory is not None and not callable(scoped_runtime_factory):
        raise ValueError("scoped_runtime_factory must be callable or None")
    if catalog is not None and not hasattr(catalog, "public"):
        raise ValueError("catalog must be a bound PersonaCatalog or None")
    if answer_review is not None and not isinstance(answer_review, ConversationReview):
        raise ValueError("answer_review must be a ConversationReview or None")
    if adaptive_retriever is not None and (
            not isinstance(adaptive_retriever, AdaptiveRetriever) or adaptive_retriever.memory is not engine):
        raise ValueError("adaptive_retriever must use this memory engine")
    if tool_retrieval is not None and (not isinstance(tool_retrieval, ConversationTools)
            or adaptive_retriever is not None):
        raise ValueError("tool_retrieval requires a compatible ConversationTools binding")
    # A catalog's sessions run the native text runtime, which streams.
    configured = runtime_factory is not None or scoped_runtime_factory is not None or catalog is not None
    def bare_runtime_available():
        return (runtime_factory is not None or scoped_runtime_factory is not None) and (runtime_available is None or runtime_available())
    if type(public_text_streaming) is not bool or (public_text_streaming and not configured):
        raise ValueError("public_text_streaming requires a configured compatible runtime")
    owned: dict[tuple[str, str], OwnedSession] = {}
    creates: dict[tuple[str, str], str] = {}
    journal = None
    shutting_down = False

    def text_window(entry, request_id):
        return entry.turns.get(request_id, {}).get("stream") if entry else None

    def end_text(entry):
        window = text_window(entry, entry.active_id)
        if window is not None:
            window.finish()

    def begin_shutdown():
        nonlocal shutting_down
        shutting_down = True
        for entry in owned.values():
            end_text(entry)

    async def close_owned(entry):
        if entry.active is not None and not entry.active.done():
            entry.active.cancel()
            await asyncio.gather(entry.active, return_exceptions=True)
        await entry.runtime.close()

    @asynccontextmanager
    async def lifespan(_app):
        nonlocal journal, shutting_down
        shutting_down = False
        descriptor: int | None = None
        try:
            try:
                import fcntl
            except ImportError as exc:
                raise RuntimeError("conversation service journal ownership requires Linux or macOS") from exc
            path = Path(journal_path).resolve()
            if path.exists() and path.stat().st_nlink != 1:
                raise InvalidInput("hard-linked conversation journals are not supported")
            # macOS can make flock on the database conflict with SQLite's locks.
            # Never unlink the sidecar: another owner might still hold its inode.
            lock_path = str(path) + ".owner.lock"
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise InvalidInput("journal ownership lock must be a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("conversation journal is already owned by another service") from exc
            journal = SessionJournal(path)
            for space in sorted(set(keys.values())):
                # A turn still marked accepted belonged to the process this
                # one replaced, so whether its provider answered is
                # unknowable; it is settled as interrupted saying that,
                # rather than left as a receipt that can never come true.
                # Safe here because the lock above makes this service the
                # journal's only owner.
                journal.recover(space)
                after = ""
                while True:
                    page = journal.sessions(space, after=after)
                    for session in page["items"]:
                        if session["state"] in {"created", "running", "stopping"}:
                            journal.transition(space, session["session_id"], "recovery:" + uuid4().hex,
                                               "interrupt", session["revision"])
                    if not page["has_more"]:
                        break
                    after = page["next_after"]
            if worker is not None:
                worker.start()
            yield
        finally:
            begin_shutdown()
            async def cleanup() -> None:
                nonlocal journal
                cleanup_errors: list[BaseException] = []
                if agent_run_service is not None:
                    try:
                        await agent_run_service.aclose()
                    except (Exception, asyncio.CancelledError) as error:
                        cleanup_errors.append(error)
                if document_import_service is not None:
                    try:
                        await document_import_service.aclose()
                    except (Exception, asyncio.CancelledError) as error:
                        cleanup_errors.append(error)
                if directory_sync_service is not None:
                    try:
                        await directory_sync_service.aclose()
                    except (Exception, asyncio.CancelledError) as error:
                        cleanup_errors.append(error)
                if worker is not None:
                    try:
                        await worker.stop()
                    except (Exception, asyncio.CancelledError) as error:
                        cleanup_errors.append(error)
                try:
                    if journal is not None:
                        for (space, sid), entry in owned.items():
                            if entry.cleanup_task is not None:
                                try:
                                    if not await asyncio.shield(entry.cleanup_task):
                                        cleanup_errors.append(RuntimeError("runtime cleanup failed"))
                                except (Exception, asyncio.CancelledError) as error:
                                    cleanup_errors.append(error)
                                continue
                            try:
                                session = journal.get(space, sid)
                                if session["state"] in {"created", "running", "stopping"}:
                                    journal.transition(space, sid, "shutdown:" + uuid4().hex, "interrupt", session["revision"])
                            except (Exception, asyncio.CancelledError) as error:
                                cleanup_errors.append(error)
                            try:
                                await asyncio.wait_for(close_owned(entry), 5)
                            except (Exception, asyncio.CancelledError) as error:
                                cleanup_errors.append(error)
                finally:
                    try:
                        if journal is not None:
                            journal.close()
                    finally:
                        journal = None
                        owned.clear()
                        creates.clear()
                        if descriptor is not None:
                            os.close(descriptor)
                if any(isinstance(error, asyncio.CancelledError) for error in cleanup_errors):
                    raise asyncio.CancelledError
                if cleanup_errors:
                    raise RuntimeError("conversation service cleanup failed") from None
            await finish_host_cleanup(cleanup())

    app = FastAPI(title="scone-conversations", docs_url=None, redoc_url=None, lifespan=lifespan)
    # Hosts must notify before draining HTTP tasks, not only at lifespan end:
    # an idle SSE response is itself one of the tasks they would wait for.
    app.state.begin_conversation_shutdown = begin_shutdown
    app.state.worker = worker

    roles = dict(roles or {})

    async def space_for(request: Request):
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer" and token:
            for key, space in keys.items():
                if hmac.compare_digest(token.encode(), key.encode()):
                    if shutting_down and request.method == "POST":
                        raise HTTPException(503, "conversation service is shutting down")
                    if not permitted(roles.get(key, "full"), request.method, request.url.path):
                        raise HTTPException(403, f"key role {roles.get(key)} cannot write")
                    # A deleted space's key answers 404 here as on the memory routes.
                    if await engine.space_deleted(space) is not None:
                        raise HTTPException(404, f"space {space!r} was deleted")
                    return space
        raise HTTPException(401, "valid bearer key required")

    @app.middleware("http")
    async def private_responses(request, call_next):
        if request.url.path.startswith("/v1/conversations"):
            if request.method == "POST":
                # Bound actual bytes rather than trusting Content-Length.
                chunks, size = [], 0
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 40000:
                        return JSONResponse({"error": "conversation request too large"}, 413, headers={"Cache-Control": "no-store"})
                    chunks.append(chunk)
                request._body = b"".join(chunks)
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def http_error(_request, error):
        return JSONResponse({"error": error.detail}, status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, _error):
        return JSONResponse({"error": "invalid conversation request"}, status_code=422)

    @app.exception_handler(InvalidInput)
    async def invalid(_request, _error):
        return JSONResponse({"error": "invalid conversation argument"}, status_code=422)

    @app.exception_handler(NotFound)
    async def missing(_request, _error):
        return JSONResponse({"error": "conversation record not found"}, status_code=404)

    @app.exception_handler(Conflict)
    async def conflict(_request, error):
        return JSONResponse({"error": "conversation revision or command conflict", "revision": error.revision}, status_code=409)

    def describe(session):
        """A session receipt names its persona from this process's catalog and
        keeps the configuration recorded at creation: an id the catalog no
        longer has gets no name, and a changed configuration is not current."""
        chosen = session.get("persona")
        recorded = session.pop("persona_fingerprint", None)
        now = catalog.fingerprint(chosen) if catalog is not None and chosen else None
        session["persona"] = ({"id": chosen, "name": catalog.name(chosen) if catalog is not None else None,
                               "fingerprint": recorded, "current": recorded is not None and recorded == now}
                              if chosen else None)
        return session

    def inspect(space, sid):
        result = describe(journal.get(space, sid))
        entry = owned.get((space, sid))
        result["active_request_id"] = entry.active_id if entry else None
        result["latest_request_id"] = journal.latest_turn_id(space, sid)
        return result

    @app.get("/v1/conversations/capabilities")
    async def capabilities(space=Depends(space_for)):
        return {"schema_version": 1,
                "text_configured": bare_runtime_available() or bool(catalog and catalog.personas),
                "tool_retrieval": {"configured": tool_retrieval is not None,
                    "protocol": tool_retrieval.mode if tool_retrieval is not None else "off",
                    "initial_search": tool_retrieval.initial_search if tool_retrieval is not None else False,
                    "computation": tool_retrieval.compute if tool_retrieval is not None else False,
                    "available": tool_retrieval is not None and (bare_runtime_available() or bool(catalog and catalog.personas)),
                    "limits": tool_retrieval.limits.model_dump(mode="json") if tool_retrieval is not None else None,
                    "applies_to": "text", "verified_accuracy": False},
                "answer_review": {"configured": answer_review is not None,
                                  "policy": answer_review.policy if answer_review is not None else "off"},
                "adaptive_retrieval": {"configured": adaptive_retriever is not None,
                    "limits": adaptive_retriever.limits.model_dump(mode="json") if adaptive_retriever is not None else None,
                    "search_history": adaptive_retriever.include_search_history if adaptive_retriever is not None else False,
                    "graph_max_hops": adaptive_retriever.graph_limits.max_hops
                        if adaptive_retriever is not None and adaptive_retriever.graph_limits is not None else 0},
                "recall_scope": scoped_runtime_factory is not None or bool(catalog and catalog.personas),
                "personas": len(catalog.personas) if catalog is not None else 0,
                "voice": bool(catalog and catalog.personas), "video": False, "streaming": public_text_streaming,
                "voice_stream": ({"schema_version": 1, "transport": "websocket", "protocol": "scone-pcm-v1",
                                  "authentication": "hello", "reconnect": False, "pcm": "s16le",
                                  "input_channels": [1, 2], "min_sample_rate": 8000,
                                  "max_sample_rate": 192000, "max_input_frame_bytes": 64000}
                                 if catalog and catalog.personas else None),
                "text_stream": ({"transport": "sse", "replay": "active_window", "max_bytes": MAX_BYTES,
                                  "max_chunks": MAX_CHUNKS} if public_text_streaming else None),
                "reply_transport": "poll", "reply_replay": "durable_receipts",
                "session_deletion": True,
                "turn_cancellation": True,
                "transcript_pagination": True,
                "provider_completion": "unverified", "max_sessions": max_sessions, "max_turns": max_turns}

    @app.get("/v1/conversations/personas")
    async def personas(space=Depends(space_for)):
        """The choices this host can run, for a client to pick from. Never the
        instructions, and nothing the registry closed over."""
        return {"schema_version": 1, "revision": catalog.revision if catalog is not None else _EMPTY_REVISION,
                "personas": catalog.public(voice=True) if catalog is not None else []}

    @app.get("/v1/conversations")
    async def sessions(after: str = "", limit: int = 100, space=Depends(space_for)):
        page = journal.sessions(space, after=after, limit=limit)
        for item in page["items"]:
            describe(item)
        return page

    @app.post("/v1/conversations")
    async def create(body: Create, space=Depends(space_for)):
        if shutting_down:
            raise HTTPException(503, "conversation service is shutting down")
        if not body.capture:
            raise HTTPException(422, "this runtime requires explicit transcript capture consent")
        scope = RecallScope.from_mapping(body.recall_scope)
        key = (space, body.request_id)
        # Identity first, judgement second. A retry of a create that already
        # made a session replays it whatever this process's catalog says now:
        # the persona may be gone or changed, and the session still exists.
        if key in creates:
            # Deleted sessions keep a process-local retry tombstone. Check it
            # before journal.create can insert a new row for the deleted key.
            current = inspect(space, creates[key])
            journal.create(space, body.request_id, body.mode, recall_scope=scope.as_dict(), persona=body.persona)
            return current
        already = journal.created(space, body.request_id)
        if already is not None:
            journal.create(space, body.request_id, body.mode, recall_scope=scope.as_dict(), persona=body.persona)
            return inspect(space, already)
        chosen = None
        if body.persona is not None:
            chosen = catalog.get(body.persona) if catalog is not None else None
            if chosen is None:
                raise HTTPException(422, f"persona is not in this host's catalog: {body.persona}")
            if body.persona_fingerprint is not None and body.persona_fingerprint != catalog.fingerprint(body.persona):
                # Nothing is behind this request id (checked above), so the
                # refusal proves no write, with a code the client can act on.
                return JSONResponse({"error": f"persona selection is stale: {body.persona} "
                                              f"(current fingerprint {catalog.fingerprint(body.persona)})",
                                     "code": "persona_selection_stale",
                                     "fingerprint": catalog.fingerprint(body.persona)}, status_code=409)
        elif not bare_runtime_available():
            if catalog is not None and catalog.personas:
                raise HTTPException(422, "this host requires a persona: " + ", ".join(p.id for p in catalog.personas))
            raise HTTPException(503, "text conversation runtime is not configured")
        if scope.as_dict() and scoped_runtime_factory is None and chosen is None:
            raise HTTPException(422, "this runtime does not support session recall constraints")
        if body.mode == "voice" and chosen is None:
            raise HTTPException(422, "a voice session needs a persona with transcription and speech choices")
        if len(creates) >= max_sessions:
            raise HTTPException(429, "conversation process capacity reached")
        receipt = journal.create(space, body.request_id, body.mode, recall_scope=scope.as_dict(), persona=body.persona,
                                 persona_fingerprint=catalog.fingerprint(body.persona) if chosen is not None else None)
        sid = receipt["session_id"]
        creates[key] = sid
        current = journal.get(space, sid)
        if current["state"] != "created" or body.mode == "voice":
            # A voice session starts when its audio socket arrives.
            return inspect(space, sid)
        try:
            fixed = RecallScope.from_mapping(current["recall_scope"])
            conversation_options: dict[str, object] = dict(answer_review.options()) if answer_review is not None else {}
            if adaptive_retriever is not None:
                conversation_options.update(adaptive_retriever=adaptive_retriever, recall_timeout=adaptive_retriever.limits.timeout_s)
            if chosen is not None:
                runtime = chosen.text(engine, space, sid, **fixed.kwargs(), **conversation_options)
            else:
                runtime = (scoped_runtime_factory(space, sid, fixed, **conversation_options)
                           if scoped_runtime_factory is not None else runtime_factory(space, sid, **conversation_options))
            owned[(space, sid)] = OwnedSession(runtime)
            journal.transition(space, sid, "start:" + uuid4().hex, "start", current["revision"])
        except Exception:
            journal.transition(space, sid, "fail:" + uuid4().hex, "fail", current["revision"])
            raise HTTPException(503, "conversation runtime failed to initialize") from None
        return inspect(space, sid)

    @app.get("/v1/conversations/{sid}")
    async def session(sid: str, space=Depends(space_for)):
        return inspect(space, sid)

    @app.get("/v1/conversations/{sid}/events")
    async def events(sid: str, after: int = 0, limit: int = 100, space=Depends(space_for)):
        return journal.events(space, sid, after, limit)

    @app.get("/v1/conversations/{sid}/transcript")
    async def transcript(sid: str, before: str | None = Query(default=None, max_length=1024),
                         limit: int = Query(default=200, ge=1, le=200), space=Depends(space_for)):
        journal.get(space, sid)
        boundary = None
        if before is not None:
            try:
                raw = base64.b64decode(before + "=" * (-len(before) % 4), altchars=b"-_", validate=True)
                cursor = json.loads(raw)
                if (not isinstance(cursor, list) or len(cursor) != 5 or type(cursor[0]) is not int or cursor[:3] != [1, space, sid]
                        or not isinstance(cursor[3], str) or normalise_time(cursor[3]) != cursor[3]
                        or type(cursor[4]) is not int or not 1 <= cursor[4] <= 2**63 - 1):
                    raise ValueError("invalid boundary")
                boundary = (cursor[3], cursor[4])
            except (ValueError, TypeError, UnicodeError, binascii.Error, InvalidInput):
                raise HTTPException(422, "invalid transcript cursor") from None
        # The engine already walks matching episodes; bound the response, not
        # the history being searched. Cursors survive deletion of their boundary.
        episodes = await engine.episodes(space, {"session_id": sid})
        if boundary is not None:
            episodes = [e for e in episodes if (e.created_at, e.episode_id) < boundary]
        page = episodes[-limit:]
        has_more = len(episodes) > limit
        next_before = None
        if has_more:
            encoded_boundary = json.dumps([1, space, sid, page[0].created_at, page[0].episode_id], separators=(",", ":"))
            next_before = base64.urlsafe_b64encode(encoded_boundary.encode()).decode().rstrip("=")
        return {"episodes": [episode_json(episode) for episode in page],
                "has_more": has_more, "next_before": next_before}

    async def refreshed_graph(space, kept, result):
        result = _without_cached_graph(result)
        context = result.get("memory_context")
        if not isinstance(context, dict) or not isinstance(context.get("evidence_fingerprints"), dict):
            return result
        context["evidence_graph_status"] = "unavailable"
        try:
            async with asyncio.timeout(1.0):
                user_id = result.get("user_episode_id")
                if type(user_id) is not int:
                    return result
                user = await engine.documents.get_episode(space, user_id)
                if user is None or user.space != space or user.episode_id != user_id:
                    return result
                if journal is None:
                    return result
                session = journal.get(space, kept["session_id"])
                scope = RecallScope.from_mapping(session["recall_scope"])
                references = context.get("references", [])
                if not isinstance(references, list):
                    return result
                pairs = [(ref["chunk_id"], ref["episode_id"]) for ref in references[:MAX_CHUNKS]
                         if isinstance(ref, dict) and type(ref.get("chunk_id")) is int
                         and type(ref.get("episode_id")) is int]
                chunks = {chunk.chunk_id: chunk for chunk in await engine.documents.get_chunks(
                    space, [chunk_id for chunk_id, _ in pairs])} if pairs else {}
                fingerprints = context["evidence_fingerprints"]
                items = []
                for chunk_id, episode_id in pairs:
                    chunk = chunks.get(chunk_id)
                    if (chunk is None or chunk.space != space or chunk.episode_id != episode_id
                            or hashlib.sha256(chunk.text.encode()).hexdigest() != fingerprints.get(str(chunk_id))):
                        continue
                    items.append(RecallItem(chunk_id=chunk_id, episode_id=episode_id, text=chunk.text,
                                            created_at=chunk.created_at, score=0))
                claim_fingerprints = context.get("claim_fingerprints", {})
                relation_fingerprints = context.get("relation_fingerprints", {})
                if not isinstance(claim_fingerprints, dict) or not isinstance(relation_fingerprints, dict):
                    return result
                facts = []
                for fact_id in list(claim_fingerprints)[:MAX_FACTS]:
                    if not isinstance(fact_id, str) or not fact_id.isdecimal():
                        continue
                    fact = await engine.documents.get_fact(space, int(fact_id))
                    if fact is not None and fact.space == space and fact.fact_id == int(fact_id):
                        facts.append(fact)
                graph = await build_query_evidence_graph(engine.documents, space, user.content,
                    RecallResult(items=items, facts=facts, event_id=context.get("recall_event_id")),
                    scope=TextFilter(**scope.kwargs()), exclude_session_id=kept["session_id"])
                records = canonical_evidence(graph)
                fact_ids = {record["fact_id"] for record in records.claims
                            if isinstance(record["fact_id"], int)
                            and fingerprint(record) == claim_fingerprints.get(str(record["fact_id"]))}
                link_ids = {record["link_id"] for record in records.relations
                            if isinstance(record["link_id"], int)
                            and record["from_fact"] in fact_ids and record["to_fact"] in fact_ids
                            and fingerprint(record) == relation_fingerprints.get(str(record["link_id"]))}
                graph = restrict_graph(graph, fact_ids, link_ids)
                if len(fact_ids) < len(claim_fingerprints) or len(link_ids) < len(relation_fingerprints):
                    graph.notices.append("Some originally supplied claims or relations are no longer retained unchanged with valid sources.")
                for node in graph.nodes:
                    if node.kind == "chunk":
                        for key in ("score", "similarity", "lanes"):
                            node.data.pop(key, None)
                if len(items) < len(references):
                    graph.notices.append("Some originally supplied evidence is no longer retained unchanged.")
                context["evidence_graph"] = graph.model_dump(mode="json")
                context["evidence_graph_status"] = "prepared"
        except Exception as error:
            logging.getLogger(__name__).warning("evidence_graph.refresh_failed", extra={"event": "evidence_graph.refresh_failed",
                "session_id": kept["session_id"], "exception_type": type(error).__name__})
        return result

    async def resolved(space, kept, entry):
        """One receipt shape, live or rehydrated, with the reply's text
        read from the episode that holds it every time.

        The text is never served from a remembered copy: a process still
        holding one must stop serving it the moment the episode is
        forgotten, or "no stale transcript copies" would hold only across
        a restart, which is the easy half. A completed turn whose episode
        is gone is terminal, not failed, and must not invite a resend; an
        episode that merely could not be read says so, because calling a
        storage failure "forgotten" would be a deletion nobody performed.
        """
        live = entry.turns.get(kept["request_id"], {}).get("receipt") if entry else None
        status = "pending" if kept["status"] == "accepted" else kept["status"]
        receipt = {"request_id": kept["request_id"], "status": status,
                   "result": None, "result_state": None, "error": kept["error"]}
        if live is not None:
            receipt = {**live, "result_state": None, "error": live.get("error", kept["error"])}
            receipt.setdefault("result", None)
            status = receipt["status"]
        if status == "pending":
            return receipt
        episode_id = kept["episode_id"]
        if episode_id is None and isinstance(receipt.get("result"), dict):
            episode_id = receipt["result"].get("assistant_episode_id")
        if status != "completed" or episode_id is None:
            return {**receipt, "result": None, "result_state": "unavailable"}
        try:
            episode = await engine.episode(space, episode_id)
        except NotFound:
            return {**receipt, "result": None, "result_state": "forgotten"}
        except Exception:
            return {**receipt, "result": None, "result_state": "unreadable"}
        result = dict(receipt.get("result") or {})
        result.update(text=episode.content, assistant_episode_id=episode_id)
        result = await refreshed_graph(space, kept, result)
        return {**receipt, "result": result, "result_state": "available"}

    def settle(space, sid, request_id, status, episode_id=None, error=None):
        """Record a turn's outcome durably, and never let that recording
        be the thing that breaks a turn: the in-process receipt is still
        correct if the journal write fails, and the next recover() call
        settles what this missed."""
        try:
            journal.finish_turn(space, sid, request_id, status, episode_id=episode_id, error=error)
        except (Conflict, NotFound, InvalidInput):
            pass

    def after_cancel(space, sid, entry, *, failed=False):
        # Only a runtime explicitly reporting reusable state can keep serving.
        # Capture/cleanup uncertainty closes the session, not just the receipt.
        if failed or getattr(entry.runtime, "closed", True) is not False:
            current = journal.get(space, sid)
            if current["state"] == "running":
                journal.transition(space, sid, "cancel-interrupted:" + uuid4().hex, "interrupt", current["revision"])
                entry.cleanup_task = asyncio.create_task(finish_cleanup(entry))

    async def run_turn(space, sid, entry, request_id, text):
        from ..runtime.diagnostics import context
        log_token = context.set({"session_id": sid, "request_id": request_id})
        started = time.perf_counter()
        logger = logging.getLogger(__name__)
        logger.info("conversation.started", extra={"event": "conversation.started"})
        receipt = entry.turns[request_id]["receipt"]
        window = text_window(entry, request_id)

        async def observe(chunk):
            if shutting_down or receipt["status"] != "pending" or journal.get(space, sid)["state"] != "running":
                raise RuntimeError("text observation is closed")
            window.append(chunk)

        try:
            result = await entry.runtime.reply(text, on_text=observe) if window is not None else await entry.runtime.reply(text)
            if receipt.get("status") == "cancelled":
                after_cancel(space, sid, entry)
                return  # a late result must not overwrite the terminal decision
            if window is not None and window.failed:
                raise RuntimeError("public text observation failed")
            if shutting_down or journal.get(space, sid)["state"] != "running":
                receipt["status"] = "interrupted"
                settle(space, sid, request_id, "interrupted")
            else:
                result = _without_cached_graph(result)
                receipt.update(status="completed", result=result)
                # The episode holding the reply's text, not the text: one
                # copy, so forgetting the episode removes it everywhere.
                settle(space, sid, request_id, "completed",
                       episode_id=result.get("assistant_episode_id") if isinstance(result, dict) else None)
        except asyncio.CancelledError:
            # The cancelled receipt survives even if capture/cleanup uncertainty
            # means this runtime cannot accept another turn.
            if receipt.get("status") == "cancelled":
                after_cancel(space, sid, entry)
                raise
            receipt["status"] = "interrupted"
            settle(space, sid, request_id, "interrupted")
            current = journal.get(space, sid)
            if current["state"] == "running":
                journal.transition(space, sid, "interrupted:" + uuid4().hex, "interrupt", current["revision"])
                entry.cleanup_task = asyncio.create_task(finish_cleanup(entry))
        except Exception as error:
            if receipt.get("status") == "cancelled":
                after_cancel(space, sid, entry, failed=True)
                return
            if shutting_down:
                receipt["status"] = "interrupted"
                settle(space, sid, request_id, "interrupted")
                return
            causes = []
            cause = error
            while cause is not None and len(causes) < 5:
                causes.append(type(cause).__name__)
                cause = cause.__cause__
            logging.getLogger(__name__).warning("Conversation turn failed: session=%s request=%s causes=%s",
                                               sid, request_id, "/".join(causes),
                                               extra={"event": "conversation.failed", "exception_type": "/".join(causes)})
            message = ("Conversation timed out before the reply completed. Check Sources to confirm what was saved, "
                       "then check the model connection and timeout before starting a new conversation."
                       if isinstance(error, TimeoutError) else "conversation turn failed; do not automatically retry")
            from ..realtime.text import _AnswerReviewFailure
            if isinstance(error, _AnswerReviewFailure):
                from ..realtime.answer_review import AnswerReviewReceipt

                # Only enum/count/status diagnostics may cross the API boundary.
                # Never publish the draft, source text or provider exception.
                try:
                    review = AnswerReviewReceipt.model_validate_json(json.dumps(error.answer_review))
                except (TypeError, ValueError):
                    review = AnswerReviewReceipt(errors=("invalid_review",), source_status="unavailable")
                receipt["answer_review"] = review.model_dump(mode="json")
                message = ("Answer review could not verify its retained sources. No assistant reply was saved."
                           if review.source_status != "retained" else
                           "Answer review did not support this reply. No assistant reply was saved.")
            receipt.update(status="failed", error=message)
            settle(space, sid, request_id, "failed", error=receipt["error"])
            current = journal.get(space, sid)
            if current["state"] == "running":
                journal.transition(space, sid, "failure:" + uuid4().hex, "fail", current["revision"])
                entry.cleanup_task = asyncio.create_task(finish_cleanup(entry))
        finally:
            logger.info("conversation.finished", extra={"event": "conversation.finished", "outcome": receipt["status"],
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)})
            context.reset(log_token)
            if window is not None:
                window.finish()
            entry.active_id = None

    @app.post("/v1/conversations/{sid}/turns", status_code=202)
    async def turn(sid: str, body: Turn, space=Depends(space_for)):
        if shutting_down:
            raise HTTPException(503, "conversation service is shutting down")
        current = journal.get(space, sid)
        if current["mode"] != "text":
            raise HTTPException(409, "a voice session takes audio over its socket, not text turns")
        entry = owned.get((space, sid))
        signature = body.model_dump()
        if entry and body.request_id in entry.turns:
            previous = entry.turns[body.request_id]
            if previous["signature"] != signature:
                raise Conflict("turn request changed", current["revision"])
            # One receipt shape everywhere: a retry answers exactly what a
            # read of the same turn answers.
            return await resolved(space, journal.turn(space, sid, body.request_id), entry)
        if current["state"] != "running" or entry is None:
            raise Conflict("conversation is not running in this process", current["revision"])
        if body.expected_revision != current["revision"] or entry.active_id is not None:
            raise Conflict("stale or busy conversation", current["revision"])
        if not body.text.strip() or len(body.text.encode()) > 32000:
            raise HTTPException(422, "message must contain 1..32000 UTF-8 bytes of nonempty text")
        if len(entry.turns) >= max_turns:
            raise HTTPException(429, "conversation turn capacity reached")
        receipt = {"request_id": body.request_id, "status": "pending"}
        # Durable half: recorded before the provider is asked anything, so a
        # process that dies mid-turn leaves a receipt rather than a gap.
        journal.start_turn(space, sid, body.request_id, signature)
        entry.turns[body.request_id] = {"signature": signature, "receipt": receipt,
                                      "stream": TextWindow() if public_text_streaming else None}
        entry.active_id = body.request_id
        entry.active = asyncio.create_task(run_turn(space, sid, entry, body.request_id, body.text))
        return await resolved(space, journal.turn(space, sid, body.request_id), entry)

    @app.get("/v1/conversations/{sid}/turns")
    async def turn_list(sid: str, after: str = "", limit: int = 100, space=Depends(space_for)):
        """Which turns this session has, so a reloaded page can find its
        receipts without knowing their request ids. Bounded and scoped
        like every other read here."""
        page = journal.turns(space, sid, after=after, limit=limit)
        entry = owned.get((space, sid))
        return {**page, "turns": [await resolved(space, kept, entry) for kept in page["turns"]]}

    @app.get("/v1/conversations/{sid}/turns/{request_id}")
    async def result(sid: str, request_id: str, space=Depends(space_for)):
        journal.get(space, sid)
        entry = owned.get((space, sid))
        # The journal answers when this process does not hold the turn, so
        # a 404 means only what it should: nobody sent that request.
        return await resolved(space, journal.turn(space, sid, request_id), entry)

    @app.get("/v1/conversations/{sid}/turns/{request_id}/stream")
    async def stream(sid: str, request_id: str, request: Request, space=Depends(space_for)):
        journal.get(space, sid)
        journal.turn(space, sid, request_id)
        if not public_text_streaming:
            raise HTTPException(501, "public text streaming is not configured")
        values = request.query_params.getlist("after")
        previous = request.headers.get("last-event-id")
        raw = values[0] if values else (previous if previous is not None else "0")
        if (set(request.query_params) - {"after"} or len(values) > 1
                or not raw.isascii() or not raw.isdecimal() or len(raw) > 19 or int(raw) > 2**63 - 1
                or (values and previous is not None and previous != raw)):
            raise HTTPException(422, "invalid stream cursor")
        cursor = int(raw)
        entry = owned.get((space, sid))
        window = text_window(entry, request_id)
        if window is not None and not window.closed and cursor > window.last_sequence:
            raise HTTPException(409, "stream cursor is ahead of observed text")

        async def output():
            after = cursor
            while True:
                if shutting_down or journal is None:
                    yield sse("end", {"request_id": request_id, "reason": "service_shutdown", "read_receipt": True})
                    return
                try:
                    current = journal.get(space, sid)
                    kept = journal.turn(space, sid, request_id)
                except NotFound:
                    yield sse("end", {"request_id": request_id, "reason": "deleted", "read_receipt": True})
                    return
                live = entry.turns.get(request_id, {}).get("receipt") if entry else None
                status = live["status"] if live else ("pending" if kept["status"] == "accepted" else kept["status"])
                if status != "pending":
                    yield sse("terminal", {"request_id": request_id, "status": status, "read_receipt": True})
                    return
                if current["state"] != "running" or window is None or window.closed:
                    yield sse("end", {"request_id": request_id, "reason": "window_unavailable", "read_receipt": True})
                    return
                gap, chunk = window.next_after(after)
                if gap is not None:
                    yield sse("gap", {"after": after, "next_sequence": gap})
                    after = gap - 1
                elif chunk is not None:
                    after, text = chunk
                    yield sse("text", {"sequence": after, "text": text, "provisional": True}, sequence=after)
                else:
                    try:
                        await asyncio.wait_for(window.wait_after(after), 10)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"

        return StreamingResponse(output(), media_type="text/event-stream", headers={
            "Cache-Control": "no-store", "X-Accel-Buffering": "no", "X-Content-Type-Options": "nosniff",
        })

    @app.delete("/v1/conversations/{sid}", status_code=204)
    async def delete(sid: str, space=Depends(space_for)):
        """The session, its transcript and its receipts. A conversation a
        process is still serving is refused rather than deleted out from
        under it: stop it first, or that process keeps answering from a
        transcript that no longer exists."""
        current = journal.get(space, sid)
        # State decides, not process ownership: an ended session may still
        # have an entry in this process, and that entry cannot serve a
        # deleted turn anyway, because every read goes through the journal.
        # A voice session waiting for its socket is served by nobody, so it can go.
        if current["state"] in ("running", "stopping") or (current["state"] == "created" and (space, sid) in owned):
            raise Conflict("stop the conversation before deleting it", current["revision"])
        # Episodes first: a receipt naming an episode that is gone reads
        # as forgotten, which is true, while the reverse leaves episodes
        # no session can account for.
        held = {episode.episode_id for episode in await engine.episodes(space, {"session_id": sid})}
        after = ""
        while True:
            page = journal.turns(space, sid, after=after, limit=200)
            held |= {turn["episode_id"] for turn in page["turns"] if turn["episode_id"] is not None}
            if not page["has_more"]:
                break
            after = page["next_after"]
        # A receipt names an episode the metadata may not: identical replies
        # deduplicate, so one episode can be two conversations' transcript.
        for episode_id in sorted(held - journal.episodes_held_elsewhere(space, sid, held)):
            try:
                await engine.forget(space, episode_id)
            except NotFound:
                pass  # an independently forgotten reply is already removed
        journal.delete_session(space, sid)
        return Response(status_code=204)

    @app.post("/v1/conversations/{sid}/turns/{request_id}/cancel")
    async def cancel(sid: str, request_id: str, space=Depends(space_for)):
        """End one turn without ending the conversation.

        Terminal and idempotent: the provider may already have been asked,
        so a later retry of the same request id answers with the
        cancellation instead of asking again. Recorded before the task is
        touched, so a reply that lands during the unwinding cannot
        overwrite the decision.
        """
        current = journal.get(space, sid)
        kept = journal.turn(space, sid, request_id)
        if kept["status"] == "cancelled":
            return await resolved(space, kept, owned.get((space, sid)))
        # A turn that already settled is refused by the journal itself,
        # with the same 409; a second guard here would be untestable,
        # because the conflict handler says one thing for every conflict.
        settled = journal.finish_turn(space, sid, request_id, "cancelled")
        entry = owned.get((space, sid))
        if entry is not None and request_id in entry.turns:
            entry.turns[request_id]["receipt"]["status"] = "cancelled"
            window = text_window(entry, request_id)
            if window is not None:
                window.finish()
        if entry is not None and entry.active_id == request_id and entry.active is not None:
            entry.active.cancel()
            await asyncio.gather(entry.active, return_exceptions=True)
        return await resolved(space, settled, entry)

    @app.post("/v1/conversations/{sid}/stop")
    async def stop(sid: str, body: Stop, space=Depends(space_for)):
        current = journal.get(space, sid)
        entry = owned.get((space, sid))
        signature = body.model_dump()
        if entry and body.request_id in entry.stops:
            if entry.stops[body.request_id] != signature:
                raise Conflict("stop request changed", current["revision"])
            return inspect(space, sid)
        if body.expected_revision != current["revision"]:
            raise Conflict("stale stop", current["revision"])
        if current["state"] != "running":
            return inspect(space, sid)
        if entry is None:
            raise Conflict("runtime not owned by this process", current["revision"])
        entry.stops[body.request_id] = signature
        stopping = journal.transition(space, sid, "stop:" + uuid4().hex, "stop", current["revision"])
        end_text(entry)
        # The service owns termination, not the HTTP request that initiated it.
        # Keep a strong reference so disconnects and matching retries cannot
        # abandon cleanup or start a second close operation.
        entry.cleanup_task = asyncio.create_task(finish_stop(space, sid, entry, stopping["revision"]))
        if not await asyncio.shield(entry.cleanup_task):
            raise HTTPException(503, "runtime cleanup failed")
        return inspect(space, sid)

    async def finish_cleanup(entry):
        try:
            await asyncio.wait_for(close_owned(entry), 5)
        except Exception:
            return False
        return True

    @app.websocket("/v1/conversations/{sid}/audio")
    async def audio(websocket: WebSocket, sid: str):
        """A voice session's one socket. A browser cannot put a bearer header
        on a WebSocket, so the first text frame is a hello carrying the key
        and the PCM format; the key is never in the URL. Refusals say why,
        naming the rule, and close with 1008 before any audio moves."""
        await websocket.accept()

        async def refuse(reason):
            try:
                await websocket.send_text(json.dumps({"type": "error", "reason": reason}))
                await websocket.close(code=1008)
            except Exception:
                pass

        hello = None
        try:
            async with asyncio.timeout(5):
                first = await websocket.receive()
            if first.get("text") is not None:
                hello = json.loads(first["text"])
        except (TimeoutError, WebSocketDisconnect, ValueError, RuntimeError):
            hello = None
        if not isinstance(hello, dict) or hello.get("type") != "hello":
            await refuse("expected a hello control first")
            return
        key = hello.get("key")
        space = keys.get(key) if isinstance(key, str) else None
        if space is None:
            await refuse("unknown key")
            return
        if not permitted(roles.get(key, "full"), "POST", "/v1/conversations"):
            await refuse(f"key role {roles.get(key)} cannot start a session")
            return
        if await engine.space_deleted(space) is not None:
            await refuse(f"space {space!r} was deleted")
            return
        if shutting_down:
            await refuse("conversation service is shutting down")
            return
        try:
            current = journal.get(space, sid)
        except (NotFound, InvalidInput):
            await refuse("unknown session")
            return
        if current["mode"] != "voice":
            await refuse("not a voice session")
            return
        if current["state"] != "created" or (space, sid) in owned:
            await refuse("session is not waiting for audio")
            return
        chosen = catalog.get(current["persona"]) if catalog is not None and current["persona"] else None
        if chosen is None:
            await refuse("persona is no longer available")
            return
        if current.get("persona_fingerprint") != catalog.fingerprint(current["persona"]):
            await refuse("model settings changed; create a new voice session")
            return
        try:
            sample_rate = hello.get("sample_rate")
            if not isinstance(sample_rate, int):
                raise ValueError("sample rate must be an integer")
            transport = WebSocketAudioTransport(websocket, sample_rate=sample_rate,
                                                channels=hello.get("channels", 1))
        except (ValueError, TypeError):
            await refuse("unsupported audio format")
            return
        try:
            fixed = RecallScope.from_mapping(current["recall_scope"])
            session = chosen.voice(engine, space, sid, transport_factory=lambda: transport, capture=True, **fixed.kwargs())
            journal.transition(space, sid, "start:" + uuid4().hex, "start", current["revision"])
        except Exception:
            await refuse("voice session failed to initialize")
            return
        owned[(space, sid)] = OwnedSession(session)
        await websocket.send_text(json.dumps({"type": "ready", "session_id": sid}))
        try:
            await session.run()
        except asyncio.CancelledError:
            # The host closed it (stop or shutdown), which the journal already
            # records; only a cancelled handler task must keep propagating.
            active_task = asyncio.current_task()
            if active_task is not None and active_task.cancelling():
                raise
        except Exception:
            pass  # the journal says failed below; the transport told the client what it could
        finally:
            try:
                latest = journal.get(space, sid) if journal is not None else None
                if latest is not None and latest["state"] == "running":
                    # The session's own verdict: ended on its input's end,
                    # interrupted when a host or this task cut it, failed
                    # otherwise. A socket the server closed on its way down
                    # ended the input, but the host did that, not the caller.
                    verdict = "interrupted" if session.state == "ended" and shutting_down else session.state
                    action = {"ended": "end", "interrupted": "interrupt"}.get(verdict, "fail")
                    journal.transition(space, sid, f"{action}:" + uuid4().hex, action, latest["revision"])
                    owned.pop((space, sid), None)
            except Exception:
                pass
            await transport.aclose()

    async def finish_stop(space, sid, entry, revision):
        if not await finish_cleanup(entry):
            journal.transition(space, sid, "failure:" + uuid4().hex, "fail", revision)
            return False
        journal.transition(space, sid, "end:" + uuid4().hex, "end", revision)
        return True


    # The mounted app answers /v1/status, so it must know the worker; its own
    # lifespan never runs under a mount, so ownership stays with this one.
    memory_app = create_app(engine, keys, conversations=True, worker=worker,
                            agent_catalog=agent_catalog, agent_plan_store=agent_plan_store,
                            agent_run_service=agent_run_service, document_ocr=document_ocr, document_media=document_media,
                            document_import_service=document_import_service,
                            directory_sync_service=directory_sync_service,
                            ingest_concurrency=ingest_concurrency, roles=roles,
                            model_connections_available=model_connections_available, vision_available=vision_available)
    app.state.memory_app = memory_app
    app.state.ingest_lane_width = memory_app.state.ingest_lane_width
    app.mount("/", memory_app)
    return app
