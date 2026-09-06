"""Optional single-process conversation service; native memory routes are mounted.

Linux/macOS local-filesystem journal ownership only. Provider factories and
engines are server configuration. Turn results are cached for this process,
not reconstructed or retried after restart; lifecycle receipts are durable.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hmac
import os
from pathlib import Path
import stat
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..engine import check_space
from ..errors import Conflict, InvalidInput, NotFound
from ..session_journal import SessionJournal
from .app import create_app, episode_json


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


class Create(Command):
    capture: bool


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


def create_conversation_app(engine, keys, journal_path, runtime_factory, *, max_sessions=100, max_turns=100):
    """The caller owns engine lifecycle; service owns journal and runtime tasks.

    runtime_factory(space, sid) supplies async reply(text) and close(). None
    advertises unavailable text. Do not use non-cooperative/untrusted runtimes:
    cancellation and cleanup use cooperative asyncio, not process termination.
    """
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
    owned: dict[tuple[str, str], OwnedSession] = {}
    creates: dict[tuple[str, str], str] = {}
    journal = None

    async def close_owned(entry):
        if entry.active is not None and not entry.active.done():
            entry.active.cancel()
            await asyncio.gather(entry.active, return_exceptions=True)
        await entry.runtime.close()

    @asynccontextmanager
    async def lifespan(_app):
        nonlocal journal
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
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise InvalidInput("journal ownership lock must be a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("conversation journal is already owned by another service") from exc
            journal = SessionJournal(path)
            for space in sorted(set(keys.values())):
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
            yield
        finally:
            cleanup_errors = []
            try:
                if journal is not None:
                    for (space, sid), entry in owned.items():
                        if entry.cleanup_task is not None:
                            try:
                                if not await asyncio.shield(entry.cleanup_task):
                                    cleanup_errors.append(RuntimeError("runtime cleanup failed"))
                            except Exception as error:
                                cleanup_errors.append(error)
                            continue
                        try:
                            session = journal.get(space, sid)
                            if session["state"] in {"created", "running", "stopping"}:
                                journal.transition(space, sid, "shutdown:" + uuid4().hex, "interrupt", session["revision"])
                        except Exception as error:
                            cleanup_errors.append(error)
                        try:
                            await asyncio.wait_for(close_owned(entry), 5)
                        except Exception as error:
                            cleanup_errors.append(error)
            finally:
                if journal is not None:
                    journal.close()
                    journal = None
                owned.clear()
                creates.clear()
                os.close(descriptor)
            if cleanup_errors:
                raise RuntimeError("conversation service cleanup failed") from None

    app = FastAPI(title="scone-conversations", docs_url=None, redoc_url=None, lifespan=lifespan)

    def space_for(request: Request):
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer" and token:
            for key, space in keys.items():
                if hmac.compare_digest(token.encode(), key.encode()):
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

    def inspect(space, sid):
        result = journal.get(space, sid)
        entry = owned.get((space, sid))
        result["active_request_id"] = entry.active_id if entry else None
        return result

    @app.get("/v1/conversations/capabilities")
    async def capabilities(space=Depends(space_for)):
        return {"schema_version": 1, "text_configured": runtime_factory is not None,
                "voice": False, "video": False, "streaming": False,
                "reply_transport": "poll", "reply_replay": "process_lifetime",
                "provider_completion": "unverified", "max_sessions": max_sessions, "max_turns": max_turns}

    @app.get("/v1/conversations")
    async def sessions(after: str = "", limit: int = 100, space=Depends(space_for)):
        return journal.sessions(space, after=after, limit=limit)

    @app.post("/v1/conversations")
    async def create(body: Create, space=Depends(space_for)):
        if not body.capture:
            raise HTTPException(422, "this runtime requires explicit transcript capture consent")
        if runtime_factory is None:
            raise HTTPException(503, "text conversation runtime is not configured")
        key = (space, body.request_id)
        if key in creates:
            return inspect(space, creates[key])
        if len(creates) >= max_sessions:
            raise HTTPException(429, "conversation process capacity reached")
        receipt = journal.create(space, body.request_id)
        sid = receipt["session_id"]
        creates[key] = sid
        current = journal.get(space, sid)
        if current["state"] != "created":
            return inspect(space, sid)
        try:
            runtime = runtime_factory(space, sid)
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
    async def transcript(sid: str, space=Depends(space_for)):
        journal.get(space, sid)
        episodes = await engine.episodes(space, {"session_id": sid}, limit=201)
        return {"episodes": [episode_json(episode) for episode in episodes[:200]],
                "has_more": len(episodes) > 200}

    async def run_turn(space, sid, entry, request_id, text):
        receipt = entry.turns[request_id]["receipt"]
        try:
            result = await entry.runtime.reply(text)
            if journal.get(space, sid)["state"] != "running":
                receipt["status"] = "interrupted"
            else:
                receipt.update(status="completed", result=result)
        except asyncio.CancelledError:
            receipt["status"] = "interrupted"
            current = journal.get(space, sid)
            if current["state"] == "running":
                journal.transition(space, sid, "interrupted:" + uuid4().hex, "interrupt", current["revision"])
                entry.cleanup_task = asyncio.create_task(finish_cleanup(entry))
        except Exception:
            receipt.update(status="failed", error="conversation turn failed; do not automatically retry")
            current = journal.get(space, sid)
            if current["state"] == "running":
                journal.transition(space, sid, "failure:" + uuid4().hex, "fail", current["revision"])
                entry.cleanup_task = asyncio.create_task(finish_cleanup(entry))
        finally:
            entry.active_id = None

    @app.post("/v1/conversations/{sid}/turns", status_code=202)
    async def turn(sid: str, body: Turn, space=Depends(space_for)):
        current = journal.get(space, sid)
        entry = owned.get((space, sid))
        signature = body.model_dump()
        if entry and body.request_id in entry.turns:
            previous = entry.turns[body.request_id]
            if previous["signature"] != signature:
                raise Conflict("turn request changed", current["revision"])
            return previous["receipt"]
        if current["state"] != "running" or entry is None:
            raise Conflict("conversation is not running in this process", current["revision"])
        if body.expected_revision != current["revision"] or entry.active_id is not None:
            raise Conflict("stale or busy conversation", current["revision"])
        if not body.text.strip() or len(body.text.encode()) > 32000:
            raise HTTPException(422, "message must contain 1..32000 UTF-8 bytes of nonempty text")
        if len(entry.turns) >= max_turns:
            raise HTTPException(429, "conversation turn capacity reached")
        receipt = {"request_id": body.request_id, "status": "pending"}
        entry.turns[body.request_id] = {"signature": signature, "receipt": receipt}
        entry.active_id = body.request_id
        entry.active = asyncio.create_task(run_turn(space, sid, entry, body.request_id, body.text))
        return receipt

    @app.get("/v1/conversations/{sid}/turns/{request_id}")
    async def result(sid: str, request_id: str, space=Depends(space_for)):
        journal.get(space, sid)
        entry = owned.get((space, sid))
        if entry is None or request_id not in entry.turns:
            raise NotFound("turn receipt unavailable in this process")
        return entry.turns[request_id]["receipt"]

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

    async def finish_stop(space, sid, entry, revision):
        if not await finish_cleanup(entry):
            journal.transition(space, sid, "failure:" + uuid4().hex, "fail", revision)
            return False
        journal.transition(space, sid, "end:" + uuid4().hex, "end", revision)
        return True

    app.mount("/", create_app(engine, keys, console=False))
    return app
