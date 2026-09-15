"""An OpenAI-compatible chat route that remembers: ``POST /v1/openai/chat/completions``.

An application that already speaks chat completions points its base URL at
``/v1/openai`` and sends its bearer key as the API key. The route recalls
from the key's space with the latest user message, puts what it found in
front of the conversation as one delimited system block, asks the server's
configured chat model, and keeps the user turn and the reply as conversation
episodes. The answer is the chat-completions shape plus a ``scone`` record
saying what was recalled, what was injected, what the request asked for that
was not forwarded, and what was kept.

The model is the server's own (SCONE_CHAT_URL and SCONE_CHAT_MODEL). A
request names no endpoint and no credential, and its ``model`` is echoed in
the record, never obeyed. The configured model takes one system prompt and
one user prompt, so earlier turns reach it flattened into the user prompt,
and sampling fields are accepted for the shape only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from typing import Optional, cast
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.errors import InvalidInput
from ..memory.engine import MemoryEngine, Record
from ..providers.llm import ChatError, ChatModel
from ..realtime.context import MemoryContext
from ..retrieval.conversation_plan import plan_conversation_retrieval
from .app import read_bounded
from .responses import LedgerJSONResponse

#: Bytes one request body may take; more is refused before it is parsed.
MAX_BODY_BYTES = 256_000
#: Bytes of text one message may carry, the same bound a native text turn has.
MAX_MESSAGE_BYTES = 32_000
#: Messages one request may carry.
MAX_MESSAGES = 200
#: Passages recalled for one request, at most; the record says how many were left out.
RECALL_LIMIT = 5
#: Bytes the recalled block may take, at most; the record says how many passages did not fit.
MAX_CONTEXT_BYTES = 8_000
#: Passages of the session's own stored turns looked at to find turns the
#: request did not resend; the record says when the probe filled.
SESSION_PROBE_LIMIT = 20

#: A session's stored identity is its name behind this prefix, so a name can
#: equal neither a document's source nor a native conversation's id.
SESSION_PREFIX = "openai:"
#: Characters a session name may take: the stored identity holds at most 128.
MAX_SESSION_CHARS = 128 - len(SESSION_PREFIX)
_SESSION = re.compile(rf"[A-Za-z0-9._:-]{{1,{MAX_SESSION_CHARS}}}")
_ROLES = ("system", "developer", "user", "assistant")


class _Message(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    role: str
    content: str | list[dict[str, object]]


class ChatCompletionBody(BaseModel):
    """The chat-completions request this route reads. Tools, images, audio
    and ``n`` are refused rather than dropped; the sampling fields are
    accepted so an ordinary client's request is not refused, and named in
    ``scone.not_forwarded`` because the configured model does not take them."""

    model_config = ConfigDict(hide_input_in_errors=True, extra="forbid")

    model: str = Field(min_length=1, max_length=256)
    messages: list[_Message] = Field(min_length=1)
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[str | list[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    user: Optional[str] = None


def _parsed(raw: bytes) -> ChatCompletionBody:
    try:
        return ChatCompletionBody.model_validate_json(raw)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first.get("loc", ()))
        raise InvalidInput(f"{where}: {first['msg']}" if where else first["msg"]) from None


def _text_of(index: int, message: _Message) -> str:
    """A message's text: a string, or text parts read in order. A part of
    any other type is refused, because this route could not carry it."""
    if message.role not in _ROLES:
        raise InvalidInput(f"messages[{index}].role: this route takes {', '.join(_ROLES)}; got {message.role!r}")
    if isinstance(message.content, str):
        text = message.content
    else:
        texts = []
        for part in message.content:
            if part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise InvalidInput(f"messages[{index}].content: this route carries text parts only; "
                                   f"got a {str(part.get('type'))[:40]!r} part")
            texts.append(str(part["text"]))
        text = "".join(texts)
    size = len(text.encode("utf-8"))
    if size > MAX_MESSAGE_BYTES:
        raise InvalidInput(f"messages[{index}] takes at most {MAX_MESSAGE_BYTES} bytes of text, got {size}")
    return text


def _session_of(request: Request) -> str | None:
    """The session name the request gave, or None for a session of its own."""
    named = request.headers.get("x-scone-session")
    if named is not None and not _SESSION.fullmatch(named):
        raise InvalidInput(f"x-scone-session is 1..{MAX_SESSION_CHARS} letters, digits, '.', '_', ':' or '-'")
    return named


def _user_prompt(turns: list[dict[str, object]]) -> str:
    """The latest user message alone, or after the earlier turns as one
    JSON value, so a turn's text can never pass itself off as another role."""
    *earlier, latest = turns
    if not earlier:
        return str(latest["content"])
    return ("Earlier turns of this conversation, oldest first, as JSON:\n"
            + json.dumps(earlier, ensure_ascii=False) + "\n\nLatest user message:\n" + str(latest["content"]))


async def _unsent_turns(engine: MemoryEngine, space: str, session_id: str,
                        turns: list[dict[str, object]]) -> dict[str, object]:
    """The session's stored turns that bear on the latest message but that
    the request does not carry, so recall may admit them: a client that
    trims its history, or sends only the latest message, still has them as
    memory. A turn is held when any of its probed passages is inside a
    message the request sent; the probe is a scoped recall, not a walk of
    the space, and says when it filled."""
    record: dict[str, object] = {"status": "probed", "probe_limit": SESSION_PROBE_LIMIT, "probed": 0,
                                 "limit_reached": False, "admitted_turn_ids": [], "error_type": None}
    query = plan_conversation_retrieval(str(turns[-1]["content"])).query
    try:
        found = await engine.recall(space, query, limit=SESSION_PROBE_LIMIT, kind="conversation",
                                    where={"session_id": session_id})
    except Exception as failure:
        logging.getLogger(__name__).warning("session_probe.failed", extra={
            "event": "session_probe.failed", "session_id": session_id, "exception_type": type(failure).__name__})
        return {**record, "status": "failed", "error_type": type(failure).__name__}
    sent = [str(turn["content"]) for turn in turns]
    held, seen = set(), []
    for item in found.items:
        turn_id = item.metadata.get("turn_id")
        if turn_id is None:
            continue
        if turn_id not in seen:
            seen.append(turn_id)
        if any(item.text.strip() in text for text in sent):
            held.add(turn_id)
    return {**record, "probed": len(found.items), "limit_reached": len(found.items) >= SESSION_PROBE_LIMIT,
            "admitted_turn_ids": [turn_id for turn_id in seen if turn_id not in held]}


async def _recalled(engine: MemoryEngine, space: str, session_id: str, named: bool,
                    turns: list[dict[str, object]]) -> tuple[dict[str, object], str | None, dict[str, object] | None]:
    """What recall found for the latest turn, the block to inject (or None)
    and the record of what was injected. Preparation is the native text
    conversation's, so bounds, exclusions and the packet format are its own;
    the session's stored turns the request did not resend are admitted, as
    a native conversation admits turns that have left its window."""
    unsent: dict[str, object] = {"status": "skipped", "probe_limit": SESSION_PROBE_LIMIT, "probed": 0,
                                 "limit_reached": False, "admitted_turn_ids": [], "error_type": None}
    if named:
        unsent = await _unsent_turns(engine, space, session_id, turns)
    context = MemoryContext(engine, space, session_id, limit=RECALL_LIMIT, max_context_bytes=MAX_CONTEXT_BYTES)
    admitted = frozenset(cast(list[str], unsent["admitted_turn_ids"]))
    prepared, receipt = await context.prepare(turns, admit_turn_ids=admitted)
    recall: dict[str, object] = {
        "status": receipt["status"], "items": [], "claim_ids": [], "omitted_count": receipt["omitted_count"],
        "limit": RECALL_LIMIT, "max_context_bytes": MAX_CONTEXT_BYTES,
        "low_confidence": receipt["low_confidence"], "degraded": receipt["degraded"],
        "error_type": receipt["error_type"], "session_turns": unsent}
    if "query_formulation" in receipt:
        recall["query_formulation"] = receipt["query_formulation"]
    if receipt["status"] != "prepared":
        return recall, None, None
    # The context puts its one block in front of the turns it was given;
    # it reaches the model as a system block between boundaries instead.
    block = str(prepared[0]["content"])
    payload = json.loads(block[block.index("{"):])
    recall["items"] = [{"episode_id": source["episode_id"], "chunk_id": source["chunk_id"],
                        "source": source["source"], "created_at": source["created_at"]}
                       for source in payload["sources"]]
    recall["claim_ids"] = [claim["fact_id"] for claim in payload.get("claims", [])]
    boundary = uuid4().hex[:16]
    wrapped = f'<scone-memory boundary="{boundary}">\n{block}\n</scone-memory boundary="{boundary}">'
    injected: dict[str, object] = {"role": "system", "boundary": boundary, "bytes": len(wrapped.encode("utf-8")),
                                   "sha256": hashlib.sha256(wrapped.encode("utf-8")).hexdigest()}
    return recall, wrapped, injected


def mount_openai_proxy_routes(app: FastAPI, engine: MemoryEngine, space_for: Callable[..., object],
                              synthesis_factory: Callable[[], ChatModel | None] | None, *,
                              assert_current_space: Callable[[Request, str], None]) -> None:
    @app.post("/v1/openai/chat/completions")
    async def chat_completions(request: Request, space: str = Depends(space_for)) -> JSONResponse:
        """Recall, inject, forward to the configured model, keep the turn.
        Streaming is refused in this version; so is a request without a
        latest user message, and a server without a model."""
        body = _parsed(await read_bounded(request, MAX_BODY_BYTES, noun="a chat completion request"))
        if body.stream:
            raise InvalidInput("stream is not supported by this route yet; send stream false or leave it out")
        if len(body.messages) > MAX_MESSAGES:
            raise InvalidInput(f"messages takes at most {MAX_MESSAGES} messages, got {len(body.messages)}")
        texts = [(message.role, _text_of(index, message)) for index, message in enumerate(body.messages)]
        if texts[-1][0] != "user" or not texts[-1][1].strip():
            raise InvalidInput("the conversation must end with a nonblank user message")
        named = _session_of(request)
        session = named if named is not None else uuid4().hex
        session_id = SESSION_PREFIX + session
        model = synthesis_factory() if synthesis_factory is not None else None
        if model is None:
            raise InvalidInput("the OpenAI-compatible route needs a model; none is configured "
                               "(SCONE_CHAT_URL and SCONE_CHAT_MODEL)")

        turns: list[dict[str, object]] = [{"role": role, "content": text} for role, text in texts
                                          if role in ("user", "assistant")]
        recall, wrapped, injected = await _recalled(engine, space, session_id, named is not None, turns)
        system_parts = [text for role, text in texts if role in ("system", "developer")]
        if wrapped is not None:
            system_parts.append(wrapped)
        try:
            reply = await model.complete("\n\n".join(system_parts), _user_prompt(turns))
        except ChatError as error:
            # The provider's words stay in the server: they can carry its
            # account, key fragments or internal hosts.
            logging.getLogger(__name__).warning("chat.failed", extra={
                "event": "chat.failed", "session_id": session_id, "exception_type": type(error).__name__})
            return LedgerJSONResponse({"error": "the configured chat model did not return a reply"}, status_code=502)

        assert_current_space(request, space)
        turn_id = uuid4().hex
        capture = await _capture(engine, space, session_id, turn_id, texts[-1][1], reply)
        name = getattr(model, "model", None)
        record = {
            "id": f"chatcmpl-{turn_id}", "object": "chat.completion", "created": int(time.time()),
            "model": name if isinstance(name, str) else "unknown",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "scone": {"session": session, "session_id": session_id, "turn_id": turn_id, "requested_model": body.model,
                      "not_forwarded": sorted(body.model_fields_set - {"messages", "stream"}),
                      "provider_completion": "unverified", "recall": recall, "injected": injected,
                      "capture": capture},
        }
        items = cast(list[dict[str, object]], recall["items"])
        headers = {"x-scone-session": session, "x-scone-recall-status": str(recall["status"]),
                   "x-scone-injected-bytes": str(injected["bytes"] if injected else 0),
                   "x-scone-recalled-episodes": ",".join(str(item["episode_id"]) for item in items),
                   "x-scone-capture-status": str(capture["status"])}
        return LedgerJSONResponse(record, headers=headers)


async def _capture(engine: MemoryEngine, space: str, session_id: str, turn_id: str,
                   question: str, reply: str) -> dict[str, object]:
    """The user turn and the reply as two conversation episodes, shaped as
    a native text conversation keeps them and written one at a time, user
    first, so a reply that cannot be kept never takes the user turn with it.
    A blank reply is not written. The model has already answered, so a
    failed write is reported, not raised. Neither write takes the ingest
    lane: a lane that refused could not be retried once the reply is out."""
    kept: dict[str, dict[str, object]] = {}
    for role, text in (("user", question), ("assistant", reply)):
        if role == "assistant" and kept["user"]["status"] != "captured":
            kept[role] = {"status": "not_attempted", "episode_id": None, "error_type": None}
            continue
        if not text.strip():
            kept[role] = {"status": "blank", "episode_id": None, "error_type": None}
            continue
        metadata = dict(integration="scone-openai-proxy", session_id=session_id, turn_id=turn_id, role=role,
                        representation="aggregated_text",
                        capture_status="submitted" if role == "user" else "aggregated")
        if role == "assistant":
            metadata["provider_completion"] = "unverified"
        record = Record(text, kind="conversation", source=session_id, metadata=metadata,
                        dedup_key=f"scone-openai:{session_id}:{turn_id}:{role}")
        try:
            [added] = await engine.remember_many(space, [record])
        except Exception as error:
            logging.getLogger(__name__).warning("capture.failed", extra={
                "event": "capture.failed", "session_id": session_id, "role": role,
                "exception_type": type(error).__name__})
            kept[role] = {"status": "failed", "episode_id": None, "error_type": type(error).__name__}
            continue
        kept[role] = {"status": "captured", "episode_id": added.episode_id, "error_type": None}
    user, assistant = kept["user"], kept["assistant"]
    status = ("failed" if user["status"] != "captured"
              else "captured" if assistant["status"] == "captured" else "partial")
    return {"status": status,
            "episode_ids": [each["episode_id"] for each in (user, assistant) if each["status"] == "captured"],
            "error_type": user["error_type"] or assistant["error_type"], "user": user, "assistant": assistant}
