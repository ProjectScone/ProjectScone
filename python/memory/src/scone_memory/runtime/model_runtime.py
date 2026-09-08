"""Bind saved self-hosted connections to fresh sessions and future extraction passes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import hmac
from ipaddress import ip_address
import json

from fastapi import HTTPException, Request

from ..core.errors import InvalidInput
from ..ingestion.derive import Deriver
from ..ingestion.distill import Distiller
from ..ingestion.worker import ConsolidationWorker
from ..providers.llm import OpenAICompatibleChat, OpenAICompatibleTextModel
from ..realtime.catalog import PersonaCatalog
from ..realtime.persona import ModelChoice, Persona, VoiceChoice
from ..realtime.providers import BoundPersona, ProviderRegistry
from ..realtime.text import TextConversation
from .config import Settings
from .model_connections import ModelConnection, ModelConnectionStore, api_key


def connection_defaults(settings: Settings) -> dict[str, ModelConnection | None]:
    if not settings.chat_url and not settings.chat_model:
        return {}
    if not settings.chat_url or not settings.chat_model:
        raise ValueError('Self-hosted model defaults require both SCONE_CHAT_URL and SCONE_CHAT_MODEL')
    connection = ModelConnection(base_url=settings.chat_url, model=settings.chat_model,
                                 timeout_s=settings.chat_timeout,
                                 api_key_env='SCONE_CHAT_API_KEY' if settings.chat_api_key else None)
    return {'chat': connection, 'extraction': connection}


def host_admin_enabled(settings: Settings) -> bool:
    return (bool(settings.model_connections) and settings.host in ('127.0.0.1', 'localhost', '::1')
            and len(settings.keys) == 1 and settings.roles.get(next(iter(settings.keys)), 'full') == 'full')


def authorize_host_admin(settings: Settings):
    """Bind bearer, socket peer and Host checks; forwarded headers grant nothing."""
    async def authorize(request: Request) -> None:
        if not host_admin_enabled(settings):
            raise HTTPException(403, 'Host administration is unavailable')
        host = request.url.hostname
        if host not in ('127.0.0.1', 'localhost', '::1'):
            raise HTTPException(403, 'Host administration requires a loopback Host')
        try:
            local_peer = request.client is not None and ip_address(request.client.host).is_loopback
        except ValueError:
            local_peer = False
        if not local_peer:
            raise HTTPException(403, 'Host administration requires a loopback client')
        scheme, _, token = request.headers.get('authorization', '').partition(' ')
        if scheme.lower() != 'bearer' or not hmac.compare_digest(token, next(iter(settings.keys))):
            raise HTTPException(401, 'Host administration requires the configured bearer key')
    return authorize


def authorize_image_write(settings: Settings, engine):
    async def authorize(request: Request) -> str:
        scheme, _, token = request.headers.get('authorization', '').partition(' ')
        if scheme.lower() != 'bearer' or token not in settings.keys:
            raise HTTPException(401, 'An authorized space bearer key is required')
        if settings.roles.get(token, 'full') not in ('write', 'full'):
            raise HTTPException(403, 'This key cannot request image understanding')
        space = settings.keys[token]
        if await engine.space_deleted(space) is not None:
            raise HTTPException(404, 'The authorized space was deleted')
        return space
    return authorize


def self_hosted_vision_factory(store: ModelConnectionStore):
    def create():
        from ..providers.vision import SelfHostedOpenAIVision

        connection = store.get('vision')
        if connection is None:
            return None
        return SelfHostedOpenAIVision(connection.base_url, connection.model, api_key=api_key(connection),
                                      timeout=connection.timeout_s)
    return create


def text_model_factory(connection: ModelConnection, *, think: bool | None = None):
    def create():
        return OpenAICompatibleTextModel(connection.base_url, connection.model, api_key=api_key(connection),
                                          timeout=connection.timeout_s, think=think, trust_env=False)
    return create


def self_hosted_text_runtime(engine, store: ModelConnectionStore, *, think: bool | None = None):
    def create(space, session_id, scope):
        connection = store.get('chat')
        if connection is None:
            raise InvalidInput('No self-hosted chat model is configured')
        # The immutable connection stays with this session, even across edits.
        return TextConversation(engine, space, session_id, text_model_factory(connection, think=think),
                                turn_timeout=connection.timeout_s, **scope.kwargs())
    return create


def _alias(role: str, connection: ModelConnection) -> str:
    value = json.dumps(connection.model_dump(), sort_keys=True, separators=(',', ':')).encode()
    return role + '-' + hashlib.sha256(value).hexdigest()[:16]


@dataclass(frozen=True)
class _SelfHostedBoundPersona(BoundPersona):
    turn_timeout: float

    def text(self, *args, **options):
        options.setdefault('turn_timeout', self.turn_timeout)
        return super().text(*args, **options)

    def voice(self, *args, **options):
        options.setdefault('turn_timeout', self.turn_timeout)
        return super().voice(*args, **options)


class DynamicSelfHostedCatalog:
    """A self-hosted voice persona exists only while all three services are configured."""

    def __init__(self, store: ModelConnectionStore, *, think: bool | None = None):
        self.store = store
        self.think = think

    def _current(self, *, legacy: bool = False) -> PersonaCatalog:
        from ..providers.speech import SelfHostedOpenAISpeech
        from ..providers.transcription import SelfHostedOpenAITranscription

        snapshot = self.store.snapshot()['connections']
        if not isinstance(snapshot, dict):
            raise ValueError('Invalid self-hosted model configuration snapshot')
        connections = {role: ModelConnection.model_validate(value) if value is not None else None
                       for role, value in snapshot.items()}
        chat, stt, tts = (connections[role] for role in ('chat', 'transcription', 'speech'))
        if chat is None or stt is None or tts is None or tts.voice is None:
            return PersonaCatalog((), {})
        selected_voice = tts.voice
        reply_id, stt_id, tts_id = (_alias(role, value) for role, value in
                                  (('chat', chat), ('transcription', stt), ('speech', tts)))
        voice_id = 'selected-voice'
        provider = 'local-openai' if legacy else 'self-hosted-openai'
        persona = Persona(id='local-voice' if legacy else 'self-hosted-voice',
                          name='Local voice' if legacy else 'Self-hosted voice',
                          instructions='You are a helpful voice assistant.',
                          reply=ModelChoice(provider=provider, model=reply_id),
                          transcription=ModelChoice(provider=provider, model=stt_id),
                          speech=VoiceChoice(provider=provider, model=tts_id, voice=voice_id))
        registry = ProviderRegistry(
            reply={(provider, reply_id): text_model_factory(chat, think=self.think)},
            transcription={(provider, stt_id): lambda: SelfHostedOpenAITranscription(
                base_url=stt.base_url, model=stt.model, api_key=api_key(stt), rate=16000, timeout=stt.timeout_s)},
            speech={(provider, tts_id, voice_id): lambda: SelfHostedOpenAISpeech(
                base_url=tts.base_url, model=tts.model, voice=selected_voice, api_key=api_key(tts),
                sample_rate=tts.sample_rate, timeout=tts.timeout_s)},
        )
        bound = registry.resolve(persona)
        selected = _SelfHostedBoundPersona(bound.persona, bound.model_factory, bound.stt_factory,
                                      bound.tts_factory, bound.activity_factory, chat.timeout_s)
        return PersonaCatalog((persona,), {persona.id: selected})

    @property
    def personas(self):
        return self._current().personas

    @property
    def revision(self):
        return self._current().revision

    def get(self, persona_id):
        return self._current(legacy=persona_id == 'local-voice').get(persona_id)

    def fingerprint(self, persona_id):
        return self._current(legacy=persona_id == 'local-voice').fingerprint(persona_id)

    def name(self, persona_id):
        return 'Self-hosted voice' if self.get(persona_id) is not None else None

    def public(self, *, voice=False):
        return self._current().public(voice=voice)


class _SelfHostedChat:
    def __init__(self, connection: ModelConnection, think: bool | None):
        self.connection, self.think = connection, think

    def _model(self):
        connection = self.connection
        return OpenAICompatibleChat(connection.base_url, connection.model, api_key=api_key(connection),
                                    timeout=connection.timeout_s, think=self.think, trust_env=False)

    async def complete(self, system: str, user: str) -> str:
        return await self._model().complete(system, user)

    async def complete_structured(self, system: str, user: str, schema: dict[str, object]) -> str:
        return await self._model().complete_structured(system, user, schema)


class SelfHostedModelWorker:
    """A stable lifecycle owner that snapshots the worker for every whole pass."""

    def __init__(self, engine, settings: Settings, store: ModelConnectionStore):
        self.engine, self.settings, self.store = engine, settings, store
        self.spaces = sorted(set(settings.keys.values()))
        self.retention = dict(settings.retention or {})
        self.batch = settings.distill_batch
        self.last: dict = {}
        self.passes = 0
        self._selected: ModelConnection | None = None
        self._current: ConsolidationWorker | None = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        self.refresh(initial=True)

    def refresh(self, *, initial=False) -> None:
        selected = self.store.get('extraction')
        if not initial and selected == self._selected:
            return
        current = None
        if selected is not None or self.retention:
            chat = _SelfHostedChat(selected, self.settings.chat_think) if selected is not None else None
            distiller = Distiller(self.engine, chat, accept_at=self.settings.distill_accept_at) if chat else None
            deriver = Deriver(self.engine, chat) if chat and self.settings.derive else None
            current = ConsolidationWorker(self.engine, distiller, self.spaces, self.settings.distill_interval_s,
                                          self.batch, retention=self.retention, deriver=deriver)
        self._current, self._selected = current, selected

    @property
    def distiller(self):
        return self._current.distiller if self._current else None

    @property
    def deriver(self):
        return self._current.deriver if self._current else None

    async def run_once(self, space):
        async with self._lock:
            self.refresh()
            current = self._current
            if current is None:
                raise InvalidInput('No extraction model or retention policy is configured')
            result = await current.run_once(space)
            self.last[space] = result
            self.passes += 1
            return result

    async def run_all(self):
        return [await self.run_once(space) for space in self.spaces]

    async def _loop(self):
        while not self._stop.is_set():
            self.refresh()
            if self._current is not None:
                await self.run_all()
            try:
                await asyncio.wait_for(self._stop.wait(), self.settings.distill_interval_s)
            except TimeoutError:
                pass

    def start(self):
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._loop(), name='scone-self-hosted-consolidation')

    async def stop(self):
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, 10)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    @property
    def running(self):
        return self._task is not None and not self._task.done()


# Existing launchers and saved integrations keep their imports.
local_admin_enabled = host_admin_enabled
authorize_local_admin = authorize_host_admin
local_vision_factory = self_hosted_vision_factory
local_text_runtime = self_hosted_text_runtime
DynamicLocalCatalog = DynamicSelfHostedCatalog
LocalModelWorker = SelfHostedModelWorker
