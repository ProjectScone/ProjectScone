"""Opt-in local agent catalog and state ownership for standard HTTP hosts."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
from ipaddress import ip_address
import json
import os
from pathlib import Path
import re
import stat
from typing import TYPE_CHECKING, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..agents.catalog import AgentCatalog, AgentDefinition, AgentModel, Identifier
from ..agents.evidence_loop import ToolModel
from ..agents.plan_store import AgentPlanStore
from ..agents.run_service import AgentRunService
from ..memory.engine import MemoryEngine
from ..providers.self_hosted import validate_self_hosted_endpoint, validate_self_hosted_identifier
from ..retrieval.recall_scope import RecallScope
from .model_connections import ModelConnection, api_key

if TYPE_CHECKING:
    from fastapi import FastAPI

_MAX_CONFIG_BYTES = 1048576
_ENV_NAME = r'^[A-Za-z_][A-Za-z0-9_]*$'


class LocalAgentModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    model_id: Identifier
    label: str = Field(min_length=1, max_length=256)
    revision: Identifier
    base_url: str
    model: str
    protocol: Literal['native', 'structured'] = 'native'
    timeout_s: float = Field(default=120.0, ge=1, le=300, allow_inf_nan=False)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    max_response_bytes: int = Field(default=128000, ge=1024, le=512000)
    think: bool | None = None
    api_key_env: str | None = Field(default=None, min_length=1, max_length=128, pattern=_ENV_NAME)

    @field_validator('base_url')
    @classmethod
    def local_endpoint(cls, value: str) -> str:
        normalized = validate_self_hosted_endpoint(value)
        host = urlsplit(normalized).hostname
        if host != 'localhost' and (host is None or not ip_address(host).is_loopback):
            raise ValueError('agent model endpoint must be loopback')
        return normalized

    @field_validator('model')
    @classmethod
    def model_identifier(cls, value: str) -> str:
        return validate_self_hosted_identifier(value)

    def create(self) -> ToolModel:
        from ..providers.tool_chat import SelfHostedToolChat
        from ..providers.structured_tool_chat import SelfHostedStructuredToolChat

        connection = ModelConnection(base_url=self.base_url, model=self.model,
            timeout_s=self.timeout_s, api_key_env=self.api_key_env)
        provider = SelfHostedStructuredToolChat if self.protocol == 'structured' else SelfHostedToolChat
        return provider(self.base_url, self.model, api_key=api_key(connection), timeout_s=self.timeout_s,
            max_tokens=self.max_tokens, max_response_bytes=self.max_response_bytes, think=self.think)

    def registered(self) -> AgentModel:
        revision = hashlib.sha256(self.model_dump_json().encode()).hexdigest()
        return AgentModel(self.model_id, self.label, revision, self.create)


class AgentRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    schema_version: Literal[1]
    state_dir: str = Field(min_length=1, max_length=4096)
    key_env: str = Field(min_length=1, max_length=128, pattern=_ENV_NAME)
    models: tuple[LocalAgentModel, ...] = Field(min_length=1, max_length=64)
    agents: tuple[AgentDefinition, ...] = Field(min_length=1, max_length=32)
    max_active: int = Field(default=4, ge=1, le=32)
    max_parallel_tasks: int = Field(default=1, ge=1, le=8)
    max_runs: int = Field(default=4096, ge=1, le=100000)
    max_plans: int = Field(default=4096, ge=1, le=100000)
    deadline_s: float = Field(default=120.0, gt=0, le=300, allow_inf_nan=False)

    @field_validator('state_dir')
    @classmethod
    def state_path(cls, value: str) -> str:
        if not value.strip() or '\0' in value:
            raise ValueError('state_dir must name a directory')
        return value

    @field_validator('schema_version', mode='before')
    @classmethod
    def exact_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError('schema_version must be 1')
        return value

    @classmethod
    def read(cls, path: Path) -> Self:
        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError('duplicate configuration key')
                result[name] = value
            return result

        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, 'rb') as source:
                info = os.fstat(source.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                    raise ValueError('private configuration required')
                raw = source.read(_MAX_CONFIG_BYTES + 1)
            if len(raw) > _MAX_CONFIG_BYTES:
                raise ValueError('configuration byte limit')
            json.loads(raw, object_pairs_hook=unique)
            return cls.model_validate_json(raw)
        except (OSError, ValueError, RecursionError):
            raise ValueError('Agent configuration must be valid bounded JSON in an owned 0600 regular file') from None


@dataclass(frozen=True)
class AgentRuntime:
    catalog: AgentCatalog
    plans: AgentPlanStore
    service: AgentRunService

    def close_idle(self) -> None:
        """Release construction resources; refuse to abandon admitted work."""
        self.service.close_idle()
        self.plans.close()

    async def aclose(self) -> None:
        try:
            await self.service.aclose()
        finally:
            self.plans.close()

    def own(self, app: FastAPI) -> FastAPI:
        original = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(host: FastAPI):
            try:
                async with original(host) as state:
                    yield state
            finally:
                await self.aclose()

        app.router.lifespan_context = lifespan
        app.state.agent_runtime = self
        return app


def load_agent_runtime(path: str | Path, memory: MemoryEngine) -> AgentRuntime:
    """Read configuration, bind factories and open private state; no model calls."""
    config_path = Path(path).expanduser().absolute()
    config = AgentRuntimeConfig.read(config_path)
    catalog = AgentCatalog(models=[model.registered() for model in config.models], agents=config.agents)
    secret = os.environ.get(config.key_env, '')
    if not re.fullmatch(r'[a-fA-F0-9]{64}', secret):
        raise ValueError('Agent encryption key environment variable must contain exactly 64 hex characters')
    key = bytes.fromhex(secret)
    target = Path(config.state_dir).expanduser()
    if not target.is_absolute():
        target = config_path.parent / target
    target.mkdir(mode=0o700, exist_ok=True)
    info = target.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Agent state directory must be owned and private')
    plans = AgentPlanStore(target / 'plans.sqlite', key=key, max_plans=config.max_plans)
    try:
        service = AgentRunService(target / 'runs', key=key, catalog=catalog, plans=plans, memory=memory,
            scope_for=lambda _: RecallScope.validated(), max_active=config.max_active,
            max_parallel_tasks=config.max_parallel_tasks, max_runs=config.max_runs, deadline_s=config.deadline_s)
    except BaseException:
        plans.close()
        raise
    return AgentRuntime(catalog, plans, service)
