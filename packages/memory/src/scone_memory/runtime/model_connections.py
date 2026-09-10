"""Private self-hosted service configuration, with atomic revision-checked updates.

The store uses POSIX file locking (Linux/macOS) and atomic local-filesystem
replacement. Defaults are read-only until a host administrator saves a change.
Only environment variable names are persisted for optional service tokens.
"""

from collections.abc import Mapping
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..providers.self_hosted import validate_self_hosted_endpoint

ModelRole = Literal['chat', 'extraction', 'vision', 'transcription', 'speech']
MODEL_ROLES: tuple[ModelRole, ...] = ('chat', 'extraction', 'vision', 'transcription', 'speech')
_MAX_FILE_BYTES = 64000


class ModelConnectionError(RuntimeError):
    """A sanitized configuration, credential or service failure."""


class ModelConnectionConflict(ModelConnectionError):
    """Another saved revision must be reviewed before replacing it."""


class ModelConnection(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid',
                              hide_input_in_errors=True, revalidate_instances='always')
    base_url: str
    model: str = Field(min_length=1, max_length=160)
    timeout_s: float = Field(default=180, ge=1, le=600, allow_inf_nan=False)
    api_key_env: str | None = Field(default=None, min_length=1, max_length=128,
                                    pattern=r'^[A-Za-z_][A-Za-z0-9_]*$')
    voice: str | None = Field(default=None, min_length=1, max_length=120)
    sample_rate: int = Field(default=24000, ge=8000, le=48000)

    @field_validator('base_url')
    @classmethod
    def self_hosted_url(cls, value: str) -> str:
        return validate_self_hosted_endpoint(value)

    @field_validator('model', 'voice')
    @classmethod
    def nonblank(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value)):
            raise ValueError('model and voice must be nonblank identifiers without control characters')
        return value


class _Document(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra='forbid', hide_input_in_errors=True)
    schema_version: Literal[1] = 1
    revision: int = Field(ge=0, le=2**63 - 2)
    connections: dict[ModelRole, ModelConnection | None]

    @field_validator('schema_version', mode='before')
    @classmethod
    def exact_schema(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError('schema_version must be 1')
        return value

    @model_validator(mode='after')
    def speech_voice(self):
        speech = self.connections.get('speech')
        if speech is not None and speech.voice is None:
            raise ValueError('speech requires a voice')
        return self


def _role(value: str) -> ModelRole:
    if value not in MODEL_ROLES:
        raise ValueError('unknown model connection role')
    return cast(ModelRole, value)


def api_key(connection: ModelConnection) -> str | None:
    """Resolve a selected environment variable at use time, never on save."""
    connection = ModelConnection.model_validate(connection)
    if connection.api_key_env is None:
        return None
    value = os.environ.get(connection.api_key_env)
    if not value or len(value) > 4096 or any(not 33 <= ord(char) <= 126 for char in value):
        raise ModelConnectionError('The configured service token environment variable is missing or invalid')
    return value


class ModelConnectionStore:
    def __init__(self, path: str | Path, defaults: Mapping[str, ModelConnection | None]):
        self.path = Path(path).expanduser()
        connections: dict[ModelRole, ModelConnection | None] = {role: None for role in MODEL_ROLES}
        for role, connection in defaults.items():
            connections[_role(role)] = ModelConnection.model_validate(connection) if connection is not None else None
        self._defaults = _Document(revision=0, connections=connections)
        self._read()  # Invalid persisted settings fail closed at host startup.

    def _read(self) -> _Document:
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return self._defaults
        except OSError:
            raise ModelConnectionError('Cannot read self-hosted model settings') from None
        try:
            with os.fdopen(fd, 'rb') as source:
                info = os.fstat(source.fileno())
                if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                    raise ModelConnectionError('Self-hosted model settings must be a private 0600 regular file')
                payload = source.read(_MAX_FILE_BYTES + 1)
            if len(payload) > _MAX_FILE_BYTES:
                raise ModelConnectionError('Self-hosted model settings exceed the file size limit')
            document = _Document.model_validate(json.loads(payload))
            connections = {role: document.connections.get(role) for role in MODEL_ROLES}
            return _Document(revision=document.revision, connections=connections)
        except (OSError, ValueError):
            raise ModelConnectionError('Self-hosted model settings are unreadable or invalid') from None

    def snapshot(self) -> dict[str, object]:
        return self._read().model_dump(mode='json')

    def get(self, role: str) -> ModelConnection | None:
        return self._read().connections[_role(role)]

    def replace(self, role: str, connection: ModelConnection | None, *, expected_revision: int) -> dict[str, object]:
        import fcntl

        selected = _role(role)
        if type(expected_revision) is not int or not 0 <= expected_revision <= 2**63 - 3:
            raise ValueError('expected_revision must be a nonnegative integer')
        validated = ModelConnection.model_validate(connection) if connection is not None else None
        if selected == 'speech' and validated is not None and validated.voice is None:
            raise ValueError('speech requires a voice')
        lock_path = self.path.with_name(self.path.name + '.lock')
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'rb+') as lock:
                info = os.fstat(lock.fileno())
                if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                    raise ModelConnectionError('Self-hosted model settings lock must be private')
                fcntl.flock(lock, fcntl.LOCK_EX)
                current = self._read()
                if current.revision != expected_revision:
                    raise ModelConnectionConflict('Model settings changed; reload the saved revision before saving')
                changed = _Document(revision=current.revision + 1,
                                    connections={**current.connections, selected: validated})
                self._write(changed)
                return changed.model_dump(mode='json')
        except OSError:
            raise ModelConnectionError('Cannot save self-hosted model settings') from None

    def _write(self, document: _Document) -> None:
        temporary: str | None = None
        try:
            fd, temporary = tempfile.mkstemp(prefix=self.path.name + '.', suffix='.tmp', dir=self.path.parent)
            with os.fdopen(fd, 'w', encoding='utf-8') as target:
                os.fchmod(target.fileno(), 0o600)
                target.write(document.model_dump_json(indent=2) + '\n')
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.path)
            temporary = None
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                os.unlink(temporary)
