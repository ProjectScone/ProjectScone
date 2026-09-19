"""Encrypted proposals and human decisions for agent-authored tool recipes.

The owning host authenticates reviewers and decides which capabilities to expose.
Models receive only a proposal adapter. Review never executes or grants a call.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import re
from pathlib import Path
import sqlite3
from typing import Literal, Self

from pydantic import Field, model_validator

from ._encrypted_store import EncryptedRecordStore
from .custom_tools import AgentTool, ToolContext, _encoded, snapshot_tools
from .recipe_models import RecipeModel, ToolRecipe
from .tool_recipes import capability_digest, recipe_dependencies, reviewed_recipe_tool
from .workflow import _integer, _name
from ..core.validation import check_space


class ToolRecipeConflict(ValueError):
    pass


def validate_proposal_id(value: str) -> None:
    _name(value)
    if value in ('.', '..'):
        raise ValueError('invalid recipe proposal identifier')


class RecipeReview(RecipeModel):
    decision: Literal['approve', 'deny', 'revoke']
    actor: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=2000)
    occurred_at: datetime

    @model_validator(mode='after')
    def valid(self) -> Self:
        _name(self.actor)
        if not self.reason.strip() or self.occurred_at.tzinfo is None:
            raise ValueError('invalid recipe review')
        return self


class ToolRecipeProposal(RecipeModel):
    space: str
    proposal_id: str
    proposed_by: str
    recipe: ToolRecipe
    dependencies_json: str = Field(max_length=64000)
    created_at: datetime
    reviews: tuple[RecipeReview, ...] = Field(default=(), max_length=2)

    @property
    def revision(self) -> int:
        return 1 + len(self.reviews)

    @property
    def status(self) -> str:
        return {'approve': 'approved', 'deny': 'denied', 'revoke': 'revoked'}[self.reviews[-1].decision] if self.reviews else 'pending'

    @property
    def requirements(self) -> str:
        return self.recipe.requirements

    def dependencies(self) -> list[dict[str, object]]:
        value = json.loads(self.dependencies_json)
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ValueError('invalid recipe dependency metadata')
        return value

    @model_validator(mode='after')
    def valid(self) -> Self:
        check_space(self.space)
        validate_proposal_id(self.proposal_id)
        _name(self.proposed_by)
        self.dependencies()
        if self.created_at.tzinfo is None:
            raise ValueError('invalid proposal timestamp')
        if self.reviews and (self.reviews[0].decision not in ('approve', 'deny')
            or self.reviews[0].actor == self.proposed_by):
            raise ValueError('independent recipe review required')
        if len(self.reviews) == 2 and (self.reviews[0].decision != 'approve' or self.reviews[1].decision != 'revoke'):
            raise ValueError('invalid recipe review transition')
        return self


@dataclass(frozen=True)
class ToolRecipePage:
    items: tuple[ToolRecipeProposal, ...]
    next_after: str | None


class ToolRecipeStore:
    def __init__(self, path: str | Path, *, key: bytes, max_proposals: int = 4096) -> None:
        _integer(max_proposals, 1, 100000)
        self._key, self._maximum = key, max_proposals
        self._path = Path(path).absolute()
        self._closed = False
        self._storage = EncryptedRecordStore(path, key=key, table='tool_recipes', metadata='tool_recipe_meta',
            application_id=0x53435250, domain='scone-reviewed-tool-recipes-v1', label='tool_recipe')

    def _prefix(self, space: str) -> str:
        check_space(space)
        return hmac.new(self._key, ('recipe-space:' + space).encode(), hashlib.sha256).hexdigest() + ':'

    def _token(self, space: str, proposal_id: str) -> str:
        validate_proposal_id(proposal_id)
        return self._prefix(space) + hmac.new(self._key, ('recipe-id:' + proposal_id).encode(), hashlib.sha256).hexdigest()

    def _decode(self, token: str, payload: object, space: str) -> ToolRecipeProposal:
        record = ToolRecipeProposal.model_validate_json(self._storage._unseal(token, payload))
        if record.space != space or token != self._token(record.space, record.proposal_id):
            raise ValueError('recipe identity changed')
        return record

    def _get(self, db: sqlite3.Connection, space: str, proposal_id: str) -> ToolRecipeProposal | None:
        token = self._token(space, proposal_id)
        row = db.execute('SELECT payload FROM tool_recipes WHERE token=?', (token,)).fetchone()
        if row is None:
            return None
        return self._decode(token, row[0], space)

    def _put(self, db: sqlite3.Connection, record: ToolRecipeProposal) -> None:
        token = self._token(record.space, record.proposal_id)
        payload = self._storage._seal(token, record.model_dump_json().encode())
        db.execute('INSERT INTO tool_recipes VALUES (?,?) ON CONFLICT(token) DO UPDATE SET payload=excluded.payload',
                   (token, payload))

    def review_actor(self, authenticated_key: str) -> str:
        if not isinstance(authenticated_key, str) or not authenticated_key or len(authenticated_key) > 8192:
            raise ValueError('authenticated review key required')
        return 'key:' + hmac.new(self._key, ('recipe-reviewer:' + authenticated_key).encode(), hashlib.sha256).hexdigest()

    def get(self, space: str, proposal_id: str) -> ToolRecipeProposal | None:
        with self._storage._access() as db:
            return self._get(db, space, proposal_id)

    def list(self, space: str, *, limit: int = 50, after: str | None = None) -> ToolRecipePage:
        _integer(limit, 1, 100)
        prefix = self._prefix(space)
        if after is not None and (not isinstance(after, str) or not after.startswith(prefix)
            or re.fullmatch(r'[a-f0-9]{64}', after[len(prefix):]) is None):
            raise ValueError('invalid recipe cursor')
        with self._storage._access() as db:
            rows = db.execute('SELECT token,payload FROM tool_recipes WHERE token>? AND token<? ORDER BY token LIMIT ?',
                (after or prefix, prefix + '~', limit + 1)).fetchall()
            return ToolRecipePage(tuple(self._decode(token, payload, space) for token, payload in rows[:limit]),
                                  rows[limit - 1][0] if len(rows) > limit else None)

    def propose(self, space: str, proposal_id: str, recipe: ToolRecipe, *, proposed_by: str,
                tools: Sequence[AgentTool]) -> ToolRecipeProposal:
        recipe = ToolRecipe.model_validate_json(recipe.model_dump_json())
        dependencies = recipe_dependencies(recipe, tools)
        metadata = json.dumps([tool.info() for tool in dependencies], sort_keys=True, ensure_ascii=False,
                              allow_nan=False, separators=(',', ':'))
        record = ToolRecipeProposal(space=space, proposal_id=proposal_id, proposed_by=proposed_by,
            recipe=recipe, dependencies_json=metadata, created_at=datetime.now(timezone.utc))
        with self._storage._access(write=True) as db:
            prior = self._get(db, space, proposal_id)
            if prior is not None:
                if (prior.recipe, prior.proposed_by, prior.dependencies_json) != (recipe, proposed_by, metadata):
                    raise ValueError('recipe proposal conflict')
                return prior
            if db.execute('SELECT COUNT(*) FROM tool_recipes').fetchone()[0] >= self._maximum:
                raise ValueError('recipe proposal capacity')
            self._put(db, record)
        return record

    def _review(self, space: str, proposal_id: str, *, decision: Literal['approve', 'deny', 'revoke'],
                actor: str, reason: str, expected_revision: int,
                admission_guard: Callable[[], None] | None = None) -> ToolRecipeProposal:
        _integer(expected_revision, 1, 2)
        review = RecipeReview(decision=decision, actor=actor, reason=reason, occurred_at=datetime.now(timezone.utc))
        with self._storage._access(write=True) as db:
            if admission_guard is not None:
                admission_guard()
            prior = self._get(db, space, proposal_id)
            if prior is None or prior.revision != expected_revision:
                raise ToolRecipeConflict('recipe review revision conflict')
            if (decision == 'revoke' and prior.status != 'approved') or (decision != 'revoke' and prior.status != 'pending'):
                raise ToolRecipeConflict('recipe review transition refused')
            record = ToolRecipeProposal.model_validate({**prior.model_dump(), 'reviews': (*prior.reviews, review)})
            self._put(db, record)
        return record

    def decide(self, space: str, proposal_id: str, *, decision: Literal['approve', 'deny'], actor: str,
               reason: str, expected_revision: int,
               admission_guard: Callable[[], None] | None = None) -> ToolRecipeProposal:
        if decision not in ('approve', 'deny'):
            raise ValueError('invalid recipe decision')
        return self._review(space, proposal_id, decision=decision, actor=actor, reason=reason,
                            expected_revision=expected_revision, admission_guard=admission_guard)

    def revoke(self, space: str, proposal_id: str, *, actor: str, reason: str, expected_revision: int,
               admission_guard: Callable[[], None] | None = None) -> ToolRecipeProposal:
        return self._review(space, proposal_id, decision='revoke', actor=actor, reason=reason,
                            expected_revision=expected_revision, admission_guard=admission_guard)

    def bind(self, space: str, proposal_id: str, *, tools: Sequence[AgentTool]) -> AgentTool:
        prior = self.get(space, proposal_id)
        if prior is None or prior.status != 'approved':
            raise ValueError('approved recipe required')
        identity = hmac.new(self._key, ('recipe-binding-v1:' + str(self._path) + ':' + prior.model_dump_json()).encode(), hashlib.sha256).hexdigest()
        def admit(context: ToolContext) -> None:
            if context.space != space:
                raise ValueError('recipe scope changed')
            current = self.get(space, proposal_id)
            if current is None or current != prior:
                raise ValueError('recipe approval changed')
        def dispatch_admit(context: ToolContext) -> None:
            if context.space != space or self._closed:
                raise ValueError('recipe admission unavailable')
            # Each worker owns its read connection. Never share SQLite connections
            # across threads or create a missing store during authorization.
            with closing(sqlite3.connect(self._path.as_uri() + '?mode=ro', uri=True, timeout=1)) as db:
                db.execute('BEGIN')
                current = self._get(db, space, proposal_id)
            if self._closed or current != prior:
                raise ValueError('recipe approval changed')
        return reviewed_recipe_tool(prior.recipe, revision=identity, tools=tools,
                                    admit=admit, dispatch_admit=dispatch_admit)

    def proposal_tool(self, *, space: str, proposed_by: str, tools: Sequence[AgentTool]) -> AgentTool:
        check_space(space)
        _name(proposed_by)
        capabilities = snapshot_tools(tools)
        digest = hmac.new(self._key, json.dumps(['recipe-proposal-v1', str(self._path), space, proposed_by,
            [tool.info() for tool in capabilities]], sort_keys=True).encode(), hashlib.sha256).hexdigest()
        async def propose(arguments: dict[str, object], context: ToolContext) -> object:
            if context.space != space:
                raise ValueError('recipe proposal scope changed')
            proposal_id, raw = arguments['proposal_id'], arguments['recipe_json']
            assert isinstance(proposal_id, str) and isinstance(raw, str)
            record = self.propose(space, proposal_id, ToolRecipe.model_validate_json(raw),
                                  proposed_by=proposed_by, tools=capabilities)
            return {'proposal_id': record.proposal_id, 'status': record.status, 'revision': record.revision}
        return AgentTool('propose_tool_recipe', 'Submit a ToolRecipe JSON proposal for independent human review; this cannot approve or execute it.',
            digest, {'type': 'object', 'properties': {
                'proposal_id': {'type': 'string', 'maxLength': 128}, 'recipe_json': {'type': 'string', 'maxLength': 16000}},
                'required': ['proposal_id', 'recipe_json'], 'additionalProperties': False}, propose)

    def capabilities_tool(self, *, space: str, tools: Sequence[AgentTool]) -> AgentTool:
        check_space(space)
        capabilities = snapshot_tools(tools)
        payload = _encoded({'recipe_schema': ToolRecipe.model_json_schema(), 'tools': [
            {'tool': tool.info(), 'digest': capability_digest(tool),
             'composable': not (tool.requires_approval or tool.return_direct)} for tool in capabilities]}, 48000)
        revision = hashlib.sha256((space + ':' + payload).encode()).hexdigest()
        async def describe(arguments: dict[str, object], context: ToolContext) -> object:
            if context.space != space:
                raise ValueError('recipe capability scope changed')
            return json.loads(payload)
        return AgentTool('recipe_capabilities', 'Inspect allowed tool contracts, pinned digests and the ToolRecipe proposal schema.',
            revision, {'type': 'object', 'properties': {}, 'additionalProperties': False}, describe,
            max_output_bytes=64000)

    def close(self) -> None:
        self._closed = True
        self._storage.close()
