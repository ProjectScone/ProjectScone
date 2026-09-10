"""Bounded DynamoDB episode catalogs, migrations, and resumable release journals.

This module performs metadata work through the owning S3 session so every
request retains its cancellation/deadline checks and transaction semantics.
"""
from __future__ import annotations

import re
from typing import Literal
from uuid import uuid4

from pydantic import Field

from ..core.errors import SconeError
from ..core.models import Attachment
from .aws_blobs import RETRIES, _Control, _Hold, _Intent, _Model, _Root, _Row, _Session, _mapping

TRANSACTION_ROWS = 90


class _EpisodeLink(_Model):
    attachment: Attachment
    generation: str = Field(pattern=r"^[0-9a-f]{32}$")
    sequence: int = Field(ge=1)


class _ReleaseResult(_Model):
    attachment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: Literal["released", "kept"]
    sequence: int = Field(default=0, ge=0)
    generation: str = Field(default="", pattern=r"^(?:[0-9a-f]{32})?$")


class Catalog:
    def __init__(self, session: _Session) -> None:
        self.s = session

    @staticmethod
    def episode_pk(space: str, episode: int) -> str:
        return f"E#{space}#{episode}"

    @staticmethod
    def journal_pk(space: str, operation: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{32}", operation):
            raise SconeError("AWS blob release operation is invalid")
        return f"J#{space}#{operation}"

    def unconditional(self, pk: str, sk: str, value: _Model | None) -> dict[str, object]:
        # The same transaction CASes the authoritative hold/control. Derived
        # rows therefore need no independent revision check and can be retried.
        change = self.s.change(pk, sk, None, value)
        _mapping(change["Delete" if value is None else "Put"]).pop("ConditionExpression")
        return change

    def link_change(self, space: str, episode: int, hold: _Hold, sequence: int) -> dict[str, object]:
        value = _EpisodeLink(attachment=hold.attachment, generation=hold.generation, sequence=sequence)
        return self.unconditional(self.episode_pk(space, episode), hold.attachment.attachment_id, value)

    def unlink_change(self, space: str, episode: int, identifier: str) -> dict[str, object]:
        return self.unconditional(self.episode_pk(space, episode), identifier, None)

    def ensure(self, space: str) -> bool:
        """Help a fenced migration finish; False means an old release must resume."""
        conflicts = 0
        while conflicts < RETRIES:
            control = self.s.control(space)
            if control is None:
                success = self.s.transaction([self.s.change("S#" + space, "C", None, _Control(catalog_version=1))])
            elif control.data.catalog_version == 1:
                return True
            elif control.data.mode in ("unlink", "release"):
                return False
            elif control.data.mode == "active":
                migrating = control.data.model_copy(update={"mode": "migrate", "catalog_after": "", "catalog_offset": 0})
                success = self.s.transaction([self.s.change(control.pk, control.sk, control, migrating)])
            else:
                success = self.migrate_page(space, control)
            conflicts = 0 if success else conflicts + 1
        raise SconeError("AWS blob catalog migration remained busy")

    def migrate_page(self, space: str, control: _Row[_Control]) -> bool:
        state = control.data
        if state.catalog_after and not re.fullmatch(r"H#[0-9a-f]{64}", state.catalog_after):
            raise SconeError("AWS blob catalog cursor is invalid")
        rows, _ = self.s.page("S#" + space, "H#", _Hold, after=state.catalog_after)
        if not rows:
            if state.catalog_offset:
                raise SconeError("AWS blob catalog offset is invalid")
            ready = state.model_copy(update={"mode": "active", "catalog_version": 1, "catalog_after": "", "catalog_offset": 0})
            return self.s.transaction([self.s.change(control.pk, control.sk, control, ready)])
        changes: list[dict[str, object]] = []
        cursor, offset = state.catalog_after, state.catalog_offset
        for hold in rows:
            self.check_hold(space, hold)
            if offset > len(hold.data.links):
                raise SconeError("AWS blob catalog offset is invalid")
            for episode, sequence in hold.data.links[offset:]:
                if len(changes) == TRANSACTION_ROWS:
                    break
                changes.append(self.link_change(space, episode, hold.data, sequence))
                offset += 1
            if offset < len(hold.data.links):
                break
            cursor, offset = hold.sk, 0
            if len(changes) == TRANSACTION_ROWS:
                break
        following = state.model_copy(update={"catalog_after": cursor, "catalog_offset": offset})
        changes.append(self.s.change(control.pk, control.sk, control, following))
        return self.s.transaction(changes)

    @staticmethod
    def check_hold(space: str, hold: _Row[_Hold]) -> None:
        if hold.pk != "S#" + space or hold.sk != "H#" + hold.data.attachment.attachment_id:
            raise SconeError("AWS blob catalog hold identity is invalid")

    def episode_rows(self, space: str, episode: int) -> list[tuple[int, _Row[_Hold]]]:
        if not self.ensure(space):
            _, rows = self.s.snapshot(space)
            return sorted((sequence, row) for row in rows for number, sequence in row.data.links if number == episode)
        for _ in range(RETRIES):
            before = self.s.control(space)
            links = self.s.query(self.episode_pk(space, episode), "", _EpisodeLink)
            result: list[tuple[int, _Row[_Hold]]] = []
            inconsistent = False
            for row in links:
                value = row.data
                if row.sk != value.attachment.attachment_id:
                    raise SconeError("AWS blob catalog link identity is invalid")
                hold = self.s.read("S#" + space, "H#" + row.sk, _Hold)
                if (hold is None or hold.data.generation != value.generation
                        or hold.data.attachment != value.attachment
                        or (episode, value.sequence) not in hold.data.links):
                    inconsistent = True
                    continue
                self.check_hold(space, hold)
                result.append((value.sequence, hold))
            if self.s.control(space) != before:
                continue
            if inconsistent:
                raise SconeError("AWS blob catalog ownership is inconsistent")
            return sorted(result, key=lambda item: (item[0], item[1].sk))
        raise SconeError("AWS blob catalog remained busy")

    def for_episode(self, space: str, episode: int) -> list[Attachment]:
        return [hold.data.attachment for _, hold in self.episode_rows(space, episode)]

    def released_by(self, space: str, episode: int) -> list[str]:
        return [hold.data.attachment.attachment_id for _, hold in self.episode_rows(space, episode) if len(hold.data.links) == 1]

    def prune_completed(self, space: str, operation: str) -> None:
        """Retire the preceding completed receipt in bounded, idempotent batches."""
        pk = self.journal_pk(space, operation)
        while True:
            rows, _ = self.s.page(pk, "R#", _ReleaseResult, limit=TRANSACTION_ROWS)
            if not rows:
                return
            if not self.s.transaction([self.unconditional(pk, row.sk, None) for row in rows]):
                raise SconeError("AWS blob completed journal cleanup remained busy")

    def start_release(self, space: str, episode: int | None) -> _Row[_Control]:
        mode: Literal["release", "unlink"] = "release" if episode is None else "unlink"
        for _ in range(RETRIES):
            control = self.s.control(space)
            if control is None:
                raise SconeError("AWS blob catalog control is missing")
            state = control.data
            if state.mode != "active":
                if state.mode != mode or state.episode != (episode or 0) or state.journal_version != 1:
                    raise SconeError("AWS blob space has another unfinished release")
                return control
            # Invalidate readers of the old receipt BEFORE pruning its rows.
            # Persist the old operation so interrupted cleanup can be resumed.
            journal = _Control(mode=mode, count=state.count, catalog_version=1, journal_version=1,
                               operation=uuid4().hex, episode=episode or 0, prior_operation=state.operation)
            if self.s.transaction([self.s.change(control.pk, control.sk, control, journal)]):
                return _Row(control.pk, control.sk, control.revision + 1, journal, journal.model_dump_json())
        raise SconeError("AWS blob release remained busy")

    def release(self, space: str, episode: int | None) -> tuple[list[str], list[str]]:
        if not self.ensure(space):
            return self.s._release_legacy(space, episode)
        started = self.start_release(space, episode)
        operation = started.data.operation
        mode = "release" if episode is None else "unlink"
        conflicts = 0
        while conflicts < RETRIES:
            control = self.s.control(space)
            if control is None or control.data.operation != operation:
                raise SconeError("AWS blob release journal changed or receipt was retired")
            if control.data.mode == "active":
                return self.completed_receipt(space, control)
            if control.data.mode != mode or control.data.episode != (episode or 0):
                raise SconeError("AWS blob release journal changed")
            self.check_release_cursor(control.data)
            if control.data.prior_operation:
                if control.data.prior_operation == operation:
                    raise SconeError("AWS blob prior release operation is invalid")
                self.prune_completed(space, control.data.prior_operation)
                following = control.data.model_copy(update={"prior_operation": ""})
                success = self.s.transaction([self.s.change(control.pk, control.sk, control, following)])
            elif control.data.release_phase == "holds":
                success = self.release_hold(space, episode, control)
            else:
                success, receipt = self.collect_page(space, control)
                if receipt is not None:
                    return receipt
            conflicts = 0 if success else conflicts + 1
        raise SconeError("AWS blob release remained busy")

    def completed_receipt(self, space: str, control: _Row[_Control]) -> tuple[list[str], list[str]]:
        receipt = self.receipt(self.journal_pk(space, control.data.operation))
        if self.s.control(space) != control:
            raise SconeError("AWS blob completed receipt changed or was retired")
        return receipt

    @staticmethod
    def check_release_cursor(state: _Control) -> None:
        cursor = state.release_cursor
        pattern = r"R#[0-9]{20}#[0-9a-f]{64}" if state.release_phase == "gc" else (
            r"H#[0-9a-f]{64}" if state.mode == "release" else r"[0-9a-f]{64}")
        if cursor and not re.fullmatch(pattern, cursor):
            raise SconeError("AWS blob release cursor is invalid")

    def release_candidates(self, space: str, episode: int | None, control: _Row[_Control]) -> list[tuple[_Row[_Hold], int]] | None:
        if episode is None:
            holds, _ = self.s.page("S#" + space, "H#", _Hold, after=control.data.release_cursor, limit=20)
            return [(hold, 0) for hold in holds]
        links, _ = self.s.page(self.episode_pk(space, episode), "", _EpisodeLink, after=control.data.release_cursor, limit=20)
        candidates: list[tuple[_Row[_Hold], int]] = []
        for link in links:
            hold = self.s.read("S#" + space, "H#" + link.sk, _Hold)
            if (hold is None or hold.data.generation != link.data.generation
                    or hold.data.attachment != link.data.attachment or link.sk != link.data.attachment.attachment_id
                    or (episode, link.data.sequence) not in hold.data.links):
                if self.s.control(space) != control:
                    return None
                raise SconeError("AWS blob release catalog is inconsistent")
            candidates.append((hold, link.data.sequence))
        return candidates

    def release_hold(self, space: str, episode: int | None, control: _Row[_Control]) -> bool:
        candidates = self.release_candidates(space, episode, control)
        if candidates is None:
            return False
        if not candidates:
            following = control.data.model_copy(update={"release_phase": "gc", "release_cursor": ""})
            return self.s.transaction([self.s.change(control.pk, control.sk, control, following)])
        changes: list[dict[str, object]] = []
        following = control.data
        for hold, sequence in candidates:
            self.check_hold(space, hold)
            if TRANSACTION_ROWS - len(changes) < 5:
                break
            planned = self.remove_links(space, episode, control, following, hold, sequence, TRANSACTION_ROWS - len(changes))
            if planned is None:
                return False
            additional, following, partial = planned
            changes.extend(additional)
            if partial:
                break
        changes.append(self.s.change(control.pk, control.sk, control, following))
        return self.s.transaction(changes)

    def remove_links(self, space: str, episode: int | None, control: _Row[_Control], state: _Control,
                     hold: _Row[_Hold], sequence: int, budget: int) -> tuple[list[dict[str, object]], _Control, bool] | None:
        identifier = hold.data.attachment.attachment_id
        removed = hold.data.links[:budget - 4] if episode is None else tuple(link for link in hold.data.links if link[0] == episode)
        removed_episodes = {number for number, _ in removed}
        remaining = tuple(link for link in hold.data.links if link[0] not in removed_episodes)
        changes = [self.unlink_change(space, number, identifier) for number, _ in removed]
        partial = episode is None and bool(remaining)
        following = state if partial else state.model_copy(update={"release_cursor": hold.sk if episode is None else identifier})
        if remaining:
            changes.append(self.s.change(hold.pk, hold.sk, hold, hold.data.model_copy(update={"links": remaining})))
        else:
            root = self.s.read("B#" + identifier, "R", _Root)
            if root is None or root.data.generation != hold.data.generation:
                if self.s.control(space) != control:
                    return None
                raise SconeError("AWS blob release ownership is inconsistent")
            changes.extend(self.remove_hold(space, episode, control, hold, root, sequence))
            if following.count < 1:
                raise SconeError("AWS blob release hold count is inconsistent")
            following = following.model_copy(update={"count": following.count - 1})
        return changes, following, partial

    def remove_hold(self, space: str, episode: int | None, control: _Row[_Control], hold: _Row[_Hold],
                    root: _Row[_Root], sequence: int) -> list[dict[str, object]]:
        identifier = hold.data.attachment.attachment_id
        last = root.data.holds == 1
        result = _ReleaseResult(attachment_id=identifier, disposition="released" if episode is not None or last else "kept",
                                sequence=sequence, generation=root.data.generation if last else "")
        changes = [self.s.change(hold.pk, hold.sk, hold, None),
                   self.s.change(root.pk, root.sk, root, None if last else root.data.model_copy(update={"holds": root.data.holds - 1})),
                   self.unconditional(self.journal_pk(space, control.data.operation), f"R#{sequence:020d}#{identifier}", result)]
        if last:
            deletion = _Intent(kind="delete", attachment_id=identifier, generation=root.data.generation,
                               key=root.data.key, version_id=root.data.version_id)
            changes.append(self.s.change("I#" + space, deletion.generation, None, deletion))
        return changes

    def collect_page(self, space: str, control: _Row[_Control]) -> tuple[bool, tuple[list[str], list[str]] | None]:
        pk = self.journal_pk(space, control.data.operation)
        results, _ = self.s.page(pk, "R#", _ReleaseResult, after=control.data.release_cursor, limit=TRANSACTION_ROWS)
        if not results:
            receipt = self.receipt(pk)
            finished = _Control(count=control.data.count, catalog_version=1, journal_version=1, operation=control.data.operation)
            success = self.s.transaction([self.s.change(control.pk, control.sk, control, finished)])
            return success, receipt if success else None
        changes: list[dict[str, object]] = []
        for row in results:
            value = row.data
            if row.sk != f"R#{value.sequence:020d}#{value.attachment_id}":
                raise SconeError("AWS blob release result identity is invalid")
            if value.generation:
                intent = self.s.intent(space, value.generation)
                if intent is not None:
                    if intent.data.kind != "delete" or intent.data.attachment_id != value.attachment_id:
                        raise SconeError("AWS blob release cleanup identity is invalid")
                    self.s.delete_object(intent.data)
                    changes.append(self.s.change(intent.pk, intent.sk, intent, None))
        following = control.data.model_copy(update={"release_cursor": results[-1].sk})
        changes.append(self.s.change(control.pk, control.sk, control, following))
        return self.s.transaction(changes), None

    def receipt(self, pk: str) -> tuple[list[str], list[str]]:
        rows = self.s.query(pk, "R#", _ReleaseResult)
        return ([row.data.attachment_id for row in rows if row.data.disposition == "released"],
                [row.data.attachment_id for row in rows if row.data.disposition == "kept"])
