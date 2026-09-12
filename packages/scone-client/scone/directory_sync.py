"""Explicit local collection scans; reads never resume or retry a write."""
from __future__ import annotations

from typing import Optional

from ._wire import ResourceClient, address, boolean, bounded_body, cursor, digest, identifier, integer, invalid, items, record
from .directory_models import (TERMINAL, SyncCollection, SyncOutcomePage, SyncPage, SyncRecord, SyncStatus)


class DirectorySyncRuns(ResourceClient):
    def collections(self) -> tuple[SyncCollection, ...]:
        self._check('documents.sync')
        row = record(self._client._request('GET', '/v1/sync-collections'))
        self._space(record(self._client._request('GET', '/v1/status')))
        values = tuple(SyncCollection.from_json(value) for value in items(row.get('items'), 256))
        if len({value.collection_id for value in values}) != len(values):
            raise invalid('duplicate sync collections')
        return values

    def status(self, run_id: str) -> SyncStatus:
        path = '/v1/sync-runs/' + address(run_id)
        self._check('documents.sync')
        return SyncStatus.from_json(self._client._request('GET', path), expected_space=self.expected_space, run_id=run_id)

    def request(self, run_id: str) -> SyncRecord:
        path = '/v1/sync-runs/' + address(run_id) + '/request'
        self._check('documents.sync')
        return SyncRecord.from_json(self._client._request('GET', path), expected_space=self.expected_space, run_id=run_id)

    def list(self, *, limit: int = 20, after: Optional[str] = None) -> SyncPage:
        params = {'limit': str(integer(limit, 1, 100))}
        if after is not None:
            params['after'] = cursor(after) or ''
        self._check('documents.sync')
        page = self._client._request('GET', '/v1/sync-runs', params=params)
        self._space(record(self._client._request('GET', '/v1/status')))
        return SyncPage.from_json(page, expected_space=self.expected_space, limit=limit, after=after)

    def start(self, run_id: str, *, collection: SyncCollection, delete_missing: bool = False) -> SyncStatus:
        if not isinstance(collection, SyncCollection):
            raise invalid('sync collection')
        body = bounded_body({'run_id': identifier(run_id), 'collection_id': identifier(collection.collection_id),
            'delete_missing': boolean(delete_missing), 'expected_configuration': digest(collection.configuration)})
        if delete_missing and not boolean(collection.allow_delete_missing):
            raise invalid('sync deletion policy')
        self._check('documents.sync', mutation=True)
        status = SyncStatus.from_json(self._client._request('POST', '/v1/sync-runs', json=body),
                                      expected_space=self.expected_space, run_id=run_id)
        spec = status.record.spec
        if spec.collection_id != collection.collection_id or spec.configuration != collection.configuration or spec.delete_missing != delete_missing:
            raise invalid('sync admission acknowledgement')
        return status

    def results(self, run_id: str, *, limit: int = 20, after: Optional[int] = None) -> SyncOutcomePage:
        path = '/v1/sync-runs/' + address(run_id) + '/result'
        params = {'limit': str(integer(limit, 1, 100))}
        if after is not None:
            params['after'] = str(integer(after, 0, 299999))
        request = self.status(run_id).record
        if request.status not in TERMINAL:
            raise invalid('sync result unavailable')
        return SyncOutcomePage.from_json(self._client._request('GET', path, params=params), request=request, limit=limit, after=after)

    def resume(self, status: SyncStatus) -> SyncStatus:
        return self._control(status, resume=True)

    def cancel(self, status: SyncStatus) -> SyncStatus:
        return self._control(status, resume=False)

    def _control(self, status: SyncStatus, *, resume: bool) -> SyncStatus:
        if not isinstance(status, SyncStatus) or status.record.space != self.expected_space:
            raise invalid('sync control identity')
        prior = status.record
        path = '/v1/sync-runs/' + address(prior.run_id) + ('/resume' if resume else '/cancel')
        body = bounded_body({'expected_revision': integer(prior.revision, 0, 2**31-1)})
        self._check('documents.sync', mutation=True)
        current = self.status(prior.run_id)
        if (current != status or prior.status in TERMINAL or current.active_elsewhere
                or (resume and (current.active_local or prior.attempt >= prior.spec.max_attempts))):
            raise invalid('sync state changed or control unavailable')
        observed = SyncStatus.from_json(self._client._request('POST', path, json=body), expected_space=self.expected_space, run_id=prior.run_id)
        saved = observed.record
        if (saved.spec != prior.spec or saved.created_at != prior.created_at or saved.revision != prior.revision+1
                or (resume and (saved.attempt != prior.attempt+1 or observed.status != 'running' or not observed.active_local or saved.cancel_requested_at is not None))
                or (not resume and (saved.attempt != prior.attempt or saved.cancel_requested_at is None))):
            raise invalid('sync control acknowledgement')
        return observed
