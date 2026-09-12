"""One explicit document operation at a time, with immutable request checks."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Optional

from ._wire import ResourceClient, address, bounded_body, cursor, digest, identifier, integer, invalid, items, record
from .document_models import (DocumentAttachment, DocumentFormats, DocumentRequest, DocumentResult, DocumentStatus, PdfOcr, check_ocr, filename)


@dataclass(frozen=True)
class DocumentPage:
    items: tuple[DocumentStatus, ...]
    next_after: Optional[str]


class DocumentJobs(ResourceClient):
    def formats(self) -> DocumentFormats:
        self._check('documents.files')
        return DocumentFormats.from_json(self._client._request('GET', '/v1/documents/formats'))

    def upload(self, data: bytes, *, media_type: str) -> DocumentAttachment:
        if not isinstance(data, bytes) or not 1 <= len(data) <= 25*1024*1024:
            raise invalid('document upload bytes')
        if (not isinstance(media_type, str) or len(media_type) > 256
                or re.fullmatch(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+", media_type) is None):
            raise invalid('document media type')
        self._check('episodes.attachments', mutation=True)
        stored = DocumentAttachment.from_json(self._client._request('POST', '/v1/attachments', data=data,
                                                                   headers={'Content-Type': media_type}))
        if stored.attachment_id != hashlib.sha256(data).hexdigest() or stored.bytes != len(data):
            raise invalid('document upload acknowledgement')
        return stored

    def request(self, import_id: str) -> DocumentRequest:
        path = '/v1/document-jobs/' + address(import_id) + '/request'
        self._check('documents.jobs')
        return DocumentRequest.from_json(self._client._request('GET', path),
                                          expected_space=self.expected_space, import_id=import_id)

    def status(self, import_id: str) -> DocumentStatus:
        path = '/v1/document-jobs/' + address(import_id)
        self._check('documents.jobs')
        return DocumentStatus.from_json(self._client._request('GET', path),
                                         expected_space=self.expected_space, import_id=import_id)

    def list(self, *, limit: int = 20, after: Optional[str] = None) -> DocumentPage:
        params = {'limit': str(integer(limit, 1, 100))}
        if after is not None:
            params['after'] = cursor(after) or ''
        self._check('documents.jobs')
        row = record(self._client._request('GET', '/v1/document-jobs', params=params))
        values = tuple(DocumentStatus.from_json(value, expected_space=self.expected_space)
                       for value in items(row.get('items'), limit))
        next_after = cursor(row.get('next_after'))
        if len({value.import_id for value in values}) != len(values) or (after is not None and next_after == after):
            raise invalid('document page')
        return DocumentPage(values, next_after)

    def start(self, import_id: str, *, attachment_id: str, filename: str,
              pdf_ocr: Optional[PdfOcr] = None) -> DocumentStatus:
        body = self._submission(import_id, attachment_id, filename, pdf_ocr)
        self._check('documents.jobs', mutation=True)
        status = DocumentStatus.from_json(self._client._request('POST', '/v1/document-jobs', json=body),
                                          expected_space=self.expected_space, import_id=import_id)
        request = self.request(import_id)
        if (request.spec.attachment_id != attachment_id or request.spec.filename != filename or request.spec.pdf_ocr != pdf_ocr):
            raise invalid('document start acknowledgement')
        status.match(request)
        return status

    @staticmethod
    def _submission(import_id: str, attachment_id: str, name: str, ocr: Optional[PdfOcr]) -> dict[str, object]:
        identifier(import_id)
        digest(attachment_id)
        filename(name)
        check_ocr(name, ocr)
        body: dict[str, object] = {'import_id': import_id, 'attachment_id': attachment_id, 'filename': name}
        if ocr is not None:
            body['pdf_ocr'] = ocr.to_json()
        return bounded_body(body)

    def result(self, import_id: str) -> DocumentResult:
        path = '/v1/document-jobs/' + address(import_id) + '/result'
        self._check('documents.jobs')
        request = self.request(import_id)
        return DocumentResult.from_json(self._client._request('GET', path), request=request)

    def resume(self, request: DocumentRequest) -> DocumentStatus:
        return self._control(request, resume=True)

    def cancel(self, request: DocumentRequest) -> DocumentStatus:
        return self._control(request, resume=False)

    def _control(self, request: DocumentRequest, *, resume: bool) -> DocumentStatus:
        if not isinstance(request, DocumentRequest) or request.space != self.expected_space:
            raise invalid('document control request')
        path = '/v1/document-jobs/' + address(request.import_id) + ('/resume' if resume else '/cancel')
        body = bounded_body({'expected_revision': integer(request.revision, 0, 2**31 - 1)})
        self._check('documents.jobs', mutation=True)
        if self.request(request.import_id) != request:
            raise invalid('document request changed before control')
        status = DocumentStatus.from_json(self._client._request('POST', path, json=body),
                                          expected_space=self.expected_space, import_id=request.import_id)
        status.match(request, control=False)
        observed = self.request(request.import_id)
        if observed.spec != request.spec or observed.created_at != request.created_at:
            raise invalid('document control request acknowledgement')
        status.match(observed)
        unchanged = status.revision == request.revision and status.attempt == request.attempt
        if status.status == 'completed' and unchanged:
            return status
        if resume:
            valid = (status.revision == request.revision + 1 and status.attempt == request.attempt + 1
                     and status.status == 'running' and status.active_local and not status.outcome_unknown
                     and observed.cancel_requested_at is None)
        else:
            valid = (status.status == 'cancelled' and status.attempt == request.attempt
                     and status.revision == request.revision + (request.cancel_requested_at is None))
        if not valid:
            raise invalid('document control acknowledgement')
        return status
