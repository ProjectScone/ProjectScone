"""Chat exports as conversation memories, over HTTP: upload the export, import it by id, read the receipt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import hashlib
import re

from ._wire import ResourceClient, boolean, bounded_body, digest, integer, invalid, record, text
from .document_models import DocumentAttachment, filename as checked_filename

PLATFORMS = ('whatsapp', 'telegram', 'discord', 'slack')
DATE_ORDERS = ('day-first', 'month-first')
#: A day and a half; the server refuses a longer session gap.
MAX_GAP_SECONDS = 36 * 3600


@dataclass(frozen=True)
class ChatImportReceipt:
    """What an import stored, and what it counted instead of storing."""
    platform: str
    chat: str
    filename: str
    attachment_id: str
    messages: int
    stored: int
    duplicates: int
    failed: int
    episodes: int
    sessions: int
    session_gap_seconds: int
    speakers: int
    first_at: Optional[str]
    last_at: Optional[str]
    messages_unread: int
    system_messages: int
    unparsed_lines: int
    media_only_messages: int
    attachments_skipped: int
    date_order: Optional[str]
    date_order_told: bool
    time_zone: Optional[str]
    failed_reason: Optional[str]

    @classmethod
    def from_json(cls, value: object) -> ChatImportReceipt:
        row = record(value)
        platform = text(row.get('platform'), 16, 'platform')
        if platform not in PLATFORMS:
            raise invalid('chat import platform')
        counts = {name: integer(row.get(name), 0) for name in (
            'messages', 'stored', 'duplicates', 'failed', 'episodes', 'sessions', 'session_gap_seconds', 'speakers',
            'messages_unread', 'system_messages', 'unparsed_lines', 'media_only_messages', 'attachments_skipped')}
        if counts['stored'] + counts['duplicates'] + counts['failed'] != counts['messages'] or counts['episodes'] != counts['stored'] + counts['duplicates']:
            raise invalid('chat import counts')
        order = row.get('date_order')
        if order is not None and order not in (*DATE_ORDERS, 'year-first', 'undecidable'):
            raise invalid('chat import date order')

        def optional(name: str, maximum: int) -> Optional[str]:
            raw = row.get(name)
            return None if raw is None else text(raw, maximum, name)

        return cls(platform, text(row.get('chat'), 200, 'chat'), text(row.get('filename'), 1024, 'filename'),
                   digest(row.get('attachment_id')), counts['messages'], counts['stored'], counts['duplicates'], counts['failed'],
                   counts['episodes'], counts['sessions'], counts['session_gap_seconds'], counts['speakers'],
                   optional('first_at', 40), optional('last_at', 40), counts['messages_unread'], counts['system_messages'],
                   counts['unparsed_lines'], counts['media_only_messages'], counts['attachments_skipped'],
                   None if order is None else str(order), boolean(row.get('date_order_told', False)),
                   optional('time_zone', 64), optional('failed_reason', 512))


class ChatImports(ResourceClient):
    """Upload a WhatsApp, Telegram, Discord or Slack export and import it as conversation memories."""

    def upload(self, data: bytes, *, filename: str, media_type: str = 'application/octet-stream') -> DocumentAttachment:
        """Retain the export's bytes under its name (the suffix decides how it
        is read); the acknowledgement names them by digest and carries the name."""
        if not isinstance(data, bytes) or not 1 <= len(data) <= 25 * 1024 * 1024:
            raise invalid('chat export upload bytes')
        if (not isinstance(media_type, str) or len(media_type) > 256
                or re.fullmatch(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+", media_type) is None):
            raise invalid('chat export media type')
        name = checked_filename(filename)
        self._check('episodes.attachments', mutation=True)
        stored = DocumentAttachment.from_json(self._client._request('POST', '/v1/attachments', data=data,
                                                                   headers={'Content-Type': media_type, 'X-Filename': name}))
        if stored.attachment_id != hashlib.sha256(data).hexdigest() or stored.bytes != len(data) or stored.filename != name:
            raise invalid('chat export upload acknowledgement')
        return stored

    def import_export(self, attachment_id: str, *, filename: Optional[str] = None, chat: Optional[str] = None,
                      time_zone: Optional[str] = None, date_order: Optional[str] = None,
                      gap_seconds: Optional[int] = None, metadata: Optional[Mapping[str, str]] = None) -> ChatImportReceipt:
        """Import an uploaded export by its id. ``filename`` names the export
        when the upload carried no name (`upload` always gives one; an
        upload made another way may not); its suffix decides the reader;
        ``date_order`` is needed for a WhatsApp file no line of which decides
        it; ``gap_seconds`` is the silence that ends a session."""
        body: dict[str, object] = {'attachment_id': digest(attachment_id)}
        if filename is not None:
            body['filename'] = text(filename, 1024, 'filename')
        if chat is not None:
            body['chat'] = text(chat, 200, 'chat')
        if time_zone is not None:
            body['time_zone'] = text(time_zone, 64, 'time_zone')
        if date_order is not None:
            if date_order not in DATE_ORDERS:
                raise invalid('chat import date order')
            body['date_order'] = date_order
        if gap_seconds is not None:
            if isinstance(gap_seconds, bool) or not isinstance(gap_seconds, int) or not 0 <= gap_seconds <= MAX_GAP_SECONDS:
                raise invalid('chat import session gap')
            body['gap_seconds'] = gap_seconds
        if metadata is not None:
            if not isinstance(metadata, Mapping) or len(metadata) > 9 or not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
                raise invalid('chat import metadata')
            body['metadata'] = dict(metadata)
        self._check('chats.imports', mutation=True)
        receipt = ChatImportReceipt.from_json(self._client._request('POST', '/v1/chat-imports', json=bounded_body(body)))
        if receipt.attachment_id != body['attachment_id']:
            raise invalid('chat import acknowledgement')
        return receipt
