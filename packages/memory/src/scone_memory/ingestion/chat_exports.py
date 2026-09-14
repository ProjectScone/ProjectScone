"""Chat exports as memory.

A chat export is not a document. Its unit is the message: who said it,
when, in which chat, in reply to what. Read as a document it would be one
`file` episode dated the day of the export, its passages carrying no
speaker and no time, and a session-capped recall would hand back two
lines of a year of conversation. Read here it becomes what the rest of
the engine already understands: one `conversation` episode per message,
dated when it was sent, keyed so a second import of a grown export adds
only the new messages, and grouped into sessions by silence so recall
and the benches see the same shape a live conversation leaves.

Four export formats are read, all local files, no credential and no
network:

- WhatsApp: the `.txt` a phone exports, or the `.zip` iOS shares it in;
  Android (`12/03/2024, 14:05 - Name: text`) and iOS (`[12/03/2024,
  14:05:33] Name: text`) shapes, continuation lines, media placeholders
  and attachment lines, system lines.
- Telegram: Telegram Desktop's `result.json`.
- Discord: the JSON DiscordChatExporter writes.
- Slack: a workspace export (`.zip` with `users.json` and one JSON file
  per channel and day) or one of its day files on its own.

WhatsApp and Telegram write local clock times with no zone; the reader
takes a zone from the caller and otherwise reads them as UTC, and the
receipt says which. WhatsApp writes dates in the phone's order, day
first or month first; the reader decides from the file, or takes the
order from the caller, and a file that decides nothing is not imported
until the caller says. A file whose lines contradict each other, or
contradict the caller, is refused.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import PurePath, PurePosixPath
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..core.errors import InvalidInput
from ..core.timeutil import format_rfc3339, parse_rfc3339
from ..core.validation import MAX_METADATA_KEYS, check_space, normalise_metadata
from .formats.types import DocumentLimits
from .records import Record

Platform = Literal['whatsapp', 'telegram', 'discord', 'slack']
DateOrder = Literal['day-first', 'month-first', 'year-first', 'undecidable']

#: Messages read from one export before the rest are counted, not read.
#: `Transcript.messages_unread` says how many were left.
MAX_MESSAGES = 50_000
#: Silence that ends a session and starts the next, unless the caller
#: says otherwise: six hours parts an evening's talk from the morning's.
DEFAULT_SESSION_GAP_SECONDS = 6 * 3600
#: Records handed to the engine per batch. A failure part-way leaves the
#: earlier batches stored and raises; nothing is lost, because a second
#: import stores only what the keys say is missing.
IMPORT_BATCH = 500
#: Metadata keys this module writes on a message: what is left of the
#: engine's limit is the caller's.
OWN_METADATA_KEYS = 7
_OWN_KEYS = frozenset({'platform', 'chat', 'speaker', 'message', 'channel', 'reply_to', 'thread'})
#: A chat's name goes into every message's metadata and key; one longer
#: than this is refused, not cut.
MAX_CHAT_NAME = 200


@dataclass(frozen=True)
class ChatMessage:
    """One message of a chat, as the export had it."""
    #: The platform's identity for it (Slack `ts`, Telegram and Discord
    #: `id`); WhatsApp has none, so a digest of when, who and what.
    key: str
    sender: str
    #: RFC 3339, UTC.
    sent_at: str
    text: str
    channel: Optional[str] = None
    reply_to: Optional[str] = None
    thread: Optional[str] = None
    #: Files, images and media the export named but did not carry.
    attachments: int = 0


@dataclass(frozen=True)
class Transcript:
    platform: Platform
    chat: str
    messages: tuple[ChatMessage, ...]
    #: Past MAX_MESSAGES, the rest are counted here, not read.
    messages_unread: int = 0
    #: Joins, leaves, pins, subject changes: counted, never memories.
    system_messages: int = 0
    #: WhatsApp lines before the first header that matched no shape, and
    #: headers whose date no calendar holds. A line that continues a
    #: system line, a dropped header or an unread message is not counted:
    #: it belongs to what it continues.
    unparsed_lines: int = 0
    #: Messages that were a media placeholder, an attachment line or a
    #: deletion notice and nothing else: nothing to remember, counted.
    media_only_messages: int = 0
    attachments_skipped: int = 0
    #: WhatsApp: how the dates were read; None for the platforms that
    #: write unambiguous dates.
    date_order: Optional[DateOrder] = None
    #: WhatsApp: the order came from the caller, not the file.
    date_order_told: bool = False
    #: How clock times without a zone were read: the zone the caller
    #: gave, or "UTC (assumed)"; None when every time carried its own.
    time_zone: Optional[str] = None


@dataclass(frozen=True)
class ChatImported:
    """What an import stored, and what it counted instead."""
    platform: Platform
    chat: str
    messages: int
    stored: int
    duplicates: int
    failed: int
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
    date_order: Optional[DateOrder]
    time_zone: Optional[str]
    #: The first failed record's reason, so a receipt with failures says
    #: what kind they were.
    failed_reason: Optional[str] = None
    date_order_told: bool = False
    #: Stored and duplicate episodes in session order -- channel by
    #: channel, then by time -- not in the export's own order.
    episode_ids: tuple[int, ...] = field(default=(), repr=False)


# --- time -----------------------------------------------------------------

def _zone(name: Optional[str]) -> tuple[tzinfo, str]:
    """The zone clock times without one are read in, and its name for the
    receipt: an IANA name, a fixed offset like `+02:00`, or UTC assumed."""
    if name is None:
        return timezone.utc, 'UTC (assumed)'
    offset = re.fullmatch(r'([+-])(\d\d):(\d\d)', name)
    if offset:
        sign, hours, minutes = offset.groups()
        delta = timedelta(hours=int(hours), minutes=int(minutes))
        return timezone(-delta if sign == '-' else delta), name
    try:
        return ZoneInfo(name), name
    except (ZoneInfoNotFoundError, ValueError):
        raise InvalidInput(f'unknown time zone {name!r}: give an IANA name like Europe/Berlin or an offset like +02:00') from None


def _instant(local: datetime, zone: tzinfo) -> str:
    # A clock time repeated at a zone's fall-back is read as its first
    # occurrence (fold 0); the docs say so.
    return format_rfc3339(local.replace(tzinfo=zone).astimezone(timezone.utc))


def _epoch(seconds: float) -> str:
    try:
        return format_rfc3339(datetime.fromtimestamp(seconds, tz=timezone.utc))
    except (OverflowError, OSError, ValueError):
        raise InvalidInput('chat export carries a timestamp outside the instants UTC can represent') from None


def _rfc3339(text: str, where: str) -> str:
    try:
        return format_rfc3339(parse_rfc3339(text))
    except ValueError:
        raise InvalidInput(f'{where} is not a timestamp: {text[:40]!r}') from None


def _chat_name(name: str) -> str:
    if not name.strip():
        raise InvalidInput('a chat needs a name')
    if len(name) > MAX_CHAT_NAME:
        raise InvalidInput(f'chat name is {len(name)} characters; at most {MAX_CHAT_NAME}')
    return name


# --- WhatsApp ---------------------------------------------------------------

_MARKS = re.compile('[‎‏‪-‮]')
#: `a.m.`, `p. m.`, `AM`: a meridian with or without dots, with or
#: without the space some locales put inside it.
_MERIDIAN = r'(?:[\s  ]?[AaPp]\.?[\s  ]?[Mm]\.?)?'
_WA_HEADER = re.compile(
    r'^\[?(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{1,4})[,\s]+'
    r'(?P<time>\d{1,2}:\d{2}(?::\d{2})?' + _MERIDIAN + r')\]?'
    r'(?:\s+-)?\s+(?P<rest>\S.*)$')
_WA_CLOCK = re.compile(r'(\d{1,2}):(\d\d)(?::(\d\d))?(?:[\s  ]?([AaPp])\.?[\s  ]?[Mm]\.?)?')
#: A sender's name cannot hold a colon: the colon is what parts it from
#: the text, and a system line is a header with no `Name: ` at all.
_WA_SENDER = re.compile(r'^(?P<sender>[^:\n]{1,100}?):\s(?P<text>.*)$', re.DOTALL)
#: What WhatsApp writes in place of what it did not export (English
#: exports; another language's placeholders are kept as text).
_WA_MEDIA = frozenset({'<media omitted>', 'image omitted', 'video omitted', 'audio omitted',
                       'sticker omitted', 'gif omitted', 'document omitted', 'contact card omitted'})
_WA_DELETED = frozenset({'this message was deleted', 'you deleted this message', 'null'})
#: A line that names an attached file: iOS with media attached
#: (`<attached: 00000042-PHOTO-2024-03-12.jpg>`), Android with a caption
#: (`IMG-20240312-WA0001.jpg (file attached)`).
_WA_ATTACHED = re.compile(r'^<attached:\s*[^>]+>$|^\S.*\(file attached\)$', re.IGNORECASE)


def _whatsapp_date(date: str, order: DateOrder) -> tuple[int, int, int]:
    parts = re.split(r'[./-]', date)
    first, second, third = (int(part) for part in parts)
    if len(parts[0]) == 4:
        return first, second, third  # year first, as some locales write
    year = third if third >= 100 else 2000 + third
    return (year, first, second) if order == 'month-first' else (year, second, first)


def _whatsapp_time(time: str) -> tuple[int, int, int]:
    match = _WA_CLOCK.fullmatch(time)
    if match is None:
        raise ValueError(time)
    hour, minute = int(match.group(1)), int(match.group(2))
    second, meridian = int(match.group(3) or 0), (match.group(4) or '').lower()
    if meridian == 'a' and hour == 12:
        hour = 0
    elif meridian == 'p' and hour != 12:
        hour += 12
    return hour, minute, second


def _whatsapp_order(dates: Iterable[str], told: Optional[DateOrder]) -> DateOrder:
    """Decided by the file: a first field over 12 is a day, a second one
    is a month; both in one file is a file that cannot be read. A file of
    year-first dates needs no deciding. The caller's word is taken when
    the file does not contradict it."""
    day_first = month_first = year_first = others = 0
    for date in dates:
        first, second, _third = re.split(r'[./-]', date)
        if len(first) == 4:
            year_first += 1
            continue
        others += 1
        if int(first) > 12:
            day_first += 1
        if int(second) > 12:
            month_first += 1
    if day_first and month_first:
        raise InvalidInput('chat export dates are day-first on some lines and month-first on others')
    found: DateOrder = ('day-first' if day_first else 'month-first' if month_first
                        else 'year-first' if year_first and not others else 'undecidable')
    if told is not None:
        if told not in ('day-first', 'month-first'):
            raise InvalidInput("date order must be 'day-first' or 'month-first'")
        if found in ('day-first', 'month-first') and found != told:
            raise InvalidInput(f'the file writes its dates {found}, not {told}')
        return told
    return found


def _whatsapp_key(sent_at: str, sender: str, text: str, seen: Counter[str]) -> str:
    digest = hashlib.sha256(f'{sent_at}\n{sender}\n{text}'.encode()).hexdigest()[:16]
    seen[digest] += 1
    return digest if seen[digest] == 1 else f'{digest}#{seen[digest]}'


def read_whatsapp(text: str, *, chat: str, time_zone: Optional[str] = None,
                  date_order: Optional[DateOrder] = None) -> Transcript:
    """The `.txt` WhatsApp exports. A line in header shape opens a
    message; a line in no shape continues what came before it -- a
    message, a system line, or a header that was dropped -- or is counted
    when nothing has; a header with no `Name: ` is a system line."""
    zone, zone_name = _zone(time_zone)
    lines = text.splitlines()
    headers = [_WA_HEADER.match(_MARKS.sub('', line)) for line in lines]
    order = _whatsapp_order((match.group('date') for match in headers if match), date_order)
    seen: Counter[str] = Counter()
    open_message: list[str] | None = None
    absorbing = False  # continuation lines of a system line, a dropped header or an unread message
    messages: list[ChatMessage] = []
    unparsed = system = media_only = attachments = unread = 0

    def close() -> None:
        nonlocal media_only, attachments
        if open_message is None:
            return
        sender, sent_at, *body = open_message
        kept: list[str] = []
        for line in body:
            if _WA_ATTACHED.match(line.strip()):
                attachments += 1
            else:
                kept.append(line)
        content = '\n'.join(kept).strip()
        lowered = content.lower()
        if lowered in _WA_MEDIA:
            attachments += 1
            content = ''
        elif lowered in _WA_DELETED:
            content = ''
        if not content:
            media_only += 1
            return
        messages.append(ChatMessage(_whatsapp_key(sent_at, sender, content, seen), sender, sent_at, content))

    for line, match in zip(lines, headers):
        clean = _MARKS.sub('', line)
        if not match:
            if open_message is not None:
                open_message.append(clean)
            elif clean.strip() and not absorbing:
                unparsed += 1
            continue
        close()
        open_message = None
        absorbing = True
        try:
            year, month, day = _whatsapp_date(match.group('date'), order)
            local = datetime(year, month, day, *_whatsapp_time(match.group('time')))
        except ValueError:
            unparsed += 1
            continue
        spoken = _WA_SENDER.match(match.group('rest'))
        if spoken is None:
            system += 1
            continue
        if len(messages) >= MAX_MESSAGES:
            unread += 1
            continue
        absorbing = False
        open_message = [spoken.group('sender').strip(), _instant(local, zone), spoken.group('text')]
    close()
    if not messages and not system and not media_only and not unread:
        raise InvalidInput('no messages found: expected WhatsApp lines like "12/03/2024, 14:05 - Name: text"')
    return Transcript('whatsapp', chat, tuple(messages), messages_unread=unread, system_messages=system,
                      unparsed_lines=unparsed, media_only_messages=media_only, attachments_skipped=attachments,
                      date_order=order, date_order_told=date_order is not None, time_zone=zone_name)


# --- JSON exports -----------------------------------------------------------

def _load_json(data: bytes) -> object:
    try:
        return json.loads(data.decode('utf-8-sig'))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise InvalidInput('chat export is not well-formed JSON') from None


def _text_of(value: object) -> str:
    """Telegram's text: a string, or a list of strings and entity dicts."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get('text'), str):
                parts.append(part['text'])
        return ''.join(parts)
    return ''


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _str(value: object, fallback: str = '') -> str:
    return value if isinstance(value, str) else str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback


_TELEGRAM_MEDIA = ('photo', 'file', 'media_type', 'contact_information', 'location_information')


def read_telegram(export: Mapping[str, Any], *, time_zone: Optional[str] = None) -> Transcript:
    """Telegram Desktop's `result.json`: one chat, its messages in order,
    service messages (joins, pins, calls) counted apart. A message
    carries at most one attachment, however many keys describe it."""
    zone, zone_name = _zone(time_zone)
    chat = _chat_name(_str(export.get('name'), 'Telegram chat'))
    messages: list[ChatMessage] = []
    system = media_only = attachments = unread = 0
    zoned = False
    for entry in export.get('messages', ()):
        if not isinstance(entry, dict):
            continue
        if entry.get('type') != 'message':
            system += 1
            continue
        if len(messages) >= MAX_MESSAGES:
            unread += 1
            continue
        unix = entry.get('date_unixtime')
        if isinstance(unix, (str, int)) and str(unix).lstrip('-').isdigit():
            sent_at = _epoch(int(unix))
        else:
            when = _str(entry.get('date'))
            try:
                local = datetime.fromisoformat(when)
            except ValueError:
                raise InvalidInput(f'Telegram message {entry.get("id")} has no readable date') from None
            if local.tzinfo is None:
                zoned = True
                sent_at = _instant(local, zone)
            else:
                sent_at = format_rfc3339(local.astimezone(timezone.utc))
        content = _text_of(entry.get('text')).strip()
        carried = 1 if any(entry.get(key) for key in _TELEGRAM_MEDIA) else 0
        attachments += carried
        if not content:
            media_only += 1
            continue
        sender = _str(entry.get('from')) or _str(entry.get('from_id'), 'unknown')
        reply = entry.get('reply_to_message_id')
        messages.append(ChatMessage(_str(entry.get('id'), f'n{len(messages) + 1}'), sender, sent_at, content,
                                    reply_to=_str(reply) if isinstance(reply, (int, str)) else None, attachments=carried))
    return Transcript('telegram', chat, tuple(messages), messages_unread=unread, system_messages=system,
                      media_only_messages=media_only, attachments_skipped=attachments,
                      time_zone=zone_name if zoned else None)


_DISCORD_SPOKEN = frozenset({'Default', 'Reply', 'ThreadStarterMessage'})


def read_discord(export: Mapping[str, Any]) -> Transcript:
    """The JSON DiscordChatExporter writes: one channel, its messages,
    joins and pins counted apart, a reply naming what it answered."""
    channel = _dict(export.get('channel'))
    guild = _dict(export.get('guild'))
    name = _str(channel.get('name'), 'channel')
    chat = _chat_name(f'{guild["name"]}/{name}' if isinstance(guild.get('name'), str) and guild['name'] else name)
    messages: list[ChatMessage] = []
    system = media_only = attachments = unread = 0
    for entry in export.get('messages', ()):
        if not isinstance(entry, dict):
            continue
        if entry.get('type') not in _DISCORD_SPOKEN:
            system += 1
            continue
        if len(messages) >= MAX_MESSAGES:
            unread += 1
            continue
        author = _dict(entry.get('author'))
        sender = _str(author.get('nickname')) or _str(author.get('name'), 'unknown')
        sent_at = _rfc3339(_str(entry.get('timestamp')), f'Discord message {entry.get("id")} timestamp')
        carried = sum(len(entry[key]) for key in ('attachments', 'stickers') if isinstance(entry.get(key), list))
        attachments += carried
        content = _str(entry.get('content')).strip()
        if not content:
            media_only += 1
            continue
        reference = _dict(entry.get('reference'))
        reply = _str(reference.get('messageId')) or None
        messages.append(ChatMessage(_str(entry.get('id'), f'n{len(messages) + 1}'), sender, sent_at, content,
                                    channel=name, reply_to=reply, attachments=carried))
    return Transcript('discord', chat, tuple(messages), messages_unread=unread, system_messages=system,
                      media_only_messages=media_only, attachments_skipped=attachments)


_SLACK_SPOKEN_SUBTYPES = frozenset({'thread_broadcast', 'file_share', 'me_message', 'bot_message'})
_SLACK_MENTION = re.compile(r'<@([A-Z0-9]+)(?:\|[^>]*)?>')
_SLACK_CHANNEL = re.compile(r'<#[A-Z0-9]+\|([^>]*)>')
_SLACK_LINK = re.compile(r'<((?:https?|mailto):[^|>]*)(?:\|([^>]*))?>')
_SLACK_DAY_FILE = re.compile(r'^(?P<channel>[^/]+)/(?P<day>\d{4}-\d\d-\d\d)\.json$')


def _slack_text(raw: str, users: Mapping[str, str]) -> str:
    text = _SLACK_MENTION.sub(lambda m: '@' + users.get(m.group(1), m.group(1)), raw)
    text = _SLACK_CHANNEL.sub(lambda m: '#' + m.group(1), text)
    text = _SLACK_LINK.sub(lambda m: m.group(2) or m.group(1), text)
    return html.unescape(text)


def _slack_users(listing: object) -> dict[str, str]:
    users: dict[str, str] = {}
    if isinstance(listing, list):
        for user in listing:
            if not isinstance(user, dict) or not isinstance(user.get('id'), str):
                continue
            profile = _dict(user.get('profile'))
            name = _str(profile.get('display_name')) or _str(profile.get('real_name')) or _str(user.get('real_name')) or _str(user.get('name'))
            if name:
                users[user['id']] = name
    return users


def _slack_messages(entries: object, channel: Optional[str], users: Mapping[str, str],
                    into: list[ChatMessage], counts: Counter[str]) -> None:
    if not isinstance(entries, list):
        raise InvalidInput('Slack day file is not a list of messages')
    for entry in entries:
        if not isinstance(entry, dict) or entry.get('type') != 'message':
            counts['system'] += 1
            continue
        subtype = entry.get('subtype')
        if subtype and subtype not in _SLACK_SPOKEN_SUBTYPES:
            counts['system'] += 1
            continue
        if len(into) >= MAX_MESSAGES:
            counts['unread'] += 1
            continue
        ts = _str(entry.get('ts'))
        try:
            sent_at = _epoch(float(ts))
        except ValueError:
            raise InvalidInput(f'Slack message has no readable ts: {ts[:40]!r}') from None
        profile = _dict(entry.get('user_profile'))
        user = _str(entry.get('user'))
        sender = (_str(profile.get('display_name')) or _str(profile.get('real_name')) or users.get(user)
                  or _str(entry.get('username')) or user or 'unknown')
        files = entry.get('files')
        carried = len(files) if isinstance(files, list) else 0
        counts['attachments'] += carried
        content = _slack_text(_str(entry.get('text')), users).strip()
        if not content:
            counts['media_only'] += 1
            continue
        thread = _str(entry.get('thread_ts')) or None
        into.append(ChatMessage(ts, sender, sent_at, content, channel=channel,
                                reply_to=thread if thread and thread != ts else None, thread=thread, attachments=carried))


def read_slack_day(entries: object, *, channel: Optional[str] = None) -> Transcript:
    """One day file of a Slack export on its own: names come from the
    `user_profile` each message carries, and mentions of users it does
    not stay as ids. The file does not say its channel; the caller does."""
    messages: list[ChatMessage] = []
    counts: Counter[str] = Counter()
    _slack_messages(entries, channel, {}, messages, counts)
    return Transcript('slack', _chat_name(channel or 'Slack channel'), tuple(messages), messages_unread=counts['unread'],
                      system_messages=counts['system'], media_only_messages=counts['media_only'],
                      attachments_skipped=counts['attachments'])


def _read_slack_archive(archive: Any, days: list[tuple[str, str, str]]) -> Transcript:
    users = _slack_users(_load_json(archive.read('users.json'))) if 'users.json' in archive.names else {}
    messages: list[ChatMessage] = []
    counts: Counter[str] = Counter()
    for channel, _day, name in days:
        _slack_messages(_load_json(archive.read(name)), channel, users, messages, counts)
    channels = sorted({channel for channel, _day, _name in days})
    chat = channels[0] if len(channels) == 1 else f'{len(channels)} channels'
    return Transcript('slack', chat, tuple(messages), messages_unread=counts['unread'],
                      system_messages=counts['system'], media_only_messages=counts['media_only'],
                      attachments_skipped=counts['attachments'])


def read_slack_export(data: bytes, *, limits: DocumentLimits = DocumentLimits()) -> Transcript:
    """A Slack workspace export: every `channel/YYYY-MM-DD.json` member
    read in order, user ids resolved through `users.json`. The
    transcript's name is the channel, or how many channels there were;
    every message's own `channel` is what identifies it."""
    from .formats.archive import SafeArchive

    with SafeArchive(data, limits) as archive:
        days = sorted((match.group('channel'), match.group('day'), name)
                      for name in archive.names for match in [_SLACK_DAY_FILE.match(name)] if match)
        if not days:
            raise InvalidInput('not a Slack export: no channel day files (channel/YYYY-MM-DD.json) in the archive')
        return _read_slack_archive(archive, days)


# --- dispatch ---------------------------------------------------------------

def _renamed(found: Transcript, chat: Optional[str]) -> Transcript:
    return found if chat is None else Transcript(**{**found.__dict__, 'chat': _chat_name(chat)})


def read_transcript(data: bytes, filename: str, *, chat: Optional[str] = None, time_zone: Optional[str] = None,
                    date_order: Optional[DateOrder] = None, limits: DocumentLimits = DocumentLimits()) -> Transcript:
    """The export at `filename`, whichever platform wrote it: `.txt` is
    WhatsApp; `.zip` is WhatsApp when it holds the `_chat.txt` iOS shares
    and a Slack export otherwise; `.json` is decided by its shape. A file
    of none of those shapes is refused, naming what was expected."""
    if len(data) > limits.max_input_bytes:
        raise InvalidInput(f'chat export is {len(data)} bytes; at most {limits.max_input_bytes}')
    path = PurePath(filename)
    suffix = path.suffix.lower()
    if suffix == '.txt':
        return read_whatsapp(_whatsapp_text(data), chat=_chat_name(chat or path.stem or 'chat'),
                             time_zone=time_zone, date_order=date_order)
    if suffix == '.zip':
        from .formats.archive import SafeArchive

        with SafeArchive(data, limits) as archive:
            texts = [name for name in archive.names if name.lower().endswith('.txt')]
            shared = [name for name in texts if PurePosixPath(name).name.lower() == '_chat.txt'] or texts
            if len(shared) == 1:
                return read_whatsapp(_whatsapp_text(archive.read(shared[0])), chat=_chat_name(chat or path.stem or 'chat'),
                                     time_zone=time_zone, date_order=date_order)
            days = sorted((match.group('channel'), match.group('day'), name)
                          for name in archive.names for match in [_SLACK_DAY_FILE.match(name)] if match)
            if not days:
                raise InvalidInput('the archive is neither a WhatsApp export (one _chat.txt) nor a Slack export '
                                   '(channel/YYYY-MM-DD.json day files)')
            return _renamed(_read_slack_archive(archive, days), chat)
    if suffix != '.json':
        raise InvalidInput('chat export must be a WhatsApp .txt or .zip, a Telegram or Discord .json, or a Slack .json or .zip')
    loaded = _load_json(data)
    if isinstance(loaded, dict) and isinstance(loaded.get('messages'), list):
        if isinstance(loaded.get('channel'), dict) and 'guild' in loaded:
            return _renamed(read_discord(loaded), chat)
        if isinstance(loaded.get('type'), str) and 'name' in loaded:
            return _renamed(read_telegram(loaded, time_zone=time_zone), chat)
        raise InvalidInput('JSON chat export is neither a Telegram result.json (name, type, messages) '
                           'nor a Discord export (guild, channel, messages)')
    if isinstance(loaded, list) and loaded and all(isinstance(e, dict) for e in loaded) \
            and any('ts' in e and e.get('type') == 'message' for e in loaded):
        return read_slack_day(loaded, channel=chat)
    raise InvalidInput('JSON chat export is neither a Telegram result.json, a Discord export, nor a Slack day file')


def _whatsapp_text(data: bytes) -> str:
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise InvalidInput('WhatsApp export is not UTF-8 text') from None


# --- sessions and import ----------------------------------------------------

def sessions(messages: Sequence[ChatMessage], gap_seconds: int = DEFAULT_SESSION_GAP_SECONDS) -> list[list[ChatMessage]]:
    """Messages grouped into sessions: one channel each, in time order,
    split where the silence between two messages exceeds the gap."""
    if gap_seconds < 0:
        raise InvalidInput('session gap must be zero or more seconds')
    grouped: list[list[ChatMessage]] = []
    by_channel: dict[Optional[str], list[ChatMessage]] = {}
    for message in messages:
        by_channel.setdefault(message.channel, []).append(message)
    for channel_messages in by_channel.values():
        ordered = sorted(channel_messages, key=lambda m: m.sent_at)
        current: list[ChatMessage] = []
        last: Optional[datetime] = None
        for message in ordered:
            try:
                when = parse_rfc3339(message.sent_at)
            except ValueError:
                raise InvalidInput(f'message {message.key} is sent at {message.sent_at[:40]!r}, not an RFC 3339 time') from None
            if current and last is not None and (when - last).total_seconds() > gap_seconds:
                grouped.append(current)
                current = []
            current.append(message)
            last = when
        if current:
            grouped.append(current)
    return grouped


def session_id(transcript: Transcript, first: ChatMessage) -> str:
    where = f'{transcript.chat}/{first.channel}' if first.channel and first.channel != transcript.chat else transcript.chat
    return f'{transcript.platform}:{where}:{first.sent_at}'


def message_identity(transcript: Transcript, message: ChatMessage) -> str:
    """The dedup key: the platform, the channel the message was in (or
    the chat's name where the platform has no channels), and the
    platform's identity for the message. A Slack `ts` is unique within a
    channel, so the channel is part of it and the export's own name --
    which changes as channels come and go -- is not."""
    return f'chat:{transcript.platform}:{message.channel or transcript.chat}:{message.key}'


async def import_transcript(memory: Any, space: str, transcript: Transcript, *,
                            gap_seconds: int = DEFAULT_SESSION_GAP_SECONDS,
                            metadata: Optional[Mapping[str, str]] = None) -> ChatImported:
    """Store every message as a `conversation` episode dated when it was
    sent, keyed by `message_identity` so a second import of a grown
    export adds only what is new, its session in `source` and its
    speaker, chat and reply in metadata. Records the engine refuses (a
    speaker's name over the metadata limit) are counted as failed with
    the first reason; the rest are stored. An engine failure part-way
    raises; the batches before it stay stored, and a second import stores
    only what is missing."""
    check_space(space)
    extra = normalise_metadata(dict(metadata or {}))
    if len(extra) > MAX_METADATA_KEYS - OWN_METADATA_KEYS:
        raise InvalidInput(f'at most {MAX_METADATA_KEYS - OWN_METADATA_KEYS} caller metadata keys on a chat import')
    for key in extra:
        if key in _OWN_KEYS:
            raise InvalidInput(f'metadata key {key!r} is written by the chat import')
    grouped = sessions(transcript.messages, gap_seconds)
    records: list[Record] = []
    for group in grouped:
        source = session_id(transcript, group[0])
        for message in group:
            own = {'platform': transcript.platform, 'chat': transcript.chat, 'speaker': message.sender, 'message': message.key}
            for name in ('channel', 'reply_to', 'thread'):
                value = getattr(message, name)
                if value:
                    own[name] = value
            records.append(Record(message.text, 'conversation', source, (), message.sent_at, {**extra, **own},
                                  dedup_key=message_identity(transcript, message)))
    stored = duplicates = failed = 0
    reason: Optional[str] = None
    episode_ids: list[int] = []
    for start in range(0, len(records), IMPORT_BATCH):
        for added in await memory.remember_many(space, records[start:start + IMPORT_BATCH], partial=True):
            if added.outcome == 'failed':
                failed += 1
                reason = reason or added.reason
                continue
            episode_ids.append(added.episode_id)
            if added.outcome == 'accepted':
                stored += 1
            else:
                duplicates += 1
    ordered = sorted(message.sent_at for message in transcript.messages)
    return ChatImported(transcript.platform, transcript.chat, len(transcript.messages), stored, duplicates, failed,
                        len(grouped), gap_seconds, len({m.sender for m in transcript.messages}),
                        ordered[0] if ordered else None, ordered[-1] if ordered else None,
                        transcript.messages_unread, transcript.system_messages, transcript.unparsed_lines,
                        transcript.media_only_messages, transcript.attachments_skipped, transcript.date_order,
                        transcript.time_zone, failed_reason=reason, date_order_told=transcript.date_order_told,
                        episode_ids=tuple(episode_ids))


async def ingest_chat_export(memory: Any, space: str, data: bytes, *, filename: str, chat: Optional[str] = None,
                             time_zone: Optional[str] = None, date_order: Optional[DateOrder] = None,
                             gap_seconds: int = DEFAULT_SESSION_GAP_SECONDS,
                             metadata: Optional[Mapping[str, str]] = None,
                             limits: DocumentLimits = DocumentLimits()) -> ChatImported:
    """Read an export and import it: see `read_transcript` and
    `import_transcript`. A WhatsApp file no line of which decides the
    date order is not imported until the caller says the order: a guess
    would date every message wrong, under keys that would then hold."""
    transcript = read_transcript(data, filename, chat=chat, time_zone=time_zone, date_order=date_order, limits=limits)
    if transcript.date_order == 'undecidable':
        raise InvalidInput('no line of the file decides whether its dates are day-first or month-first; '
                           "say which with date_order ('day-first' or 'month-first') and import again")
    return await import_transcript(memory, space, transcript, gap_seconds=gap_seconds, metadata=metadata)
