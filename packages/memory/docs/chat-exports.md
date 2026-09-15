# Chat exports as memory

[Package overview](../README.md) · [File ingestion](file-ingestion.md) · [Text conversations](text-conversations.md)

A chat export is not a document. Its unit is the message: who said it, when,
in which chat, in reply to what. Read as a document it would be one `file`
episode dated the day of the export, with passages that carry no speaker and
no time, and a recall capped at two passages per episode would hand back two
lines of a year of conversation. Read as a chat it becomes what the engine
already understands from live conversations: one `conversation` episode per
message, dated when it was sent, keyed so a second import of a grown export
adds only what is new, and grouped into sessions by silence.

Everything here reads local files. No credential, no network, no platform
API: the reference frameworks' Slack, Discord and Telegram readers call the
platforms with a token, which a local memory should not need.

## What is read

| Export | File | Shape |
| --- | --- | --- |
| WhatsApp | the `.txt` a phone exports, or the `.zip` iOS shares it in (`_chat.txt` and the media) | Android `12/03/2024, 14:05 - Name: text` and iOS `[12/03/2024, 14:05:33] Name: text`, 12- and 24-hour clocks (`p. m.` included), continuation lines, media placeholders, attachment lines and system lines |
| Telegram | Telegram Desktop's `result.json` | one chat; `message` entries kept, `service` entries counted; text as a string or a list of strings and entities |
| Discord | the JSON DiscordChatExporter writes | one channel; `Default` and `Reply` messages kept, joins and pins counted; a reply names the message it answered |
| Slack | a workspace export `.zip`, or one `channel/YYYY-MM-DD.json` day file | user ids resolved through `users.json` when the zip carries it, mentions and links unwrapped, `channel_join` and the like counted; a thread reply names its root |

```bash
scone-memory --space chats import-chat "Family chat.txt" --time-zone Europe/London --meta user_id=mark
scone-memory --space work import-chat slack-export.zip --gap-hours 2
scone-memory --space work import-chat result.json --chat "Garden club" --json
```

```python
from scone_memory.ingestion import ingest_chat_export

receipt = await ingest_chat_export(engine, "chats", data, filename="Family chat.txt",
                                   time_zone="Europe/London", metadata={"user_id": "mark"})
receipt.stored, receipt.duplicates, receipt.sessions   # 412, 0, 37
```

`read_transcript` reads without storing, for a caller that wants to look
first; `import_transcript` stores a transcript it was given.

Over HTTP the export is uploaded first, as a document is, then imported by
its attachment id with a small JSON body; the receipt is the same one the
command prints, plus `episodes`, `attachment_id` and `filename`. The server
advertises the route as the `chats.imports` capability, and the client has a
typed namespace for it:

```bash
curl -X POST $SCONE/v1/attachments -H "authorization: Bearer $KEY" -H "x-filename: Family chat.txt" --data-binary @"Family chat.txt"
curl -X POST $SCONE/v1/chat-imports -H "authorization: Bearer $KEY" -H "content-type: application/json" \
     -d '{"attachment_id": "<sha256 from the upload>", "time_zone": "Europe/London", "metadata": {"user_id": "mark"}}'
```

```python
chats = client.chat_imports(expected_space="chats")
stored = chats.upload(open("Family chat.txt", "rb").read(), filename="Family chat.txt")
receipt = chats.import_export(stored.attachment_id, time_zone="Europe/London", metadata={"user_id": "mark"})
receipt.stored, receipt.duplicates, receipt.sessions
```

A WhatsApp file no line of which decides the date order is refused with 422
until `date_order` says, as every refusal from the reader is (400 is a
malformed request body); `gap_seconds` is at most 36 hours; the caller's
metadata is at most nine keys. The export's name decides how it is read:
give it at upload (`x-filename`, or the client's `filename=`) or in the
request. The uploaded export stays retained as an attachment in the space
until it is forgotten.

A file is at most 25 MB, and an archive at most 10,000 members: a Slack
workspace export of many channels over years can exceed both, and is
refused whole, naming the number. Export a narrower range.

## What each message becomes

One `conversation` episode: its content the message text as the export had
it, `created_at` when it was sent, `source` the session it belongs to, and
metadata `platform`, `chat`, `speaker`, `message` (the platform's identity for
it), plus `channel`, `reply_to` and `thread` when the export says so. The
caller's own metadata (`user_id`, an `agent_id`) goes on every message
alongside; those seven names are the import's, and a caller who passes one
is refused rather than overwritten.

The dedup key is the platform, the channel the message was in, and the
platform's identity for the message (Slack `ts`, which is unique within a
channel; Telegram and Discord `id`; WhatsApp has none, so a digest of when,
who and what). Importing the same export twice stores nothing the second
time; a grown export, or a Slack export that gained a channel, stores only
the new messages. The receipt says `stored` and `duplicates` apart.

Where the platform has no channels the chat's name stands in for one, so a
WhatsApp or Telegram export imported under another name (`--chat`, or a
renamed file, since the file's name is the default) is another chat to the
engine and is stored again. Import a chat under one name.

## Sessions

Messages are grouped into sessions per channel, in time order, split where
the silence between two messages exceeds the gap (`--gap-hours`, six by
default). A session's id is `platform:chat:first message time`, and it is the
episode's `source`, which is how the benches and recall's per-session
handling already read a conversation.

## Times and dates

Discord and Slack write instants; they are read as they are. WhatsApp and
Telegram write the phone's clock time with no zone. `--time-zone` (an IANA
name like `Europe/Berlin` or an offset like `+02:00`) says which zone that
was; without it the times are read as UTC and the receipt's `time_zone` says
`UTC (assumed)`. Telegram exports that carry `date_unixtime` are instants and
need no zone. A WhatsApp message's identity is a digest of its time, so import
a chat with the same zone each time: the same file read in another zone is
another set of messages to the engine, and would be stored again.

WhatsApp writes dates in the phone's order, day first or month first, or
year first in a few locales. The reader decides from the file: a first field
over 12 is a day, a second one is a month, a four-digit first field is a
year. A file no line decides is not imported until `--date-order` says which
(`day-first` or `month-first`), because a guess would date every message
wrong under keys that would then hold; `read_transcript` reads such a file
and says `undecidable`. A file whose lines say both, or that contradicts
`--date-order`, is refused. The receipt's `date_order_told` says when the
order came from the caller rather than the file.

A clock time repeated at a zone's fall-back (the second 01:30 of the year) is
read as its first occurrence; the export does not say which it was.

## What is counted, not stored

Every receipt says what did not become a memory, so a count of stored
messages is never mistaken for the export's size:

- `system_messages`: joins, leaves, pins, subject changes, WhatsApp's
  encryption notice.
- `media_only_messages`: a message that was a media placeholder (`<Media
  omitted>`, `image omitted`), an attachment line (`<attached: …>`, `… (file
  attached)`), a deletion notice, or an attachment with no text. A caption
  under an attachment line is the message. `attachments_skipped` counts the
  files, images and media the export named but did not carry; nothing is
  read from them.
- `unparsed_lines`: WhatsApp lines before the first header that matched no
  shape, and headers whose date no calendar holds. A line that continues a
  system line or a dropped header belongs to it and is not counted.
- `messages_unread`: past 50,000 messages the rest are counted, not read.
- `failed`: messages the engine refused, with the first `failed_reason` (a
  message over the content limit); the rest are stored.

WhatsApp's placeholders are recognised in English; another language's are
kept as text. A line in header shape is a message and cannot be told from a
message that quoted one: the export does not mark the difference. A saved
name that contains a colon cannot be told from a system line either, since
the colon is what parts a name from its text; such a contact's lines are
counted as system lines.

The command exits 1 when the import stored nothing and refused something,
and 2 when the file could not be read at all.
