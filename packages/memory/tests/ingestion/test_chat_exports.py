"""Chat exports become conversation memories: one per message, dated when
it was sent, keyed by the platform's identity, grouped into sessions."""
import io
import json
import zipfile

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion import chat_exports as chat
from scone_memory.ingestion.chat_exports import (ChatMessage, Transcript, import_transcript, ingest_chat_export,
                                                  message_identity, read_discord, read_slack_day, read_slack_export,
                                                  read_telegram, read_transcript, read_whatsapp, sessions)

ANDROID = """12/03/2024, 14:05 - Messages and calls are end-to-end encrypted. No one outside of this chat can read them.
12/03/2024, 14:05 - Alice: Hello Bob
12/03/2024, 14:06 - Bob: Hi! The plan:
12/03/2024 was the deadline, remember
so we slipped a day
13/03/2024, 09:00 - Alice: <Media omitted>
13/03/2024, 09:01 - Alice: This message was deleted
13/03/2024, 09:02 - Bob added Carol
13/03/2024, 09:03 - Carol: Morning all
"""


def test_whatsapp_android_lines_open_messages_and_lines_in_no_shape_continue_them():
    found = read_whatsapp(ANDROID, chat="Team")
    assert found.platform == "whatsapp" and found.chat == "Team"
    assert [m.sender for m in found.messages] == ["Alice", "Bob", "Carol"]
    assert found.messages[1].text == "Hi! The plan:\n12/03/2024 was the deadline, remember\nso we slipped a day", \
        "a line that starts with a date but has no clock is a continuation, not a message"
    assert found.messages[0].sent_at == "2024-03-12T14:05:00.000Z" and found.time_zone == "UTC (assumed)"
    assert found.date_order == "day-first", "13/03 decides it"
    assert found.system_messages == 2, "the encryption notice and the join"
    assert found.media_only_messages == 2 and found.attachments_skipped == 1, "the omitted media and the deletion notice; only the media was a file"
    assert found.unparsed_lines == 0
    assert len({m.key for m in found.messages}) == 3


def test_whatsapp_meridians_with_dots_and_spaces_are_clock_times_not_speakers():
    spanish = ("[12/3/24, 2:05:33 p. m.] Alice: buenas tardes\n"
               "[12/3/24, 9:05:33 a. m.] Alice: buenos días\n"
               "13/3/24, 11:59 p.m. - Bob: casi medianoche\n")
    found = read_whatsapp(spanish, chat="es")
    assert [(m.sender, m.sent_at) for m in found.messages] == [
        ("Alice", "2024-03-12T14:05:33.000Z"), ("Alice", "2024-03-12T09:05:33.000Z"), ("Bob", "2024-03-13T23:59:00.000Z")]
    assert found.system_messages == 0 and len({m.sender for m in found.messages}) == 2


def test_whatsapp_attachment_lines_are_counted_and_the_caption_is_the_message():
    text = ("[12/03/2024, 14:05:33] Alice: \u200e<attached: 00000042-PHOTO-2024-03-12-14-05-33.jpg>\n"
            "12/03/2024, 14:06 - Bob: IMG-20240312-WA0001.jpg (file attached)\nlook at this\n"
            "12/03/2024, 14:07 - Bob: plain words\n")
    found = read_whatsapp(text, chat="x")
    assert [m.text for m in found.messages] == ["look at this", "plain words"], "the file's name is not what was said"
    assert found.attachments_skipped == 2 and found.media_only_messages == 1
    assert not any("attached" in m.text for m in found.messages)


def test_whatsapp_senders_with_quotes_are_kept_and_iso_dates_decide_themselves():
    found = read_whatsapp('12/03/2024, 14:05 - Jo "Jojo" Smith: hi\n', chat="x")
    assert [m.sender for m in found.messages] == ['Jo "Jojo" Smith']
    iso = read_whatsapp("2024-03-12, 14:05 - Alice: hi\n2024-03-13, 09:00 - Bob: yo\n", chat="x")
    assert iso.date_order == "year-first" and iso.messages[1].sent_at == "2024-03-13T09:00:00.000Z"
    told = read_whatsapp("01/02/2024, 10:00 - Alice: hi\n", chat="x", date_order="month-first")
    assert told.date_order == "month-first" and told.date_order_told and told.messages[0].sent_at == "2024-01-02T10:00:00.000Z"
    with pytest.raises(InvalidInput, match="writes its dates day-first, not month-first"):
        read_whatsapp("13/02/2024, 10:00 - Alice: hi\n", chat="x", date_order="month-first")


def test_whatsapp_lines_that_continue_a_system_line_or_a_dropped_header_are_not_unparsed():
    text = ("12/03/2024, 14:05 - Alice changed the group description to:\nWe plant on Fridays.\nBring gloves.\n"
            "31/02/2024, 10:00 - Alice: no such day\nstill part of it\n"
            "12/03/2024, 14:06 - Bob: hello\n")
    found = read_whatsapp(text, chat="x")
    assert found.system_messages == 1 and found.unparsed_lines == 1, "the impossible header is counted once; its lines belong to it"
    assert [m.text for m in found.messages] == ["hello"]


def test_whatsapp_ios_shape_with_direction_marks_and_meridian_reads_month_first_when_the_file_says_so():
    ios = ("‎[3/12/24, 2:05:33 PM] Alice: Hi\n"
           "[3/13/24, 9:00:00 AM] Bob: ‎image omitted\n"
           "[3/13/24, 12:30:00 AM] Bob: past midnight\n"
           "[3/13/24, 12:30:00 PM] Bob: noon\n")
    found = read_whatsapp(ios, chat="iOS")
    assert found.date_order == "month-first"
    assert [m.sent_at for m in found.messages] == ["2024-03-12T14:05:33.000Z", "2024-03-13T00:30:00.000Z", "2024-03-13T12:30:00.000Z"]
    assert found.media_only_messages == 1 and found.attachments_skipped == 1


def test_whatsapp_dates_nobody_can_order_are_said_and_contradicting_ones_refused():
    found = read_whatsapp("01/02/2024, 10:00 - Alice: hi\n03/04/2024, 10:00 - Bob: yo\n", chat="x")
    assert found.date_order == "undecidable" and found.messages[0].sent_at == "2024-02-01T10:00:00.000Z", "read day-first, and the receipt says it was a guess"
    with pytest.raises(InvalidInput, match="day-first on some lines and month-first on others"):
        read_whatsapp("13/02/2024, 10:00 - Alice: hi\n02/13/2024, 10:00 - Bob: yo\n", chat="x")
    with pytest.raises(InvalidInput, match="expected WhatsApp lines"):
        read_whatsapp("just a text file\nwith two lines\n", chat="x")


def test_whatsapp_clock_times_are_read_in_the_zone_the_caller_gives():
    text = "12/03/2024, 14:05 - Alice: hi\n"
    assert read_whatsapp(text, chat="x", time_zone="+02:00").messages[0].sent_at == "2024-03-12T12:05:00.000Z"
    berlin = read_whatsapp(text, chat="x", time_zone="Europe/Berlin")
    assert berlin.messages[0].sent_at == "2024-03-12T13:05:00.000Z" and berlin.time_zone == "Europe/Berlin"
    with pytest.raises(InvalidInput, match="unknown time zone"):
        read_whatsapp(text, chat="x", time_zone="Mars/Olympus")


def test_whatsapp_lines_a_header_shape_matches_but_no_calendar_holds_are_counted_not_guessed():
    found = read_whatsapp("31/02/2024, 10:00 - Alice: not a day\n12/03/2024, 25:00 - Alice: not an hour\n12/03/2024, 10:00 - Alice: real\n", chat="x")
    assert found.unparsed_lines == 2 and [m.text for m in found.messages] == ["real"]


def test_the_message_bound_leaves_the_rest_counted(monkeypatch):
    monkeypatch.setattr(chat, "MAX_MESSAGES", 2)
    found = read_whatsapp(ANDROID, chat="Team")
    assert len(found.messages) == 2 and found.messages_unread == 3, "past the bound nothing is read, media placeholders included"
    telegram = read_telegram({"name": "g", "type": "private_group", "messages": [
        {"id": i, "type": "message", "date": "2024-03-12T10:00:00", "from": "A", "text": f"m{i}"} for i in range(4)]})
    assert len(telegram.messages) == 2 and telegram.messages_unread == 2
    discord = read_discord({"guild": {}, "channel": {"name": "c"}, "messages": [
        {"id": str(i), "type": "Default", "timestamp": "2024-03-12T10:00:00+00:00", "content": f"m{i}", "author": {"name": "a"}} for i in range(3)]})
    assert len(discord.messages) == 2 and discord.messages_unread == 1
    slack = read_slack_day([{"type": "message", "user": "U1", "text": f"m{i}", "ts": f"171023760{i}.000100"} for i in range(5)], channel="c")
    assert len(slack.messages) == 2 and slack.messages_unread == 3


def test_telegram_result_json_reads_entity_lists_service_messages_and_replies():
    export = {"name": "Garden club", "type": "private_group", "id": 7, "messages": [
        {"id": 1, "type": "service", "date": "2024-03-12T10:00:00", "actor": "Alice", "action": "create_group", "text": ""},
        {"id": 2, "type": "message", "date": "2024-03-12T10:01:00", "date_unixtime": "1710237660", "from": "Alice", "from_id": "user1",
         "text": ["Seeds arrive ", {"type": "bold", "text": "Friday"}, "."]},
        {"id": 3, "type": "message", "date": "2024-03-12T10:02:00", "from": "Bob", "from_id": "user2",
         "reply_to_message_id": 2, "text": "Great, I will bring trays"},
        {"id": 4, "type": "message", "date": "2024-03-12T10:03:00", "from": "Bob", "from_id": "user2", "photo": "photos/1.jpg", "text": ""},
    ]}
    found = read_telegram(export, time_zone="+01:00")
    assert found.chat == "Garden club" and found.system_messages == 1
    assert found.messages[0].text == "Seeds arrive Friday." and found.messages[0].sent_at == "2024-03-12T10:01:00.000Z", "date_unixtime is an instant; the zone does not move it"
    assert found.messages[1].sent_at == "2024-03-12T09:02:00.000Z" and found.messages[1].reply_to == "2", "a naive date is read in the caller's zone"
    assert found.media_only_messages == 1 and found.attachments_skipped == 1 and found.time_zone == "+01:00"
    assert [m.key for m in found.messages] == ["2", "3"]
    voice = read_telegram({"name": "g", "type": "private_group", "messages": [
        {"id": 9, "type": "message", "date": "2024-03-12T10:03:00", "from": "Bob", "file": "voice_messages/1.ogg", "media_type": "voice_message", "text": ""}]})
    assert voice.attachments_skipped == 1, "one file, however many keys describe it"


def test_discord_export_reads_replies_and_counts_joins_and_attachments():
    export = {"guild": {"id": "1", "name": "Makers"}, "channel": {"id": "2", "name": "lasers", "category": "shop"}, "messages": [
        {"id": "10", "type": "GuildMemberJoin", "timestamp": "2024-03-12T10:00:00+00:00", "content": "", "author": {"name": "bot"}},
        {"id": "11", "type": "Default", "timestamp": "2024-03-12T10:01:00.250+00:00", "content": "Lens is fogged again",
         "author": {"name": "alice", "nickname": "Ada"}, "attachments": [{"url": "x.png"}]},
        {"id": "12", "type": "Reply", "timestamp": "2024-03-12T11:01:00+01:00", "content": "Clean it with the blue cloth",
         "author": {"name": "bob"}, "reference": {"messageId": "11"}},
        {"id": "13", "type": "Default", "timestamp": "2024-03-12T10:03:00+00:00", "content": "", "author": {"name": "bob"}, "attachments": [{"url": "y.png"}]},
    ]}
    found = read_discord(export)
    assert found.chat == "Makers/lasers" and found.system_messages == 1
    assert [(m.sender, m.reply_to, m.channel) for m in found.messages] == [("Ada", None, "lasers"), ("bob", "11", "lasers")]
    assert found.messages[0].sent_at == "2024-03-12T10:01:00.250Z" and found.messages[0].attachments == 1
    assert found.messages[1].sent_at == "2024-03-12T10:01:00.000Z", "an offset is folded to UTC"
    assert found.media_only_messages == 1 and found.attachments_skipped == 2
    with pytest.raises(InvalidInput, match="timestamp"):
        read_discord({"guild": {}, "channel": {}, "messages": [{"id": "1", "type": "Default", "timestamp": "yesterday", "content": "x", "author": {}}]})


SLACK_DAY = [
    {"type": "message", "subtype": "channel_join", "user": "U1", "text": "<@U1> has joined the channel", "ts": "1710237600.000100"},
    {"type": "message", "user": "U1", "user_profile": {"display_name": "ada", "real_name": "Ada L"},
     "text": "Deploy at <https://example.test/run/9|run 9> &amp; ping <@U2>, see <#C9|ops>", "ts": "1710237660.000200"},
    {"type": "message", "user": "U2", "text": "on it", "ts": "1710237720.000300", "thread_ts": "1710237660.000200"},
    {"type": "message", "user": "U2", "text": "", "ts": "1710237780.000400", "files": [{"name": "log.txt"}]},
]


def test_a_slack_day_file_on_its_own_names_who_it_can_and_keeps_the_rest_as_ids():
    found = read_slack_day(SLACK_DAY, channel="deploys")
    assert found.system_messages == 1 and found.chat == "deploys"
    first, second = found.messages
    assert first.sender == "ada" and first.text == "Deploy at run 9 & ping @U2, see #ops"
    assert second.sender == "U2" and second.reply_to == "1710237660.000200" and second.thread == "1710237660.000200"
    assert first.thread is None and first.sent_at == "2024-03-12T10:01:00.000Z"
    assert found.media_only_messages == 1 and found.attachments_skipped == 1


def test_a_slack_export_zip_resolves_users_across_channels_and_days():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("users.json", json.dumps([{"id": "U1", "name": "ada", "profile": {"display_name": "", "real_name": "Ada L"}},
                                                    {"id": "U2", "name": "bob", "profile": {"display_name": "bobby"}}]))
        archive.writestr("channels.json", json.dumps([{"id": "C1", "name": "deploys"}]))
        archive.writestr("deploys/2024-03-12.json", json.dumps(SLACK_DAY))
        archive.writestr("random/2024-03-13.json", json.dumps([{"type": "message", "user": "U2", "text": "lunch?", "ts": "1710324000.000100"}]))
    found = read_slack_export(buffer.getvalue())
    assert found.chat == "2 channels" and [m.channel for m in found.messages] == ["deploys", "deploys", "random"]
    assert [m.sender for m in found.messages] == ["ada", "bobby", "bobby"], "the day file's own profile first, then users.json"
    assert found.messages[0].text.endswith("ping @bobby, see #ops")
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("users.json", "[]")
    with pytest.raises(InvalidInput, match="no channel day files"):
        read_slack_export(empty.getvalue())


def test_read_transcript_decides_by_suffix_and_shape_and_refuses_the_rest():
    assert read_transcript(ANDROID.encode(), "Family chat.txt").chat == "Family chat"
    assert read_transcript(ANDROID.encode(), "Family chat.txt", chat="Fam").chat == "Fam"
    telegram = json.dumps({"name": "g", "type": "private_group", "messages": [{"id": 1, "type": "message", "date": "2024-03-12T10:00:00", "from": "A", "text": "m"}]}).encode()
    assert read_transcript(telegram, "result.json").platform == "telegram"
    discord = json.dumps({"guild": {"name": "G"}, "channel": {"name": "c"}, "messages": []}).encode()
    assert read_transcript(discord, "c.json").platform == "discord"
    assert read_transcript(json.dumps(SLACK_DAY).encode(), "2024-03-12.json", chat="deploys").platform == "slack"
    shared = io.BytesIO()
    with zipfile.ZipFile(shared, "w") as archive:
        archive.writestr("_chat.txt", ANDROID)
        archive.writestr("00000042-PHOTO.jpg", b"not read")
    ios = read_transcript(shared.getvalue(), "WhatsApp Chat - Team.zip")
    assert ios.platform == "whatsapp" and ios.chat == "WhatsApp Chat - Team" and len(ios.messages) == 3
    with pytest.raises(InvalidInput, match="at most 200"):
        read_transcript(ANDROID.encode(), "t.txt", chat="x" * 201)
    with pytest.raises(InvalidInput, match="neither a Telegram"):
        read_transcript(b'{"items": [1, 2, 3]}', "data.json")
    with pytest.raises(InvalidInput, match="neither a Telegram"):
        read_transcript(b'{"messages": [], "hello": 1}', "data.json")
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("readme.md", "nothing")
    with pytest.raises(InvalidInput, match="neither a WhatsApp export .* nor a Slack export"):
        read_transcript(empty.getvalue(), "x.zip")
    with pytest.raises(InvalidInput, match="must be a WhatsApp"):
        read_transcript(b"a,b\n1,2\n", "table.csv")
    with pytest.raises(InvalidInput, match="not well-formed JSON"):
        read_transcript(b"{", "broken.json")


def _message(key, sender, sent_at, text="t", channel=None):
    return ChatMessage(key, sender, sent_at, text, channel=channel)


def test_sessions_split_on_silence_within_a_channel_in_time_order():
    messages = [_message("1", "a", "2024-03-12T10:00:00Z"), _message("2", "b", "2024-03-12T10:30:00Z"),
                _message("3", "a", "2024-03-12T17:00:00Z"), _message("0", "b", "2024-03-12T09:59:00Z"),
                _message("4", "c", "2024-03-12T10:10:00Z", channel="other")]
    grouped = sessions(messages, gap_seconds=6 * 3600)
    assert [[m.key for m in group] for group in grouped] == [["0", "1", "2"], ["3"], ["4"]]
    assert [[m.key for m in group] for group in sessions(messages, gap_seconds=0)] == [["0"], ["1"], ["2"], ["3"], ["4"]]
    with pytest.raises(InvalidInput, match="zero or more"):
        sessions(messages, gap_seconds=-1)


async def test_import_stores_one_dated_keyed_conversation_episode_per_message_and_a_second_import_adds_only_the_new(engine):
    first = read_whatsapp(ANDROID, chat="Team")
    receipt = await import_transcript(engine, "chat", first, metadata={"user_id": "mark"})
    assert (receipt.stored, receipt.duplicates, receipt.failed, receipt.messages) == (3, 0, 0, 3)
    assert receipt.sessions == 2 and receipt.speakers == 3 and receipt.first_at == "2024-03-12T14:05:00.000Z"
    assert receipt.system_messages == 2 and receipt.date_order == "day-first" and receipt.time_zone == "UTC (assumed)"
    episode = await engine.episode("chat", receipt.episode_ids[0])
    assert episode.kind == "conversation" and episode.content == "Hello Bob" and episode.created_at == "2024-03-12T14:05:00.000Z"
    assert episode.source == "whatsapp:Team:2024-03-12T14:05:00.000Z"
    assert episode.metadata["speaker"] == "Alice" and episode.metadata["chat"] == "Team" and episode.metadata["user_id"] == "mark"
    assert episode.metadata["platform"] == "whatsapp" and episode.metadata["message"] == first.messages[0].key
    later = await engine.episode("chat", receipt.episode_ids[2])
    assert later.source == "whatsapp:Team:2024-03-13T09:03:00.000Z", "the next morning is a second session"
    again = await import_transcript(engine, "chat", first)
    assert (again.stored, again.duplicates) == (0, 3), "the same export again stores nothing"
    grown = read_whatsapp(ANDROID + "13/03/2024, 09:04 - Alice: Coffee?\n", chat="Team")
    third = await import_transcript(engine, "chat", grown)
    assert (third.stored, third.duplicates) == (1, 3), "a grown export adds only the new message"
    with pytest.raises(InvalidInput, match="written by the chat import"):
        await import_transcript(engine, "chat", first, metadata={"speaker": "me"})
    with pytest.raises(InvalidInput, match="at most 9 caller metadata keys"):
        await import_transcript(engine, "chat", first, metadata={f"k{i}": "v" for i in range(10)})
    with pytest.raises(InvalidInput, match="metadata key must match"):
        await import_transcript(engine, "chat", first, metadata={"User": "mark"})


async def test_slack_messages_are_keyed_by_channel_so_a_grown_export_adds_only_the_new_channel(engine):
    def export(channels):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("users.json", "[]")
            for channel in channels:
                archive.writestr(f"{channel}/2024-03-12.json", json.dumps(SLACK_DAY))
        return buffer.getvalue()
    one = read_slack_export(export(["deploys"]))
    two = read_slack_export(export(["deploys", "random"]))
    assert one.chat == "deploys" and two.chat == "2 channels"
    assert message_identity(one, one.messages[0]) == message_identity(two, two.messages[0]) == "chat:slack:deploys:1710237660.000200"
    first = await import_transcript(engine, "chat", one)
    second = await import_transcript(engine, "chat", two)
    assert (first.stored, second.stored, second.duplicates) == (2, 2, 2), "the channel already imported is known; the new one is stored"


async def test_an_engine_failure_part_way_leaves_the_earlier_batches_stored_and_a_second_import_finishes(engine, monkeypatch):
    monkeypatch.setattr(chat, "IMPORT_BATCH", 1)
    transcript = read_whatsapp(ANDROID, chat="Team")
    real = engine.remember_many
    calls = 0
    async def flaky(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("embedder went away")
        return await real(*args, **kwargs)
    monkeypatch.setattr(engine, "remember_many", flaky)
    with pytest.raises(RuntimeError, match="embedder went away"):
        await import_transcript(engine, "chat", transcript)
    monkeypatch.setattr(engine, "remember_many", real)
    receipt = await import_transcript(engine, "chat", transcript)
    assert (receipt.stored, receipt.duplicates) == (2, 1), "the batch stored before the failure is known; the rest are stored now"


async def test_ingest_refuses_a_whatsapp_file_whose_date_order_nobody_decided_until_told(engine):
    data = b"01/02/2024, 10:00 - Alice: hi\n03/04/2024, 10:00 - Bob: yo\n"
    with pytest.raises(InvalidInput, match="decides whether its dates are day-first or month-first"):
        await ingest_chat_export(engine, "chat", data, filename="quiet.txt")
    receipt = await ingest_chat_export(engine, "chat", data, filename="quiet.txt", date_order="month-first")
    assert receipt.stored == 2 and receipt.date_order == "month-first" and receipt.date_order_told
    assert receipt.first_at == "2024-01-02T10:00:00.000Z"


async def test_a_message_the_engine_refuses_is_counted_failed_with_its_reason_and_the_rest_stored(engine):
    transcript = Transcript("telegram", "g", (_message("1", "a", "2024-03-12T10:00:00Z", "fine"),
                                              _message("2", "a", "2024-03-12T10:01:00Z", "   ")))
    receipt = await import_transcript(engine, "chat", transcript)
    assert (receipt.stored, receipt.failed) == (1, 1) and receipt.failed_reason and "content" in receipt.failed_reason
    with pytest.raises(InvalidInput, match="not an RFC 3339 time"):
        await import_transcript(engine, "chat", Transcript("telegram", "g", (_message("3", "a", "yesterday", "hi"),)))
