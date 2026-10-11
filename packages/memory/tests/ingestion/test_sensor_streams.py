"""Sensor streams keep the readings that say something changed, dated when they were read."""
from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.sensor_streams import SensorStream, sensor_events, sensor_state


async def engine() -> MemoryEngine:
    return await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()


def at(second: int) -> str:
    return f'2026-10-08T12:{second // 60:02d}:{second % 60:02d}Z'


async def test_a_number_is_kept_when_it_moves_by_the_deadband_from_the_last_kept_reading():
    stream = SensorStream(await engine(), 'plant', 'pump-3-temperature', unit='°C', deadband=5)
    first = await stream.observe(70.0, observed_at=at(0))
    near = await stream.observe(73.0, observed_at=at(1))
    drifted = await stream.observe(75.0, observed_at=at(2))
    assert (first.kept, first.reason, first.change) == (True, 'first', None)
    assert (near.kept, near.reason) == (False, 'steady') and near.change == pytest.approx(3.0)
    assert (drifted.kept, drifted.reason) == (True, 'changed'), 'measured from the last kept reading, 70, not from 73'
    assert drifted.change == pytest.approx(5.0)
    assert (stream.seen, stream.kept, stream.dropped) == (3, 2, 1)
    assert near.added is None and near.fact is None


async def test_without_a_deadband_any_different_number_is_kept():
    stream = SensorStream(await engine(), 'plant', 'counter')
    await stream.observe(10, observed_at=at(0))
    assert (await stream.observe(10, observed_at=at(1))).kept is False
    assert (await stream.observe(11, observed_at=at(2))).reason == 'changed'


async def test_crossing_a_limit_is_kept_whatever_the_deadband_and_staying_past_it_is_not():
    stream = SensorStream(await engine(), 'plant', 'pump-3-temperature', unit='°C', deadband=50, low=10, high=90)
    await stream.observe(89.0, observed_at=at(0))
    over = await stream.observe(91.0, observed_at=at(1))
    still = await stream.observe(92.0, observed_at=at(2))
    back = await stream.observe(88.0, observed_at=at(3))
    under = await stream.observe(9.0, observed_at=at(4))
    assert (over.kept, over.reason, over.band) == (True, 'limit_crossed', 'above_high')
    assert (still.kept, still.band) == (False, 'above_high')
    assert (back.kept, back.reason, back.band) == (True, 'limit_crossed', 'in_range')
    assert (under.reason, under.band) == ('limit_crossed', 'below_low')
    assert over.added is not None
    stored = await stream.engine.documents.get_episode('plant', over.added.episode_id)
    assert stored is not None and stored.content == (
        'pump-3-temperature read 91 °C on 2026-10-08T12:00:01.000Z. It rose above the high limit of 90 °C.')


async def test_a_state_is_kept_when_it_differs_and_numbers_and_states_do_not_mix():
    stream = SensorStream(await engine(), 'plant', 'valve-7')
    await stream.observe('closed', observed_at=at(0))
    same = await stream.observe('closed', observed_at=at(1))
    opened = await stream.observe('open', observed_at=at(2))
    assert same.kept is False
    assert (opened.kept, opened.reason, opened.band, opened.change) == (True, 'state_changed', None, None)
    with pytest.raises(InvalidInput, match='numbers or states'):
        await stream.observe(3.5, observed_at=at(3))
    flag = SensorStream(stream.engine, 'plant', 'guard-door')
    await flag.observe(True, observed_at=at(0))
    assert (await flag.observe(False, observed_at=at(1))).reason == 'state_changed'
    assert (await sensor_state(stream.engine, 'plant', 'guard-door')).value == 'false'  # type: ignore[union-attr]


async def test_a_steady_sensor_is_kept_again_when_the_heartbeat_is_due():
    stream = SensorStream(await engine(), 'plant', 'tank-level', heartbeat_seconds=60)
    await stream.observe(40.0, observed_at=at(0))
    early = await stream.observe(40.0, observed_at=at(59))
    due = await stream.observe(40.0, observed_at=at(60))
    after = await stream.observe(40.0, observed_at=at(61))
    assert early.kept is False
    assert (due.kept, due.reason) == (True, 'heartbeat')
    assert due.fact is None, 'the value did not change, so the ledger keeps the fact it has'
    assert after.kept is False, 'the heartbeat is measured from the last kept reading'


async def test_a_kept_reading_is_an_episode_dated_when_it_was_read_and_a_fact_from_then():
    memory = await engine()
    stream = SensorStream(memory, 'plant', 'pump-3-temperature', unit='°C', place='station-4')
    kept = await stream.observe(71.25, observed_at='2026-10-08T14:00:05+02:00')
    assert kept.added is not None and kept.fact is not None
    episode = await memory.documents.get_episode('plant', kept.added.episode_id)
    assert episode is not None
    assert episode.created_at == '2026-10-08T12:00:05.000Z'
    assert episode.source == 'sensor:pump-3-temperature'
    assert episode.content == 'pump-3-temperature read 71.25 °C at station-4 on 2026-10-08T12:00:05.000Z.'
    assert {k: episode.metadata[k] for k in ('document_format', 'sensor', 'sensor_event', 'sensor_value',
                                             'sensor_unit', 'sensor_place')} == {
        'document_format': 'sensor_reading', 'sensor': 'pump-3-temperature', 'sensor_event': 'first',
        'sensor_value': '71.25', 'sensor_unit': '°C', 'sensor_place': 'station-4'}
    assert (kept.fact.subject, kept.fact.predicate, kept.fact.object) == ('pump-3-temperature', 'reads', '71.25 °C')
    assert kept.fact.valid_from == '2026-10-08T12:00:05.000Z' and kept.fact.source_episode_id == episode.episode_id


async def test_a_value_the_sensor_returns_to_is_a_new_record_and_an_exact_retry_is_not():
    memory = await engine()
    stream = SensorStream(memory, 'plant', 'valve-7')
    first = await stream.observe('closed', observed_at=at(0))
    await stream.observe('open', observed_at=at(1))
    again = await stream.observe('closed', observed_at=at(2))
    assert first.added is not None and again.added is not None
    assert again.added.episode_id != first.added.episode_id and again.added.deduplicated is False
    assert [e.value for e in (await sensor_events(memory, 'plant')).items] == ['closed', 'open', 'closed']
    restarted = SensorStream(memory, 'plant', 'valve-7')
    retry = await restarted.observe('closed', observed_at=at(2))
    assert retry.added is not None and retry.added.deduplicated is True
    assert retry.added.episode_id == again.added.episode_id
    assert len((await sensor_events(memory, 'plant')).items) == 3


async def test_sensor_state_reads_the_ledger_now_and_at_a_past_time():
    memory = await engine()
    stream = SensorStream(memory, 'plant', 'pump-3-temperature', unit='°C')
    assert await sensor_state(memory, 'plant', 'pump-3-temperature') is None
    await stream.observe(70.0, observed_at=at(0))
    await stream.observe(80.0, observed_at=at(10))
    await stream.observe(95.5, observed_at=at(20))
    now = await sensor_state(memory, 'plant', 'pump-3-temperature')
    then = await sensor_state(memory, 'plant', 'pump-3-temperature', as_of=at(15))
    before = await sensor_state(memory, 'plant', 'pump-3-temperature', as_of='2026-10-08T11:59:59Z')
    assert now is not None and (now.value, now.since, now.until) == ('95.5 °C', '2026-10-08T12:00:20.000Z', None)
    assert then is not None and (then.value, then.since, then.until) == (
        '80 °C', '2026-10-08T12:00:10.000Z', '2026-10-08T12:00:20.000Z')
    assert before is None
    assert await sensor_state(memory, 'plant', 'another-sensor') is None


async def test_sensor_events_are_newest_first_and_narrow_by_sensor_place_reason_and_time():
    memory = await engine()
    pump = SensorStream(memory, 'plant', 'pump-3-temperature', unit='°C', place='station-4', high=90)
    valve = SensorStream(memory, 'plant', 'valve-7', place='station-9')
    await pump.observe(70.0, observed_at=at(0))
    await valve.observe('closed', observed_at=at(5))
    await pump.observe(95.0, observed_at=at(10))
    await valve.observe('open', observed_at=at(15))
    await pump.observe(80.0, observed_at=at(20))
    everything = await sensor_events(memory, 'plant')
    assert [(e.sensor, e.value) for e in everything.items] == [
        ('pump-3-temperature', '80'), ('valve-7', 'open'), ('pump-3-temperature', '95'), ('valve-7', 'closed'),
        ('pump-3-temperature', '70')]
    assert everything.more is False
    assert [e.value for e in (await sensor_events(memory, 'plant', sensor='valve-7')).items] == ['open', 'closed']
    assert [e.sensor for e in (await sensor_events(memory, 'plant', place='station-9')).items] == ['valve-7', 'valve-7']
    crossed = await sensor_events(memory, 'plant', reason='limit_crossed')
    assert [(e.value, e.band, e.unit, e.place) for e in crossed.items] == [
        ('80', 'in_range', '°C', 'station-4'), ('95', 'above_high', '°C', 'station-4')]
    window = await sensor_events(memory, 'plant', since=at(5), until=at(15))
    assert [e.observed_at[17:19] for e in window.items] == ['15', '10', '05'], 'both ends are included'
    cut = await sensor_events(memory, 'plant', limit=2)
    assert [e.value for e in cut.items] == ['80', 'open'] and cut.more is True


async def test_a_question_in_words_finds_the_event():
    memory = await engine()
    pump = SensorStream(memory, 'plant', 'pump-3-temperature', unit='°C', place='station-4', high=90)
    await pump.observe(70.0, observed_at=at(0))
    await pump.observe(95.0, observed_at=at(10))
    found = await memory.recall('plant', 'when did the temperature rise above the high limit', limit=1,
                                where={'document_format': 'sensor_reading'})
    assert found.items[0].created_at == '2026-10-08T12:00:10.000Z'
    assert 'rose above the high limit of 90 °C' in found.items[0].text


async def test_bad_values_and_bad_settings_are_refused():
    memory = await engine()
    stream = SensorStream(memory, 'plant', 'tank-level')
    await stream.observe(40.0, observed_at=at(10))
    with pytest.raises(InvalidInput, match='RFC 3339'):
        await stream.observe(41.0, observed_at='noon')
    with pytest.raises(InvalidInput, match='finite number'):
        await stream.observe(float('nan'), observed_at=at(11))
    assert (stream.seen, stream.kept) == (1, 1), 'a refused reading is not counted'
    for bad in ({'deadband': 0}, {'low': 5, 'high': 5}, {'heartbeat_seconds': 0}, {'unit': ''}, {'high': float('inf')}):
        with pytest.raises(InvalidInput):
            SensorStream(memory, 'plant', 'tank-level', **bad)  # type: ignore[arg-type]
    with pytest.raises(InvalidInput, match='limit'):
        await sensor_events(memory, 'plant', limit=501)
    with pytest.raises(InvalidInput, match='reason'):
        await sensor_events(memory, 'plant', reason='steady')
    with pytest.raises(InvalidInput, match='since'):
        await sensor_events(memory, 'plant', since=at(20), until=at(10))


async def test_an_engine_that_keeps_several_values_per_sensor_is_refused():
    """With ``reads`` many-valued the ledger would not close a value when the next arrives, and the state would
    name one that no longer holds."""
    memory = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                many_valued=('reads',)).open()
    with pytest.raises(InvalidInput, match='single-valued'):
        SensorStream(memory, 'plant', 'valve-7')
    with pytest.raises(InvalidInput, match='single-valued'):
        await sensor_state(memory, 'plant', 'valve-7')


async def test_a_reading_dated_before_the_last_one_is_recorded_as_late_and_moves_nothing():
    """A clock that steps back must not stop the stream, lose the reading, or rewrite what the sensor reads now."""
    memory = await engine()
    stream = SensorStream(memory, 'plant', 'machine-temperature', unit='°F', deadband=5, heartbeat_seconds=60)
    await stream.observe(90.0, observed_at=at(100))
    late = await stream.observe(40.0, observed_at=at(50))
    assert (late.kept, late.reason, late.fact) == (True, 'late', None) and late.added is not None
    assert (stream.seen, stream.kept, stream.dropped, stream.late) == (2, 2, 0, 1)
    now = await sensor_state(memory, 'plant', 'machine-temperature')
    assert now is not None and now.value == '90 °F', 'the ledger still says what was read last in time order'
    episode = await memory.documents.get_episode('plant', late.added.episode_id)
    assert episode is not None and episode.created_at == '2026-10-08T12:00:50.000Z'
    assert episode.content.endswith('It arrived out of order: a later reading had already been recorded.')
    near = await stream.observe(92.0, observed_at=at(101))
    assert (near.kept, near.reason) == (False, 'steady'), 'compared with 90, not with the late 40'
    not_due = await stream.observe(90.0, observed_at=at(159))
    assert not_due.kept is False, 'the heartbeat still runs from second 100, not from the late reading'
    assert [e.reason for e in (await sensor_events(memory, 'plant', reason='late')).items] == ['late']


async def test_a_stream_that_starts_past_a_limit_says_so_and_is_found_by_its_band():
    memory = await engine()
    hot = SensorStream(memory, 'plant', 'pump-3-temperature', unit='°C', deadband=1, high=90)
    cold = SensorStream(memory, 'plant', 'freezer', unit='°C', deadband=1, low=-20)
    fine = SensorStream(memory, 'plant', 'office', unit='°C', deadband=1, low=10, high=30)
    first = await hot.observe(92.4, observed_at=at(0))
    await cold.observe(-25, observed_at=at(1))
    await fine.observe(21, observed_at=at(2))
    assert (first.reason, first.band) == ('first', 'above_high'), 'nothing was crossed: there was no reading before'
    texts = {event.sensor: event.text for event in (await sensor_events(memory, 'plant')).items}
    assert texts['pump-3-temperature'].endswith('It was above the high limit of 90 °C.')
    assert texts['freezer'].endswith('It was below the low limit of -20 °C.')
    assert 'limit' not in texts['office']
    assert (await sensor_events(memory, 'plant', reason='limit_crossed')).items == ()
    assert [event.sensor for event in (await sensor_events(memory, 'plant', band='above_high')).items] == ['pump-3-temperature']
    assert [event.sensor for event in (await sensor_events(memory, 'plant', band='below_low')).items] == ['freezer']
    assert [event.sensor for event in (await sensor_events(memory, 'plant', band='in_range')).items] == ['office']
    with pytest.raises(InvalidInput, match='band must be one of'):
        await sensor_events(memory, 'plant', band='hot')  # type: ignore[arg-type]


async def test_a_reading_exactly_at_a_limit_is_within_it():
    stream = SensorStream(await engine(), 'plant', 'pump-3-temperature', deadband=100, low=10, high=90)
    assert (await stream.observe(50, observed_at=at(0))).band == 'in_range'
    for second, value in ((1, 90), (2, 10)):
        exact = await stream.observe(value, observed_at=at(second))
        assert (exact.kept, exact.band) == (False, 'in_range'), 'the limits themselves are allowed values'
    assert (await stream.observe(90.5, observed_at=at(3))).reason == 'limit_crossed'


async def test_a_late_reading_does_not_become_the_time_later_readings_are_judged_against():
    stream = SensorStream(await engine(), 'plant', 'counter')
    await stream.observe(1, observed_at=at(10))
    assert (await stream.observe(2, observed_at=at(2))).reason == 'late'
    still_late = await stream.observe(3, observed_at=at(5))
    assert still_late.reason == 'late' and still_late.fact is None, 'second 5 is before second 10, the last reading in order'
    assert stream.late == 2
    assert (await sensor_state(stream.engine, 'plant', 'counter')).value == '1'
