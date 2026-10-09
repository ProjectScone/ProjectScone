"""Sensor streams: readings over time, kept when they say something changed.

A controller reports a temperature every second and a valve its state every cycle. Almost every reading repeats
the last one. A ``SensorStream`` accepts each reading and keeps it only when it is the first, crosses a limit,
changes state, moves by at least the deadband, or the heartbeat is due. A kept reading becomes two things:

- an **episode**: one sentence saying what was read, where and when, dated at the instant it was read, so the
  text lanes, ``since``, ``as_of`` and forgetting apply to it like any other record;
- a **fact** in the ledger, ``<sensor> reads <value>``, valid from that instant. A new value closes the one
  before, so ``sensor_state(..., as_of=)`` answers what the sensor read at any past time.

``sensor_events`` lists the kept readings newest first, narrowed by sensor, place, reason and time.

Every bound says when it bit: a receipt names why a reading was kept or dropped, the stream counts both, and
``SensorEvents.more`` says the list was cut at its limit.

Three things are deliberately the caller's:

- **Limits and deadband.** They are engineering values of the process, not something to infer from the data.
- **Clocks.** A reading dated before the last one (a clock stepped back, a batch replayed) is kept as a record
  marked ``late`` and counted, but it does not move the gate or the ledger: what the sensor reads now is
  decided by readings in time order.
- **The gate's memory.** The last kept value lives in the process. A new ``SensorStream`` keeps its first
  reading whatever an earlier one saw.

This is a record of what was read. It is not a control loop: nothing here is fast or certain enough to stop a
machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
from typing import TYPE_CHECKING, Literal, cast

from ..core.errors import InvalidInput
from ..core.models import Added, Fact
from ..core.timeutil import format_rfc3339, parse_rfc3339
from ..core.validation import check_space

if TYPE_CHECKING:
    from ..core import forget_after as schedule
    from ..memory.engine import MemoryEngine

__all__ = ['DEFAULT_HEARTBEAT_SECONDS', 'EVENT_LIMIT', 'MAX_EVENT_LIMIT', 'SensorEvent', 'SensorEvents', 'SensorRead',
           'SensorState', 'SensorStream', 'sensor_events', 'sensor_state']

#: A reading is kept after this long without one, so a steady sensor still leaves a trace.
DEFAULT_HEARTBEAT_SECONDS = 300.0
#: Events one ``sensor_events`` call returns.
EVENT_LIMIT = 50
MAX_EVENT_LIMIT = 500
SOURCE_PREFIX = 'sensor:'
FORMAT = 'sensor_reading'
PREDICATE = 'reads'

Value = float | int | str | bool
Reason = Literal['first', 'limit_crossed', 'state_changed', 'changed', 'heartbeat', 'late', 'steady']
Band = Literal['below_low', 'in_range', 'above_high']
REASONS: tuple[Reason, ...] = ('first', 'limit_crossed', 'state_changed', 'changed', 'heartbeat', 'late')


def _name(value: object, what: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit or any(ord(c) < 32 for c in value):
        raise InvalidInput(f'{what} must be 1..{limit} printable characters')
    return value.strip()


def _tag(kind: str, name: str) -> str:
    # Engine tags are casefolded; hashing keeps "Pump-3" and "pump-3" apart.
    return hashlib.sha256(f'sensor-{kind}:{name}'.encode()).hexdigest()


def _instant(text: object, what: str = 'observed_at') -> datetime:
    if not isinstance(text, str):
        raise InvalidInput(f'{what} must be an RFC 3339 timestamp')
    try:
        return parse_rfc3339(text)
    except ValueError:
        raise InvalidInput(f'{what} must be an RFC 3339 timestamp') from None


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidInput(f'{what} must be a finite number')
    return float(value)


def _check_single_valued(engine: MemoryEngine) -> None:
    # A new value has to close the one before it. A many-valued predicate keeps both, and the state would lie.
    if PREDICATE in engine.many_valued:
        raise InvalidInput(f'sensor streams need {PREDICATE!r} to be single-valued; this engine lists it in many_valued')


def _shown(value: float) -> str:
    """A number as a person would write it: 90 not 90.0, 92.4 not 92.40000000000001."""
    return format(value, '.10g')


@dataclass(frozen=True)
class SensorRead:
    """What the stream did with one reading."""

    kept: bool
    #: ``first``: nothing kept yet. ``limit_crossed``: it moved into or out of the limits. ``state_changed``: a
    #: state (text or true/false) differs from the last kept one. ``changed``: a number moved by at least the
    #: deadband. ``heartbeat``: unchanged, but the heartbeat was due. ``late``: dated before the last reading,
    #: so it was recorded without moving the gate or the ledger. ``steady``: unchanged, so it was dropped.
    reason: Reason
    #: Where a number sits against the limits; ``None`` for a state, or when the stream has no limits.
    band: Band | None
    #: A number's difference from the last kept reading; ``None`` for the first reading and for states.
    change: float | None
    #: The stored episode, when the reading was kept.
    added: Added | None = None
    #: The ledger fact, when the reading was kept in time order and its value differs from the last kept one.
    fact: Fact | None = None


@dataclass(frozen=True)
class SensorState:
    """What a sensor read at one time."""

    sensor: str
    #: The value as it was written, with its unit: ``"92.4 °C"``, ``"open"``.
    value: str
    #: When the sensor started reading this.
    since: str
    #: When it stopped; ``None`` while it still reads this.
    until: str | None
    fact_id: int
    #: The kept reading that recorded it.
    episode_id: int | None


@dataclass(frozen=True)
class SensorEvent:
    """One kept reading."""

    episode_id: int
    sensor: str
    observed_at: str
    reason: Reason
    #: The value as it was written, without its unit.
    value: str
    unit: str | None
    place: str | None
    band: Band | None
    #: The sentence stored for the reading.
    text: str


@dataclass(frozen=True)
class SensorEvents:
    """Kept readings, newest first."""

    items: tuple[SensorEvent, ...]
    #: Readings matched beyond ``limit``: older ones exist that this list leaves out.
    more: bool


class SensorStream:
    """One sensor in one space. See the module."""

    def __init__(self, engine: MemoryEngine, space: str, sensor: str, *, unit: str | None = None,
                 place: str | None = None, deadband: float | None = None, low: float | None = None,
                 high: float | None = None, heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
                 forget_after: str | schedule.Resolved | None = None) -> None:
        check_space(space)
        _check_single_valued(engine)
        self.engine = engine
        self.space = space
        self.sensor = _name(sensor, 'sensor')
        self.unit = None if unit is None else _name(unit, 'unit', 32)
        self.place = None if place is None else _name(place, 'place')
        self.deadband = None if deadband is None else _number(deadband, 'deadband')
        if self.deadband is not None and self.deadband <= 0:
            raise InvalidInput('deadband must be above 0')
        self.low = None if low is None else _number(low, 'low')
        self.high = None if high is None else _number(high, 'high')
        if self.low is not None and self.high is not None and self.low >= self.high:
            raise InvalidInput('low must be below high')
        self.heartbeat_seconds = _number(heartbeat_seconds, 'heartbeat_seconds')
        if self.heartbeat_seconds <= 0:
            raise InvalidInput('heartbeat_seconds must be a positive number')
        self.forget_after = forget_after
        self._numeric: bool | None = None
        self._last_value: float | str | None = None
        self._last_band: Band | None = None
        self._last_seen: datetime | None = None
        self._last_kept: datetime | None = None
        #: Readings offered, kept and dropped by this object since it was made.
        self.seen = 0
        self.kept = 0
        self.dropped = 0
        #: Of the kept readings, those dated before the reading that preceded them.
        self.late = 0

    def _band(self, value: float) -> Band | None:
        if self.low is None and self.high is None:
            return None
        if self.high is not None and value > self.high:
            return 'above_high'
        if self.low is not None and value < self.low:
            return 'below_low'
        return 'in_range'

    def _decide(self, value: float | str, band: Band | None, at: datetime) -> tuple[Reason, float | None]:
        if self._last_kept is None or self._last_value is None:
            return 'first', None
        if isinstance(value, str):
            if value != self._last_value:
                return 'state_changed', None
            change = None
        else:
            # One stream is all numbers or all states, so the last kept value is a number too.
            change = value - cast(float, self._last_value)
            if band != self._last_band:
                return 'limit_crossed', change
            moved = change != 0 if self.deadband is None else abs(change) >= self.deadband
            if moved:
                return 'changed', change
        if (at - self._last_kept).total_seconds() >= self.heartbeat_seconds:
            return 'heartbeat', change
        return 'steady', change

    def _sentence(self, shown: str, reason: Reason, band: Band | None, change: float | None, stamp: str) -> str:
        unit = '' if self.unit is None else f' {self.unit}'
        where = '' if self.place is None else f' at {self.place}'
        text = f'{self.sensor} read {shown}{unit}{where} on {stamp}.'
        if reason == 'limit_crossed':
            if band == 'above_high':
                text += f' It rose above the high limit of {_shown(cast(float, self.high))}{unit}.'
            elif band == 'below_low':
                text += f' It fell below the low limit of {_shown(cast(float, self.low))}{unit}.'
            else:
                text += ' It returned within its limits.'
        elif reason == 'state_changed':
            text += f' It changed from {self._last_value}.'
        elif reason == 'changed' and change is not None:
            text += f' It {"rose" if change > 0 else "fell"} by {_shown(abs(change))}{unit} since the last kept reading.'
        elif reason == 'heartbeat':
            text += ' A routine reading: nothing had changed enough to record since the last kept one.'
        elif reason == 'late':
            text += ' It arrived out of order: a later reading had already been recorded.'
        return text

    async def observe(self, value: Value, *, observed_at: str) -> SensorRead:
        """Offer one reading: a number, or a state as text or true/false."""
        at = _instant(observed_at)
        if isinstance(value, bool):
            read: float | str = 'true' if value else 'false'
        elif isinstance(value, str):
            read = _name(value, 'a state reading', 64)
        else:
            read = _number(value, 'a reading')
        numeric = isinstance(read, float)
        if self._numeric is not None and numeric != self._numeric:
            raise InvalidInput('one sensor stream holds numbers or states, not both')
        band = self._band(read) if isinstance(read, float) else None
        late = self._last_seen is not None and at < self._last_seen
        reason, change = ('late', None) if late else self._decide(read, band, at)
        self._numeric = numeric
        self.seen += 1
        if not late:
            self._last_seen = at
        if reason == 'steady':
            self.dropped += 1
            return SensorRead(False, reason, band, change)
        stamp = format_rfc3339(at)
        shown = _shown(read) if isinstance(read, float) else read
        metadata = {'document_format': FORMAT, 'sensor': self.sensor, 'sensor_event': reason, 'sensor_value': shown}
        for key, item in (('sensor_unit', self.unit), ('sensor_place', self.place), ('sensor_band', band)):
            if item is not None:
                metadata[key] = item
        tags = [_tag('any', 'reading'), _tag('sensor', self.sensor)]
        if self.place is not None:
            tags.append(_tag('place', self.place))
        added = await self.engine.remember(
            self.space, self._sentence(shown, reason, band, change, stamp), kind='note',
            source=SOURCE_PREFIX + self.sensor, tags=tags, created_at=stamp, metadata=metadata,
            # One reading per sensor per instant: an exact retry is the same record, and a value the sensor
            # returns to later is a new one.
            dedup_key=f'sensor-v1:{self.sensor}:{stamp}', forget_after=self.forget_after)
        self.kept += 1
        if late:
            self.late += 1
            return SensorRead(True, reason, band, change, added)
        fact = None
        if read != self._last_value:
            written = shown if self.unit is None else f'{shown} {self.unit}'
            fact = await self.engine.assert_fact(self.space, self.sensor, PREDICATE, written, valid_from=stamp,
                                                 source_episode_id=added.episode_id, quote=written)
        self._last_value = read
        self._last_band = band
        self._last_kept = at
        return SensorRead(True, reason, band, change, added, fact)


async def sensor_state(engine: MemoryEngine, space: str, sensor: str, *, as_of: str | None = None) -> SensorState | None:
    """What ``sensor`` reads now, or read at ``as_of``; ``None`` when the ledger holds nothing for that time."""
    check_space(space)
    _check_single_valued(engine)
    name = _name(sensor, 'sensor')
    if as_of is not None:
        as_of = format_rfc3339(_instant(as_of, 'as_of'))
    held = [fact for fact in await engine.facts(space, as_of=as_of)
            if fact.subject == name and fact.predicate == PREDICATE]
    if not held:
        return None
    [fact] = held  # single-valued: one value holds at a time
    return SensorState(name, fact.object, fact.valid_from, fact.valid_until, fact.fact_id, fact.source_episode_id)


async def sensor_events(engine: MemoryEngine, space: str, *, sensor: str | None = None, place: str | None = None,
                        reason: Reason | None = None, since: str | None = None, until: str | None = None,
                        limit: int = EVENT_LIMIT) -> SensorEvents:
    """Kept readings, newest first, narrowed by sensor, place, reason and time (both ends included).

    This walks the space's episodes, as ``MemoryEngine.episodes`` does: it is a listing, not an index."""
    check_space(space)
    if type(limit) is not int or not 1 <= limit <= MAX_EVENT_LIMIT:
        raise InvalidInput(f'limit must be an integer from 1 to {MAX_EVENT_LIMIT}')
    if reason is not None and reason not in REASONS:
        raise InvalidInput(f'reason must be one of {", ".join(REASONS)}')
    where = {'document_format': FORMAT}
    for key, item, what in (('sensor', sensor, 'sensor'), ('sensor_place', place, 'place')):
        if item is not None:
            where[key] = _name(item, what)
    if reason is not None:
        where['sensor_event'] = reason
    start = None if since is None else _instant(since, 'since')
    end = None if until is None else _instant(until, 'until')
    if start is not None and end is not None and start > end:
        raise InvalidInput('since must not be after until')
    matched = []
    for episode in await engine.episodes(space, where):
        at = parse_rfc3339(episode.created_at)
        if (start is None or at >= start) and (end is None or at <= end):
            matched.append(episode)
    matched.reverse()  # ``episodes`` is oldest first
    items = tuple(
        SensorEvent(episode.episode_id, episode.metadata['sensor'], episode.created_at,
                    cast(Reason, episode.metadata['sensor_event']), episode.metadata['sensor_value'],
                    episode.metadata.get('sensor_unit'), episode.metadata.get('sensor_place'),
                    cast('Band | None', episode.metadata.get('sensor_band')), episode.content)
        for episode in matched[:limit])
    return SensorEvents(items, len(matched) > limit)
