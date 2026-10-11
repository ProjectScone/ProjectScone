# Sensor streams

[Scheduled forgetting](scheduled-forgetting.md) · [Recall semantics](retrieval-and-storage.md) · [Storage adapters](storage-adapters.md)

A controller reports a temperature every second and a valve its state every cycle. Almost every reading repeats
the one before. A sensor stream accepts each reading and stores it only when it says something changed. What is
stored is an ordinary record plus a dated fact, so search, `since`, `as_of` and forgetting apply to sensor data
the same way they apply to documents.

```python
from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion import SensorStream, sensor_events, sensor_state

engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()

pump = SensorStream(engine, "plant", "pump-3-temperature", unit="°C", place="line-2",
                    deadband=0.5, high=90)
receipt = await pump.observe(92.4, observed_at="2026-10-08T12:00:05Z")
receipt.kept, receipt.reason, receipt.band     # True, "first", "above_high"

now = await sensor_state(engine, "plant", "pump-3-temperature")
then = await sensor_state(engine, "plant", "pump-3-temperature", as_of="2026-10-08T11:00:00Z")
crossings = await sensor_events(engine, "plant", reason="limit_crossed", since="2026-10-08T00:00:00Z")
too_hot = await sensor_events(engine, "plant", band="above_high")
```

A reading is a number, or a state given as text or true/false. One stream holds numbers or states, not both.
Nothing here calls a network service.

## What the gate keeps

| `reason` | Kept | When |
| --- | --- | --- |
| `first` | yes | This stream object has kept nothing yet. |
| `limit_crossed` | yes | A number moved into or out of the `low`/`high` limits. |
| `state_changed` | yes | A state differs from the last kept one. |
| `changed` | yes | A number moved by at least `deadband` from the last kept reading (by any amount when no deadband is set). |
| `heartbeat` | yes | Nothing changed, but `heartbeat_seconds` (300 by default) passed since the last kept reading, so a steady sensor still leaves a trace. |
| `late` | yes | The reading is dated before the one that preceded it. It is recorded, but it does not move the gate or the ledger. |
| `steady` | no | Nothing changed and the heartbeat is not due. |

`stream.seen`, `stream.kept`, `stream.dropped` and `stream.late` count the readings offered since the stream
object was made.

## What a kept reading becomes

- **A record.** One sentence saying what was read, where and when, dated at the instant it was read:
  `pump-3-temperature read 92.4 °C at line-2 on 2026-10-08T12:00:05.000Z. It was above the high limit of 90 °C.`
  It is tagged with the sensor and the place, and carries the value, unit, band and reason as metadata.
- **A fact.** `<sensor> reads <value>` in the ledger, valid from that instant. A new value closes the one
  before it, which is what lets `sensor_state(..., as_of=)` answer what the sensor read at a past time. A
  heartbeat that repeats the last value writes a record and no new fact.

`sensor_events` lists kept readings newest first, narrowed by `sensor`, `place`, `reason`, `band`, `since` and
`until`. It returns 50 by default and at most 500; `more` is true when older matches were left out.

`reason="limit_crossed"` finds the moments a sensor went past a limit or came back. A stream whose first reading
is already past a limit crossed nothing, so that reading has the reason `first`; `band="above_high"` or
`band="below_low"` finds it, along with every other kept reading taken past the limit.

## Measured on a real series

The machine temperature series from the Numenta Anomaly Benchmark: 22,695 readings at five-minute steps, with
four labelled anomaly windows. In-memory stores, a six-hour heartbeat, on a laptop.

| Deadband (°F) | Kept | Share of readings |
| ---: | ---: | ---: |
| 1 | 8,068 | 35.5% |
| 2 | 1,844 | 8.1% |
| 5 | 725 | 3.2% |
| 10 | 467 | 2.1% |

With a low limit of 50 °F the reading fell below the limit 29 times: 12, 4, 0 and 5 times inside the four
labelled windows, and 8 times outside them. A fixed limit found three of the four labelled periods and missed
the third. It is a threshold, not an anomaly detector.

The series contains 11 readings dated before the one that preceded them. They were recorded as `late`.

## Limits

- **Limits and deadband are yours to set.** They are engineering values of the process. The stream does not
  infer them from the data.
- **Late readings do not rewrite history.** A reading that arrives out of order is kept as a record marked
  `late` and counted. It does not change what `sensor_state` answers for that time.
- **The gate's memory lives in the process.** A new `SensorStream` keeps its first reading whatever an earlier
  one saw. After a restart, expect one extra kept reading per stream.
- **`reads` must hold one value at a time.** A stream refuses an engine configured with `reads` as a
  many-valued predicate, because the ledger could then no longer say what the sensor reads now.
- **`sensor_events` and `sensor_state` are listings, not indexes.** `sensor_events` walks the space's records
  and `sensor_state` reads the space's facts current at that time. Both grow with the space.
- **Readings can expire.** `forget_after` on the stream schedules each kept reading's forgetting, which bounds
  the disk a stream uses.
- **This is a record, not a control loop.** Nothing here is fast or certain enough to stop a machine.
