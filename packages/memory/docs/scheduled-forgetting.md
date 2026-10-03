# Scheduled forgetting

A memory can carry the time it is to be forgotten. From that time recall does
not return it, whether or not anything has swept it yet, and a sweep forgets it
through the ordinary forget.

```python
await engine.remember("home", "the door code is 4411", forget_after="30d")
await engine.remember("home", "guest wifi: attic-5G", forget_after="2026-10-01")
report = await engine.forget_due("home")          # bounded; say when it cut
```

```sh
echo "the door code is 4411" | scone-memory --space home remember --forget-after 30d
scone-memory --space home forget-due --dry-run
scone-memory --space home forget-due --limit 100 --with-claims keep
```

```http
POST /v1/episodes            {"content": "the door code is 4411", "forget_after": "30d"}
POST /v1/episodes/batch      {"records": [{"content": "...", "forget_after": "2026-10-01T09:00:00Z"}]}
POST /v1/episodes/forget-due {"limit": 100, "dry_run": false, "with_claims": "keep"}
```

`/v1/capabilities` advertises `episodes.forget_after` when the document store
can forget and walk its sources.

## What a schedule may say

`forget_after` is one of:

- an RFC 3339 time, with `Z` or an offset: `2026-10-01T09:00:00+02:00`;
- a bare date, meaning midnight UTC: `2026-10-01`;
- a duration counted from the engine's clock, whole numbers of weeks, days,
  hours, minutes or seconds written together: `30d`, `1d12h`, `90m`, `2w`.
  Months and years are not units, because their length depends on where they
  start. A duration is at most 36,525 days; name a date for a time further off.

The episode keeps the resolved instant, in the engine's timestamp form
(`2026-10-01T07:00:00.000Z`), on its metadata under `forget_after`. That is why
every store carries it with no schema change, an export carries it, and the
write's clock does not have to be remembered to read it back. The key counts
against the sixteen metadata keys an episode may hold.

Refused at the write, with nothing stored:

- a time that is not after the engine's clock (`now` itself included);
- text that is none of the forms above (`tomorrow`, `5y`, `1.5d`, `0d`);
- `metadata={"forget_after": ...}` that disagrees with the `forget_after`
  argument. Metadata carrying the key on its own is held to the same rules.

A batch refuses whole on one bad schedule unless it is `partial`, in which case
that record comes back `failed` with the reason.

The write's receipt (`Added.forget_after`) says when the stored episode is to
be forgotten. A duplicate reports the schedule of the episode already there,
which the write did not change: sending the same words again with a schedule
does not schedule the existing memory, and `replace=True` with unchanged words
does not either. Either way the receipt shows the schedule that holds, so a
caller that asked for another can see it was not applied.

Writing the same words over a memory that is already past its time forgets the
overdue one first (the ordinary forget) and stores the new write afresh, because
otherwise the next sweep would take the new write with it. The receipt says so:
`Added.forgot_overdue` names the episode forgotten, when it was due, the reason
and the forget's receipt, and `remember` on the command line prints a line for
it. `replace=True` over an overdue keyed record forgets it as any replaced
record, reports `updated` and returns the forget's receipt as `replaced`.

## From ingestion

Every ingestion path that creates an episode takes the same `forget_after`, in
the same forms, and stores it on the episode exactly as `remember` does:

| path | Python | HTTP | command |
|---|---|---|---|
| an image and its context | `ingest_image(..., forget_after=)` | `POST /v1/images` | none (`remember --image` has `--forget-after`) |
| a document | `ingest_document`, `store_document` | `POST /v1/documents` | none |
| a page by URL | `ingest_url` | `POST /v1/documents/from-url` | `import-url --forget-after` |
| a directory (`sync`) | `sync_directory` | none, by design | `sync --forget-after` |
| a journaled directory | `DirectorySync.synchronize` | `POST /v1/sync-runs` | `sync-directory --forget-after` |

```python
await ingest_image(engine, "home", png, media_type="image/png", context=ctx, forget_after="30d")
await ingest_document(engine, "home", data, filename="lease.pdf", forget_after="2027-06-30")
await sync_directory(engine, "home", "~/scans", apply=True, forget_after="90d")
```

**Refused before anything is stored.** Each path resolves the schedule against
the engine's clock before it stores anything of its own, and refuses a past or
unreadable one with `remember`'s `InvalidInput` (HTTP 422): no image, original
or manifest attachment is left behind, a URL import fetches nothing, `POST
/v1/documents` parses nothing, a sync reads nothing and a journaled run writes no
journal, and `POST /v1/sync-runs` admits no run. A value that is not text at all
(`30`, `true`, a list) is refused the same way, with 422, on every route. Bytes
the caller uploaded through `POST /v1/attachments` beforehand stay, as they
would for any refusal.

**One instant per call.** The schedule is resolved once and that instant is
passed on (as `core.forget_after.Resolved`, which the layers below store without
checking it against their own later clock), so a duration names one time for
everything the call writes: every file a sync adds or updates carries the same
`forget_after`. The work is not refused part way when it outlasts a short
schedule -- a parse or OCR longer than `2m`, a walk longer than `1h`, a sync run
near its deadline: what it stores after the instant is stored with that instant,
is due at once (reads withhold it), and the next sweep forgets it with the
attachments it carries. The receipts name
it: `Added.forget_after` (inside `ImageIngested.added` and
`DocumentIngested.added`), `forget_after` in the URL import's record,
`SyncReceipt.forget_after`, `DirectorySyncResult.forget_after` and each
`SourceReceipt.forget_after`, and a sync run's `spec.forget_after` beside
`spec.forget_after_asked`, the text as sent.

**A re-import keeps the schedule the content holds.** Unchanged content is a
duplicate, and a duplicate keeps the schedule its episode holds, whatever the
re-import asked for, none included; its receipt reports the schedule that holds,
so a caller who asked for another can see it was not applied. This is
`remember`'s rule. It also means no store has to rewrite an episode's metadata,
and a sync on a timer does not push a schedule back for ever. A sync does not
write an unchanged file at all: `schedule_kept` (on `SyncReceipt`,
`DirectorySyncResult` and the run record) counts the unchanged files whose
memory holds another schedule than the run's, or holds one when the run asked
for none. To give content already held a new schedule, forget it and import it
again.

**Content already past its time is stored afresh.** An image, document, page
or video stored as sampled frames only whose episode is past its `forget_after`
(not yet swept) is stored again as a new episode: the overdue one is forgotten
first, `Added.forgot_overdue` names it, and the image or file is still linked to
the new episode -- a write puts back the attachment bytes that the overdue
memory's forget released. `sync` does the same for a file (reads leave the
overdue memory out, so the file is written again with the run's schedule).

A journaled `DirectorySync` holds the same window. A source whose revision's
time has come is read afresh from the file by the next run that sees it, changed
or not, with that run's schedule: an ordinary replacement, reported `updated`,
which closes the claims the file no longer states and forgets the old revision
if the sweep has not. A missing file whose revision was swept is `absent`, its
claims closed as for a deletion, and may come back. Only a forget *before* the
revision's time -- a forget by hand -- suppresses the path, as for any managed
source; the journal keeps each revision's schedule, and the tombstone's time
says which it was (a forget by hand after the time reads as the schedule's).

Recovery keeps the schedule. A pending revision keeps the schedule of the run
that prepared it, whichever run finishes it -- one asking for another schedule
or none included -- and is stored with it even if the time has passed since. A
pending revision swept before its replacement finished (stored, then taken at
its time, before the run that stored it could retire the last revision) is
abandoned: the claims both revisions were cited for are closed as for a
deletion, a last revision still held is forgotten, and the file is read afresh.
A run admitted over HTTP keeps its resolved instant, so every attempt writes
the same one; a retry of the same request (the same text, even a duration) is
the same run, even after its instant has passed; and a resume after the instant
has passed is refused with `sync_schedule_passed` (409).

**The sweep takes what ingestion attached.** `forget_due` forgets an ingested
episode through the ordinary forget: its chunks and text vectors, its image
vector when the engine has the image lane (keyed by the same chunk ids), and the
image, original and manifest attachments nothing else carries. The same image
held by another occurrence stays with that occurrence. Before the sweep, recall
withholds an overdue image from every lane, the image lane included.

Not covered yet: durable document jobs (`POST /v1/document-jobs`) and the
Python client's `DirectorySyncRuns.start` take no schedule.

A frames-only video interrupted between its store and the clearing of its
unfinished mark is finished by recovery when the engine next opens, whether or
not its time has come since; the sweep then forgets it.

## Before the sweep: the read

A memory past its `forget_after` reads as forgotten:

- `recall` leaves its passages out *before* the limit, so a withheld passage
  gives its place to the next one. The result's `past_forget_after` (omitted
  when nothing was withheld) says how many passages were withheld, which
  episodes and the time they were judged at. Without a reranker only the
  candidates needed to fill the answer are read, which are the episodes the
  answer reads anyway; with one, every candidate is judged. The lanes'
  candidate windows are not widened for it: a space whose windows are filled by
  overdue passages can answer short until it is swept.
- `engine.episode` and `engine.episode_by_key` (`GET /v1/episodes/{id}`,
  `GET /v1/episodes/by-key`) raise `Gone` (HTTP 410) naming the scheduled time.
  Summary expansion, which reads sources through `engine.episode`, therefore
  refuses an overdue source as `source_gone`.
- `engine.episodes(where)` (session turns) leaves it out before its limit.
- `engine.overview` (the recent evidence a routed conversation turn hands the
  model) leaves it out and counts it in `past_forget_after`.
- `engine.source_page` (`GET /v1/sources`) passes over it and still fills the
  page, walking on as a page with `conditions` does; `past_forget_after` counts
  what was passed over (the HTTP key appears only when some were).
- `build_chunk_questions` over the whole space does not show its chunks to the
  question model, and counts it in `episodes_past_forget_after`. Named
  `episode_ids` go through `engine.episode`, which refuses an overdue one.

`impact`, `forget`, `forget_status` and `forget_matching` still reach it: they
are how it goes.

Not yet filtered: `scopes`, `profile`'s recent excerpts, and `export`. They show
an overdue memory, with its `forget_after`, until a sweep forgets it.

A stored `forget_after` that cannot be read as a time (a key written before
this feature, say) is never treated as due: it is not withheld, not forgotten,
and the sweep names it under `unreadable`.

## The sweep

`engine.forget_due(space, now=None, *, limit=100, dry_run=False,
with_claims="keep", before=None)` forgets, through `engine.forget`, the
episodes whose `forget_after` is at or before `now`, most overdue first. Each
forget is the ordinary one -- retirement record, retry, tombstone, `forget`
event and receipt -- so chunks, vectors, attachments nothing else carries and,
with `with_claims="exclude"`, the claims only that source supported go exactly
as a manual forget takes them. Claims stand by default, as they do for
`forget`. A document's stored summaries are not forgotten with it; that is
`forget`'s behaviour today, and a sweep does not add to it.

`now` defaults to the engine's clock. An earlier `now` sweeps less; a later one
is refused, because a sweep forgets what is due, not what will be.

The report (`ForgetDueReport`):

| field | meaning |
|---|---|
| `items` | each episode the pass took: `episode_id`, `forget_after`, `reason` ("forget_after T had passed at now"), `outcome` (`forgotten`, `would_forget` in a dry run, `skipped` when it was already gone when its forget ran) and the forget's `receipt` |
| `forgotten` | the ids forgotten |
| `due` | episodes found due in what the walk read |
| `limited` | more were due than `limit` (1 to 1000) lets one pass take |
| `remaining` | due episodes this pass did not take |
| `scanned`, `scan_complete` | the walk reads at most 10,000 episodes, newest id first; `false` means it stopped before the oldest, so `due` is not the whole space |
| `resume_before` | pass as `before` to walk on from where the walk stopped |
| `unreadable` | episodes whose stored `forget_after` is not a time |

A pass that took anything (forgotten, or skipped because it was already gone)
appends a `forget_due` event with the counts, beside the `forget` event each
forget writes. A pass that took nothing, and a dry run, write no event.

### On a timer

When `scone-memory serve` runs a consolidation worker (a chat model, or a
`SCONE_RETAIN` policy), every pass first sweeps each configured space, at most
`SCONE_DISTILL_BATCH` episodes per pass. The pass report (and the
`consolidation.finished` log line) says:

| field | meaning |
|---|---|
| `forgotten_due` | episodes this pass forgot |
| `forget_due_limited` | more were due in the window walked than the pass takes |
| `forget_due_scan_cut` | the walk stopped at its 10,000-episode bound before the oldest |
| `forget_due_skipped` | why the pass did not sweep at all: the store cannot walk its sources |

The worker carries the walk across passes, per space and in memory. A window
whose due episodes were all taken hands the next pass the place its walk
stopped; a walk that reached the oldest episode starts the next from the
newest; a window with more due than one pass takes is walked again until it is
drained. So every episode of a large space is reached in turn, and one that
comes due behind the window being walked waits for the walk to come round
(recall withholds it meanwhile). A restarted worker starts from the newest.

A sweep that fails is the pass's `error` (the exception class only, for an
error that is not Scone's own), but it is recorded after retention and
derivation have run: a failed sweep stops neither. A store without source
inventory is not swept, says so in `forget_due_skipped`, and still expires.

Each pass walks up to 10,000 episodes' metadata even when nothing in the space
is scheduled; there is no index of schedules to ask instead. Without a worker
nothing sweeps on its own: call the route or the CLI from a scheduler. Recall
withholds overdue memories either way.

## Moving a space

An export carries the schedule inside each episode's metadata, and an import
keeps it. An archived episode whose time has already come is not restored:
`ImportSummary.past_forget_after` counts those (the field appears only when
non-zero), alongside `tombstoned`. A space merge leaves such episodes behind
too and counts them in its receipt's `past_forget_after`; they go when the
source space is deleted. If one comes due between a merge's selection and its
import, the merge refuses and the source stays open.
