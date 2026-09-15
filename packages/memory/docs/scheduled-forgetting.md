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
