# Complete source inventories for archive export

Archive export reads all episode and fact pages rather than using ordinary
recent-item or fact listings that may stop at a backend result window. Episode
pages are newest ID first, fact pages are checked newest first and emitted in
ascending ID order. Original timestamps remain unchanged. Short pages do not
signal completion: traversal continues until an empty page.

The exporter validates each page's shape, size, space, strict positive row IDs
and cursor order. It detaches and validates model payloads before emission, and
checks the total episode inventory against the initial source count. A missing,
repeated or foreign page refuses instead of producing a successful partial dump.

Links use `ArchiveLinkInventory.space_fact_links(space)`: one complete inventory
for the space, ascending by ID. Every built-in document store implements it.
This includes stored links whose endpoints are missing; walking links from
existing facts would silently omit those rows. Legacy archive/1 retains such a
row, while archive/2's reference validation explicitly refuses the dangling
archive. Neither path invents replacement endpoints. Link and affirmation
inventories validate source space, strict row identities and unique ascending
order before removing the space field from exported records. Conflicting
repeated IDs refuse; they are not deduplicated by keeping an arbitrary payload.

Elasticsearch's all-link and all-affirmation operations now walk bounded ID-range
queries across result windows. Its export preparation refreshes episode, fact,
link and affirmation visibility before counts and pages, even when bulk refresh
is disabled; the configured setting remains unchanged. Search responses marked
timed out, ended early or containing failed shards raise instead of being treated
as complete results. Bounded graph projection methods keep their own limits.

## Custom stores

Export requires callable `EpisodeInventory`, `LedgerPager` and
`ArchiveLinkInventory` methods. Missing capabilities refuse before the archive
header. Custom stores must return complete inventories with acknowledged writes
visible; `prepare_archive_read(space)` is an optional hook for deferred-visibility
backends. A store that supports affirmations must return every source affirmation
from `space_affirmations`, ordered by its identity. Preparation and read failures
propagate.

## Consistency and output handling

Quiesce writers to the source for the full export. Paging, counts and cross-store
checks are separate observations, not an atomic snapshot or distributed writer
lock. Count equality does not prove the source was unchanged; for example, a
concurrent edit need not change the number of episodes. In-flight readers and
writers are not canceled by export.

Archive/1 streams rows but buffers fact and relationship metadata for inventory
validation. It has no new whole-space row ceiling and is not a constant-memory
backup interface. Archive/2 retains its existing row, attachment and byte limits;
it refuses overflow rather than truncating evidence.

A later error can occur after archive/1 has emitted earlier rows. Discard that
partial output and retry after fixing the underlying problem. For files, write to
a temporary destination and publish it only when the export command succeeds.
There is no completion footer or full-file checksum in these profiles.

Complete source enumeration does not make a full backup. The existing profile
rules still govern omitted attachments, jobs, events, tombstones, runtime policy
and configuration. Cross-language profile compatibility and exact ledger-policy
restoration remain separate work. See [attachment transfer](attachment-archives.md)
and [replacement history](archive-replacement-history.md).

## Measured link-read cost

In a local in-memory export with 4,000 facts and 3,999 links, replacing per-fact
incident reads with the space inventory reduced link-read calls from 4,000 to
one. Export elapsed time in the same fixture was 1.474 seconds before and 0.033
seconds after, with all 3,999 links emitted. At 1,000 facts and 999 links, it was
0.095 seconds before and 0.008 seconds after. These are local measurements, not
network-backend throughput guarantees.
