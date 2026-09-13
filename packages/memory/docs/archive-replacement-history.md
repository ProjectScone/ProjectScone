# Preserve replacement history during archive transfer

Both `scone.archive/1` and `scone.archive/2` imports preserve explicit fact
`superseded_by` edges. Fact IDs belong to their source store: the importer first
resolves or inserts destination facts, then connects those destination IDs.
Reordered archive records and existing destination IDs do not change the chain.
The original `closed_reason` text is retained verbatim; any source-local numeric
ID mentioned in that prose is not rewritten or treated as a destination ID.

A nonzero `ImportSummary.supersessions` counts replacement edges restored by
that call. JSON receipts include this field when nonzero; zero is omitted so
ordinary legacy archive receipts keep their existing shape. Reimporting an
already complete chain restores zero edges. The import still advances the space
revision to settle any prior interrupted attempt; idempotence applies to the
facts and edges, not to that cache-invalidation counter.
This count describes repaired relationships, not newly inserted facts.

## Validation and destination policy

An archive that names replacement edges must provide unique positive 64-bit
identities for the relevant fact records, complete endpoints and an acyclic
chain. Missing endpoints, repeated IDs, boolean IDs, self-links and cycles
refuse before attachment staging or episode ingestion.

The destination inventory uses complete keyset pages, including short pages,
until an empty page reports exhaustion. It is bounded at 100,000 destination
facts; exceeding that bound or using a custom store without the paged-ledger
contract refuses rather than validating a truncated graph. Pagination is required
because ordinary backend listings may have result-window caps. Elasticsearch
refreshes fact visibility before this scan even when bulk refresh is disabled;
it leaves that configuration unchanged. Custom pagers must expose acknowledged
writes; an optional `prepare_ledger_read(space)` hook can settle deferred
visibility before scanning. Preparation errors propagate.

After episode IDs have been remapped, the importer plans fact identities before
inserting any facts. It refuses edges that collapse into a self-link or a cycle
through deduplication, conflicting successors, and ambiguous destination facts
with the same import identity. A destination fact already naming a different
replacement is a conflict; importing the archive does not overwrite it.
Incoming records without a replacement edge do not clear a destination edge.
Existing destination exclusions also remain in effect.

## Interruptions and consistency

Serialize import with destination writers. Planning and storing are separate
operations, not an atomic graph transaction. An episode or attachment may already
have copied before a destination-history conflict is detected. A later storage
failure can leave imported facts or some restored edges behind. Retry the same
archive to deduplicate those facts and fill missing edges.

The importer validates destination identities again before updating facts and
advances the space revision before the first edge write and after the write
attempt, including failures with lost acknowledgments. The final advance
invalidates readers that cached between the first advance and the write.
Every retry carrying replacement edges advances it again, even when the edges
are already present, so a process interruption that skipped finalization can be
settled. Repair storage availability and retry if a revision write fails. This
does not cancel concurrent reads or provide a snapshot across stores.

Replacement history is one part of provenance. This change does not make an
archive a lossless full backup: existing target tombstone and exclusion policies,
source-reference omissions, untransported jobs/events/configuration, and
cross-language profile limitations still apply. See
[attachment archives](attachment-archives.md) and [space movement](space-merge.md).
