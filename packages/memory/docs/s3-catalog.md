# S3 attachment catalog and recovery

`S3BlobStore` stores immutable attachment generations in S3 and ownership metadata in an existing DynamoDB table with `pk` and `sk` string keys. No additional table or secondary index is required. The adapter does not provision AWS resources. The validation below uses Moto locally, not AWS.

## Indexed episode lookup

Each episode has a DynamoDB partition `E#<space>#<episode_id>`. Each attachment link occupies one row keyed by its digest and contains its attachment metadata, immutable generation, and original link sequence. Link creation changes the episode row, authoritative hold, and space control in one transaction. Unlink and release remove the episode rows in the transactions that change their holds.

`for_episode()` and `released_by()` query the episode partition with strongly consistent reads. They read only the matching authoritative holds and check the control before and after collection, retrying when a concurrent space mutation invalidates the read. Generation mismatches or inconsistent catalog ownership raise errors. Results preserve link order across restarts. Metadata lookup does not fetch S3 bodies; link creation still verifies the version-specific checksum, with the existing download-and-hash fallback when checksum HEAD is unavailable.

## Upgrade and migration

Upgrade all writers for a catalog together. Concurrent mixed-version writers are **not supported**: old binaries may reject the added control fields or migration mode. Keep the prior build stopped once migration begins; there is no automatic downgrade migration.

Existing controls without a catalog version enter a resumable `migrate` mode. This fences ordinary writes. New-build workers encountering that mode help finish migration using the durable hold cursor and within-hold link offset. Backfill reads at most 100 holds per query and writes at most 90 derived rows plus one control per transaction. Duplicate retries are safe because each backfill transaction checks the original control. The catalog is marked ready only after all hold pages have been covered. Original link sequences are preserved.

A timeout or lost response leaves the checkpoint in DynamoDB. Retry the same operation to continue. Existing unfinished release journals retain their original recovery path; finish the recorded release before migration. Compare-and-swap checks use the exact stored JSON, including legacy serialization, rather than a reserialized model with new default fields.

## Space capacity and bounded release journals

The fixed 1,024-holds-per-space limit has been removed. The **separate limit of 1,024 episode references per individual hold remains**, as do the 25 MiB attachment and 300 KiB metadata-item limits.

New releases store scalar progress in the space control and one result row per removed hold in `J#<space>#<operation>`. They do not append every attachment ID to one control item. Hold/index mutations use at most 90 data-row changes plus one control per transaction. A hold with many episode references is removed over several transactions; its authoritative remaining links are the resume point. Concurrent changes to roots shared across spaces are checked transactionally.

Last-holder removal creates the existing deletion intent before S3 cleanup. Cleanup verifies the immutable generation is unpublished and deletes that exact generation/version. Intent acknowledgements and the cleanup cursor advance in bounded transactions only after deletion attempts succeed. Failures or cancellation leave resumable metadata. Cancellation cannot interrupt an in-flight SDK call; the worker remains occupied until it returns.

The latest completed operation retains its result rows. Helpers of that exact operation can return the same completed receipt. Starting the next release first changes control to a new operation and records `prior_operation`; only then are old receipt rows pruned in bounded batches. Interrupted pruning resumes from this field. A reader whose old receipt was retired raises an explicit error instead of returning incomplete/empty results. This is not a permanent history or an exactly-once API: a later fresh release may retire the previous receipt, and callers do not supply a durable idempotency key.

## Request-count verification

The following local Moto tests keep **one attachment in the target episode** while varying total holds in the space. Fixture creation is excluded. Migration and the subsequent lookup are measured separately; these are API request counts, not latency or production throughput measurements.

| Total holds | Migration Query / GetItem / transactions | Ready lookup Query / GetItem / transactions |
| ---: | ---: | ---: |
| 1 | 2 / 4 / 3 | 1 / 4 / 0 |
| 101 | 3 / 5 / 4 | 1 / 4 / 0 |
| 1,030 | 12 / 14 / 13 | 1 / 4 / 0 |

The previous lookup queried all hold pages: 1 and 101 holds required 1 and 2 queries; 1,024 holds required 11 queries. A 1,030-hold space was unsupported by the old adapter. The ready indexed lookup uses one query in all three cases, with additional point reads for control/hold consistency. Episodes with more than one page of attachments still require multiple episode queries and corresponding hold reads.

Tests also cover migration response loss, concurrent migration/linking, repeated link/unlink, scope and generation checks, malformed cursors, legacy release recovery, cancellation during cleanup, lost prune responses, cooperative completion, concurrent pruning, a 1,024-reference hold, and full release of 1,030 independent holds across deterministic stop checkpoints and bounded retry attempts. An earlier single-attempt version took about 179 seconds in Moto on the development host. That emulator duration is not an AWS performance result; retryable deadlines remain relevant.

## Remaining scale limits

- All mutations in one space contend on its control row. Requests use one bounded worker per store instance, finite deadlines, and bounded optimistic conflict retries. The catalog is not a claim of high write throughput for one heavily contended space.
- `held()`, `linked()`, preview, and release receipts retain their list/set return contracts. They materialize all matching results in process; bounded DynamoDB requests do not make their complete results constant-memory.
- Space preview reads all holds and their roots. The engine's `delete_space()` performs this preview before starting the resumable release. A sufficiently large catalog can repeatedly exceed its deadline in this preflight before deletion begins. This change does not bypass preview or remove those timeouts.
- Migration and release can require retries when their request deadline expires. Only migration/release progress is checkpointed; whole-space read/preview calls restart their read on retry.
- No ten-million-attachment, AWS service throughput, or end-to-end engine erasure claim is established by these contract tests. AWS IAM, table capacity, service throttling, and real concurrency still require deployment-specific validation.
