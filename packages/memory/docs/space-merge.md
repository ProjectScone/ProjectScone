# Move a space with its retained evidence

`engine.merge_space("old", into="new", preview=True)` inspects the proposed
movement without writes. To execute, pass `confirm="old"`. The CLI uses:

```sh
scone-memory --space old merge-space --into new --dry-run
scone-memory --space old merge-space --into new --confirm old
```

Episodes, claims and linked attachments use the verified attachment archive path.
Unlinked attachment holds move explicitly too, retaining their bytes, filenames
and media types without inventing episode links. Shared bytes are verified once
per destination hold and linked to the corresponding remapped episodes. Existing
destination metadata or episode identity conflicts refuse rather than overwriting
that destination's evidence. Textless video sources use retained manifest
verification; copying them requires no decoder, OCR or language-model call.

The receipt's `moved` distinguishes execution from preview. `episodes` counts
accepted source episodes, including deduplicated ones; `tombstoned` counts source
episodes skipped because the destination intentionally forgot them. `facts` is
the number of source claims considered, including claims already in the target.
`attachments` and `attachment_bytes` count verified destination holds, including
existing ones; these are not newly allocated storage counts.
`unlinked_attachments` counts holds that were unlinked in the source.
`attachments_skipped` counts linked blobs used only by tombstoned source episodes.
A blob shared with an accepted episode still moves.

## Claims whose sources were forgotten

Forgetting a source leaves its claims standing in Scone. Merge keeps that existing
behavior: a source reference backed by the old space's tombstone is omitted,
with the claim, quote and other ledger fields still handled by normal import.
`forgotten_source_references` counts those omitted references across facts,
fact links and affirmations. No replacement provenance is fabricated. A dangling
reference with no matching episode or tombstone refuses the merge.

Destination tombstones still take precedence. Claims tied to source episodes
skipped by those target tombstones follow the existing archive remapping policy.
Fact supersession pointers are not reconstructed. Target exclusions are preserved.
This operation is not a lossless ledger/policy backup: jobs, event logs, tombstones
and runtime configuration are not transported. Keep full storage backups when
those histories must survive independently of the original space.

## Consistency, limits and failures

**Quiesce writers to both spaces for the entire operation.** Merge copies data,
verifies destination episode identities, links and exact attachment bytes,
re-reads the source to detect changed records/holds/link assignments, then checks
the destination evidence again before deleting the source. These are separate
observations across stores. Changes after a final check are not locked out;
this is not a distributed transaction or a concurrent atomic cutover.

The attachment archive's record, attachment-count and byte limits apply to the
combined linked and unlinked evidence. Preview reads and validates retained bytes;
it is not just a cheap count query. Buffering is bounded, but this is not a
streaming whole-repository migration interface.

Staging, ingestion or linking failures leave the original space open. A retry
reuses completed destination writes and fills missing links. An observed source
change or missing/changed destination evidence refuses source closure; the
partially populated destination is retained. Resolve concurrent changes before
retrying. Successful completion closes the source name permanently.

Once source deletion begins, failures follow the existing `delete_space` backend
semantics. The copy is already verified, but cleanup across the document, vector,
blob and event stores is not transactional, and retrying the merge may no longer
be possible if the source has been marked deleted. Inspect both stores and use
the backend's cleanup/recovery facilities; merge does not claim a durable
whole-space deletion journal.

## HTTP authorization

`POST /v1/spaces/{source}/merge` requires two explicit credentials, for both
preview and execution:

- `Authorization: Bearer <source-key>` must authorize the source with the full role.
- `X-Scone-Destination-Authorization: Bearer <destination-key>` must authorize
  `into` with the full role. Copying a ledger can carry review decisions, so a
  destination write-only key is insufficient.

A key for the source alone cannot inject records or attachments into another
space. The header is also described in the generated OpenAPI schema. Locally
invoked engine/CLI operations use the caller's direct storage authority.

The host rechecks both keys and their current roles/scopes at stage boundaries,
before source closure and before returning a receipt. Revocation detected after
a partial copy leaves the source open and the already copied target data in place.
These checks do not roll back writes already in flight, and revocation during
source deletion cannot undo completed cleanup. Inspect the destination if a
request loses authorization after copying began.
