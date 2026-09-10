# Retained image attachments and S3 storage

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

## Preserve an original image from the CLI

```sh
scone-memory remember source-note.txt --image screenshot.png \
  --space my-project --source manual-import \
  --meta session_id=my-session --json
```

The note is searchable text; the image is an original attachment, not OCR or a
model-generated description. The CLI reads only the explicitly selected file,
never paths mentioned inside the note or an agent transcript. One image can be
linked per invocation; `--jsonl` and `--image` cannot be combined. Files must be
nonempty regular files up to 25 MB. PNG, JPEG, GIF and WebP are identified by
their byte signatures rather than filename extensions. Signature recognition
does not validate decoding or bound decoded image dimensions.

The command stores exact bytes through the configured blob store, links them to
the episode and reads back the link before reporting success. With `--json`, the
receipt additionally contains the selected attachment's digest, type, byte count
and basename; text-only receipt shape is unchanged. Identical note text may reuse
an episode. Repeated identical images reuse their content-addressed attachment.
Source and session metadata follow the existing episode deduplication rules;
reusing a note does not create a new session record or replace its metadata.

Inspect the retained originals using the episode ID returned by `remember`:

```sh
scone-memory attachments 42 --space my-project
scone-memory attachments 42 --space my-project --json
```

This lists the episode's linked attachment IDs, media types, byte counts and
filenames. JSON returns one object with `space`, `episode_id` and `attachments`
(an empty array for an existing text-only episode). Missing, forgotten or
other-space episodes fail with exit status 2, not an empty success. This is a
metadata read: it does not download, decode, or verify the current blob bytes,
and does not describe image contents. Use the attachment ID with the native
`engine.attachment(space, id)` API or an authorized `GET /v1/attachments/{id}`
request when you want the original bytes.

This CLI opens the configured native stores, **not** the browser's HTTP server.
To inspect the result in the webapp, both must use the same authorized memory
space and document/blob configuration. SQLite defaults keep originals in an
attachment directory beside the database; `SCONE_BLOB_DIR` selects another
location. In-memory blob configuration does not persist originals across runs.

Image storage and episode linking are separate operations. An error after a
write can leave bytes or an episode behind; the command reports an unconfirmed
save and does not retry. Inspect the store before repeating it. There is no
automatic host image capture, missing-image reconstruction or Rust CLI parity
claim. Export/import still carries references, not a portable copy of image bytes.

## Optional S3 attachment storage

Install `scone-memory[aws]` and select pre-provisioned resources explicitly:

```sh
SCONE_BLOBS=s3
SCONE_S3_BUCKET=your-attachment-bucket
SCONE_DYNAMODB_BLOB_TABLE=your-blob-metadata-table
SCONE_AWS_REGION=us-east-1
SCONE_S3_PREFIX=attachments/
```

S3 stores attachment bytes; DynamoDB stores their space ownership, episode
references, and cleanup journals. This is an attachment backend, not a DynamoDB
replacement for the document ledger, conversation journal, or retrieval index.
Configure those stores separately. `SCONE_BLOBS=auto` preserves existing defaults;
`file` requires `SCONE_BLOB_DIR`, and `memory` explicitly selects ephemeral bytes.
Contradictory storage settings fail at startup.

The adapter uses the AWS SDK credential chain; prefer workload IAM roles.
Constructing it does not resolve credentials or contact AWS. Supply a private,
versioned S3 bucket dedicated to attachments with default SSE-KMS encryption and a DynamoDB table with
string partition key `pk` and string sort key `sk`. Configure table encryption,
PITR, and least-privilege access to that table, bucket prefix, and KMS key.
The role also needs `s3:ListBucket` on the dedicated bucket so S3 HEAD can report
missing objects as 404 during interrupted-upload recovery; 403 is treated as an
error, never proof that bytes are absent.
Deployment container instructions are in [`deploy/aws`](../../../deploy/aws/README.md).
Reusable Terraform configuration lives in the repository-root `terraform/`
directory. Supply deployment values through the environment; private `.env`
files, populated variable files, plans, and state stay ignored.

Each S3 upload has an immutable generation key. DynamoDB transactions publish
ownership and persist deletion intents before S3 cleanup. Cleanup targets the
recorded version, so a delayed delete cannot remove a newly published generation.
S3 and DynamoDB do not provide a shared atomic transaction: failed operations
raise and retain recovery state. Operators can call
`await engine.blobs.recover_uploads(space, after=cursor, limit=100)` on this
adapter to fence unpublished uploads and retry cleanup. Each call returns a
`RecoveryPage` with `visited` and `next_cursor`; follow the cursor until it is
`None` to complete a sweep. Repeat sweeps after paused writers finish; an attempt
is not proof that no late write can arrive. Pending upload intents are retained
for that purpose. No automatic background recovery is configured.

The original adapter limited each space to 1,024 held attachments. The
[indexed catalog](s3-catalog.md) removes that per-space cap with versioned
migration and paged release journals. Limits remain: 25 MiB per attachment,
1,024 episode references per hold, and 300 KiB per metadata item. Over-budget
operations fail; cap removal is not proof of unbounded throughput or capacity.
Episode lookups use strongly consistent indexed queries. SDK calls run off the
event loop with bounded waits; cancellation cannot interrupt an already running
SDK request. Large list/preview operations still need explicit capacity testing.

Run emulator tests with `pip install -e '.[aws-test]'` followed by
`pytest tests/backends/test_aws_blobs.py tests/runtime/test_aws_blob_config.py`. Tests use synthetic
resources and credentials through Moto; they do not provision AWS resources or
measure AWS throughput.
