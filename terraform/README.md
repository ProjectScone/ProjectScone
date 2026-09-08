# Scone AWS foundation

Terraform source and mock tests are versioned. Downloaded providers and the
generated provider lock remain ignored; the reviewed AWS provider version is
pinned in `versions.tf`. Operator
`.env` files, populated tfvars, state, plans and `.terraform/` are ignored and must
remain private. Image and architecture documentation live in `../deploy/aws/`.

The module defines attachment storage and an image registry. It does not deploy
compute, expose a public endpoint, configure a model, or replace the document or
vector backends. Nothing has been provisioned by creating or validating it.

Resources: an S3 bucket with public access blocked, bucket-owner enforcement,
TLS-only policy, versioning and KMS encryption; a DynamoDB table with string
`pk`/`sk`, on-demand billing, KMS, point-in-time recovery and deletion protection;
an immutable-tag ECR repository; an ECS application task role scoped to this
bucket's attachment prefix, the metadata table and the storage key via S3/DDB.
There is no account lookup data source. `account_id` also restricts the provider's
target account. No IAM user, credentials, build principal, or execution role is
created. The ECR push policy is only an output for an operator-controlled identity.

Supply deployment identifiers in an explicitly selected owned mode-0600 `.env`
file. `scripts/terraform_env.py` reuses the strict local environment parser:
single-line assignments and quoting are supported, without shell execution or
expansion. Duplicate assignments and symlink files are rejected. It maps only these inputs,
never API keys or AWS credentials, to Terraform variables:

| Operator environment | Terraform variable |
| --- | --- |
| `SCONE_AWS_ACCOUNT_ID` | `account_id` |
| `SCONE_AWS_REGION` | `aws_region` |
| `SCONE_AWS_NAME_PREFIX` | `name_prefix` |
| `SCONE_S3_BUCKET` | `bucket_name` |
| `SCONE_DYNAMODB_BLOB_TABLE` | `blob_table_name` |
| `SCONE_S3_PREFIX` | `s3_prefix` |
| `SCONE_AWS_TAGS` (optional JSON string map, default `{}`) | `tags` |

All six identifiers are required by the wrapper. Use the same bucket, table,
region and prefix in the application environment, with `SCONE_BLOBS=s3`.
Identifiers and tags are not secret fields: they appear in ordinary Terraform
plans/state and AWS resource metadata. Never put credentials or private content
in them. Authentication remains the operator's normal AWS provider credential
chain for an explicitly requested cloud action; no AWS credentials are Terraform
variables or outputs.

From the repository root, validate an operator file without starting any process:

```sh
python3 scripts/terraform_env.py --env-file /path/to/operator.env --check
python3 scripts/terraform_env.py --env-file /path/to/operator.env init
python3 scripts/terraform_env.py --env-file /path/to/operator.env fmt
python3 scripts/terraform_env.py --env-file /path/to/operator.env validate
python3 scripts/terraform_env.py --env-file /path/to/operator.env test
```

`init` downloads providers with `-backend=false -input=false`. The validation
commands remove ambient credentials, disable metadata/config credential lookup,
and clear ambient Terraform argument/variable overrides. Tests use
`mock_provider "aws"`; their mock apply creates no AWS resources. Checks print
no dotenv values. The wrapper refuses automatic `terraform.tfvars`,
`terraform.tfvars.json` and `*.auto.tfvars[.json]` in the module because those
would override environment inputs. It writes no tfvars or plan file.

Only an explicit final `plan` or `apply` command invokes the AWS provider against
the selected account. Neither is part of validation; `apply` retains Terraform's
interactive confirmation and accepts no auto-approve flags through this wrapper.
Choose an encrypted, access-controlled state backend with locking before an
authorized deployment. The operator-specific backend configuration is unset.
Deployment identifiers come from `.env`; fixed schema/security settings stay in
reviewed source. No deployment has been performed by adding or validating it.

The `attachment_environment` output contains the adapter's bucket, table, region
and prefix settings. It is not a complete server environment. Do not set
`SCONE_DOCUMENTS=dynamodb`: the full document adapter is not implemented. The
container needs a runtime API key injected as a secret, document/vector/event
backends and their persistent storage, plus an external model configuration if
conversation generation is wanted. The hash embedder default is a smoke/demo
setting, not a semantic embedding model.

Versioning is intentionally enabled. The blob adapter must delete specific S3
versions after the final space releases a blob; plain DeleteObject creates a
delete marker and retains old bytes. No automatic noncurrent-version expiry is
configured because it could destroy a version still referenced by metadata.
PITR and independent backups require an explicit retention/erasure policy too.
The role has no ListObjectVersions or table Scan permission. It does have
ListBucket on this dedicated attachment bucket: S3 HEAD returns 404 for missing
objects only with this permission, otherwise 403 is indistinguishable from
denied access. A prefix condition does not establish this permission for HEAD.
Do not share this bucket with unrelated raw/artifact data. Upload/delete recovery
uses keyed intent records, not a bucket enumeration. See
[S3 HeadObject permissions](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html).

This is a commercial AWS-partition foundation. Network controls, TLS ingress,
ECS task/service sizing, task execution role, secrets, durable SQL/vector/journal
storage, backups/restore exercises and monitoring remain operator deployment
work. A role ARN does not by itself make ephemeral container SQLite safe for
production. IAM access is service-level; tenant/space authorization remains in
the application and attachment metadata transactions.

## Intended data-engine deployment after the attachment slice

The broader target is S3 for raw input and extracted artifacts, DynamoDB for
metadata, job state and conversation state, SQS for retryable ingestion work,
and ECR images running API/worker tasks on Fargate. A built static frontend can
live in a separate private S3 bucket behind CloudFront with `/v1` requests routed
to the API origin. The current combined image remains useful for local smoke
and a single-origin API/UI deployment. A static deployment needs deep-link
fallback and same-origin routing; it must never embed server keys in static HTML.
These resources and the full DynamoDB application ports are not yet implemented
by this foundation.

The recommended target keeps API and worker services on Fargate and places only
Qdrant on ECS EC2 capacity in the same cluster, as separate services/tasks.
Qdrant requires a POSIX-compatible block filesystem while running; S3 or NFS/EFS
is not a live database-volume substitute. Its encrypted EBS volume must have a
lifecycle independent of the task and instance, with delete-on-instance-termination
disabled, same-AZ placement/reattachment recovery, single-writer fencing and
readiness only after a valid mount. A recreated task must never format or start
against an unintended empty data directory. Replication, snapshots, restore
tests and explicit recovery objectives remain required; a single persistent
volume alone is not high availability.

An all-Fargate Qdrant alternative remains future work: Fargate task storage and
ECS service-managed EBS do not preserve the live database across service task
replacement. That alternative requires immutable S3 snapshots plus a durable
DynamoDB/S3 indexing outbox, consistent snapshot watermarks, idempotent replay
including deletions, and restore-before-ready behavior. These recovery
controllers and either compute topology are not implemented in the attachment
adapter or this module. See
[Qdrant storage requirements](https://qdrant.tech/documentation/installation/)
and [ECS EBS configuration](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/configure-ebs-volume.html).

S3 raw/artifact storage needs its own provenance, authorization, deletion and
version-retention contract before widening the attachment prefix or role.
SQS jobs must be idempotent and acknowledge only after durable state publication.
Fargate API tasks become replaceable only after SQLite documents and local
conversation journals are replaced by the shared adapters. This architecture
does not require an AWS-hosted model: the current self-hosted model runtime stays
an independently configured service.
