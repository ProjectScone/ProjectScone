# AWS container and storage foundation

The image combines the Python API with the current built web console. Build it
from the repository root:

```sh
bash deploy/aws/build.sh scone-aws:local
python3 deploy/aws/smoke.py scone-aws:local
```

The build defaults to `linux/amd64`; set `SCONE_BUILD_PLATFORM=linux/arm64` for an
ARM deployment or local Apple Silicon smoke. Node and pnpm exist only in the web
build stage. Like CI, that stage uses pnpm 9.9.0 and installs directly from
`Webapp/pnpm-lock.yaml` with `--frozen-lockfile`; no lock or generated web
artifact is rewritten in the workspace. The runtime is Python
3.14, UID/GID 10001, and includes `api,agents,remote-embed,aws,qdrant` extras. No model weights
are downloaded or bundled. The Dockerfile-specific context allowlist excludes
Terraform files, state, credentials, local datasets and model caches.

The smoke runner uses an existing local image (`--pull=never`), generates a
temporary test key, disables container networking, drops Linux capabilities,
makes the image filesystem read-only and supplies disposable tmpfs storage. It
checks Python 3.14 and AWS/Qdrant client imports, health, the built UI,
authentication, absence of the key from public HTML,
and a synthetic SQLite write/recall. It removes its own container afterward.
It does not call AWS, start a model, push an image or exercise cloud storage.

The repository-root `terraform/` source and mock tests belong in version control.
Operator `.env` files, populated tfvars, state, plans and `.terraform/` stay ignored.
The module defines encrypted private S3 attachment storage,
a DynamoDB `pk`/`sk` attachment metadata table, an ECR repository and a scoped ECS
task role. It creates no running service. Terraform mock-provider validation
uses no AWS account. Real planning, applying and ECR publishing are separate
operator actions; none is performed by the build or smoke scripts. Use the
[safe environment wrapper](../../terraform/README.md) for infrastructure checks
and explicitly requested Terraform commands.

The attachment bucket is dedicated to this adapter. Its task role can list that
bucket so S3 HEAD can distinguish a missing object (404) from denied access (403)
during upload recovery; object read/write/delete access stays inside the selected
prefix. Do not reuse it for unrelated raw data without redesigning those policies.
[S3 HeadObject permissions](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html).

The image starts `scone-memory serve` on port 7437. Inject `SCONE_API_KEY` or
`SCONE_API_KEYS` at runtime through the deployment's secret mechanism; never bake
keys into the image or Terraform variables/state. `/healthz` is process liveness,
not a model or backend readiness guarantee. Terminate HTTPS at the operator's
ingress and restrict direct container access. The public console asks for an API
key; memory routes require it.

Defaults use SQLite at `/data/memory.db` for documents, vectors and events. Mount
durable storage owned by UID 10001 for a single-instance deployment, or select
existing remote backends and include their Python extras when building. Do not
run multiple tasks with independent ephemeral SQLite volumes and expect shared
memory. Existing settings such as `SCONE_EMBEDDER=remote`, `SCONE_EMBED_URL`,
`SCONE_EMBED_MODEL` and the conversation/model connection configuration continue
to select externally managed, self-hosted models. The default hash embedder is
useful for smoke tests; it provides no semantic embedding quality claim.

The AWS attachment adapter is a separate optional `BlobStore`, selected explicitly
with `SCONE_BLOBS=s3`. Its settings are
`SCONE_S3_BUCKET`, `SCONE_DYNAMODB_BLOB_TABLE`, `SCONE_AWS_REGION` and
`SCONE_S3_PREFIX` (default `attachments/`). The image includes its optional SDK
dependency. Those settings do not select a
DynamoDB `DocumentStore`, event log or vector index. Full port requirements and
emulator testing are mapped in [architecture.md](architecture.md).

For a later authorized ECR publication, use a unique immutable image tag and
record its resolved image digest and installed dependency manifest. The module
outputs an ECR URL and a narrowly scoped push policy for an explicitly
authorized build identity; it does not create credentials or log in. Base image
tags and Python dependency ranges currently resolve at build time, so rebuilds
are not byte-for-byte reproducible. Pin reviewed base image digests and a Python
dependency lock before promoting a production release.
