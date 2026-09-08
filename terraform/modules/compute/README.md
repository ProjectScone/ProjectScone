# Optional ECS compute module

This child module defines a private Fargate API service behind an HTTPS ALB and a separate single-writer Qdrant service on one ECS EC2 host in the same cluster. It is **not wired into the root deployment**. Validation uses mocked AWS metadata; no AWS resources have been provisioned or inspected for this module.

The default service counts are zero. Applying the module would still create the ALB, EC2 host, attachment, roles, logs and discovery resources. This is a bounded compute foundation, not a complete AWS application deployment or a high-availability Qdrant cluster.

## Required runtime work

Scone currently has an S3 attachment adapter with DynamoDB blob metadata. It does **not** have a DynamoDB `DocumentStore`, DynamoDB `EventLog`, distributed conversation journal, durable ingestion queue/outbox or indexing replay controller. The existing conversation journal and model-connection configuration use local files. This module rejects those local configuration paths and defaults the document/event adapters to an invalid sentinel so accidentally starting a task cannot silently use in-memory storage.

To enable the API, provide and test supported remote document/event backends, `SCONE_BLOBS=s3` with its bucket/table/region settings, an API authentication secret reference, enabled Qdrant, and `runtime_configuration_verified=true`. That flag records an operator assertion; Terraform cannot verify application consistency or backend reachability. A DynamoDB-only deployment and durable native conversations remain blocked pending those adapters. Mongo, PostgreSQL or Elasticsearch require the appropriate optional dependencies in the reviewed API image; the current AWS image installs AWS and Qdrant extras, not every remote database driver.

Configure the existing self-hosted embedding/chat/model endpoints explicitly and provide their network access. A production readiness probe that exercises required backends is still needed: the existing `/healthz` endpoint is process liveness. There is no worker/SQS service, automatic backup controller, frontend CloudFront distribution, DNS alias, remote database, model server or VPC/NAT creation here.

## Inputs and future root wiring

All deployment identifiers are inputs: account, region, name prefix, existing VPC/subnets, client/egress CIDRs, certificate, image digests, task role, secret references, AMI, instance type, external volume and filesystem UUID. Only bounded sizing and behavioral defaults are embedded. API CPU/memory default to 1024 units/2048 MiB; Qdrant defaults to 1024/2048. Fargate pairs are validated for the supported 0.25–4 vCPU range. Qdrant task sizing must leave host capacity for the OS and ECS agent; EC2 instance capacity and image architecture are operator-verified requirements.

The child inherits its AWS provider from a future caller. Both images must be private ECR digest references in the supplied account/region. The API task uses the existing attachment foundation's application role; it does not receive the execution role or host role's permissions. Execution permissions cover only the two image repositories, module log groups and supplied Secrets Manager/KMS ARNs (ECR token issuance itself requires `Resource="*"`). Secret contents are never read by Terraform. Environment values enter state: put credentials, authenticated connection URLs and keys in `api_secret_arns`, not `api_environment`.

Future caller shape (illustrative, deliberately not a root configuration):

```hcl
module "compute" {
  source = "./modules/compute"

  account_id    = var.account_id
  aws_region    = var.aws_region
  name_prefix   = var.name_prefix
  vpc_id        = var.compute_vpc_id
  api_subnet_ids = var.compute_api_subnet_ids
  alb_subnet_ids = var.compute_alb_subnet_ids
  ingress_cidrs  = var.compute_ingress_cidrs
  https_egress_cidrs = var.compute_https_egress_cidrs
  api_backend_egress = var.compute_backend_egress
  certificate_arn = var.compute_certificate_arn
  api_image       = var.compute_api_image
  api_architecture = var.compute_api_architecture
  api_cpu         = var.compute_api_cpu
  api_memory_mib  = var.compute_api_memory_mib
  api_task_role_arn = var.application_task_role_arn
  api_environment = var.compute_api_environment
  api_secret_arns = var.compute_api_secret_arns
  secret_kms_key_arns = var.compute_secret_kms_key_arns

  qdrant_image             = var.compute_qdrant_image
  qdrant_secret_arn        = var.compute_qdrant_secret_arn
  qdrant_subnet_id         = var.compute_qdrant_subnet_id
  qdrant_availability_zone = var.compute_qdrant_availability_zone
  qdrant_ami_id            = var.compute_qdrant_ami_id
  qdrant_instance_type     = var.compute_qdrant_instance_type
  qdrant_volume_id         = var.compute_qdrant_volume_id
  qdrant_filesystem_uuid   = var.compute_qdrant_filesystem_uuid
  qdrant_filesystem_type   = var.compute_qdrant_filesystem_type
  qdrant_cpu               = var.compute_qdrant_cpu
  qdrant_memory_mib        = var.compute_qdrant_memory_mib

  api_desired_count               = 0
  qdrant_enabled                  = false
  runtime_configuration_verified = false
}
```

The existing root dotenv wrapper does not yet map these compute inputs. A later root integration should add explicit `SCONE_AWS_COMPUTE_*` mappings through the existing strict parser, keeping private `.env` values, tfvars, state and plans ignored. Do not source an operator dotenv file as executable shell or put AWS credentials in Terraform variables. Use the AWS provider's external credential chain only for a separately authorized deployment.

Exact-ID subnet and volume metadata lookups run during a future real plan. They require read permission for EC2 subnet/volume descriptions and enforce VPC, at least two API/ALB AZs, volume encryption, disabled multi-attach and matching volume/host/subnet AZ. They do not prove route tables, certificate DNS coverage, image compatibility, secret contents or application readiness. The account/region inputs must match the inherited provider's deployment target.

Use existing public ALB subnets with internet routing and private API/host subnets with approved NAT or VPC endpoints. The supplied CIDRs explicitly control HTTPS egress; extra API backend TCP ports are separately bounded. Qdrant is reachable only from the API security group on port 6333. There is no SSH ingress. Configure a certificate-covered hostname as a DNS alias to `api_alb_dns_name`; the AWS ALB hostname itself does not match the supplied certificate. API-to-Qdrant HTTP stays inside the VPC; this module does not configure internal TLS.

## External disk preparation and boot gate

Supply an encrypted, **non-multi-attach**, whole-device XFS or ext4 EBS volume in the host's exact AZ. A separate operator-owned storage process must create, format, back up and restore it. Its filesystem root must already be owned by UID/GID `10001:10001`, with permissions allowing that user to write. This module never formats, repairs, changes ownership, snapshots or deletes the data volume. The Qdrant process has no AWS permissions and cannot perform its own S3 backup upload.

Use a reviewed Amazon Linux 2023 ECS-optimized Nitro AMI with Python 3, cloud-init, systemd, util-linux (`lsblk`, `findmnt`, `mount`) and Docker. Its ECS unit must retain the standard dependency on cloud-init completion. First-boot cloud-init masks ECS before writing its configuration. A persistent systemd dependency runs the disk gate before ECS registration on first boot and every reboot. The gate waits up to ten minutes for the separately attached volume and udev filesystem metadata, matches the Nitro EBS serial to the exact supplied volume ID, matches filesystem UUID/type, mounts with `nodev,nosuid`, rechecks the actual mounted device/UUID/type/read-write flags, and verifies filesystem ownership. Any failure blocks ECS. It never accepts a path merely because it exists or falls back to the root disk.

The task is placed only on the host advertising that verified volume ID. It bind-mounts `/var/lib/scone-qdrant` into `/qdrant/data`. Qdrant runs as UID/GID 10001 with a read-only root and writable data plus ephemeral `/tmp`. The current `qdrant/qdrant:v1.19.1` image passed a local network-disabled readiness smoke with these settings; its attempt to write the optional `.qdrant-initialized` marker emits a read-only warning. Operator-mirrored digests must be retested; the ECS health command requires Bash and `/readyz` support.

The EC2 root disk is disposable. The external data disk is attached after instance launch: AWS defaults such attachments to `DeleteOnTermination=false`. Verify that flag before enabling Qdrant and after any manual attachment changes. Both the Terraform host and attachment have `prevent_destroy=true`; the module has no ASG, launch-before-stop replacement, or forced detach. IAM for the host does not permit attaching/detaching/formatting disks or making snapshots. Backup and host replacement permissions belong to a separate operator role.

## Single-writer replacement procedure

1. Set API desired count and Qdrant desired count to zero. Confirm the Qdrant task has fully stopped, then take and verify an operator-owned backup/snapshot according to the retention policy. Service updates use minimum healthy 0 / maximum 100 so the previous writer stops before the replacement task starts. Automatic rollback is disabled to avoid an unreviewed storage-format downgrade.
2. Stop the old EC2 host and confirm it is stopped. Verify the external volume has `DeleteOnTermination=false`. Remove/review the specific Terraform lifecycle protection only as part of an explicit replacement change; a routine apply is intentionally unable to replace the host or detach the protected attachment.
3. Detach only from the stopped host. Never force-detach a live writer, enable multi-attach, or launch a second writer on a cloned volume using the same service identity. Retain the original volume and snapshot independently of the host lifecycle.
4. Replace the host in the same AZ and attach the original volume. Restore the lifecycle protection. The boot gate must validate the same volume ID and filesystem UUID before the ECS agent registers. If recovery instead requires a snapshot restore, deliberately review the new volume ID/UUID inputs. A mismatch must be investigated; do not format it to make startup succeed.
5. Enable the single Qdrant task and verify backend health and stored collections. Then restore API count and run scoped write/recall/erase checks before reopening client access.

This procedure entails downtime. It is not automated fencing against an administrator with EC2 root access. Same-AZ manual recovery and independent backups remain operator responsibilities. Fargate-only Qdrant would require a separate tested snapshot restore plus durable indexing outbox/replay protocol; ECS service-managed EBS is not a substitute for this external volume lifecycle, and EFS/NFS is not a supported substitute for Qdrant block storage.

## Offline checks

From the repository root, with Terraform 1.16.1 and pinned AWS provider 6.63.0:

```sh
terraform -chdir=terraform/modules/compute init -backend=false -input=false
terraform -chdir=terraform/modules/compute fmt -check -recursive
terraform -chdir=terraform/modules/compute validate
terraform -chdir=terraform/modules/compute test -no-color
python3 -m unittest discover -s terraform/modules/compute/tests -p 'test_*.py'
mypy --strict terraform/modules/compute/mount_volume.py terraform/modules/compute/tests/test_mount_volume.py
```

The Terraform test provider and all metadata are mocked. Bootstrap tests use synthetic subprocess results and do not invoke mount, block-device commands, network or AWS. These checks validate configuration and host-gate behavior; actual ECS registration/IAM, EBS attachment, DNS, ALB routing and cloud recovery remain deployment tests.

## Primary references

- [ECS task CPU/memory combinations](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-tasks-services.html).
- [ECS agent installation and cloud-init/systemd ordering](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-agent-install.html).
- [ECS IAM role boundaries](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/security-ecs-iam-role-overview.html).
- [ECS bind mounts and host source paths](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/bind-mounts.html).
- [EC2 volume preservation and attachment defaults](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/preserving-volumes-on-termination.html).
- [ECS-managed EBS lifecycle](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/configure-ebs-volume.html).
- [Qdrant storage requirements](https://qdrant.tech/documentation/installation/).
