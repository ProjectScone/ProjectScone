# Adopt an existing Hetzner host

This independent Terraform root adopts one existing server and its already
assigned IPv4 Primary IP. It creates no additional hosting architecture and does
not invoke the AWS roots above it. It contains no firewall, volume, DNS, SSH-key,
provisioner or application deployment resources. Existing host firewall rules,
web proxy restrictions, application files, encryption keys and services remain
outside this configuration. An unused project firewall stays unused.

The first acceptance condition is an import-only plan with **zero remote
changes**, followed by a no-op plan after a separately reviewed state adoption.
No import, infrastructure apply or service restart is performed by validation.

## Inventory before any live plan

Use a token scoped to the verified existing project. A read-only token is enough
for discovery and planning; keep it solely in `HCLOUD_TOKEN`, supplied through
the operator's private credential mechanism. The provider has no project-ID
allowlist option. Independently verify the project and the exact numeric server
and Primary IP IDs; their identity and attachment are checked again in the plan.
Do not paste a token into commands, tfvars, source, output or a Terraform variable.

Record every field from `adoption.tfvars.example` in an owned mode-0600 private
tfvars file. The example is synthetic and must never be used against a live
account. Names, IDs, labels, backup status, disk size and protection settings must
match the observed resources exactly. The IP resource ID is a numeric identifier,
not the dotted IPv4 address. Check that this IP is assigned to the intended server,
that there are no attached cloud firewalls and that its current settings agree
with the server inventory. Do not guess absent fields or replace them with safer
defaults: changing a setting belongs in a later, explicitly reviewed plan.

The server image is deliberately omitted. An import must not recreate an existing
OS from a distribution image, even when the image name differs from the running
OS version. Creation-only `image`, `user_data` and `ssh_keys` are ignored for
adoption; there is no blanket `ignore_changes = all`. The server's `public_net`
block is omitted to avoid managing its existing IPv6 or competing over attachment.
The IPv4 resource specifies `assignee_id` and `assignee_type`; it must not also
specify `location` because the pinned provider accepts only one placement method.

## Offline validation

Use Terraform 1.16.1 or later compatible 1.x. The hcloud provider is pinned exactly
to 1.68.0. Downloaded providers and the generated lock are ignored, following this
repository's existing convention. Initialization downloads the provider; mock
tests do not contact the Cloud API or use credentials.

From the repository root:

```sh
env -u HCLOUD_TOKEN TF_CLI_CONFIG_FILE=/dev/null terraform -chdir=terraform/hetzner init -backend=false -input=false
terraform -chdir=terraform/hetzner fmt -check
terraform -chdir=terraform/hetzner validate
env -u HCLOUD_TOKEN TF_CLI_CONFIG_FILE=/dev/null terraform -chdir=terraform/hetzner test
python3 -m unittest discover -s terraform/hetzner/tests -p test_check_plan.py
python3 -m mypy --strict terraform/hetzner/check_plan.py
```

## Prepare an import-only plan

Use a dedicated operator shell, `umask 077`, no `TF_LOG`/`TF_LOG_PATH`, no ambient
`TF_CLI_ARGS*` or `TF_VAR_*` overrides, and no automatic `terraform.tfvars` or
`*.auto.tfvars` files. Keep the example separate from the real inventory. Replace
the absolute paths below with the private operator directory; do not use a shared
temporary directory for real plans or state.

```sh
umask 077
terraform -chdir=terraform/hetzner plan -input=false -refresh=true \
  -var-file=/private/operator/observed.tfvars \
  -out=/private/operator/adoption.tfplan
terraform -chdir=terraform/hetzner show -json /private/operator/adoption.tfplan \
  > /private/operator/adoption.plan.json
python3 terraform/hetzner/check_plan.py /private/operator/adoption.plan.json
```

Import blocks target only the two supplied existing IDs. Plan review must show
two imports and no create, update, replace or delete actions. The checker rejects
every non-no-op resource action or provider action invocation, extra or missing resources, mismatched IDs/IPs,
changed disk/settings/attachment, refreshed drift, deferred work, unknown values
or checks that did not pass. It accepts only owned mode-0600 regular JSON files
and prints counts, never their values. After adoption, the same checker accepts
zero imports with zero changes. A passing check is evidence about that saved
plan, not authorization to apply it; any later changes require a fresh plan.

If any check fails, compare configuration with the observed resource and correct
the adoption inputs. Do not apply a resize, protection change, new backup charge,
firewall attachment, IP reassignment or replacement merely to make adoption fit.
Do not generate configuration containing cloud-init or application secrets.

State uses Terraform's local backend in this directory. No remote backend or
cloud state service is configured. Use restrictive permissions on the directory,
keep its disk encrypted, and back up state privately before any later authorized
state operation. `.terraform/`, `*.tfvars`, `*.tfstate*`, `*.tfplan`, plan JSON and
crash logs are gitignored. State and saved plans may contain sensitive provider
metadata; neither belongs in commits, PR comments or CI artifacts.

## Destruction and in-place changes

Both resources have `prevent_destroy`. Retain their resource blocks: deleting a
block removes that lifecycle protection. This guard does not prevent all in-place
changes, so the zero-change plan policy remains mandatory for adoption. Provider
delete/rebuild protection is mirrored from inventory; enabling it is a distinct
future operation. The provider can power off a server when changing its type or
network, and its Primary IP deletion path can unassign an address before a delete
protection failure. Never use `destroy`, `-replace`, targeted apply or state removal
as an adoption shortcut. The existing OS and its primary disk are not recreated.

References: [server resource](https://registry.terraform.io/providers/hetznercloud/hcloud/1.68.0/docs/resources/server),
[Primary IP resource](https://registry.terraform.io/providers/hetznercloud/hcloud/1.68.0/docs/resources/primary_ip),
[pinned provider implementation](https://github.com/hetznercloud/terraform-provider-hcloud/tree/v1.68.0/internal),
[Terraform imports](https://developer.hashicorp.com/terraform/language/import),
[lifecycle limits](https://developer.hashicorp.com/terraform/language/meta-arguments/lifecycle#prevent_destroy).
