"""Map an explicitly selected private .env file to nonsecret Terraform inputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from local_env import private_environment

MODULE = Path(__file__).resolve().parents[1] / "terraform"
INPUTS = {
    "SCONE_AWS_ACCOUNT_ID": "account_id",
    "SCONE_AWS_REGION": "aws_region",
    "SCONE_AWS_NAME_PREFIX": "name_prefix",
    "SCONE_S3_BUCKET": "bucket_name",
    "SCONE_DYNAMODB_BLOB_TABLE": "blob_table_name",
    "SCONE_S3_PREFIX": "s3_prefix",
}
_SAFE_PROCESS_ENV = ("PATH", "HOME", "TMPDIR", "SYSTEMROOT", "SSL_CERT_FILE", "SSL_CERT_DIR")


def terraform_inputs(values: Mapping[str, str]) -> dict[str, str]:
    """Only deployment identifiers and string tags can become TF_VAR values."""
    if any(not values.get(name) for name in INPUTS):
        raise ValueError("missing required deployment input")
    checks = {
        "SCONE_AWS_ACCOUNT_ID": r"[0-9]{12}",
        "SCONE_AWS_REGION": r"(?!cn-)[a-z]{2}-[a-z]+-[0-9]+",
        "SCONE_AWS_NAME_PREFIX": r"[a-z][a-z0-9-]{1,30}[a-z0-9]",
        "SCONE_S3_BUCKET": r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]",
        "SCONE_DYNAMODB_BLOB_TABLE": r"[A-Za-z0-9_.-]{3,255}",
        "SCONE_S3_PREFIX": r"(?:[A-Za-z0-9_-]+/)+",
    }
    if any(re.fullmatch(pattern, values[name]) is None for name, pattern in checks.items()):
        raise ValueError("invalid deployment input")
    bucket = values["SCONE_S3_BUCKET"]
    if bucket.startswith(("xn--", "sthree-", "amzn-s3-demo-")) or bucket.endswith(
        ("--x-s3", "--table-s3", "-s3alias", "--ol-s3")
    ) or len(values["SCONE_S3_PREFIX"]) > 128:
        raise ValueError("invalid deployment input")
    tags: object = json.loads(values.get("SCONE_AWS_TAGS", "{}"))
    if not isinstance(tags, dict) or len(tags) > 48 or any(
        not isinstance(key, str) or not isinstance(value, str)
        or not 1 <= len(key) <= 128 or len(value) > 256
        or key.lower().startswith("aws:")
        or any(ord(char) < 32 for char in key + value)
        for key, value in tags.items()
    ):
        raise ValueError("invalid deployment tags")
    result = {"TF_VAR_" + target: values[source] for source, target in INPUTS.items()}
    result["TF_VAR_tags"] = json.dumps(tags, sort_keys=True, separators=(",", ":"))
    return result


def process_environment(inputs: Mapping[str, str], inherited: Mapping[str, str], *, cloud: bool) -> dict[str, str]:
    if cloud:
        # Credential resolution remains the AWS provider's standard chain. Never
        # translate credentials from the dotenv file into Terraform variables.
        result = {key: value for key, value in inherited.items() if not key.startswith("TF_")}
    else:
        result = {key: inherited[key] for key in _SAFE_PROCESS_ENV if key in inherited}
        result.update(AWS_EC2_METADATA_DISABLED="true", AWS_CONFIG_FILE=os.devnull,
                      AWS_SHARED_CREDENTIALS_FILE=os.devnull, TF_CLI_CONFIG_FILE=os.devnull)
    result.update(inputs)
    result["TF_IN_AUTOMATION"] = "1"
    return result


def command_arguments(executable: str, command: str) -> list[str]:
    args = [executable, "-chdir=" + str(MODULE), command]
    if command == "init":
        return [*args, "-backend=false", "-input=false"]
    if command == "fmt":
        return [*args, "-check", "-recursive"]
    if command == "plan":
        return [*args, "-input=false"]
    # Apply retains Terraform's interactive confirmation. No auto-approve flag.
    return args


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="validate inputs without starting Terraform")
    parser.add_argument("--terraform", default="terraform", help="Terraform executable or absolute path")
    parser.add_argument("command", nargs="?", choices=("init", "fmt", "validate", "test", "plan", "apply"))
    args = parser.parse_args(argv)
    if args.check and args.command or not args.check and args.command is None:
        parser.error("choose --check or exactly one Terraform command")
    try:
        values = private_environment(args.env_file, {})
        inputs = terraform_inputs(values)
        if any(next(MODULE.glob(pattern), None) is not None for pattern in
               ("terraform.tfvars", "terraform.tfvars.json", "*.auto.tfvars", "*.auto.tfvars.json")):
            raise ValueError("automatic Terraform input files are not allowed with the environment wrapper")
    except (OSError, ValueError):
        print("Invalid deployment environment; check required identifiers, tags, format and owned 0600 permissions.", file=sys.stderr)
        return 2
    if args.check:
        print("Deployment inputs validated; no process started and no values printed.")
        return 0
    try:
        result = subprocess.run(command_arguments(args.terraform, args.command),
                                env=process_environment(inputs, os.environ, cloud=args.command in {"plan", "apply"}),
                                check=False)
    except OSError:
        print("Cannot start Terraform; check its installation.", file=sys.stderr)
        return 2
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
