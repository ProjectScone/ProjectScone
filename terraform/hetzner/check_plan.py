"""Reject every remote action in a private saved Terraform adoption plan."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import stat
import sys
from typing import cast

MAX_PLAN_BYTES = 10 * 1024 * 1024
PROVIDER = "registry.terraform.io/hetznercloud/hcloud"
RESOURCES = {"hcloud_server.existing": "hcloud_server", "hcloud_primary_ip.existing": "hcloud_primary_ip"}


def object_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Expected object")
    return cast(dict[str, object], value)


def resource_id(value: object) -> str:
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is not int or value <= 0:
        raise ValueError("Invalid resource identity")
    return str(value)


def has_unknown(value: object) -> bool:
    if isinstance(value, dict):
        return any(has_unknown(item) for item in value.values())
    if isinstance(value, list):
        return any(has_unknown(item) for item in value)
    return value is not False and value is not None


def check_plan(value: object) -> int:
    plan = object_value(value)
    if plan.get("format_version") != "1.2" or plan.get("complete") is not True or plan.get("errored") is not False:
        raise ValueError("Unsupported or incomplete plan")
    if any(plan.get(field) for field in ("deferred_changes", "resource_drift", "action_invocations", "deferred_action_invocations")):
        raise ValueError("Provider actions, deferred work or refreshed drift need review")
    checks = plan.get("checks")
    if not isinstance(checks, list) or not checks or any(object_value(item).get("status") != "pass" for item in checks):
        raise ValueError("Every plan check must pass")
    variables = object_value(plan.get("variables"))
    server = object_value(object_value(variables.get("server")).get("value"))
    primary = object_value(object_value(variables.get("primary_ipv4")).get("value"))
    server_fields = {"id", "name", "server_type", "location", "ipv4_address", "primary_disk_size", "backups", "delete_protection", "rebuild_protection", "labels"}
    primary_fields = {"id", "name", "auto_delete", "delete_protection", "labels"}
    if set(server) != server_fields or set(primary) != primary_fields:
        raise ValueError("Incomplete expected inventory")
    ids = {"hcloud_server.existing": resource_id(server["id"]), "hcloud_primary_ip.existing": resource_id(primary["id"])}
    address = server["ipv4_address"]
    if not isinstance(address, str) or str(ipaddress.IPv4Address(address)) != address:
        raise ValueError("Invalid expected IPv4")
    for inventory, boolean_fields in ((server, ("backups", "delete_protection", "rebuild_protection")),
                                     (primary, ("auto_delete", "delete_protection"))):
        if any(type(inventory[field]) is not bool for field in boolean_fields):
            raise ValueError("Invalid observed setting")
        labels = object_value(inventory["labels"])
        if any(not isinstance(item, str) for item in labels.values()):
            raise ValueError("Invalid observed labels")
    changes = plan.get("resource_changes")
    if not isinstance(changes, list) or len(changes) != 2:
        raise ValueError("Unexpected resource set")
    seen: set[str] = set()
    imports = 0
    for entry in changes:
        resource = object_value(entry)
        name = resource.get("address")
        if not isinstance(name, str) or name not in RESOURCES or name in seen:
            raise ValueError("Unexpected or duplicate resource")
        seen.add(name)
        if resource.get("mode") != "managed" or resource.get("type") != RESOURCES[name] or resource.get("provider_name") != PROVIDER:
            raise ValueError("Unexpected resource provider or type")
        change = object_value(resource.get("change"))
        if change.get("actions") != ["no-op"] or has_unknown(change.get("after_unknown")):
            raise ValueError("Remote actions or unknown values are forbidden")
        if "importing" in change:
            importing = object_value(change["importing"])
            if resource_id(importing.get("id")) != ids[name]:
                raise ValueError("Wrong import identity")
            imports += 1
        after = object_value(change.get("after"))
        if resource_id(after.get("id")) != ids[name]:
            raise ValueError("Wrong planned identity")
        expected = server if name == "hcloud_server.existing" else primary
        if any(after.get(field) != item for field, item in expected.items() if field != "id"):
            raise ValueError("Planned inventory changed")
        if name == "hcloud_server.existing":
            if after.get("firewall_ids") != []:
                raise ValueError("Firewall attachment needs separate review")
        elif after.get("type") != "ipv4" or after.get("assignee_type") != "server" or resource_id(after.get("assignee_id")) != ids["hcloud_server.existing"] or after.get("ip_address") != address:
            raise ValueError("Primary IP attachment changed")
    return imports


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON field")
        result[name] = value
    return result


def private_plan(path: Path) -> object:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) != 0o600 or before.st_nlink != 1 or before.st_size > MAX_PLAN_BYTES:
            raise ValueError("Plan must be an owned private regular file")
        raw = stream.read(MAX_PLAN_BYTES + 1)
        after = os.fstat(stream.fileno())
        if len(raw) > MAX_PLAN_BYTES or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("Plan changed while reading")
    return json.loads(raw, object_pairs_hook=unique_object)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Owned mode-0600 Terraform show -json file")
    args = parser.parse_args()
    try:
        imports = check_plan(private_plan(args.plan))
    except (OSError, ValueError, RecursionError):
        print("Plan refused: require private valid input, exact inventory, passed checks and zero remote actions.", file=sys.stderr)
        return 1
    print(f"Plan accepted: {imports} imports; zero remote changes. No apply was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
