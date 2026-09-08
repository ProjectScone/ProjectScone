#!/usr/bin/env python3
"""Fail closed before ECS registration; never discover, format or repair storage."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

MOUNT = "/var/lib/scone-qdrant"


class MountError(ValueError):
    """Sanitized storage gate failure."""


class DeviceNotReady(MountError):
    """The exact external device has not appeared yet."""


@dataclass(frozen=True)
class Result:
    code: int
    text: str


Runner = Callable[[Sequence[str]], Result]


def run_command(arguments: Sequence[str]) -> Result:
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=10, check=False)
    if len(result.stdout.encode("utf-8")) > 131072:
        raise MountError("command_output_limit")
    return Result(result.returncode, result.stdout)


def records(text: str, key: str) -> list[dict[str, object]]:
    try:
        payload: object = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise MountError("invalid_device_metadata") from error
    if not isinstance(payload, dict):
        raise MountError("invalid_device_metadata")
    entries: object = payload.get(key)
    if not isinstance(entries, list) or len(entries) > 256:
        raise MountError("invalid_device_metadata")
    output: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not all(isinstance(name, str) for name in entry):
            raise MountError("invalid_device_metadata")
        output.append(entry)
    return output


def device_path(result: Result, volume: str, uuid: str, filesystem: str) -> str:
    if result.code != 0:
        raise MountError("device_lookup_failed")
    matches = [item for item in records(result.text, "blockdevices") if item.get("serial") == volume.replace("-", "")]
    if not matches:
        raise DeviceNotReady("device_not_ready")
    if len(matches) != 1:
        raise MountError("ambiguous_device")
    item = matches[0]
    name = item.get("name")
    if not isinstance(name, str) or re.fullmatch(r"nvme[0-9]+n[0-9]+", name) is None or item.get("type") != "disk":
        raise MountError("unsupported_device")
    if item.get("uuid") is None or item.get("fstype") is None:
        raise DeviceNotReady("filesystem_metadata_not_ready")
    if item.get("uuid") != uuid or item.get("fstype") != filesystem:
        raise MountError("filesystem_identity_mismatch")
    mounts = item.get("mountpoints")
    if not isinstance(mounts, list) or any(point is not None and point != MOUNT for point in mounts):
        raise MountError("device_already_mounted_elsewhere")
    return "/dev/" + name


def mounted(result: Result, device: str, uuid: str, filesystem: str) -> bool:
    if result.code == 1 and not result.text.strip():
        return False
    if result.code != 0:
        raise MountError("mount_lookup_failed")
    entries = records(result.text, "filesystems")
    if len(entries) != 1:
        raise MountError("ambiguous_mount")
    entry = entries[0]
    options = entry.get("options")
    if (entry.get("source") != device or entry.get("uuid") != uuid or entry.get("fstype") != filesystem
            or not isinstance(options, str) or not {"rw", "nodev", "nosuid"}.issubset(options.split(","))):
        raise MountError("mounted_filesystem_mismatch")
    return True


def mount_once(volume: str, uuid: str, filesystem: str, runner: Runner = run_command,
               owner: Callable[[str], tuple[int, int]] = lambda path: (os.stat(path).st_uid, os.stat(path).st_gid)) -> None:
    if (re.fullmatch(r"vol-[0-9a-f]{8,17}", volume) is None
            or re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", uuid) is None
            or filesystem not in {"ext4", "xfs"}):
        raise MountError("invalid_configuration")
    device = device_path(runner(["lsblk", "--json", "--nodeps", "--output", "NAME,SERIAL,FSTYPE,UUID,TYPE,MOUNTPOINTS"]), volume, uuid, filesystem)
    lookup = ["findmnt", "--json", "--mountpoint", MOUNT, "--output", "SOURCE,UUID,FSTYPE,OPTIONS"]
    if not mounted(runner(lookup), device, uuid, filesystem):
        if runner(["install", "-d", "-m", "0755", MOUNT]).code != 0:
            raise MountError("mount_directory_failed")
        if runner(["mount", "-t", filesystem, "-o", "nodev,nosuid", device, MOUNT]).code != 0:
            raise MountError("mount_failed")
        if not mounted(runner(lookup), device, uuid, filesystem):
            raise MountError("mount_not_present")
    if owner(MOUNT) != (10001, 10001):
        raise MountError("filesystem_owner_mismatch")


def wait_for_mount(action: Callable[[], None], *, clock: Callable[[], float] = time.monotonic,
                   sleep: Callable[[float], None] = time.sleep) -> None:
    deadline = clock() + 600
    while True:
        try:
            action()
            return
        except DeviceNotReady:
            remaining = deadline - clock()
            if remaining <= 0:
                raise MountError("device_wait_expired") from None
            sleep(min(2, remaining))


def main(arguments: Sequence[str]) -> int:
    try:
        if len(arguments) != 3 or Path(MOUNT).is_symlink():
            raise MountError("invalid_configuration")
        wait_for_mount(lambda: mount_once(arguments[0], arguments[1], arguments[2]))
        return 0
    except (MountError, OSError, subprocess.SubprocessError):
        print("Qdrant storage verification failed; ECS remains blocked.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
