"""Synthetic command responses only: never mount or access a real block device."""
from __future__ import annotations

import json
import sys
import unittest
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mount_volume import DeviceNotReady, MountError, Result, mount_once, wait_for_mount

VOLUME = "vol-0123456789abcdef0"
UUID = "12345678-1234-1234-1234-123456789abc"
MOUNT = "/var/lib/scone-qdrant"


def disk(**changes: object) -> dict[str, object]:
    return dict({"name": "nvme1n1", "serial": VOLUME.replace("-", ""), "uuid": UUID,
                 "fstype": "xfs", "type": "disk", "mountpoints": [None]}, **changes)


def found(**changes: object) -> Result:
    return Result(0, json.dumps({"filesystems": [dict({"source": "/dev/nvme1n1", "uuid": UUID,
                   "fstype": "xfs", "options": "rw,nodev,nosuid,relatime"}, **changes)]}))


class Commands:
    def __init__(self, devices: list[dict[str, object]] | None = None, mounts: list[Result] | None = None) -> None:
        self.devices = [disk()] if devices is None else devices
        self.mounts = [Result(1, ""), found()] if mounts is None else mounts
        self.calls: list[list[str]] = []

    def __call__(self, args: Sequence[str]) -> Result:
        self.calls.append(list(args))
        if args[0] == "lsblk":
            return Result(0, json.dumps({"blockdevices": self.devices}))
        if args[0] == "findmnt":
            return self.mounts.pop(0)
        return Result(0, "")


class MountTests(unittest.TestCase):
    def mount(self, commands: Commands, owner: tuple[int, int] = (10001, 10001)) -> None:
        mount_once(VOLUME, UUID, "xfs", commands, lambda _: owner)

    def test_mounts_only_verified_whole_device_then_rechecks(self) -> None:
        commands = Commands()
        self.mount(commands)
        self.assertEqual([call[0] for call in commands.calls], ["lsblk", "findmnt", "install", "mount", "findmnt"])
        self.assertEqual(commands.calls[3], ["mount", "-t", "xfs", "-o", "nodev,nosuid", "/dev/nvme1n1", MOUNT])

    def test_existing_verified_mount_is_not_remounted(self) -> None:
        commands = Commands(mounts=[found()])
        self.mount(commands)
        self.assertEqual([call[0] for call in commands.calls], ["lsblk", "findmnt"])

    def test_wrong_or_duplicate_volume_never_mounts(self) -> None:
        for devices in [[disk(serial="volwrong")], [disk(), disk(name="nvme2n1")]]:
            with self.subTest(devices=devices):
                commands = Commands(devices)
                with self.assertRaises(MountError):
                    self.mount(commands)
                self.assertEqual(len(commands.calls), 1)

    def test_wrong_uuid_filesystem_device_or_existing_mount_never_mounts(self) -> None:
        changes: list[dict[str, object]] = [{"uuid": "wrong"}, {"fstype": "ext4"}, {"name": "nvme1n1p1"},
                       {"type": "part"}, {"mountpoints": ["/somewhere"]}, {"mountpoints": "bad"}]
        for change in changes:
            with self.subTest(change=change):
                commands = Commands([disk(**change)])
                with self.assertRaises(MountError):
                    self.mount(commands)
                self.assertEqual(len(commands.calls), 1)

    def test_wrong_mount_identity_or_readonly_is_rejected(self) -> None:
        for change in [{"source": "/dev/nvme2n1"}, {"uuid": "wrong"}, {"fstype": "ext4"},
                       {"options": "ro,nodev,nosuid"}, {"options": "rw"}]:
            with self.subTest(change=change):
                commands = Commands(mounts=[found(**change)])
                with self.assertRaises(MountError):
                    self.mount(commands)
                self.assertEqual(len(commands.calls), 2)

    def test_failed_final_verification_blocks_registration(self) -> None:
        with self.assertRaises(MountError):
            self.mount(Commands(mounts=[Result(1, ""), Result(1, "")]))

    def test_owner_must_be_prepared_and_never_repaired(self) -> None:
        commands = Commands(mounts=[found()])
        with self.assertRaisesRegex(MountError, "filesystem_owner_mismatch"):
            self.mount(commands, (0, 0))
        self.assertEqual(len(commands.calls), 2)

    def test_invalid_cli_input_never_runs_command(self) -> None:
        commands = Commands()
        with self.assertRaises(MountError):
            mount_once("vol-12345678;touch /tmp/x", UUID, "xfs", commands)
        self.assertEqual(commands.calls, [])

    def test_device_absence_retries_with_a_deadline(self) -> None:
        now = [0.0]
        attempts = [0]
        def action() -> None:
            attempts[0] += 1
            raise DeviceNotReady("device_not_ready")
        def sleep(seconds: float) -> None:
            now[0] += seconds
        with self.assertRaisesRegex(MountError, "device_wait_expired"):
            wait_for_mount(action, clock=lambda: now[0], sleep=sleep)
        self.assertEqual(now[0], 600)
        self.assertEqual(attempts[0], 301)

    def test_udev_metadata_may_arrive_after_device(self) -> None:
        stages = [Commands([]), Commands([disk(uuid=None, fstype=None)]), Commands()]
        def action() -> None:
            self.mount(stages.pop(0))
        wait_for_mount(action, sleep=lambda _: None)
        self.assertEqual(stages, [])

    def test_identity_failure_does_not_retry(self) -> None:
        def action() -> None:
            raise MountError("filesystem_identity_mismatch")
        with self.assertRaisesRegex(MountError, "filesystem_identity_mismatch"):
            wait_for_mount(action, sleep=lambda _: self.fail("must not sleep"))


if __name__ == "__main__":
    unittest.main()
