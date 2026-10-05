"""Synthetic saved-plan policy tests. No provider, token or real inventory."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

GUARD = Path(__file__).resolve().parents[1] / "check_plan.py"
spec = importlib.util.spec_from_file_location("hetzner_plan_guard", GUARD)
assert spec is not None and spec.loader is not None
target = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)


def plan() -> dict[str, object]:
    server = {"id": 1001, "name": "synthetic-server", "server_type": "cpx22",
              "location": "hel1", "ipv4_address": "192.0.2.10", "primary_disk_size": 80,
              "backups": False, "delete_protection": False, "rebuild_protection": False,
              "labels": {}}
    primary = {"id": 2001, "name": "synthetic-ip", "auto_delete": True,
               "delete_protection": False, "labels": {}}
    server_after = {**server, "id": "1001", "firewall_ids": []}
    primary_after = {**primary, "type": "ipv4", "assignee_type": "server",
                     "assignee_id": 1001, "ip_address": "192.0.2.10"}
    return {
        "format_version": "1.2", "terraform_version": "1.16.1",
        "complete": True, "errored": False,
        "variables": {"server": {"value": server}, "primary_ipv4": {"value": primary}},
        "resource_changes": [
            {"address": "hcloud_server.existing", "mode": "managed", "type": "hcloud_server",
             "provider_name": "registry.terraform.io/hetznercloud/hcloud",
             "change": {"actions": ["no-op"], "after": server_after,
                        "after_unknown": {}, "importing": {"id": "1001"}}},
            {"address": "hcloud_primary_ip.existing", "mode": "managed", "type": "hcloud_primary_ip",
             "provider_name": "registry.terraform.io/hetznercloud/hcloud",
             "change": {"actions": ["no-op"], "after": primary_after,
                        "after_unknown": {}, "importing": {"id": "2001"}}},
        ],
        "checks": [{"status": "pass"}],
    }


class PlanPolicyTests(unittest.TestCase):
    def test_import_only_and_existing_state_noop_are_accepted(self) -> None:
        self.assertEqual(target.check_plan(plan()), 2)
        value = plan()
        for resource in value["resource_changes"]:
            del resource["change"]["importing"]
        self.assertEqual(target.check_plan(value), 0)

    def test_every_remote_action_including_inplace_update_is_rejected(self) -> None:
        for actions in (["create"], ["update"], ["delete"], ["delete", "create"],
                        ["create", "delete"], ["forget"], ["read"], []):
            value = plan()
            value["resource_changes"][0]["change"]["actions"] = actions
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                target.check_plan(value)

    def test_exact_resource_set_and_provider_are_required(self) -> None:
        variants = []
        missing = plan(); missing["resource_changes"].pop(); variants.append(missing)
        duplicate = plan(); duplicate["resource_changes"].append(deepcopy(duplicate["resource_changes"][0])); variants.append(duplicate)
        for field, wrong in (("address", "hcloud_firewall.unreviewed"), ("mode", "data"),
                             ("type", "hcloud_volume"), ("provider_name", "other/provider")):
            value = plan(); value["resource_changes"][0][field] = wrong; variants.append(value)
        for value in variants:
            with self.assertRaises(ValueError): target.check_plan(value)

    def test_import_and_planned_identities_must_match_private_inventory(self) -> None:
        for index in (0, 1):
            for part, field, wrong in (("importing", "id", "9999"), ("after", "id", 9999)):
                value = plan(); value["resource_changes"][index]["change"][part][field] = wrong
                with self.assertRaises(ValueError): target.check_plan(value)
        for field, wrong in (("assignee_id", 1002), ("assignee_type", "other"),
                             ("type", "ipv6"), ("ip_address", "192.0.2.11")):
            value = plan(); value["resource_changes"][1]["change"]["after"][field] = wrong
            with self.assertRaises(ValueError): target.check_plan(value)

    def test_changed_disk_address_settings_or_firewall_are_not_hidden(self) -> None:
        for field, wrong in (("primary_disk_size", 81), ("ipv4_address", "192.0.2.11"),
                             ("server_type", "cpx32"), ("location", "fsn1"),
                             ("backups", True), ("firewall_ids", [123]), ("labels", {"new": "label"})):
            value = plan(); value["resource_changes"][0]["change"]["after"][field] = wrong
            with self.assertRaises(ValueError): target.check_plan(value)
        for field, wrong in (("auto_delete", False), ("delete_protection", True)):
            value = plan(); value["resource_changes"][1]["change"]["after"][field] = wrong
            with self.assertRaises(ValueError): target.check_plan(value)

    def test_incomplete_deferred_drift_or_unknown_checks_are_rejected(self) -> None:
        for field, wrong in (("complete", False), ("errored", True), ("format_version", "2.0"),
                             ("deferred_changes", [{}]), ("resource_drift", [{}]),
                             ("checks", [{"status": "unknown"}]), ("checks", [{"status": "fail"}]),
                             ("checks", [])):
            value = plan(); value[field] = wrong
            with self.assertRaises(ValueError): target.check_plan(value)
        value = plan(); value["resource_changes"][0]["change"]["after_unknown"] = {"id": True}
        with self.assertRaises(ValueError): target.check_plan(value)
        value = plan(); del value["complete"]
        with self.assertRaises(ValueError): target.check_plan(value)

    def test_provider_actions_and_deferred_provider_actions_are_rejected(self) -> None:
        for field in ("action_invocations", "deferred_action_invocations"):
            value = plan(); value[field] = [{"address": "action.hcloud_synthetic.unreviewed"}]
            with self.subTest(field=field), self.assertRaises(ValueError): target.check_plan(value)

    def test_cli_accepts_only_private_regular_bounded_json_without_echoing_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.plan.json"
            path.write_text(json.dumps(plan())); path.chmod(0o600)
            accepted = subprocess.run([sys.executable, str(GUARD), str(path)], capture_output=True, text=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertNotIn("192.0.2.10", accepted.stdout + accepted.stderr)
            path.chmod(0o644)
            self.assertNotEqual(subprocess.run([sys.executable, str(GUARD), str(path)], capture_output=True).returncode, 0)
            path.chmod(0o600)
            link = Path(directory) / "link.json"; link.symlink_to(path)
            self.assertNotEqual(subprocess.run([sys.executable, str(GUARD), str(link)], capture_output=True).returncode, 0)
            path.write_text('{"format_version":"1.2","format_version":"1.2"}')
            self.assertNotEqual(subprocess.run([sys.executable, str(GUARD), str(path)], capture_output=True).returncode, 0)

    def test_oversized_or_hardlinked_plan_is_rejected_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.plan.json"
            path.write_bytes(b" " * (target.MAX_PLAN_BYTES + 1)); path.chmod(0o600)
            with self.assertRaises(ValueError): target.private_plan(path)
            path.write_text(json.dumps(plan()))
            os.link(path, Path(directory) / "second.json")
            with self.assertRaises(ValueError): target.private_plan(path)


if __name__ == "__main__":
    unittest.main()
