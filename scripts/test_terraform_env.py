from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import terraform_env as target


VALUES = {
    "SCONE_AWS_ACCOUNT_ID": "123456789012",
    "SCONE_AWS_REGION": "us-east-2",
    "SCONE_AWS_NAME_PREFIX": "scone-test",
    "SCONE_S3_BUCKET": "scone-synthetic-env-test",
    "SCONE_DYNAMODB_BLOB_TABLE": "scone-synthetic-blob-test",
    "SCONE_S3_PREFIX": "attachments/",
}


class TerraformEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / ".env"
        self.path.write_text("\n".join(f"{key}={value}" for key, value in VALUES.items()) + "\n")
        self.path.chmod(0o600)

    def test_exact_mapping_and_no_secret_forwarding(self) -> None:
        values = {**VALUES, "AWS_SECRET_ACCESS_KEY": "synthetic-private", "SCONE_API_KEY": "synthetic-api"}
        actual = target.terraform_inputs(values)
        self.assertEqual(actual["TF_VAR_blob_table_name"], VALUES["SCONE_DYNAMODB_BLOB_TABLE"])
        self.assertEqual(set(actual), {"TF_VAR_" + name for name in target.INPUTS.values()} | {"TF_VAR_tags"})
        self.assertNotIn("synthetic-private", json.dumps(actual))
        self.assertNotIn("synthetic-api", json.dumps(actual))

    def test_tags_are_json_string_map(self) -> None:
        actual = target.terraform_inputs({**VALUES, "SCONE_AWS_TAGS": '{"Owner":"synthetic","Stage":"test"}'})
        self.assertEqual(json.loads(actual["TF_VAR_tags"]), {"Owner": "synthetic", "Stage": "test"})
        for tags in ('[]', '{"a":1}', '{"aws:reserved":"x"}', '{"":"x"}', '{"x":"\\n"}'):
            with self.subTest(tags=tags), self.assertRaises(ValueError):
                target.terraform_inputs({**VALUES, "SCONE_AWS_TAGS": tags})

    def test_missing_required_inputs(self) -> None:
        for name in VALUES:
            with self.subTest(name=name), self.assertRaises(ValueError):
                target.terraform_inputs({key: value for key, value in VALUES.items() if key != name})

    def test_invalid_identifiers_and_shell_payloads(self) -> None:
        for name, value in (("SCONE_AWS_ACCOUNT_ID", "000"), ("SCONE_AWS_REGION", "cn-north-1"),
                            ("SCONE_AWS_NAME_PREFIX", "$(touch marker)"), ("SCONE_S3_BUCKET", "a*bucket"),
                            ("SCONE_S3_BUCKET", "amzn-s3-demo-invalid"),
                            ("SCONE_S3_PREFIX", "../"), ("SCONE_S3_PREFIX", "x" * 129 + "/"),
                            ("SCONE_DYNAMODB_BLOB_TABLE", "table/foreign")):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                target.terraform_inputs({**VALUES, name: value})

    def test_check_never_launches_process_or_prints_values(self) -> None:
        output = StringIO()
        with patch.object(subprocess, "run") as run, redirect_stdout(output):
            result = target.main(["--env-file", str(self.path), "--check"])
        self.assertEqual(result, 0)
        run.assert_not_called()
        for value in VALUES.values():
            self.assertNotIn(value, output.getvalue())

    def test_dotenv_command_substitution_stays_data(self) -> None:
        marker = self.path.with_name("must-not-exist")
        with self.path.open("a") as stream:
            stream.write(f"SCONE_API_KEY='$(touch {marker})'\n")
        with redirect_stdout(StringIO()):
            self.assertEqual(target.main(["--env-file", str(self.path), "--check"]), 0)
        self.assertFalse(marker.exists())

    def test_duplicate_invalid_file_and_permissions_fail_without_value_output(self) -> None:
        with self.path.open("a") as stream:
            stream.write("SCONE_AWS_REGION=synthetic-secret-marker\n")
        output = StringIO()
        with patch.object(subprocess, "run") as run, redirect_stderr(output):
            result = target.main(["--env-file", str(self.path), "--check"])
        self.assertEqual(result, 2)
        run.assert_not_called()
        self.assertNotIn("synthetic-secret-marker", output.getvalue())
        self.path.chmod(0o644)
        with redirect_stderr(StringIO()):
            self.assertEqual(target.main(["--env-file", str(self.path), "--check"]), 2)

    def test_symlink_is_not_followed(self) -> None:
        link = self.path.with_name("link.env")
        link.symlink_to(self.path)
        with redirect_stderr(StringIO()):
            self.assertEqual(target.main(["--env-file", str(link), "--check"]), 2)

    def test_automatic_tfvars_cannot_override_dotenv(self) -> None:
        module = self.path.parent / "module"
        module.mkdir()
        (module / "operator.auto.tfvars").write_text('account_id="999999999999"')
        with patch.object(target, "MODULE", module), patch.object(subprocess, "run") as run, redirect_stderr(StringIO()):
            self.assertEqual(target.main(["--env-file", str(self.path), "validate"]), 2)
        run.assert_not_called()

    def test_offline_environment_drops_ambient_credentials_and_terraform_overrides(self) -> None:
        inherited = {"PATH": "/bin", "HOME": "/synthetic", "AWS_ACCESS_KEY_ID": "private",
                     "AWS_WEB_IDENTITY_TOKEN_FILE": "/private", "TF_VAR_account_id": "foreign",
                     "TF_CLI_ARGS": "-var account_id=foreign", "TF_LOG": "TRACE"}
        result = target.process_environment(target.terraform_inputs(VALUES), inherited, cloud=False)
        self.assertNotIn("AWS_ACCESS_KEY_ID", result)
        self.assertNotIn("AWS_WEB_IDENTITY_TOKEN_FILE", result)
        self.assertNotIn("TF_CLI_ARGS", result)
        self.assertNotIn("TF_LOG", result)
        self.assertEqual(result["TF_VAR_account_id"], VALUES["SCONE_AWS_ACCOUNT_ID"])
        self.assertEqual(result["AWS_SHARED_CREDENTIALS_FILE"], os.devnull)
        self.assertEqual(result["AWS_EC2_METADATA_DISABLED"], "true")

    def test_cloud_command_keeps_provider_chain_but_drops_tf_overrides(self) -> None:
        result = target.process_environment(target.terraform_inputs(VALUES),
                                            {"AWS_PROFILE": "synthetic-sso", "TF_VAR_secret": "private",
                                             "TF_CLI_ARGS_apply": "-auto-approve"}, cloud=True)
        self.assertEqual(result["AWS_PROFILE"], "synthetic-sso")
        self.assertNotIn("TF_VAR_secret", result)
        self.assertNotIn("TF_CLI_ARGS_apply", result)

    def test_init_is_backend_disabled_and_apply_requires_normal_confirmation(self) -> None:
        self.assertEqual(target.command_arguments("terraform", "init")[-2:], ["-backend=false", "-input=false"])
        self.assertNotIn("-auto-approve", target.command_arguments("terraform", "apply"))

    def test_validate_uses_argv_no_shell_and_return_code(self) -> None:
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as run:
            result = target.main(["--env-file", str(self.path), "--terraform", "/synthetic/terraform", "validate"])
        self.assertEqual(result, 7)
        self.assertEqual(run.call_args.args[0], ["/synthetic/terraform", "-chdir=" + str(target.MODULE), "validate"])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["env"]["TF_VAR_bucket_name"], VALUES["SCONE_S3_BUCKET"])


if __name__ == "__main__":
    unittest.main()
