mock_provider "aws" {
  mock_resource "aws_kms_key" {
    defaults = {
      arn    = "arn:aws:kms:us-east-2:123456789012:key/00000000-0000-0000-0000-000000000001"
      key_id = "00000000-0000-0000-0000-000000000001"
    }
  }
}

variables {
  account_id      = "123456789012"
  bucket_name     = "scone-synthetic-foundation-test"
  blob_table_name = "scone-synthetic-blob-metadata"
  aws_region      = "us-east-2"
  name_prefix     = "scone-test"
  s3_prefix       = "attachments/"
}

run "private_foundation" {
  # Mock apply resolves generated resource ARNs without calling AWS.
  command = apply
  assert {
    condition     = aws_dynamodb_table.blobs.hash_key == "pk" && aws_dynamodb_table.blobs.range_key == "sk" && aws_dynamodb_table.blobs.billing_mode == "PAY_PER_REQUEST"
    error_message = "The blob adapter requires on-demand string pk/sk metadata."
  }
  assert {
    condition     = aws_dynamodb_table.blobs.deletion_protection_enabled && aws_dynamodb_table.blobs.point_in_time_recovery[0].enabled && aws_dynamodb_table.blobs.server_side_encryption[0].enabled
    error_message = "Metadata needs deletion protection, recovery and encryption."
  }
  assert {
    condition     = aws_s3_bucket_public_access_block.attachments.block_public_acls && aws_s3_bucket_public_access_block.attachments.block_public_policy && aws_s3_bucket_public_access_block.attachments.ignore_public_acls && aws_s3_bucket_public_access_block.attachments.restrict_public_buckets
    error_message = "Attachments must not be public."
  }
  assert {
    condition     = aws_s3_bucket_versioning.attachments.versioning_configuration[0].status == "Enabled" && !aws_s3_bucket.attachments.force_destroy
    error_message = "Versioned attachments must not be force-destroyed."
  }
  assert {
    condition     = jsondecode(aws_iam_role_policy.storage.policy).Statement[0].Resource == "arn:aws:s3:::scone-synthetic-foundation-test/attachments/*"
    error_message = "Runtime object permissions must stay inside the attachment prefix."
  }
  assert {
    condition     = !contains(jsondecode(aws_iam_role_policy.storage.policy).Statement[1].Action, "dynamodb:Scan")
    error_message = "The attachment adapter must use keyed reads, not table-wide scans."
  }
  assert {
    condition     = jsondecode(aws_iam_role_policy.storage.policy).Statement[3].Resource == "arn:aws:s3:::scone-synthetic-foundation-test" && contains(jsondecode(aws_iam_role_policy.storage.policy).Statement[3].Action, "s3:ListBucket")
    error_message = "Missing-object recovery needs ListBucket on the dedicated attachment bucket only."
  }
  assert {
    condition     = aws_ecr_repository.application.image_tag_mutability == "IMMUTABLE" && aws_ecr_repository.application.image_scanning_configuration[0].scan_on_push
    error_message = "Published image tags must be immutable and scanned."
  }
}

run "reject_wildcard_prefix" {
  command = plan
  variables { s3_prefix = "*" }
  expect_failures = [var.s3_prefix]
}
