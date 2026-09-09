output "attachment_environment" {
  description = "Attachment adapter settings; these do not configure a DynamoDB DocumentStore or a vector backend."
  value = {
    SCONE_BLOBS               = "s3"
    SCONE_S3_BUCKET           = aws_s3_bucket.attachments.id
    SCONE_DYNAMODB_BLOB_TABLE = aws_dynamodb_table.blobs.name
    SCONE_AWS_REGION          = var.aws_region
    SCONE_S3_PREFIX           = var.s3_prefix
  }
}

output "storage_kms_key_arn" { value = aws_kms_key.storage.arn }
output "application_task_role_arn" { value = aws_iam_role.task.arn }
output "ecr_repository_url" { value = aws_ecr_repository.application.repository_url }

output "ecr_push_policy" {
  description = "Operator attaches to an explicitly authorized build principal; no identity or trust is created."
  value = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
      {
        Effect   = "Allow"
        Action   = ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage"]
        Resource = aws_ecr_repository.application.arn
      }
    ]
  })
}
