locals {
  log_arns = [for name in ["api", "qdrant"] : "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/scone/${var.name_prefix}/${name}"]
  task_trust = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "ecs-tasks.amazonaws.com" }
      Condition = {
        StringEquals = { "aws:SourceAccount" = var.account_id }
        ArnLike      = { "aws:SourceArn" = "arn:aws:ecs:${var.aws_region}:${var.account_id}:*" }
      }
    }]
  })
}

resource "aws_kms_key" "logs" {
  description             = "Scone compute logs"
  enable_key_rotation     = true
  deletion_window_in_days = 30
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Principal = { AWS = "arn:aws:iam::${var.account_id}:root" }, Action = "kms:*", Resource = "*" },
      {
        Effect    = "Allow", Principal = { Service = "logs.${var.aws_region}.amazonaws.com" }
        Action    = ["kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:DescribeKey"], Resource = "*"
        Condition = { ArnEquals = { "kms:EncryptionContext:aws:logs:arn" = local.log_arns } }
      }
    ]
  })
  tags = local.tags
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/scone/${var.name_prefix}/api"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.logs.arn
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "qdrant" {
  name              = "/scone/${var.name_prefix}/qdrant"
  retention_in_days = 30
  kms_key_id        = aws_kms_key.logs.arn
  tags              = local.tags
}

resource "aws_iam_role" "execution" {
  name               = "${var.name_prefix}-execution"
  assume_role_policy = local.task_trust
  tags               = local.tags
}

resource "aws_iam_role_policy" "execution" {
  name = "pull-log-secret-references"
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat([
      { Sid = "RegistryToken", Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
      { Sid = "ReviewedImages", Effect = "Allow", Action = ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"], Resource = local.repositories },
      { Sid = "ApplicationLogs", Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = [for arn in local.log_arns : "${arn}:*"] },
      { Sid = "ReferencedSecrets", Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = local.secret_arns }
      ], length(var.secret_kms_key_arns) == 0 ? [] : [{
        Sid       = "ReferencedSecretKeys", Effect = "Allow", Action = ["kms:Decrypt"], Resource = var.secret_kms_key_arns
        Condition = { StringEquals = { "kms:ViaService" = "secretsmanager.${var.aws_region}.amazonaws.com", "kms:CallerAccount" = var.account_id } }
    }])
  })
}

resource "aws_iam_role" "qdrant_task" {
  name               = "${var.name_prefix}-qdrant-task"
  assume_role_policy = local.task_trust
  tags               = local.tags
  # The Qdrant process has no AWS API permissions. Snapshot lifecycle is external.
}

resource "aws_iam_role" "host" {
  name = "${var.name_prefix}-qdrant-host"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "ec2.amazonaws.com" } }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy" "host" {
  name = "ecs-registration"
  role = aws_iam_role.host.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["ecs:RegisterContainerInstance"], Resource = local.cluster_arn },
      { Effect = "Allow", Action = ["ecs:DeregisterContainerInstance", "ecs:Poll", "ecs:StartTelemetrySession", "ecs:UpdateContainerInstancesState"], Resource = local.instance_arns },
      { Effect = "Allow", Action = ["ecs:SubmitAttachmentStateChanges", "ecs:SubmitContainerStateChange", "ecs:SubmitTaskStateChange"], Resource = local.cluster_arn },
      { Effect = "Allow", Action = ["ecs:DiscoverPollEndpoint"], Resource = "*" },
      { Effect = "Allow", Action = ["ecs:TagResource"], Resource = local.instance_arns, Condition = { StringEquals = { "ecs:CreateAction" = "RegisterContainerInstance" } } }
    ]
  })
}

resource "aws_iam_instance_profile" "host" {
  name = "${var.name_prefix}-qdrant-host"
  role = aws_iam_role.host.name
  tags = local.tags
}
