mock_provider "aws" {
  mock_data "aws_subnet" {
    defaults = { vpc_id = "vpc-0123456789abcdef0", availability_zone = "us-east-1a" }
  }
  mock_data "aws_ebs_volume" {
    defaults = { encrypted = true, multi_attach_enabled = false, availability_zone = "us-east-1a" }
  }
  mock_resource "aws_iam_role" {
    defaults = { arn = "arn:aws:iam::123456789012:role/mock-role" }
  }
  mock_resource "aws_kms_key" {
    defaults = { arn = "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789abc" }
  }
  mock_resource "aws_lb" {
    defaults = { arn = "arn:aws:elasticloadbalancing:us-east-1:123456789012:loadbalancer/app/mock/123456789abcdef0", dns_name = "mock.us-east-1.elb.amazonaws.com" }
  }
  mock_resource "aws_lb_target_group" {
    defaults = { arn = "arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/mock/123456789abcdef0" }
  }
  mock_resource "aws_ecs_cluster" {
    defaults = { arn = "arn:aws:ecs:us-east-1:123456789012:cluster/scone-test-compute", id = "arn:aws:ecs:us-east-1:123456789012:cluster/scone-test-compute" }
  }
  mock_resource "aws_ecs_task_definition" {
    defaults = { arn = "arn:aws:ecs:us-east-1:123456789012:task-definition/mock:1" }
  }
  mock_resource "aws_service_discovery_service" {
    defaults = { arn = "arn:aws:servicediscovery:us-east-1:123456789012:service/srv-mock" }
  }
}

override_data {
  target = data.aws_subnet.api["subnet-22222222222222222"]
  values = { vpc_id = "vpc-0123456789abcdef0", availability_zone = "us-east-1b" }
}
override_data {
  target = data.aws_subnet.alb["subnet-44444444444444444"]
  values = { vpc_id = "vpc-0123456789abcdef0", availability_zone = "us-east-1b" }
}

variables {
  account_id               = "123456789012"
  aws_region               = "us-east-1"
  name_prefix              = "scone-test"
  vpc_id                   = "vpc-0123456789abcdef0"
  api_subnet_ids           = ["subnet-11111111111111111", "subnet-22222222222222222"]
  alb_subnet_ids           = ["subnet-33333333333333333", "subnet-44444444444444444"]
  ingress_cidrs            = ["192.0.2.0/24"]
  https_egress_cidrs       = ["10.0.0.0/16"]
  certificate_arn          = "arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789abc"
  api_image                = "123456789012.dkr.ecr.us-east-1.amazonaws.com/scone@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  api_task_role_arn        = "arn:aws:iam::123456789012:role/scone-foundation"
  qdrant_image             = "123456789012.dkr.ecr.us-east-1.amazonaws.com/qdrant@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  qdrant_secret_arn        = "arn:aws:secretsmanager:us-east-1:123456789012:secret:qdrant-example"
  qdrant_subnet_id         = "subnet-11111111111111111"
  qdrant_availability_zone = "us-east-1a"
  qdrant_ami_id            = "ami-0123456789abcdef0"
  qdrant_instance_type     = "t4g.medium"
  qdrant_volume_id         = "vol-0123456789abcdef0"
  qdrant_filesystem_uuid   = "12345678-1234-1234-1234-123456789abc"
}

run "guarded_defaults" {
  command = apply
  assert {
    condition     = aws_ecs_service.api.desired_count == 0 && aws_ecs_service.qdrant.desired_count == 0
    error_message = "Unconfigured services must not start."
  }
  assert {
    condition     = aws_volume_attachment.qdrant.volume_id == var.qdrant_volume_id && !aws_volume_attachment.qdrant.force_detach && aws_volume_attachment.qdrant.stop_instance_before_detaching
    error_message = "The external disk must never be force-detached from a live host."
  }
  assert {
    condition     = aws_ecs_service.qdrant.deployment_minimum_healthy_percent == 0 && aws_ecs_service.qdrant.deployment_maximum_percent == 100
    error_message = "Rolling deployments must stop the old writer before starting a new one."
  }
  assert {
    condition     = aws_ecs_task_definition.api.task_role_arn == var.api_task_role_arn && jsondecode(aws_ecs_task_definition.api.container_definitions)[0].readonlyRootFilesystem
    error_message = "The API must use the supplied application role and a read-only root filesystem."
  }
  assert {
    condition     = one(aws_ecs_task_definition.qdrant.volume).host_path == "/var/lib/scone-qdrant" && strcontains(aws_instance.qdrant.user_data, "Requires=scone-qdrant-volume.service") && strcontains(aws_instance.qdrant.user_data, "ExecStartPre=/usr/bin/mountpoint")
    error_message = "Only the verified host mount may back the Qdrant container."
  }
  assert {
    condition     = alltrue([for item in jsondecode(aws_ecs_task_definition.api.container_definitions)[0].secrets : startswith(item.valueFrom, "arn:aws:secretsmanager:")])
    error_message = "Task definitions must contain secret references, never plaintext secrets."
  }
  assert {
    condition     = jsondecode(aws_iam_role_policy.host.policy).Statement[2].Resource == local.cluster_arn
    error_message = "ECS state reporting requires the cluster ARN resource."
  }
}

run "unconfigured_api_is_blocked" {
  command = plan
  variables { api_desired_count = 1 }
  expect_failures = [aws_ecs_service.api]
}

run "wrong_volume_zone_is_blocked" {
  command = plan
  override_data {
    target = data.aws_ebs_volume.qdrant
    values = { encrypted = true, multi_attach_enabled = false, availability_zone = "us-east-1b" }
  }
  expect_failures = [aws_instance.qdrant]
}

run "unencrypted_volume_is_blocked" {
  command = plan
  override_data {
    target = data.aws_ebs_volume.qdrant
    values = { encrypted = false, multi_attach_enabled = false, availability_zone = "us-east-1a" }
  }
  expect_failures = [aws_instance.qdrant]
}

run "multi_attach_is_blocked" {
  command = plan
  override_data {
    target = data.aws_ebs_volume.qdrant
    values = { encrypted = true, multi_attach_enabled = true, availability_zone = "us-east-1a" }
  }
  expect_failures = [aws_instance.qdrant]
}

run "foreign_subnet_is_blocked" {
  command = plan
  override_data {
    target = data.aws_subnet.alb["subnet-44444444444444444"]
    values = { vpc_id = "vpc-fffffffffffffffff", availability_zone = "us-east-1b" }
  }
  expect_failures = [aws_ecs_cluster.this]
}

run "single_zone_alb_is_blocked" {
  command = plan
  override_data {
    target = data.aws_subnet.alb["subnet-44444444444444444"]
    values = { vpc_id = "vpc-0123456789abcdef0", availability_zone = "us-east-1a" }
  }
  expect_failures = [aws_ecs_cluster.this]
}

run "plaintext_secrets_are_blocked" {
  command = plan
  variables { api_environment = { SCONE_API_KEY = "synthetic-only" } }
  expect_failures = [var.api_environment]
}

run "credential_url_is_blocked" {
  command = plan
  variables { api_environment = { SCONE_MONGO_URL = "mongodb://synthetic:example@db.invalid" } }
  expect_failures = [var.api_environment]
}

run "mutable_image_is_blocked" {
  command = plan
  variables { api_image = "123456789012.dkr.ecr.us-east-1.amazonaws.com/scone:latest" }
  expect_failures = [var.api_image]
}

run "ephemeral_blobs_are_blocked" {
  command = plan
  variables {
    api_desired_count              = 1
    runtime_configuration_verified = true
    qdrant_enabled                 = true
    api_environment                = { SCONE_DOCUMENTS = "mongo", SCONE_EVENTS = "mongo" }
    api_secret_arns                = { SCONE_API_KEY = "arn:aws:secretsmanager:us-east-1:123456789012:secret:api-example" }
  }
  expect_failures = [aws_ecs_service.api]
}

run "explicit_remote_runtime" {
  command = apply
  variables {
    api_desired_count              = 1
    runtime_configuration_verified = true
    qdrant_enabled                 = true
    api_environment = {
      SCONE_DOCUMENTS = "mongo", SCONE_EVENTS = "mongo", SCONE_BLOBS = "s3"
      SCONE_S3_BUCKET = "synthetic-scone-attachments", SCONE_DYNAMODB_BLOB_TABLE = "synthetic-blobs", SCONE_AWS_REGION = "us-east-1"
    }
    api_secret_arns = {
      SCONE_API_KEY   = "arn:aws:secretsmanager:us-east-1:123456789012:secret:api-example"
      SCONE_MONGO_URL = "arn:aws:secretsmanager:us-east-1:123456789012:secret:mongo-example"
    }
  }
  assert {
    condition     = aws_ecs_service.api.desired_count == 1 && aws_ecs_service.qdrant.desired_count == 1
    error_message = "Only an explicitly verified remote runtime may start both services."
  }
}

run "invalid_fargate_pair_is_blocked" {
  command = plan
  variables {
    api_cpu        = 256
    api_memory_mib = 4096
  }
  expect_failures = [aws_ecs_task_definition.api]
}

run "configured_task_sizes" {
  command = plan
  variables {
    api_cpu           = 2048
    api_memory_mib    = 4096
    qdrant_cpu        = 512
    qdrant_memory_mib = 1024
  }
  assert {
    condition     = aws_ecs_task_definition.api.cpu == "2048" && aws_ecs_task_definition.api.memory == "4096" && aws_ecs_task_definition.qdrant.cpu == "512" && aws_ecs_task_definition.qdrant.memory == "1024"
    error_message = "Task definitions must preserve configured resource limits."
  }
}
