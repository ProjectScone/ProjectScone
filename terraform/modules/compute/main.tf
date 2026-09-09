locals {
  cluster_name  = "${var.name_prefix}-compute"
  cluster_arn   = "arn:aws:ecs:${var.aws_region}:${var.account_id}:cluster/${local.cluster_name}"
  instance_arns = "arn:aws:ecs:${var.aws_region}:${var.account_id}:container-instance/${local.cluster_name}/*"
  dns_name      = "${var.name_prefix}.internal"
  qdrant_url    = "http://qdrant.${local.dns_name}:6333"
  ecr_prefix    = "${var.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/"
  repositories  = toset([for image in [var.api_image, var.qdrant_image] : "arn:aws:ecr:${var.aws_region}:${var.account_id}:repository/${join("/", slice(split("/", split("@", image)[0]), 1, length(split("/", split("@", image)[0]))))}"])
  secret_arns   = toset(concat(values(var.api_secret_arns), [var.qdrant_secret_arn]))
  api_environment = merge({
    SCONE_DOCUMENTS = "deployment-unconfigured"
    SCONE_EVENTS    = "deployment-unconfigured"
    }, var.api_environment, {
    SCONE_HOST       = "0.0.0.0"
    SCONE_PORT       = "7437"
    SCONE_VECTORS    = "qdrant"
    SCONE_QDRANT_URL = local.qdrant_url
  })
  api_secrets = merge(var.api_secret_arns, { SCONE_QDRANT_API_KEY = var.qdrant_secret_arn })
  tags        = { Application = "Scone", Component = "compute", ManagedBy = "Terraform" }
}

# Metadata only, and mocked in tests. No secret contents are read into state.
data "aws_subnet" "api" {
  for_each = var.api_subnet_ids
  id       = each.value
}

data "aws_subnet" "alb" {
  for_each = var.alb_subnet_ids
  id       = each.value
}

data "aws_subnet" "qdrant" { id = var.qdrant_subnet_id }

data "aws_ebs_volume" "qdrant" {
  filter {
    name   = "volume-id"
    values = [var.qdrant_volume_id]
  }
}

resource "aws_ecs_cluster" "this" {
  name = local.cluster_name
  tags = local.tags
  lifecycle {
    precondition {
      condition     = alltrue([for subnet in concat(values(data.aws_subnet.api), values(data.aws_subnet.alb), [data.aws_subnet.qdrant]) : subnet.vpc_id == var.vpc_id])
      error_message = "Every supplied subnet must belong to the supplied VPC."
    }
    precondition {
      condition     = length(toset([for subnet in values(data.aws_subnet.api) : subnet.availability_zone])) >= 2 && length(toset([for subnet in values(data.aws_subnet.alb) : subnet.availability_zone])) >= 2
      error_message = "API and ALB subnet sets must each span at least two AZs."
    }
    precondition {
      condition     = startswith(var.api_image, local.ecr_prefix) && startswith(var.qdrant_image, local.ecr_prefix)
      error_message = "Both reviewed image digests must be in this account and region's ECR registry."
    }
    precondition {
      condition     = startswith(var.certificate_arn, "arn:aws:acm:${var.aws_region}:${var.account_id}:") && startswith(var.api_task_role_arn, "arn:aws:iam::${var.account_id}:")
      error_message = "The ACM certificate and application task role must belong to the configured account/region."
    }
  }
}

resource "aws_service_discovery_private_dns_namespace" "this" {
  name = local.dns_name
  vpc  = var.vpc_id
  tags = local.tags
}

resource "aws_service_discovery_service" "qdrant" {
  name = "qdrant"
  dns_config {
    namespace_id   = aws_service_discovery_private_dns_namespace.this.id
    routing_policy = "MULTIVALUE"
    dns_records {
      ttl  = 10
      type = "A"
    }
  }
  health_check_custom_config {}
  tags = local.tags
}

resource "aws_lb" "api" {
  name                       = "${var.name_prefix}-api"
  internal                   = false
  load_balancer_type         = "application"
  subnets                    = var.alb_subnet_ids
  security_groups            = [aws_security_group.alb.id]
  drop_invalid_header_fields = true
  enable_deletion_protection = true
  tags                       = local.tags
}

resource "aws_lb_target_group" "api" {
  name        = "${var.name_prefix}-api"
  port        = 7437
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id
  health_check {
    path                = "/healthz"
    matcher             = "200"
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
  deregistration_delay = 30
  tags                 = local.tags
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

resource "aws_ecs_task_definition" "api" {
  lifecycle {
    precondition {
      condition = contains(lookup({
        "256"  = [512, 1024, 2048]
        "512"  = [1024, 2048, 3072, 4096]
        "1024" = range(2048, 8193, 1024)
        "2048" = range(4096, 16385, 1024)
        "4096" = range(8192, 30721, 1024)
      }, tostring(var.api_cpu), []), var.api_memory_mib)
      error_message = "API CPU and memory must be a valid AWS Fargate pair."
    }
  }

  family                   = "${var.name_prefix}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.api_cpu)
  memory                   = tostring(var.api_memory_mib)
  task_role_arn            = var.api_task_role_arn
  execution_role_arn       = aws_iam_role.execution.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = var.api_architecture
  }
  container_definitions = jsonencode([{
    name                   = "api"
    image                  = var.api_image
    essential              = true
    user                   = "10001:10001"
    readonlyRootFilesystem = true
    portMappings           = [{ containerPort = 7437, protocol = "tcp" }]
    environment            = [for name, value in local.api_environment : { name = name, value = value }]
    secrets                = [for name, arn in local.api_secrets : { name = name, valueFrom = arn }]
    linuxParameters        = { capabilities = { drop = ["ALL"] }, initProcessEnabled = true }
    healthCheck            = { command = ["CMD", "python", "/opt/healthcheck.py"], interval = 30, timeout = 5, retries = 3, startPeriod = 30 }
    logConfiguration = {
      logDriver = "awslogs"
      options   = { awslogs-group = aws_cloudwatch_log_group.api.name, awslogs-region = var.aws_region, awslogs-stream-prefix = "api" }
    }
  }])
  tags = local.tags
}

resource "aws_ecs_service" "api" {
  name                              = "api"
  cluster                           = aws_ecs_cluster.this.id
  task_definition                   = aws_ecs_task_definition.api.arn
  launch_type                       = "FARGATE"
  desired_count                     = var.api_desired_count
  health_check_grace_period_seconds = 60
  enable_ecs_managed_tags           = true
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  network_configuration {
    subnets          = var.api_subnet_ids
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = false
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 7437
  }
  depends_on = [aws_lb_listener.https, aws_iam_role_policy.execution]
  lifecycle {
    precondition {
      condition = var.api_desired_count == 0 || (
        var.runtime_configuration_verified && var.qdrant_enabled &&
        contains(["mongo", "postgres", "elasticsearch"], local.api_environment.SCONE_DOCUMENTS) &&
        contains(["mongo", "postgres", "elasticsearch"], local.api_environment.SCONE_EVENTS) &&
        lookup(local.api_environment, "SCONE_BLOBS", "") == "s3" &&
        alltrue([for key in ["SCONE_S3_BUCKET", "SCONE_DYNAMODB_BLOB_TABLE", "SCONE_AWS_REGION"] : length(lookup(local.api_environment, key, "")) > 0]) &&
        (contains(keys(var.api_secret_arns), "SCONE_API_KEY") || contains(keys(var.api_secret_arns), "SCONE_API_KEYS"))
      )
      error_message = "Starting API tasks requires explicit verified runtime configuration, enabled Qdrant, durable remote documents/events, S3 attachment configuration and an API-key secret reference. DynamoDB documents/journals are not implemented."
    }
    precondition {
      condition     = length(setintersection(toset(keys(var.api_environment)), toset(keys(local.api_secrets)))) == 0
      error_message = "Environment settings and secret references must not define the same name."
    }
  }
}
