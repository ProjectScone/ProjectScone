locals {
  volume_mount = "/var/lib/scone-qdrant"
  cloud_config = {
    bootcmd = [["cloud-init-per", "once", "scone-ecs-gate", "systemctl", "mask", "ecs.service"]]
    write_files = [
      { path = "/usr/local/libexec/scone_mount.py", permissions = "0755", content = file("${path.module}/mount_volume.py") },
      {
        path    = "/etc/systemd/system/scone-qdrant-volume.service", permissions = "0644"
        content = <<-UNIT
          [Unit]
          Description=Verify and mount the owned Qdrant EBS filesystem
          Before=ecs.service
          [Service]
          Type=oneshot
          RemainAfterExit=yes
          TimeoutStartSec=660
          ExecStart=/usr/bin/python3 /usr/local/libexec/scone_mount.py ${var.qdrant_volume_id} ${var.qdrant_filesystem_uuid} ${var.qdrant_filesystem_type}
          [Install]
          WantedBy=multi-user.target
        UNIT
      },
      {
        path    = "/etc/systemd/system/ecs.service.d/scone-volume.conf", permissions = "0644"
        content = <<-UNIT
          [Unit]
          Requires=scone-qdrant-volume.service
          After=scone-qdrant-volume.service
          [Service]
          ExecStartPre=/usr/bin/mountpoint --quiet ${local.volume_mount}
        UNIT
      },
      {
        path    = "/etc/ecs/ecs.config", permissions = "0600"
        content = <<-CONFIG
          ECS_CLUSTER=${local.cluster_name}
          ECS_INSTANCE_ATTRIBUTES={"scone.qdrant_volume":"${var.qdrant_volume_id}"}
          ECS_ENABLE_AWSLOGS_EXECUTIONROLE_OVERRIDE=true
          ECS_AWSVPC_BLOCK_IMDS=true
          ECS_AVAILABLE_LOGGING_DRIVERS=["json-file","awslogs"]
        CONFIG
      }
    ]
    runcmd = [
      ["systemctl", "daemon-reload"],
      ["systemctl", "unmask", "ecs.service"],
      ["systemctl", "enable", "scone-qdrant-volume.service", "ecs.service"],
      ["systemctl", "--no-block", "start", "ecs.service"]
    ]
  }
}

resource "aws_instance" "qdrant" {
  ami                         = var.qdrant_ami_id
  instance_type               = var.qdrant_instance_type
  subnet_id                   = var.qdrant_subnet_id
  availability_zone           = var.qdrant_availability_zone
  associate_public_ip_address = false
  iam_instance_profile        = aws_iam_instance_profile.host.name
  vpc_security_group_ids      = [aws_security_group.host.id]
  user_data                   = "#cloud-config\n${yamlencode(local.cloud_config)}"
  user_data_replace_on_change = true
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }
  root_block_device {
    encrypted             = true
    volume_type           = "gp3"
    volume_size           = 30
    delete_on_termination = true
  }
  tags       = merge(local.tags, { Name = "${var.name_prefix}-qdrant" })
  depends_on = [aws_iam_role_policy.host]
  lifecycle {
    prevent_destroy = true
    precondition {
      condition     = data.aws_ebs_volume.qdrant.encrypted && !data.aws_ebs_volume.qdrant.multi_attach_enabled && data.aws_ebs_volume.qdrant.availability_zone == var.qdrant_availability_zone && data.aws_subnet.qdrant.availability_zone == var.qdrant_availability_zone
      error_message = "The external data volume must be encrypted, prohibit multi-attach and share the host subnet's exact availability zone."
    }
  }
}

resource "aws_volume_attachment" "qdrant" {
  device_name                    = "/dev/sdf"
  volume_id                      = var.qdrant_volume_id
  instance_id                    = aws_instance.qdrant.id
  force_detach                   = false
  stop_instance_before_detaching = true
  lifecycle { prevent_destroy = true }
}

resource "aws_ecs_task_definition" "qdrant" {
  family                   = "${var.name_prefix}-qdrant"
  requires_compatibilities = ["EC2"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.qdrant_cpu)
  memory                   = tostring(var.qdrant_memory_mib)
  task_role_arn            = aws_iam_role.qdrant_task.arn
  execution_role_arn       = aws_iam_role.execution.arn
  volume {
    name      = "qdrant-data"
    host_path = local.volume_mount
  }
  placement_constraints {
    type       = "memberOf"
    expression = "attribute:scone.qdrant_volume == ${var.qdrant_volume_id}"
  }
  container_definitions = jsonencode([{
    name                   = "qdrant"
    image                  = var.qdrant_image
    essential              = true
    user                   = "10001:10001"
    readonlyRootFilesystem = true
    portMappings           = [{ containerPort = 6333, protocol = "tcp" }]
    mountPoints            = [{ sourceVolume = "qdrant-data", containerPath = "/qdrant/data", readOnly = false }]
    environment = [
      { name = "QDRANT__STORAGE__STORAGE_PATH", value = "/qdrant/data/storage" },
      { name = "QDRANT__STORAGE__SNAPSHOTS_PATH", value = "/qdrant/data/snapshots" },
      { name = "QDRANT__TELEMETRY_DISABLED", value = "true" }
    ]
    secrets         = [{ name = "QDRANT__SERVICE__API_KEY", valueFrom = var.qdrant_secret_arn }]
    linuxParameters = { capabilities = { drop = ["ALL"] }, initProcessEnabled = true, tmpfs = [{ containerPath = "/tmp", size = 32, mountOptions = ["rw", "nosuid", "noexec"] }] }
    healthCheck = {
      command  = ["CMD-SHELL", "bash -c 'exec 3<>/dev/tcp/127.0.0.1/6333; printf \"GET /readyz HTTP/1.1\\r\\nHost: localhost\\r\\nConnection: close\\r\\n\\r\\n\" >&3; read -r status <&3; [[ \"$status\" == *\" 200 \"* ]]'"]
      interval = 30, timeout = 5, retries = 3, startPeriod = 60
    }
    logConfiguration = {
      logDriver = "awslogs"
      options   = { awslogs-group = aws_cloudwatch_log_group.qdrant.name, awslogs-region = var.aws_region, awslogs-stream-prefix = "qdrant" }
    }
  }])
  tags = local.tags
}

resource "aws_ecs_service" "qdrant" {
  name                               = "qdrant"
  cluster                            = aws_ecs_cluster.this.id
  task_definition                    = aws_ecs_task_definition.qdrant.arn
  launch_type                        = "EC2"
  desired_count                      = var.qdrant_enabled ? 1 : 0
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100
  enable_ecs_managed_tags            = true
  deployment_circuit_breaker {
    enable   = true
    rollback = false
  }
  network_configuration {
    subnets         = [var.qdrant_subnet_id]
    security_groups = [aws_security_group.qdrant.id]
  }
  service_registries { registry_arn = aws_service_discovery_service.qdrant.arn }
  depends_on = [aws_volume_attachment.qdrant, aws_iam_role_policy.execution]
}
