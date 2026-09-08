variable "account_id" {
  type = string
  validation {
    condition     = can(regex("^[0-9]{12}$", var.account_id))
    error_message = "account_id must contain 12 digits."
  }
}

variable "aws_region" {
  type = string
  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-[0-9]+$", var.aws_region)) && !startswith(var.aws_region, "cn-")
    error_message = "Use a commercial AWS region."
  }
}

variable "name_prefix" {
  type = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,18}[a-z0-9]$", var.name_prefix))
    error_message = "name_prefix must be 3-20 lowercase alphanumeric/hyphen characters."
  }
}

variable "vpc_id" {
  type = string
  validation {
    condition     = can(regex("^vpc-[0-9a-f]{8,17}$", var.vpc_id))
    error_message = "vpc_id must be an AWS VPC ID."
  }
}

variable "api_subnet_ids" {
  description = "Existing private subnets with routed access to required AWS endpoints and configured remote backends."
  type        = set(string)
  validation {
    condition     = length(var.api_subnet_ids) >= 2 && alltrue([for id in var.api_subnet_ids : can(regex("^subnet-[0-9a-f]{8,17}$", id))])
    error_message = "Provide at least two private subnet IDs in different AZs."
  }
}

variable "alb_subnet_ids" {
  description = "Existing public subnets in different AZs for HTTPS ingress."
  type        = set(string)
  validation {
    condition     = length(var.alb_subnet_ids) >= 2 && alltrue([for id in var.alb_subnet_ids : can(regex("^subnet-[0-9a-f]{8,17}$", id))])
    error_message = "Provide at least two ALB subnet IDs in different AZs."
  }
}

variable "ingress_cidrs" {
  description = "IPv4 clients allowed to reach the HTTPS ALB."
  type        = set(string)
  validation {
    condition     = length(var.ingress_cidrs) > 0 && alltrue([for cidr in var.ingress_cidrs : can(cidrnetmask(cidr))])
    error_message = "Provide explicit IPv4 ingress CIDRs."
  }
}

variable "https_egress_cidrs" {
  description = "IPv4 networks reachable on HTTPS for ECR, logs, Secrets Manager and remote HTTPS backends. Use VPC endpoint CIDRs when available."
  type        = set(string)
  validation {
    condition     = length(var.https_egress_cidrs) > 0 && alltrue([for cidr in var.https_egress_cidrs : can(cidrnetmask(cidr))])
    error_message = "Provide explicit HTTPS egress CIDRs."
  }
}

variable "api_backend_egress" {
  description = "Additional explicit IPv4 TCP access for external SQL/document/model services; no ingress is created."
  type        = list(object({ cidr = string, port = number }))
  default     = []
  validation {
    condition     = length(var.api_backend_egress) <= 16 && alltrue([for rule in var.api_backend_egress : can(cidrnetmask(rule.cidr)) && rule.port >= 1 && rule.port <= 65535 && floor(rule.port) == rule.port])
    error_message = "Use at most 16 explicit IPv4 TCP endpoints with integer ports."
  }
}

variable "certificate_arn" {
  type = string
  validation {
    condition     = can(regex("^arn:aws:acm:[a-z0-9-]+:[0-9]{12}:certificate/[0-9a-f-]+$", var.certificate_arn))
    error_message = "certificate_arn must reference an existing ACM certificate."
  }
}

variable "api_image" {
  description = "Private ECR image in this account/region, pinned by sha256 digest."
  type        = string
  validation {
    condition     = can(regex("^[0-9]{12}\\.dkr\\.ecr\\.[a-z0-9-]+\\.amazonaws\\.com/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$", var.api_image))
    error_message = "api_image must be a private ECR digest reference."
  }
}

variable "api_architecture" {
  type    = string
  default = "ARM64"
  validation {
    condition     = contains(["ARM64", "X86_64"], var.api_architecture)
    error_message = "api_architecture must match the image: ARM64 or X86_64."
  }
}

variable "api_desired_count" {
  type    = number
  default = 0
  validation {
    condition     = floor(var.api_desired_count) == var.api_desired_count && var.api_desired_count >= 0 && var.api_desired_count <= 10
    error_message = "api_desired_count must be an integer from 0 to 10."
  }
}

variable "runtime_configuration_verified" {
  description = "Explicit operator assertion after testing durable documents/events, image drivers, backend readiness and model connectivity. Does not add missing adapters."
  type        = bool
  default     = false
}

variable "api_environment" {
  description = "Nonsecret SCONE settings only; document/event backends must be remote before tasks can start. This appears in Terraform state."
  type        = map(string)
  default     = {}
  validation {
    condition = alltrue([for name, value in var.api_environment :
      can(regex("^SCONE_[A-Z0-9_]+$", name)) && length(value) <= 4096 &&
      !can(regex("(?i)(KEY|TOKEN|PASSWORD|SECRET|CREDENTIAL)", name)) &&
      !can(regex("[\\x00-\\x1F]", value)) &&
      !can(regex("://[^/]*@", value)) &&
      !contains(["SCONE_HOST", "SCONE_PORT", "SCONE_VECTORS", "SCONE_QDRANT_URL", "SCONE_CONVERSATIONS_JOURNAL", "SCONE_MODEL_CONNECTIONS", "SCONE_LOG_PATH"], name)
    ])
    error_message = "Use bounded nonsecret SCONE settings, without reserved routing, local journals or local model settings. Put sensitive values in secret references."
  }
}

variable "api_secret_arns" {
  description = "Environment name -> existing Secrets Manager ARN. Only references enter state; never pass secret contents."
  type        = map(string)
  default     = {}
  validation {
    condition = alltrue([for name, arn in var.api_secret_arns :
      can(regex("^SCONE_[A-Z0-9_]+$", name)) && can(regex("^arn:aws:secretsmanager:[a-z0-9-]+:[0-9]{12}:secret:[A-Za-z0-9/_+=.@-]+$", arn)) &&
      !contains(["SCONE_HOST", "SCONE_PORT", "SCONE_VECTORS", "SCONE_QDRANT_URL", "SCONE_QDRANT_API_KEY", "SCONE_DOCUMENTS", "SCONE_EVENTS", "SCONE_CONVERSATIONS_JOURNAL", "SCONE_MODEL_CONNECTIONS", "SCONE_LOG_PATH"], name)
    ])
    error_message = "Use SCONE environment names and Secrets Manager ARNs only, without reserved routing/backend/local-state settings."
  }
}

variable "api_task_role_arn" {
  description = "Existing application role, for example the attachment foundation role. Execution permissions remain separate."
  type        = string
  validation {
    condition     = can(regex("^arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9/+=,.@_-]+$", var.api_task_role_arn))
    error_message = "api_task_role_arn must reference an IAM role."
  }
}

variable "secret_kms_key_arns" {
  description = "Customer-managed keys encrypting referenced secrets, if any; no key values."
  type        = set(string)
  default     = []
  validation {
    condition     = alltrue([for arn in var.secret_kms_key_arns : can(regex("^arn:aws:kms:[a-z0-9-]+:[0-9]{12}:key/[0-9a-f-]+$", arn))])
    error_message = "Use KMS key ARNs only."
  }
}

variable "qdrant_image" {
  description = "Reviewed Qdrant image mirrored to private ECR in this account/region and pinned by digest."
  type        = string
  validation {
    condition     = can(regex("^[0-9]{12}\\.dkr\\.ecr\\.[a-z0-9-]+\\.amazonaws\\.com/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$", var.qdrant_image))
    error_message = "qdrant_image must be a private ECR digest reference."
  }
}

variable "qdrant_secret_arn" {
  description = "Existing Secrets Manager secret containing the Qdrant API key."
  type        = string
  validation {
    condition     = can(regex("^arn:aws:secretsmanager:[a-z0-9-]+:[0-9]{12}:secret:[A-Za-z0-9/_+=.@-]+$", var.qdrant_secret_arn))
    error_message = "qdrant_secret_arn must be a Secrets Manager reference."
  }
}

variable "qdrant_subnet_id" {
  description = "Existing private subnet in the external EBS volume's AZ."
  type        = string
  validation {
    condition     = can(regex("^subnet-[0-9a-f]{8,17}$", var.qdrant_subnet_id))
    error_message = "qdrant_subnet_id must be a subnet ID."
  }
}

variable "qdrant_availability_zone" {
  type = string
  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-[0-9]+[a-z]$", var.qdrant_availability_zone))
    error_message = "Use the external EBS volume's commercial-region AZ."
  }
}

variable "qdrant_ami_id" {
  description = "Explicit reviewed Amazon Linux 2023 ECS-optimized Nitro AMI with cloud-init, systemd, lsblk, blkid and Docker. No AMI lookup occurs."
  type        = string
  validation {
    condition     = can(regex("^ami-[0-9a-f]{8,17}$", var.qdrant_ami_id))
    error_message = "Use an explicit reviewed ECS-optimized AMI ID."
  }
}

variable "qdrant_instance_type" {
  description = "Nitro instance type matching the AMI/Qdrant image architecture, with at least 4 GiB RAM."
  type        = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9]*[0-9][a-z0-9-]*\\.[a-z0-9]+$", var.qdrant_instance_type))
    error_message = "Use an explicit EC2 instance type."
  }
}

variable "qdrant_volume_id" {
  description = "Externally owned, encrypted, preformatted whole-device EBS volume. This module neither creates nor formats it."
  type        = string
  validation {
    condition     = can(regex("^vol-[0-9a-f]{8,17}$", var.qdrant_volume_id))
    error_message = "Use an external EBS volume ID."
  }
}

variable "qdrant_filesystem_uuid" {
  type = string
  validation {
    condition     = can(regex("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", var.qdrant_filesystem_uuid))
    error_message = "Provide the preformatted volume's filesystem UUID."
  }
}

variable "qdrant_filesystem_type" {
  type    = string
  default = "xfs"
  validation {
    condition     = contains(["xfs", "ext4"], var.qdrant_filesystem_type)
    error_message = "Use a prepared xfs or ext4 filesystem."
  }
}

variable "qdrant_enabled" {
  description = "Start the single Qdrant task only after external volume preparation and deployment review. The host is still created when false."
  type        = bool
  default     = false
}

variable "api_cpu" {
  description = "Fargate CPU units; this bounded module supports 0.25 through 4 vCPU."
  type        = number
  default     = 1024
  validation {
    condition     = contains([256, 512, 1024, 2048, 4096], var.api_cpu)
    error_message = "api_cpu must be 256, 512, 1024, 2048 or 4096."
  }
}

variable "api_memory_mib" {
  type    = number
  default = 2048
  validation {
    condition     = floor(var.api_memory_mib) == var.api_memory_mib && var.api_memory_mib >= 512 && var.api_memory_mib <= 30720
    error_message = "api_memory_mib must be an integer from 512 through 30720, paired with the selected Fargate CPU."
  }
}

variable "qdrant_cpu" {
  description = "EC2 task CPU units; operator must leave sufficient host capacity for the ECS agent and OS."
  type        = number
  default     = 1024
  validation {
    condition     = var.qdrant_cpu >= 256 && var.qdrant_cpu <= 16384 && var.qdrant_cpu % 256 == 0
    error_message = "qdrant_cpu must be a multiple of 256 from 256 through 16384."
  }
}

variable "qdrant_memory_mib" {
  description = "EC2 task hard memory limit; must fit the supplied host with OS/agent headroom."
  type        = number
  default     = 2048
  validation {
    condition     = var.qdrant_memory_mib >= 512 && var.qdrant_memory_mib <= 131072 && var.qdrant_memory_mib % 128 == 0
    error_message = "qdrant_memory_mib must be a multiple of 128 from 512 through 131072."
  }
}
