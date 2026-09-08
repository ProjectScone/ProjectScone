variable "account_id" {
  description = "Target AWS account ID; explicit to avoid identity lookups during validation."
  type        = string
  validation {
    condition     = can(regex("^[0-9]{12}$", var.account_id))
    error_message = "account_id must contain exactly 12 digits."
  }
}

variable "aws_region" {
  type = string
  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-[0-9]+$", var.aws_region)) && !startswith(var.aws_region, "cn-")
    error_message = "Use a commercial AWS region; this foundation uses the aws partition."
  }
}

variable "name_prefix" {
  type = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,30}[a-z0-9]$", var.name_prefix))
    error_message = "name_prefix must be 3-32 lowercase letters, digits or hyphens, starting with a letter and ending alphanumeric."
  }
}

variable "bucket_name" {
  description = "Globally unique, private attachment bucket name; no bucket is inferred or reused."
  type        = string
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$", var.bucket_name)) && !startswith(var.bucket_name, "xn--") && !startswith(var.bucket_name, "sthree-") && !startswith(var.bucket_name, "amzn-s3-demo-") && !endswith(var.bucket_name, "--x-s3") && !endswith(var.bucket_name, "--table-s3") && !endswith(var.bucket_name, "-s3alias") && !endswith(var.bucket_name, "--ol-s3")
    error_message = "Use a valid 3-63 character S3 bucket name containing lowercase letters, digits and hyphens, without reserved prefixes/suffixes."
  }
}

variable "s3_prefix" {
  type = string
  validation {
    condition     = length(var.s3_prefix) <= 128 && can(regex("^([A-Za-z0-9_-]+/)+$", var.s3_prefix))
    error_message = "s3_prefix must be a nonempty slash-terminated path of letters, digits, underscores and hyphens, at most 128 characters."
  }
}

variable "blob_table_name" {
  description = "Attachment metadata table name, matching SCONE_DYNAMODB_BLOB_TABLE."
  type        = string
  validation {
    condition     = can(regex("^[A-Za-z0-9_.-]{3,255}$", var.blob_table_name))
    error_message = "blob_table_name must be a valid 3-255 character DynamoDB table name."
  }
}

variable "tags" {
  type    = map(string)
  default = {}
}
