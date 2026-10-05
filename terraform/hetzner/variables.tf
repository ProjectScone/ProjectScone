variable "server" {
  description = "Observed existing server inventory. Supply private local inputs; do not infer or change settings during adoption."
  type = object({
    id                 = number
    name               = string
    server_type        = string
    location           = string
    ipv4_address       = string
    primary_disk_size  = number
    backups            = bool
    delete_protection  = bool
    rebuild_protection = bool
    labels             = map(string)
  })
  nullable = false
  validation {
    condition     = var.server.id > 0 && floor(var.server.id) == var.server.id && var.server.primary_disk_size > 0 && floor(var.server.primary_disk_size) == var.server.primary_disk_size
    error_message = "Use observed positive integer server ID and primary disk size."
  }
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$", var.server.name)) && can(regex("^[a-z0-9-]+$", var.server.server_type)) && can(regex("^[a-z0-9]+$", var.server.location))
    error_message = "Use observed server name, server type and location identifiers."
  }
  validation {
    condition     = can(cidrnetmask("${var.server.ipv4_address}/32")) && try(cidrhost("${var.server.ipv4_address}/32", 0) == var.server.ipv4_address, false)
    error_message = "Use the existing canonical IPv4 address, without a prefix."
  }
  validation {
    condition     = var.server.delete_protection == var.server.rebuild_protection
    error_message = "The provider requires the observed delete and rebuild protection settings to match."
  }
}

variable "primary_ipv4" {
  description = "Observed attached IPv4 Primary IP inventory. Its numeric resource ID is not its IP address."
  type = object({
    id                = number
    name              = string
    auto_delete       = bool
    delete_protection = bool
    labels            = map(string)
  })
  nullable = false
  validation {
    condition     = var.primary_ipv4.id > 0 && floor(var.primary_ipv4.id) == var.primary_ipv4.id && length(var.primary_ipv4.name) > 0 && length(var.primary_ipv4.name) <= 128
    error_message = "Use the existing numeric Primary IP ID and its observed name."
  }
}
