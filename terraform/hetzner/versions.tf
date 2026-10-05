terraform {
  required_version = ">= 1.16.1, < 2.0"
  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "= 1.68.0"
    }
  }
}

# Authentication uses HCLOUD_TOKEN in the operator's process environment only.
provider "hcloud" {}
