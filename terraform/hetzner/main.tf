resource "hcloud_server" "existing" {
  name               = var.server.name
  server_type        = var.server.server_type
  location           = var.server.location
  backups            = var.server.backups
  labels             = var.server.labels
  delete_protection  = var.server.delete_protection
  rebuild_protection = var.server.rebuild_protection
  firewall_ids       = []

  # Creation-only image, user_data and ssh_keys are intentionally omitted.
  # public_net remains unmanaged here; the existing Primary IP owns attachment.
  lifecycle {
    prevent_destroy = true
    ignore_changes  = [image, user_data, ssh_keys]
    postcondition {
      condition     = self.id == tostring(var.server.id) && self.ipv4_address == var.server.ipv4_address && self.primary_disk_size == var.server.primary_disk_size
      error_message = "The imported server identity, IPv4 or primary disk differs from the verified inventory."
    }
  }
}

resource "hcloud_primary_ip" "existing" {
  name              = var.primary_ipv4.name
  type              = "ipv4"
  assignee_type     = "server"
  assignee_id       = var.server.id
  auto_delete       = var.primary_ipv4.auto_delete
  delete_protection = var.primary_ipv4.delete_protection
  labels            = var.primary_ipv4.labels

  # An attached IP uses assignee_id, not location; the provider forbids both.
  lifecycle {
    prevent_destroy = true
    postcondition {
      condition     = self.id == var.primary_ipv4.id && self.ip_address == var.server.ipv4_address
      error_message = "The imported Primary IP identity or address differs from the verified inventory."
    }
  }
}

import {
  to = hcloud_server.existing
  id = tostring(var.server.id)
}

import {
  to = hcloud_primary_ip.existing
  id = tostring(var.primary_ipv4.id)
}
