mock_provider "hcloud" {}

override_resource {
  target = hcloud_server.existing
  values = {
    id                = "1001"
    ipv4_address      = "192.0.2.10"
    primary_disk_size = 80
  }
}

override_resource {
  target = hcloud_primary_ip.existing
  values = {
    id         = 2001
    ip_address = "192.0.2.10"
  }
}

variables {
  server = {
    id           = 1001, name = "synthetic-server", server_type = "cpx22", location = "hel1",
    ipv4_address = "192.0.2.10", primary_disk_size = 80,
    backups      = false, delete_protection = false, rebuild_protection = false, labels = {}
  }
  primary_ipv4 = {
    id = 2001, name = "synthetic-ip", auto_delete = true, delete_protection = false, labels = {}
  }
}

run "preserve_observed_adoption_settings" {
  command = plan
  assert {
    condition     = hcloud_server.existing.name == var.server.name && hcloud_server.existing.server_type == var.server.server_type && hcloud_server.existing.location == var.server.location
    error_message = "Adoption must preserve observed server sizing and placement."
  }
  assert {
    condition     = !hcloud_server.existing.backups && !hcloud_server.existing.delete_protection && !hcloud_server.existing.rebuild_protection && length(hcloud_server.existing.firewall_ids) == 0
    error_message = "Adoption must not enable services, change protections or attach a firewall."
  }
  assert {
    condition     = hcloud_primary_ip.existing.assignee_id == var.server.id && hcloud_primary_ip.existing.assignee_type == "server" && hcloud_primary_ip.existing.type == "ipv4" && hcloud_primary_ip.existing.auto_delete && !hcloud_primary_ip.existing.delete_protection
    error_message = "Adoption must retain existing IP attachment and policy."
  }
}

run "reject_non_ipv4_inventory" {
  command = plan
  variables {
    server = {
      id           = 1001, name = "synthetic-server", server_type = "cpx22", location = "hel1",
      ipv4_address = "2001:db8::1", primary_disk_size = 80,
      backups      = false, delete_protection = false, rebuild_protection = false, labels = {}
    }
  }
  expect_failures = [var.server]
}

run "preserve_enabled_policies_and_labels" {
  command = plan
  variables {
    server = {
      id           = 1001, name = "synthetic-server", server_type = "cpx22", location = "hel1",
      ipv4_address = "192.0.2.10", primary_disk_size = 80,
      backups      = true, delete_protection = true, rebuild_protection = true, labels = { existing = "retained" }
    }
    primary_ipv4 = {
      id = 2001, name = "synthetic-ip", auto_delete = false, delete_protection = true, labels = { existing = "retained" }
    }
  }
  assert {
    condition     = hcloud_server.existing.backups && hcloud_server.existing.delete_protection && hcloud_server.existing.rebuild_protection && hcloud_server.existing.labels["existing"] == "retained" && !hcloud_primary_ip.existing.auto_delete && hcloud_primary_ip.existing.delete_protection && hcloud_primary_ip.existing.labels["existing"] == "retained"
    error_message = "Already enabled policies and labels must survive adoption."
  }
}

run "reject_wrong_import_identity" {
  command = plan
  variables {
    primary_ipv4 = {
      id = 0, name = "synthetic-ip", auto_delete = true, delete_protection = false, labels = {}
    }
  }
  expect_failures = [var.primary_ipv4]
}
