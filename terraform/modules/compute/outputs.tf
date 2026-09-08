output "cluster_arn" { value = aws_ecs_cluster.this.arn }
output "api_service_name" { value = aws_ecs_service.api.name }
output "api_alb_dns_name" { value = aws_lb.api.dns_name }
output "qdrant_internal_url" { value = local.qdrant_url }
output "qdrant_instance_id" { value = aws_instance.qdrant.id }
output "qdrant_external_volume_id" { value = var.qdrant_volume_id }
output "api_enabled" { value = var.api_desired_count > 0 }
