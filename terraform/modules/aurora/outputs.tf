output "cluster_id" {
  description = "Aurora cluster ID"
  value       = aws_rds_cluster.aurora.id
}

output "cluster_arn" {
  description = "Aurora cluster ARN"
  value       = aws_rds_cluster.aurora.arn
}

output "cluster_endpoint" {
  description = "Aurora cluster endpoint"
  value       = aws_rds_cluster.aurora.endpoint
}

output "reader_endpoint" {
  description = "Aurora reader endpoint"
  value       = aws_rds_cluster.aurora.reader_endpoint
}

output "cluster_port" {
  description = "Aurora cluster port"
  value       = aws_rds_cluster.aurora.port
}

output "database_name" {
  description = "Aurora database name"
  value       = aws_rds_cluster.aurora.database_name
}

output "master_username" {
  description = "Aurora master username"
  value       = aws_rds_cluster.aurora.master_username
  sensitive   = true
}

output "security_group_id" {
  description = "Aurora security group ID"
  value       = aws_security_group.aurora.id
}

output "secrets_manager_secret_arn" {
  description = "ARN of the Secrets Manager secret containing Aurora credentials"
  value       = aws_secretsmanager_secret.aurora_credentials.arn
}