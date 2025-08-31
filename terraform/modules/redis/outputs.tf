output "cluster_id" {
  description = "Redis cluster ID"
  value       = aws_elasticache_replication_group.redis.id
}

output "primary_endpoint" {
  description = "Redis primary endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "configuration_endpoint" {
  description = "Redis configuration endpoint"
  value       = aws_elasticache_replication_group.redis.configuration_endpoint_address
}

output "port" {
  description = "Redis port"
  value       = aws_elasticache_replication_group.redis.port
}

output "security_group_id" {
  description = "Redis security group ID"
  value       = aws_security_group.redis.id
}

output "secrets_manager_secret_arn" {
  description = "ARN of the Secrets Manager secret containing Redis credentials"
  value       = aws_secretsmanager_secret.redis_credentials.arn
}

output "sns_topic_arn" {
  description = "ARN of the SNS topic for Redis notifications"
  value       = aws_sns_topic.redis_notifications.arn
}