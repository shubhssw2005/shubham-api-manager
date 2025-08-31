# Disaster Recovery Module Outputs

output "global_cluster_id" {
  description = "Aurora Global Cluster identifier"
  value       = var.enable_global_database ? aws_rds_global_cluster.main[0].id : null
}

output "primary_cluster_id" {
  description = "Primary Aurora cluster identifier"
  value       = var.enable_global_database ? aws_rds_cluster.primary[0].id : null
}

output "primary_cluster_endpoint" {
  description = "Primary Aurora cluster endpoint"
  value       = var.enable_global_database ? aws_rds_cluster.primary[0].endpoint : null
}

output "primary_cluster_reader_endpoint" {
  description = "Primary Aurora cluster reader endpoint"
  value       = var.enable_global_database ? aws_rds_cluster.primary[0].reader_endpoint : null
}

output "primary_s3_bucket_id" {
  description = "Primary S3 bucket ID"
  value       = aws_s3_bucket.primary.id
}

output "primary_s3_bucket_arn" {
  description = "Primary S3 bucket ARN"
  value       = aws_s3_bucket.primary.arn
}

output "dr_kms_key_id" {
  description = "Disaster recovery KMS key ID"
  value       = aws_kms_key.dr_key.key_id
}

output "dr_kms_key_arn" {
  description = "Disaster recovery KMS key ARN"
  value       = aws_kms_key.dr_key.arn
}

output "s3_replication_role_arn" {
  description = "S3 replication IAM role ARN"
  value       = aws_iam_role.s3_replication.arn
}

output "rds_monitoring_role_arn" {
  description = "RDS monitoring IAM role ARN"
  value       = aws_iam_role.rds_monitoring.arn
}