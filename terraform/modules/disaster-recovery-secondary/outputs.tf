# Disaster Recovery Secondary Region Module Outputs

output "secondary_cluster_id" {
  description = "Secondary Aurora cluster identifier"
  value       = var.enable_secondary_cluster ? aws_rds_cluster.secondary[0].id : null
}

output "secondary_cluster_endpoint" {
  description = "Secondary Aurora cluster endpoint"
  value       = var.enable_secondary_cluster ? aws_rds_cluster.secondary[0].endpoint : null
}

output "secondary_cluster_reader_endpoint" {
  description = "Secondary Aurora cluster reader endpoint"
  value       = var.enable_secondary_cluster ? aws_rds_cluster.secondary[0].reader_endpoint : null
}

output "replica_s3_bucket_id" {
  description = "Replica S3 bucket ID"
  value       = aws_s3_bucket.replica.id
}

output "replica_s3_bucket_arn" {
  description = "Replica S3 bucket ARN"
  value       = aws_s3_bucket.replica.arn
}

output "dr_secondary_kms_key_id" {
  description = "Disaster recovery secondary KMS key ID"
  value       = aws_kms_key.dr_secondary_key.key_id
}

output "dr_secondary_kms_key_arn" {
  description = "Disaster recovery secondary KMS key ARN"
  value       = aws_kms_key.dr_secondary_key.arn
}

output "failover_lambda_function_name" {
  description = "Failover automation Lambda function name"
  value       = var.enable_failover_automation ? aws_lambda_function.failover_automation[0].function_name : null
}

output "failover_lambda_function_arn" {
  description = "Failover automation Lambda function ARN"
  value       = var.enable_failover_automation ? aws_lambda_function.failover_automation[0].arn : null
}