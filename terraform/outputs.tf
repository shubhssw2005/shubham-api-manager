# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnet_ids
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnet_ids
}

# EKS Outputs
output "eks_cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "eks_cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "eks_node_groups" {
  description = "EKS node group information"
  value       = module.eks.node_groups
}

# Aurora Outputs
output "aurora_cluster_id" {
  description = "Aurora cluster ID"
  value       = module.aurora.cluster_id
}

output "aurora_cluster_endpoint" {
  description = "Aurora cluster endpoint"
  value       = module.aurora.cluster_endpoint
  sensitive   = true
}

output "aurora_reader_endpoint" {
  description = "Aurora reader endpoint"
  value       = module.aurora.reader_endpoint
  sensitive   = true
}

# Redis Outputs
output "redis_cluster_id" {
  description = "Redis cluster ID"
  value       = module.redis.cluster_id
}

output "redis_primary_endpoint" {
  description = "Redis primary endpoint"
  value       = module.redis.primary_endpoint
  sensitive   = true
}

output "redis_configuration_endpoint" {
  description = "Redis configuration endpoint"
  value       = module.redis.configuration_endpoint
  sensitive   = true
}

# S3 Outputs
output "s3_media_bucket" {
  description = "S3 media bucket name"
  value       = module.s3.media_bucket_name
}

output "s3_backup_bucket" {
  description = "S3 backup bucket name"
  value       = module.s3.backup_bucket_name
}

output "s3_logs_bucket" {
  description = "S3 logs bucket name"
  value       = module.s3.logs_bucket_name
}

# KMS Outputs
output "kms_key_id" {
  description = "KMS key ID"
  value       = module.kms.key_id
}

output "kms_key_arn" {
  description = "KMS key ARN"
  value       = module.kms.key_arn
}

# IAM Outputs
output "tenant_role_arn" {
  description = "IAM role ARN for tenant isolation"
  value       = module.iam.tenant_role_arn
}

output "eks_service_account_role_arn" {
  description = "IAM role ARN for EKS service accounts"
  value       = module.iam.eks_service_account_role_arn
}

# CloudFront Outputs
output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = module.cloudfront.distribution_id
}

output "cloudfront_distribution_arn" {
  description = "CloudFront distribution ARN"
  value       = module.cloudfront.distribution_arn
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  value       = module.cloudfront.distribution_domain_name
}

output "cloudfront_hosted_zone_id" {
  description = "CloudFront distribution hosted zone ID"
  value       = module.cloudfront.distribution_hosted_zone_id
}

output "cloudfront_public_key_id" {
  description = "CloudFront public key ID for signed URLs"
  value       = module.cloudfront.public_key_id
  sensitive   = true
}

output "cloudfront_key_group_id" {
  description = "CloudFront key group ID"
  value       = module.cloudfront.key_group_id
}

# SSL Certificate Outputs
output "ssl_certificate_arn" {
  description = "SSL certificate ARN for CloudFront"
  value       = module.acm_certificate.certificate_arn
}