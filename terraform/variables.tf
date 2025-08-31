# Core variables
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "strapi-platform"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# VPC variables
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# EKS variables
variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "general_node_desired_size" {
  description = "Desired number of general purpose nodes"
  type        = number
  default     = 3
}

variable "general_node_max_size" {
  description = "Maximum number of general purpose nodes"
  type        = number
  default     = 10
}

variable "general_node_min_size" {
  description = "Minimum number of general purpose nodes"
  type        = number
  default     = 1
}

variable "memory_node_desired_size" {
  description = "Desired number of memory optimized nodes"
  type        = number
  default     = 2
}

variable "memory_node_max_size" {
  description = "Maximum number of memory optimized nodes"
  type        = number
  default     = 5
}

variable "memory_node_min_size" {
  description = "Minimum number of memory optimized nodes"
  type        = number
  default     = 0
}

# Aurora variables
variable "aurora_engine_version" {
  description = "Aurora PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

variable "aurora_instance_class" {
  description = "Aurora instance class"
  type        = string
  default     = "db.serverless"
}

variable "aurora_backup_retention_days" {
  description = "Number of days to retain Aurora backups"
  type        = number
  default     = 7
}

# Redis variables
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes in Redis cluster"
  type        = number
  default     = 3
}

# CloudFront variables
variable "cloudfront_custom_domains" {
  description = "List of custom domains for CloudFront distribution"
  type        = list(string)
  default     = []
}

variable "route53_zone_id" {
  description = "Route53 hosted zone ID for custom domain validation"
  type        = string
  default     = null
}

variable "cloudfront_public_key_content" {
  description = "Public key content for CloudFront signed URLs"
  type        = string
  sensitive   = true
}

variable "cloudfront_price_class" {
  description = "CloudFront price class"
  type        = string
  default     = "PriceClass_100"
  validation {
    condition = contains([
      "PriceClass_All",
      "PriceClass_200", 
      "PriceClass_100"
    ], var.cloudfront_price_class)
    error_message = "Price class must be one of: PriceClass_All, PriceClass_200, PriceClass_100."
  }
}

variable "cloudfront_geo_restriction_type" {
  description = "Type of geo restriction for CloudFront"
  type        = string
  default     = "none"
  validation {
    condition = contains([
      "none",
      "whitelist",
      "blacklist"
    ], var.cloudfront_geo_restriction_type)
    error_message = "Geo restriction type must be one of: none, whitelist, blacklist."
  }
}

variable "cloudfront_geo_restriction_locations" {
  description = "List of country codes for CloudFront geo restriction"
  type        = list(string)
  default     = []
}#
 CloudFront vari# Securit
y variables
variable "enable_shield_advanced" {
  description = "Enable AWS Shield Advanced protection"
  type        = bool
  default     = false
}

variable "blocked_countries" {
  description = "List of country codes to block in WAF"
  type        = list(string)
  default     = ["CN", "RU", "KP"]
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for security alerts"
  type        = string
  default     = ""
}

variable "replica_region" {
  description = "AWS region for secret replication"
  type        = string
  default     = "us-west-2"
}

# Database credentials
variable "db_username" {
  description = "Database master username"
  type        = string
  sensitive   = true
  default     = "postgres"
}

variable "db_password" {
  description = "Database master password"
  type        = string
  sensitive   = true
}

# JWT secrets
variable "jwt_access_secret" {
  description = "JWT access token secret"
  type        = string
  sensitive   = true
}

variable "jwt_refresh_secret" {
  description = "JWT refresh token secret"
  type        = string
  sensitive   = true
}

variable "jwt_encryption_key" {
  description = "JWT encryption key"
  type        = string
  sensitive   = true
}

# External API keys
variable "stripe_secret_key" {
  description = "Stripe secret key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "sendgrid_api_key" {
  description = "SendGrid API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "cloudflare_api_token" {
  description = "Cloudflare API token"
  type        = string
  sensitive   = true
  default     = ""
}

variable "github_webhook_secret" {
  description = "GitHub webhook secret"
  type        = string
  sensitive   = true
  default     = ""
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  sensitive   = true
  default     = ""
}