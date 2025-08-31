# API Gateway Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where resources will be created"
  type        = string
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for Lambda functions"
  type        = list(string)
}

variable "backend_url" {
  description = "Backend service URL for proxy integration"
  type        = string
}

variable "jwt_secret" {
  description = "JWT secret for token validation"
  type        = string
  sensitive   = true
}

variable "redis_cluster_endpoint" {
  description = "Redis cluster endpoint for rate limiting"
  type        = string
}

variable "allowed_ip_ranges" {
  description = "List of allowed IP ranges for API access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "rate_limit_per_second" {
  description = "Rate limit per second for API Gateway"
  type        = number
  default     = 1000
}

variable "burst_limit" {
  description = "Burst limit for API Gateway"
  type        = number
  default     = 2000
}

variable "monthly_quota_limit" {
  description = "Monthly quota limit for usage plan"
  type        = number
  default     = 1000000
}

variable "rate_limit_window" {
  description = "Rate limit window in seconds"
  type        = number
  default     = 3600
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "log_level" {
  description = "Log level for Lambda functions"
  type        = string
  default     = "INFO"
  
  validation {
    condition     = contains(["DEBUG", "INFO", "WARN", "ERROR"], var.log_level)
    error_message = "Log level must be one of: DEBUG, INFO, WARN, ERROR."
  }
}

# Rate limiting tiers
variable "rate_limit_tiers" {
  description = "Rate limiting configuration for different tenant tiers"
  type = map(object({
    requests_per_hour = number
    burst_limit      = number
  }))
  default = {
    free = {
      requests_per_hour = 1000
      burst_limit      = 50
    }
    pro = {
      requests_per_hour = 10000
      burst_limit      = 200
    }
    enterprise = {
      requests_per_hour = 100000
      burst_limit      = 1000
    }
  }
}

# Request validation settings
variable "max_request_size" {
  description = "Maximum request size in bytes"
  type        = number
  default     = 10485760 # 10MB
}

variable "allowed_content_types" {
  description = "List of allowed content types"
  type        = list(string)
  default = [
    "application/json",
    "application/x-www-form-urlencoded",
    "multipart/form-data",
    "text/plain"
  ]
}

# Security settings
variable "enable_xray_tracing" {
  description = "Enable X-Ray tracing for API Gateway"
  type        = bool
  default     = true
}

variable "enable_waf" {
  description = "Enable WAF for API Gateway"
  type        = bool
  default     = true
}

variable "cors_allowed_origins" {
  description = "List of allowed origins for CORS"
  type        = list(string)
  default     = ["*"]
}

variable "cors_allowed_headers" {
  description = "List of allowed headers for CORS"
  type        = list(string)
  default = [
    "Content-Type",
    "X-Amz-Date",
    "Authorization",
    "X-Api-Key",
    "X-Amz-Security-Token",
    "X-Tenant-ID"
  ]
}

variable "cors_allowed_methods" {
  description = "List of allowed methods for CORS"
  type        = list(string)
  default = [
    "DELETE",
    "GET",
    "HEAD",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT"
  ]
}