# CloudFront Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for media storage"
  type        = string
}

variable "s3_bucket_domain_name" {
  description = "Domain name of the S3 bucket"
  type        = string
}

variable "api_gateway_domain" {
  description = "Domain name of the API Gateway"
  type        = string
  default     = ""
}

variable "custom_domains" {
  description = "List of custom domain names for the CloudFront distribution"
  type        = list(string)
  default     = []
}

variable "ssl_certificate_arn" {
  description = "ARN of the SSL certificate for custom domains"
  type        = string
  default     = ""
}

variable "price_class" {
  description = "CloudFront price class"
  type        = string
  default     = "PriceClass_100"
  
  validation {
    condition = contains([
      "PriceClass_All",
      "PriceClass_200", 
      "PriceClass_100"
    ], var.price_class)
    error_message = "Price class must be PriceClass_All, PriceClass_200, or PriceClass_100."
  }
}

variable "geo_restriction_type" {
  description = "Type of geographic restriction (none, whitelist, blacklist)"
  type        = string
  default     = "none"
  
  validation {
    condition = contains([
      "none",
      "whitelist",
      "blacklist"
    ], var.geo_restriction_type)
    error_message = "Geo restriction type must be none, whitelist, or blacklist."
  }
}

variable "geo_restriction_locations" {
  description = "List of country codes for geographic restrictions"
  type        = list(string)
  default     = []
}

variable "logging_bucket" {
  description = "S3 bucket for CloudFront access logs"
  type        = string
  default     = ""
}

variable "waf_web_acl_arn" {
  description = "ARN of the WAF Web ACL to associate with CloudFront"
  type        = string
  default     = ""
}

variable "public_key_content" {
  description = "Content of the public key for signed URLs"
  type        = string
}

variable "cors_allowed_origins" {
  description = "List of allowed origins for CORS"
  type        = list(string)
  default     = ["*"]
}

variable "cache_behaviors" {
  description = "Additional cache behaviors for the distribution"
  type = list(object({
    path_pattern           = string
    target_origin_id       = string
    viewer_protocol_policy = string
    allowed_methods        = list(string)
    cached_methods         = list(string)
    compress               = bool
    ttl_settings = object({
      default_ttl = number
      max_ttl     = number
      min_ttl     = number
    })
  }))
  default = []
}

variable "enable_ipv6" {
  description = "Enable IPv6 support for CloudFront distribution"
  type        = bool
  default     = true
}

variable "retain_on_delete" {
  description = "Retain the distribution when destroying the resource"
  type        = bool
  default     = false
}

variable "wait_for_deployment" {
  description = "Wait for the distribution deployment to complete"
  type        = bool
  default     = true
}