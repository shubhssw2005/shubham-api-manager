# CloudFront Configuration Variables

variable "domain_name" {
  description = "Primary domain name for the application"
  type        = string
}

variable "subject_alternative_names" {
  description = "Additional domain names for the SSL certificate"
  type        = list(string)
  default     = []
}

variable "route53_zone_id" {
  description = "Route53 hosted zone ID for the domain"
  type        = string
}

variable "create_wildcard_cert" {
  description = "Create a wildcard SSL certificate"
  type        = bool
  default     = false
}

variable "certificate_alarm_actions" {
  description = "SNS topic ARNs for certificate expiry alarms"
  type        = list(string)
  default     = []
}

variable "custom_domains" {
  description = "List of custom domain names for CloudFront"
  type        = list(string)
  default     = []
}

variable "api_gateway_domain" {
  description = "API Gateway custom domain name"
  type        = string
  default     = ""
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
    error_message = "CloudFront price class must be PriceClass_All, PriceClass_200, or PriceClass_100."
  }
}

variable "geo_restriction_type" {
  description = "Type of geographic restriction"
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

variable "cloudfront_logging_bucket" {
  description = "S3 bucket for CloudFront access logs"
  type        = string
  default     = ""
}

variable "waf_web_acl_arn" {
  description = "WAF Web ACL ARN to associate with CloudFront"
  type        = string
  default     = ""
}

variable "cors_allowed_origins" {
  description = "List of allowed origins for CORS"
  type        = list(string)
  default     = ["*"]
}

variable "api_subdomain_enabled" {
  description = "Enable API subdomain"
  type        = bool
  default     = true
}

variable "media_subdomain_enabled" {
  description = "Enable media subdomain"
  type        = bool
  default     = true
}

variable "cdn_subdomain_enabled" {
  description = "Enable CDN subdomain"
  type        = bool
  default     = true
}

variable "additional_subdomains" {
  description = "Map of additional subdomains to create"
  type        = map(string)
  default     = {}
}

variable "enable_health_check" {
  description = "Enable Route53 health check"
  type        = bool
  default     = true
}

variable "health_check_path" {
  description = "Path for health check"
  type        = string
  default     = "/health"
}

variable "health_check_failure_threshold" {
  description = "Number of consecutive failures before marking unhealthy"
  type        = number
  default     = 3
}

variable "health_check_alarm_actions" {
  description = "SNS topic ARNs for health check alarms"
  type        = list(string)
  default     = []
}

variable "mx_records" {
  description = "List of MX records for email"
  type        = list(string)
  default     = []
}

variable "spf_record" {
  description = "SPF record for email security"
  type        = string
  default     = ""
}

variable "dmarc_record" {
  description = "DMARC record for email security"
  type        = string
  default     = ""
}

variable "caa_records" {
  description = "List of CAA records for certificate authority authorization"
  type        = list(string)
  default     = [
    "0 issue \"amazon.com\"",
    "0 issue \"amazontrust.com\"",
    "0 issue \"awstrust.com\"",
    "0 issue \"amazonaws.com\""
  ]
}

variable "domain_verification_records" {
  description = "Map of domain verification TXT records"
  type        = map(string)
  default     = {}
}

variable "cloudfront_alarm_actions" {
  description = "SNS topic ARNs for CloudFront alarms"
  type        = list(string)
  default     = []
}

# Environment-specific defaults
locals {
  environment_defaults = {
    production = {
      cloudfront_price_class = "PriceClass_All"
      enable_health_check   = true
      create_wildcard_cert  = true
    }
    staging = {
      cloudfront_price_class = "PriceClass_100"
      enable_health_check   = true
      create_wildcard_cert  = false
    }
    development = {
      cloudfront_price_class = "PriceClass_100"
      enable_health_check   = false
      create_wildcard_cert  = false
    }
  }
  
  current_defaults = local.environment_defaults[var.environment]
}